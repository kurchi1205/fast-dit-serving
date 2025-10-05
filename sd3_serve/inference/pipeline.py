import os
import torch
import math
import numpy as np
from PIL import Image
import sys
from inference.tokenizer import SD3Tokenizer
from inference.text_encoder import T5XXL, ClipL, ClipG
from inference.sd3 import SD3, SD3LatentFormat, SkipLayerCFGDenoiser, CFGDenoiser
from inference.vae import VAE
from tqdm import tqdm
import time
PROFILE_GPU = os.getenv("PROFILE_GPU", "false").lower() == "true"

class SD3Inferencer:

    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model="models/sd3_medium.safetensors",
        vae=None,
        shift=3.0,
        controlnet_ckpt=None,
        model_folder: str = "models",
        text_encoder_device: str = "cpu",
        device: str = "cuda",
        verbose=False,
        load_tokenizers: bool = True,
        custom_scheduler = True
    ):
        self.verbose = verbose
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        if load_tokenizers:
            print("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, text_encoder_device, torch.float32)
            print("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder, device=text_encoder_device)
            print("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder, text_encoder_device)
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, controlnet_ckpt, verbose, device, custom_scheduler)
        print("Loading VAE model...")
        self.vae = VAE(vae or model, device=device)
        print("Models loaded.")
        self.custom_scheduler = custom_scheduler
        self.device = device

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        # print(l_out.size())
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().to(device=self.device), cond[1].half().to(device=self.device))
        return {"c_crossattn": cond, "y": pooled}


    def prepare_for_first_timestep(self, empty_latent, prompt, neg_cond, steps, seed_type="rand", seed=None):
        controlnet_cond = None
        if seed is None:
            seed = 50
        latent = empty_latent.to(device=self.device)
        seed_num = None
        if seed_type == "roll":
            seed_num = seed if seed_num is None else seed_num + 1
        elif seed_type == "rand":
            seed_num = torch.randint(0, 100000, (1,)).item()
        else:  # fixed
            seed_num = seed
        if PROFILE_GPU:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            conditioning = self.get_cond(prompt)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
        else:
            conditioning = self.get_cond(prompt)
        latent = latent.half().to(device=self.device)
        self.sd3.model = self.sd3.model.to(device=self.device)
        noise = self.get_noise(seed_num, latent).to(device=self.device)
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).to(device=self.device)
        conditioning = self.fix_cond(conditioning)
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        if not PROFILE_GPU:
            return noise_scaled, sigmas, conditioning, neg_cond, seed_num
        return noise_scaled, sigmas, conditioning, neg_cond, seed_num, elapsed_time_ms / 1000


    def denoise_each_step_1(self, model, x, sigma, prev_sigma, next_sigma, old_denoised, extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        denoised = model(x, sigma * s_in, **extra_args)
        t, t_next = t_fn(sigma), t_fn(next_sigma)
        h = t_next - t
        
        if old_denoised is None or next_sigma == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(prev_sigma)
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        return x, old_denoised


    def denoise_each_step(self, model, x, sigma, prev_sigma, next_sigma, old_denoised, extra_args=None):
        # st = time.time()
        extra_args = extra_args if extra_args is not None else {}
        s_in = x.new_ones([x.shape[0]])  # Ensure size matches batch size in x

        # Functions to compute sigma and time values
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        # Model denoising step
        adaptive_sigma = sigma
        if self.custom_scheduler:
            adaptive_sigma, x = model.model.model_sampling.sigma_from_latent(x, sigma)
        # adaptive_sigma = adaptive_sigma.view(-1, 1, 1, 1)
        if PROFILE_GPU:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            denoised = model(x, adaptive_sigma * s_in, sigma * s_in, **extra_args)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
        else:
            denoised = model(x, adaptive_sigma * s_in, sigma * s_in, **extra_args)
        # print("Denoised: ", time.time() - st)
        # print("Just Denoising: ", time.time() - st_2)
        # Compute time values
        t = t_fn(sigma)
        t_next = t_fn(next_sigma)

        # Compute the exponential moving average components
        h = t_next - t
        sigma_ratio = (sigma_fn(t_next) / sigma_fn(t)).view(-1, 1, 1, 1)
        expm1_h = (-h).expm1().view(-1, 1, 1, 1)

        # Calculate the ratio for blending denoised versions if applicable
        
        if old_denoised is not None:
            h_last = t - t_fn(prev_sigma)
            r = (h_last / h).view(-1, 1, 1, 1)
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        else:
            denoised_d = denoised  # Fallback to current denoised if old_denoised is None

        # Compute the new x with conditional checks integrated
        x = torch.where(next_sigma.view(-1, 1, 1, 1) == 0,
                        sigma_ratio * x - expm1_h * denoised,
                        sigma_ratio * x - expm1_h * denoised_d)
        if not PROFILE_GPU:
            return x, denoised
        return x, denoised, elapsed_time_ms/1000



    def vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).to(device=self.device)
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.to(device=self.device)
        self.vae.model = self.vae.model.to(device=self.device)
        latent = self.vae.model.encode(image_torch).to(device=self.device)
        self.vae.model = self.vae.model.to(device=self.device)
        self.print("Encoded")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        image = self.vae.model.decode(latent)
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.detach().cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        del image 
        self.print("Decoded")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent
