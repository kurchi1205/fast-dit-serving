import torch
import json
import time
from collections import defaultdict
from pathlib import Path
from diffusers import StableDiffusion3Pipeline



def load_pipe():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    return pipe

def compile_pipe(pipe):
    pipe.vae = torch.compile(pipe.vae, fullgraph=True, mode="reduce-overhead")
    pipe.transformer = torch.compile(pipe.transformer, fullgraph=False, mode="reduce-overhead")
    return pipe

def infer(pipe, prompt, num_inference_steps, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        generator=generator
    ).images[0]
    return image

def run_batch_inference(prompt_json_path, output_dir, num_inference_steps=28, compile=True, seed=42):
    with open(prompt_json_path, "r") as f:
        prompts = json.load(f)

    # with open(challenge_json_path, "r") as f:
    #     challenges_raw = json.load(f)

    # # Map base prompt key to challenge
    # challenges = {
    #     key.replace("_challenge", ""): value
    #     for key, value in challenges_raw.items()
    #     if key.endswith("_challenge")
    # }

    # # Group keys by challenge type
    # grouped = defaultdict(list)
    # for key, challenge in challenges.items():
    #     if key in prompts:
    #         grouped[challenge].append(key)

    # # Select first 20 prompts per challenge type
    # selected_keys = []
    # for challenge_type, keys in grouped.items():
    #     selected_keys.extend(keys[:20])

    # print(f"Selected {len(selected_keys)} prompts (first 20 from each challenge type)")

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and optionally compile pipeline
    pipe = load_pipe()
    if compile:
        pipe = compile_pipe(pipe)

    # Run inference
    for key, prompt in prompts.items():
        if prompt:
            image = infer(pipe, prompt, num_inference_steps, seed)
            image.save(output_dir / f"{key}.jpeg")


if __name__ == "__main__":
    import fire
    fire.Fire(run_batch_inference)
