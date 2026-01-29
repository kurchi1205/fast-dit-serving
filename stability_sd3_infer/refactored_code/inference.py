import os
import fire
import time
from PIL import Image
from pipeline import SD3Inferencer

# Assuming the SD3Inferencer and necessary classes are already imported

def run_inference(prompt, model_path="../../sd3_model/sd3_medium.safetensors", model_folder="../../sd3_model/", output_dir="outputs", width=1024, height=1024, steps=30, cfg_scale=5.0, seed=42, sampler="dpmpp_2m"):
    """
    Run SD3 model inference to generate an image based on the provided prompt.

    Args:
        prompt (str): Text prompt for image generation.
        model_path (str): Path to the SD3 model checkpoint.
        vae_path (str): Path to the VAE model checkpoint.
        model_folder (str): Path to the folder containing text encoder models.
        output_dir (str): Directory to save the generated image. Default is "outputs".
        width (int): Width of the generated image. Default is 512.
        height (int): Height of the generated image. Default is 512.
        steps (int): Number of denoising steps. Default is 50.
        cfg_scale (float): Guidance scale for CFG. Default is 7.5.
        seed (int): Seed for random number generation. Default is 42.
        sampler (str): Sampler type. Default is "dpmpp_2m".
    """
    # Initialize the inferencer
    inferencer = SD3Inferencer()

    # Load the model
    inferencer.load(
        model=model_path,
        model_folder=model_folder,
        text_encoder_device="cuda",
        verbose=True,
        shift=5
    )

    # Generate image
    print("Generating image...")
    st = time.time()
    images = inferencer.gen_image(
        prompts=[prompt],
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        seed_type="fixed",
        sampler=sampler
    )
    print("Time taken:", time.time() - st)
    # Save and display the generated image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_image.png")

    if images:
        images[0].save(output_path)
        print(f"Image saved to {output_path}")
        images[0].show()  # Open the generated image
    else:
        print("No image was generated.")

if __name__ == "__main__":
    fire.Fire(run_inference)
