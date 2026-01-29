import os
import fire
import json
import time
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from pipeline import SD3Inferencer


def batch_inference(
    prompt_json_path,
    model_path="/home/fast-dit-serving/sd3_model/sd3_medium.safetensors",
    model_folder="/home/fast-dit-serving/sd3_model/",
    output_dir="outputs",
    width=1024,
    height=1024,
    steps=30,
    cfg_scale=5.0,
    seed=42,
    sampler="dpmpp_2m"
):
    """
    Run SD3 inference for each prompt in a JSON file and save output images with key-named files.

    Args:
        prompt_json_path (str): Path to JSON file with {key: prompt} entries.
    """
    # Load prompts
    with open(prompt_json_path, "r") as f:
        prompts_dict = json.load(f)

    # with open(challenge_json_path, "r") as f:
    #     challenge_dict = json.load(f)

    # # Group prompts by challenge type
    # challenge_groups = defaultdict(list)
    # for key, prompt in prompts_dict.items():
    #     challenge_key = f"{key}_challenge"
    #     challenge = challenge_dict.get(challenge_key)
    #     if challenge:
    #         challenge_groups[challenge].append((key, prompt))

    # # Keep only first 20 prompts per challenge
    # filtered_prompts = {}
    # for challenge, entries in challenge_groups.items():
    #     for key, prompt in entries[:20]:
    #         filtered_prompts[key] = prompt

    # print(f"🧠 Total prompts selected: {len(filtered_prompts)}")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize and load model
    inferencer = SD3Inferencer()
    inferencer.load(
        model=model_path,
        model_folder=model_folder,
        text_encoder_device="cuda",
        verbose=True,
        shift=5
    )

    for key, prompt in tqdm(prompts_dict.items()):
        # if i == 10:
        #     break
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
        if images:
            output_path = os.path.join(output_dir, f"{key}.png")
            images[0].save(output_path)
        else:
            print(f"Failed to generate image for {key}")

if __name__ == "__main__":
    fire.Fire(batch_inference)
