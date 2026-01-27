import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.functional.multimodal import clip_score
from functools import partial

# Set up CLIP score function
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(image_tensor, prompts):
    images_uint8 = (image_tensor * 255).clamp(0, 255).to(torch.uint8)
    score = clip_score_fn(images_uint8, prompts).detach()
    return round(float(score), 4)


def compute_clip_scores(image_dir, prompts_path, output_path, cache_interval=None):
    image_dir = Path(image_dir)

    # Load prompts
    with open(prompts_path, "r") as f:
        prompt_dict = json.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

    clip_scores = {}

    if cache_interval is None:
        # Calculate CLIP scores for all images in the folder
        for img_path in sorted(image_dir.glob("*.png")):
            filename = img_path.stem  # e.g., 'p2_prompt_0_cache_2'
            
            # Extract the key by removing '_cache*' from filename
            if "_cache_" in filename:
                key = filename.split("_cache_")[0]
            else:
                key = filename

            prompt_text = prompt_dict.get(key)
            if prompt_text is None:
                print(f"Prompt not found for key: {key}")
                clip_scores[filename] = None
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

                score = calculate_clip_score(image_tensor, [prompt_text])
                clip_scores[filename] = score
                print(f"{filename} → CLIP score: {score:.4f}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                clip_scores[filename] = None
    else:
        # Calculate CLIP scores for images with specific cache interval
        for img_path in sorted(image_dir.glob(f"*_cache_{cache_interval}.png")):
            filename = img_path.stem  # e.g., 'p2_prompt_0_cache_2'
            
            # Extract the key by removing '_cache*' from filename
            if "_cache_" in filename:
                key = filename.split("_cache_")[0]
            else:
                key = filename

            prompt_text = prompt_dict.get(key)
            if prompt_text is None:
                print(f"Prompt not found for key: {key}")
                clip_scores[filename] = None
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

                score = calculate_clip_score(image_tensor, [prompt_text])
                clip_scores[filename] = score
                print(f"{filename} → CLIP score: {score:.4f}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                clip_scores[filename] = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clip_scores, f, indent=2)

    print(f"\nCLIP scores saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP scores for images using prompts.")
    parser.add_argument("--image_dir", required=True, help="Directory with images (e.g., image_cache_*.png)")
    parser.add_argument("--prompts", required=True, help="Path to prompts.json")
    parser.add_argument("--output", default="clip_scores.json", help="Path to save output JSON")
    parser.add_argument("--cache_interval", type=int, help="Cache interval to filter images (optional)")

    args = parser.parse_args()
    compute_clip_scores(args.image_dir, args.prompts, args.output, args.cache_interval)
