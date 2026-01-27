import os
import json
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def compute_ssim_scores(src_dir, tgt_dir, cache_intervals, output_path):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssim_scores = {}

    src_images = list(src_dir.glob("p2_prompt_*.png"))
    all_prompt_keys = sorted([img.stem for img in src_images])  # e.g., 'p2_prompt_0'

    for interval in cache_intervals:
        print(f"\n=== Cache Interval: {interval} ===")
        ssim_scores[interval] = {}

        for key in all_prompt_keys:
            src_path = src_dir / f"{key}.png"
            tgt_path = tgt_dir / f"{key}_cache_{interval}.png"

            if not src_path.exists():
                print(f"Missing source: {src_path}")
                continue
            if not tgt_path.exists():
                print(f"Missing target: {tgt_path}")
                continue

            try:
                ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

                # Load and transform images
                src_img = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
                tgt_img = transform(Image.open(tgt_path).convert("RGB")).unsqueeze(0).to(device)

                # Ensure same shape
                if src_img.shape != tgt_img.shape:
                    print(f"Image shape mismatch for {key}, skipping.")
                    continue

                score = ssim_metric(src_img, tgt_img).item()
                ssim_scores[interval][key] = score
                print(f"{key} (cache_{interval}) → SSIM: {score:.4f}")

            except Exception as e:
                print(f"Error computing SSIM for {key} (cache_{interval}): {e}")
                ssim_scores[interval][key] = None

    # Reorganize results: prompt -> interval -> score
    final_scores = {}
    for interval, results in ssim_scores.items():
        for prompt, score in results.items():
            if prompt not in final_scores:
                final_scores[prompt] = {}
            final_scores[prompt][str(interval)] = score

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_scores, f, indent=2)

    print(f"\nSSIM scores saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSIM scores for image prompts across caching intervals.")
    parser.add_argument("--src_dir", required=True, help="Path to source image directory (originals)")
    parser.add_argument("--tgt_dir", required=True, help="Path to target image directory (generated)")
    parser.add_argument("--output", default="ssim_scores_by_prompt.json", help="Path to save the SSIM scores JSON")
    parser.add_argument("--intervals", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6], help="List of cache intervals")

    args = parser.parse_args()
    compute_ssim_scores(args.src_dir, args.tgt_dir, args.intervals, args.output)
