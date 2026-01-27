import os
import json
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_fid_scores(src_dir, tgt_dir, cache_intervals, output_path):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid_scores = {}

    src_images = list(src_dir.glob("p2_prompt_*.png"))
    all_prompt_keys = sorted([img.stem for img in src_images])  # e.g., 'p2_prompt_0'

    for interval in cache_intervals:
        print(f"\n=== Cache Interval: {interval} ===")
        fid_scores[interval] = {}

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
                fid_metric = FrechetInceptionDistance(feature=64, input_img_size=(3, 1024, 1024)).to(device)

                # Source image
                src_img = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
                src_img = (src_img * 255).byte()
                fid_metric.update(torch.cat([src_img, src_img], dim=0), real=True)

                # Target image
                tgt_img = transform(Image.open(tgt_path).convert("RGB")).unsqueeze(0).to(device)
                tgt_img = (tgt_img * 255).byte()
                fid_metric.update(torch.cat([tgt_img, tgt_img], dim=0), real=False)

                fid_score = fid_metric.compute().item()
                fid_scores[interval][key] = fid_score
                print(f"{key} (cache_{interval}) → FID: {fid_score:.2f}")

            except Exception as e:
                print(f"Error computing FID for {key} (cache_{interval}): {e}")
                fid_scores[interval][key] = None

    # Reorganize results: prompt -> interval -> score
    final_scores = {}
    for interval, results in fid_scores.items():
        for prompt, score in results.items():
            if prompt not in final_scores:
                final_scores[prompt] = {}
            final_scores[prompt][str(interval)] = score

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_scores, f, indent=2)

    print(f"\nFID scores saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID scores for image prompts across caching intervals.")
    parser.add_argument("--src_dir", required=True, help="Path to source image directory (originals)")
    parser.add_argument("--tgt_dir", required=True, help="Path to target image directory (generated)")
    parser.add_argument("--output", default="fid_scores_by_prompt.json", help="Path to save the FID scores JSON")
    parser.add_argument("--intervals", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6], help="List of cache intervals")

    args = parser.parse_args()
    compute_fid_scores(args.src_dir, args.tgt_dir, args.intervals, args.output)
