import json
import numpy as np
import argparse

def compute_ssim_statistics(json_path):
    with open(json_path, "r") as f:
        ssim_data = json.load(f)

    interval_scores = {}
    for prompt, interval_dict in ssim_data.items():
        for interval_str, score in interval_dict.items():
            if score is not None:
                interval = int(interval_str)
                interval_scores.setdefault(interval, []).append(score)

    print("\n=== SSIM Score Statistics per Cache Interval ===")
    for interval in sorted(interval_scores):
        scores = interval_scores[interval]
        stats = {
            "count": len(scores),
            "mean": np.mean(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores)
        }
        print(f"[cache_{interval}] -> Count: {stats['count']}, "
              f"Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
              f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")

    # Overall stats
    all_scores = [score for scores in interval_scores.values() for score in scores]
    print("\n=== Overall SSIM Statistics ===")
    print(f"Total Prompts: {len(all_scores)}")
    print(f"Mean SSIM: {np.mean(all_scores):.4f}")
    print(f"Median SSIM: {np.median(all_scores):.4f}")
    print(f"Min SSIM: {np.min(all_scores):.4f}")
    print(f"Max SSIM: {np.max(all_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute summary statistics from SSIM scores JSON.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to ssim_scores_by_prompt.json")
    args = parser.parse_args()

    compute_ssim_statistics(args.json_path)
