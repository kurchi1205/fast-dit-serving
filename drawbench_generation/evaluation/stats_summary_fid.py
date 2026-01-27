import json
import numpy as np
import argparse

def compute_fid_statistics(json_path):
    with open(json_path, "r") as f:
        fid_data = json.load(f)

    interval_scores = {}
    for prompt, interval_dict in fid_data.items():
        for interval_str, score in interval_dict.items():
            if score is not None:
                interval = int(interval_str)
                interval_scores.setdefault(interval, []).append(score)

    print("\n=== FID Score Statistics per Cache Interval ===")
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
              f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, "
              f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

    # Overall stats
    all_scores = [score for scores in interval_scores.values() for score in scores]
    print("\n=== Overall FID Statistics ===")
    print(f"Total Prompts: {len(all_scores)}")
    print(f"Mean FID: {np.mean(all_scores):.2f}")
    print(f"Median FID: {np.median(all_scores):.2f}")
    print(f"Min FID: {np.min(all_scores):.2f}")
    print(f"Max FID: {np.max(all_scores):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute summary statistics from FID scores JSON.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to fid_scores_by_prompt.json")
    args = parser.parse_args()

    compute_fid_statistics(args.json_path)
