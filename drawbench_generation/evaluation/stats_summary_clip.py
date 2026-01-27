import json
import argparse
import statistics
from collections import defaultdict

def compute_clip_score_stats(json_path):
    with open(json_path, "r") as f:
        scores = json.load(f)

    grouped = defaultdict(list)

    for full_key, score in scores.items():
        if score is None:
            continue
        if "_cache_" in full_key:
            base_key = full_key.split("_cache_")[0]
        else:
            base_key = full_key
        grouped[base_key].append(score)

    overall_scores = []

    print("\n=== Per-Prompt Stats ===")
    for key, values in sorted(grouped.items()):
        mean_val = round(statistics.mean(values), 4)
        min_val = round(min(values), 4)
        max_val = round(max(values), 4)
        std_val = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        overall_scores.extend(values)

        print(f"{key}: mean={mean_val}, std={std_val}, min={min_val}, max={max_val}, n={len(values)}")

    # Overall stats
    print("\n=== Overall Stats ===")
    if overall_scores:
        print(f"Total samples: {len(overall_scores)}")
        print(f"Mean: {round(statistics.mean(overall_scores), 4)}")
        print(f"Std Dev: {round(statistics.stdev(overall_scores), 4)}")
        print(f"Min: {round(min(overall_scores), 4)}")
        print(f"Max: {round(max(overall_scores), 4)}")
    else:
        print("No valid scores found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics from a CLIP score JSON.")
    parser.add_argument("--json_path", required=True, help="Path to the clip_scores.json file")
    args = parser.parse_args()

    compute_clip_score_stats(args.json_path)
