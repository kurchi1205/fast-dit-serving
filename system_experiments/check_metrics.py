import json
from datetime import datetime
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict


def parse_time(t):
    """Parses ISO formatted timestamp into a datetime object."""
    return datetime.fromisoformat(t)


def safe_stats(data):
    """Calculates average, p95, and max safely for a list of values."""
    if not data:
        return {"avg": None, "p95": None, "max": None}
    return {
        "avg": round(float(np.mean(data)), 3),
        "p95": round(float(np.percentile(data, 95)), 3),
        "max": round(float(np.max(data)), 3),
    }


def merged_active_duration(start_times, end_times):
    """Compute total active duration accounting for overlaps."""
    intervals = sorted(
        [(s, e) for s, e in zip(start_times, end_times) if e > s],
        key=lambda x: x[0]
    )
    if not intervals:
        return 0.0

    merged = []
    current_start, current_end = intervals[0]

    for s, e in intervals[1:]:
        if s <= current_end:
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    total_active = sum((e - s).total_seconds() for s, e in merged)
    return total_active


def build_metric_block(total_latencies, queue_latencies, inference_latencies, gpu_latencies,
                       all_start_times, all_end_times, failed_requests_count):
    # Time span & throughput (per the window covered by these requests)
    if all_end_times:
        time_span = merged_active_duration(all_start_times, all_end_times)
        throughput_rps = len(all_end_times) / time_span if time_span > 0 else 0.0
    else:
        time_span, throughput_rps = 0.0, 0.0

    total_gpu_minutes = sum(gpu_latencies) / 60.0
    throughput_per_gpu_min = (len(all_end_times) / total_gpu_minutes) if total_gpu_minutes > 0 else 0.0

    return {
        "latency": {
            "total": safe_stats(total_latencies),
            "queue": safe_stats(queue_latencies),
            "inference": safe_stats(inference_latencies),
            "gpu_inference": safe_stats(gpu_latencies),
        },
        "throughput": {
            "requests_per_second": round(throughput_rps, 3),
            "gpu_minutes": round(total_gpu_minutes, 3),
            "requests_per_gpu_minute": round(throughput_per_gpu_min, 3),
            "time_span_sec": round(time_span, 2),
            "failed_requests": failed_requests_count,
            "total_completed_requests": len(all_end_times),
        },
    }


def save_latency_throughput_metrics(
    overall_block,
    per_gpu_blocks,
    output_file
):
    """Saves calculated latency and throughput metrics to a JSON file."""
    metrics = {
        "overall": overall_block,
        "per_gpu": per_gpu_blocks,  # dict keyed by gpu_id
    }

    try:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_file}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")


def analyze_requests(requests, output_file):
    """
    Analyzes latency and throughput metrics overall and per GPU from a list of request logs.
    """
    # Overall accumulators
    total_latencies, queue_latencies, inference_latencies, gpu_latencies = [], [], [], []
    all_start_times, all_end_times = [], []
    failed_requests_count_overall = 0

    # Per-GPU accumulators
    per_gpu = defaultdict(lambda: {
        "total_latencies": [],
        "queue_latencies": [],
        "inference_latencies": [],
        "gpu_latencies": [],
        "all_start_times": [],
        "all_end_times": [],
        "failed_requests": 0,
    })

    for req in requests:
        gpu_id = req.get("gpu_id", "unknown")

        status = str(req.get("status", "")).lower()
        if status == "failed":
            failed_requests_count_overall += 1
            per_gpu[gpu_id]["failed_requests"] += 1
            # Skip latency calculations for failed requests
            continue

        try:
            t_add = parse_time(req["timestamp"])
            t_start = parse_time(req["processing_time_start"])
            t_end = parse_time(req["time_completed"])

            tot = (t_end - t_add).total_seconds()
            que = (t_start - t_add).total_seconds()
            inf = (t_end - t_start).total_seconds()

            # GPU time from request if present, fallback to inference time
            gpu_time = float(req.get("elapsed_gpu_time", inf))

            # Overall
            total_latencies.append(tot)
            queue_latencies.append(que)
            inference_latencies.append(inf)
            gpu_latencies.append(gpu_time)
            all_start_times.append(t_add)
            all_end_times.append(t_end)

            # Per GPU
            g = per_gpu[gpu_id]
            g["total_latencies"].append(tot)
            g["queue_latencies"].append(que)
            g["inference_latencies"].append(inf)
            g["gpu_latencies"].append(gpu_time)
            g["all_start_times"].append(t_add)
            g["all_end_times"].append(t_end)

        except Exception as e:
            print(f"Skipping request {req.get('request_id', '?')} due to error: {e}")
            continue

    # Build overall block
    overall_block = build_metric_block(
        total_latencies,
        queue_latencies,
        inference_latencies,
        gpu_latencies,
        all_start_times,
        all_end_times,
        failed_requests_count_overall
    )

    # Build per-GPU blocks
    per_gpu_blocks = {}
    for gid, acc in per_gpu.items():
        block = build_metric_block(
            acc["total_latencies"],
            acc["queue_latencies"],
            acc["inference_latencies"],
            acc["gpu_latencies"],
            acc["all_start_times"],
            acc["all_end_times"],
            acc["failed_requests"]
        )
        per_gpu_blocks[str(gid)] = block

    # Save
    save_latency_throughput_metrics(overall_block, per_gpu_blocks, output_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze latency and throughput metrics (overall and per GPU) from completed requests.")
    parser.add_argument("--log_file", required=True, help="Path to the completed requests JSON file.")
    parser.add_argument("--output_file", default="latency_metrics.json", help="Path to save the metrics output JSON.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    try:
        with open(log_path, "r") as f:
            data = json.load(f)
            requests = data.get("completed_requests", data)
    except Exception as e:
        print(f"Failed to read log file: {e}")
        sys.exit(1)

    analyze_requests(requests, args.output_file)


if __name__ == "__main__":
    main()
