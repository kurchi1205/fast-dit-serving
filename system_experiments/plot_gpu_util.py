import json
import matplotlib.pyplot as plt
from datetime import datetime
import collections
import argparse

def plot_gpu_memory(log_file, save_path=None):
    """
    Plot GPU memory usage (in GB) vs elapsed time (seconds).

    Args:
        log_file (str): Path to the JSON log file.
        save_path (str, optional): If provided, saves the plot to this file.
        show (bool): Whether to display the plot interactively.
    """
    gpu_data = collections.defaultdict(lambda: {"t": [], "mem": []})

    # --- Load JSON lines ---
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ts = record.get("timestamp")
                if not ts:
                    continue
                ts = datetime.fromisoformat(ts)
                for gpu in record.get("gpus", []):
                    gpu_data[gpu["gpu"]]["t"].append(ts)
                    gpu_data[gpu["gpu"]]["mem"].append(gpu["memory_used_MB"] / 1024)  # MB → GB
            except json.JSONDecodeError:
                continue

    if not gpu_data:
        print("No valid GPU data found in log.")
        return

    # --- Convert timestamps to elapsed seconds ---
    for gpu_id, data in gpu_data.items():
        start_time = data["t"][0]
        data["elapsed_sec"] = [(t - start_time).total_seconds() for t in data["t"]]
        # filtered_indices = [
        #     i for i, t in enumerate(data["elapsed_sec"]) if 0 <= t <= 1000
        # ]
        # data["elapsed_sec"] = [data["elapsed_sec"][i] for i in filtered_indices]
        # data["mem"] = [data["mem"][i] for i in filtered_indices]


    # --- Plot ---
    plt.figure(figsize=(18, 6))
    colors = plt.cm.tab10.colors

    for i, (gpu_id, data) in enumerate(gpu_data.items()):
        plt.plot(
            data["elapsed_sec"], data["mem"],
            label=f"GPU {gpu_id}",
            color=colors[i % len(colors)],
            alpha=0.8
        )

    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("Memory Used (GB)")
    plt.title("GPU Memory Usage Over Time")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Plot saved to: {save_path}")


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPU memory usage vs time.")
    parser.add_argument("--log_file", required=True, help="Path to the GPU utilization JSON log.")
    parser.add_argument("--output", default=None, help="Optional path to save the output plot (e.g., gpu_memory.png).")
    args = parser.parse_args()
    plot_gpu_memory(args.log_file, save_path=args.output)
