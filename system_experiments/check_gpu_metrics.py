import pynvml
import time
import csv
from datetime import datetime

import pynvml
import time
import json
from datetime import datetime
import argparse

pynvml.nvmlInit()
n_gpus = pynvml.nvmlDeviceGetCount()


def log_gpu_utilization(log_file, dump_interval, sample_interval):
    all_records = []
    try:
        while True:
            batch = []
            for _ in range(dump_interval):  # collect multiple samples before writing
                timestamp = datetime.now().isoformat()
                snapshot = {"timestamp": timestamp, "gpus": []}
                for i in range(n_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    snapshot["gpus"].append({
                        "gpu": i,
                        "utilization_percent": util.gpu,
                        "memory_used_MB": round(mem.used / 1024**2, 2),
                        "memory_total_MB": round(mem.total / 1024**2, 2)
                    })
                batch.append(snapshot)
                time.sleep(sample_interval)

            # append batch to file periodically
            with open(log_file, "a") as f:
                for record in batch:
                    f.write(json.dumps(record) + "\n")

            print(f"[{datetime.now().isoformat()}] Dumped {len(batch)} samples to {log_file}")

    except KeyboardInterrupt:
        print("Stopped GPU monitoring.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPU memory usage vs time.")
    parser.add_argument("--log_file", required=True, help="Path to the GPU utilization JSON log.")
    parser.add_argument(
        "--dump_interval",
        type=int,
        default=5,
        help="Number of seconds between writing batches of GPU samples to the log file (default: 5)."
    )

    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Sampling interval in seconds between GPU utilization checks (default: 1)."
    )

    args = parser.parse_args()
    log_gpu_utilization(args.log_file, args.dump_interval, args.sample_interval)


