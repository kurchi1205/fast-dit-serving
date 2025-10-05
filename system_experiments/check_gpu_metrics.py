import pynvml
import time
import csv
from datetime import datetime

import pynvml
import time
import json
from datetime import datetime

pynvml.nvmlInit()
n_gpus = pynvml.nvmlDeviceGetCount()

LOG_FILE = "outputs/gpu_burst_utilization_log.json"
DUMP_INTERVAL = 5     # seconds between dumps
SAMPLE_INTERVAL = 1   # seconds between samples


def log_gpu_utilization():
    all_records = []
    try:
        while True:
            batch = []
            for _ in range(DUMP_INTERVAL):  # collect multiple samples before writing
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
                time.sleep(SAMPLE_INTERVAL)

            # append batch to file periodically
            with open(LOG_FILE, "a") as f:
                for record in batch:
                    f.write(json.dumps(record) + "\n")

            print(f"[{datetime.now().isoformat()}] Dumped {len(batch)} samples to {LOG_FILE}")

    except KeyboardInterrupt:
        print("Stopped GPU monitoring.")

# pynvml.nvmlInit()
# n_gpus = pynvml.nvmlDeviceGetCount()

# def log_gpu_utilization():
#     with open("gpu_utilization_log.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["timestamp", "gpu", "utilization_percent", "memory_used_MB", "memory_total_MB"])

#         try:
#             while True:
#                 timestamp = datetime.now().isoformat()
#                 for i in range(n_gpus):
#                     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#                     util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#                     mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
#                     writer.writerow([timestamp, i, util.gpu, mem.used / 1024**2, mem.total / 1024**2])
#                 f.flush()
#                 time.sleep(1)   # sample every second
#         except KeyboardInterrupt:
#             print("Stopped logging.")


if __name__ == "__main__":
    log_gpu_utilization()


