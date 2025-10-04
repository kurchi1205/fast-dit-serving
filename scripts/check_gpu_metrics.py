import pynvml
import time
import csv
from datetime import datetime

pynvml.nvmlInit()
n_gpus = pynvml.nvmlDeviceGetCount()

def log_gpu_utilization():
    with open("gpu_utilization_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gpu", "utilization_percent", "memory_used_MB", "memory_total_MB"])

        try:
            while True:
                timestamp = datetime.now().isoformat()
                for i in range(n_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    writer.writerow([timestamp, i, util.gpu, mem.used / 1024**2, mem.total / 1024**2])
                f.flush()
                time.sleep(1)   # sample every second
        except KeyboardInterrupt:
            print("Stopped logging.")


if __name__ == "__main__":
    log_gpu_utilization()


