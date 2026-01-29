
LOG_DIR="outputs"
LOG_FILE="$LOG_DIR/gpu_burst_utilization_log_3_gpus_cache_0.json"
DUMP_INTERVAL=5     # seconds between dumps
SAMPLE_INTERVAL=1   # seconds between samples

mkdir -p "$LOG_DIR"


python check_gpu_metrics.py --log_file $LOG_FILE --dump_interval "$DUMP_INTERVAL" --sample_interval "$SAMPLE_INTERVAL"