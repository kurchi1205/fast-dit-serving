LOG_FILE="outputs/gpu_burst_utilization_log_3_gpus.json"
OUTPUT_IMAGE="outputs/gpu_memory_plot_load_3_gpus.png"

# Run the Python plotter
python3 plot_gpu_util.py \
  --log_file "$LOG_FILE" \
  --output "$OUTPUT_IMAGE"