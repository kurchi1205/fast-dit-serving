LOG_FILE="outputs/gpu_load_2_utilization_log.json"
OUTPUT_IMAGE="outputs/gpu_memory_plot_load_2.png"

# Run the Python plotter
python3 plot_gpu_util.py \
  --log_file "$LOG_FILE" \
  --output "$OUTPUT_IMAGE"