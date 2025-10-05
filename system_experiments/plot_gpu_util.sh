LOG_FILE="outputs/gpu_burst_utilizatio_log.json"
OUTPUT_IMAGE="outputs/gpu_memory_plot.png"

# Run the Python plotter
python3 plot_gpu_util.py \
  --log_file "$LOG_FILE" \
  --output "$OUTPUT_IMAGE"