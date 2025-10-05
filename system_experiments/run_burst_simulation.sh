PROMPT_FILE="/home/fast-dit-serving/partiprompts_generation/parti_prompts.json"
COMPLETED_LOG="/home/fast-dit-serving/system_experiments/outputs/completed_requests_burst.json"
HOST="http://localhost:8000"

SAVE_INTERVAL=5        # Save completed requests every N results
BURST_SIZE=20          # Number of requests in each burst
BURST_INTERVAL=10      # Seconds between bursts
DURATION=100           # Total duration (seconds)
MODE="burst"           # Use burst mode

python simulate_load.py \
  --prompt_file "$PROMPT_FILE" \
  --completed_log "$COMPLETED_LOG" \
  --host "$HOST" \
  --save_interval $SAVE_INTERVAL \
  --mode $MODE \
  --burst_size $BURST_SIZE \
  --burst_interval $BURST_INTERVAL \
  --duration $DURATION

echo "Burst load simulation completed."
