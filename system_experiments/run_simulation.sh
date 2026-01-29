#!/bin/bash

PROMPT_FILE="/home/fast-dit-serving/partiprompts_generation/parti_prompts.json"
COMPLETED_LOG="/home/fast-dit-serving/system_experiments/outputs/completed_requests_rr_1_sec_100_a100_3.json"
HOST="http://localhost:8000"
SAVE_INTERVAL=1
RATE=1              # requests per second
DURATION=100        # total duration in seconds
MODE="constant"     # "constant" or "burst"

python simulate_load.py \
  --prompt_file "$PROMPT_FILE" \
  --completed_log "$COMPLETED_LOG" \
  --host "$HOST" \
  --save_interval $SAVE_INTERVAL \
  --rate $RATE \
  --duration $DURATION \
  --mode $MODE
