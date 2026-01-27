#!/bin/bash

# Define the arguments
INTERVAL_LIST=(5)
PROMPT_PATH="../drawbench_prompts.json"

# Run background process and client
# python start_custom_sd3_bck_proc.py &
python infer_for_custom_sd3.py --interval_list "${INTERVAL_LIST[@]}" --prompt_path "$PROMPT_PATH"