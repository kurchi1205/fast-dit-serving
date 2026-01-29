IMG_DIR="/home/fast-dit-serving/assets/drawbench_sd3"
PROMPTS_JSON="../drawbench_prompts.json"
OUTPUT_JSON="/home/fast-dit-serving/drawbench_generation/outputs/clip_scores_drawbench_by_prompt_sd3_og.json"
CACHE_INTERVAL=None

python compute_clip_scores.py --image_dir "$IMG_DIR" --prompts "$PROMPTS_JSON" --output "$OUTPUT_JSON"
python stats_summary_clip.py --json_path "$OUTPUT_JSON"