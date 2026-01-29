SRC_DIR="/home/fast-dit-serving/assets/drawbench_sd3"
TGT_DIR="/home/fast-dit-serving/assets/drawbench_sd3_custom"
OUTPUT_JSON="/home/fast-dit-serving/drawbench_generation/outputs/fid_scores_by_prompt_drawbench_sd3_custom.json"
INTERVALS="5"

# Run the Python script
python compute_fid_scores.py \
  --src_dir "$SRC_DIR" \
  --tgt_dir "$TGT_DIR" \
  --output "$OUTPUT_JSON" \
  --intervals $INTERVALS

python stats_summary_fid.py --json_path "/home/fast-dit-serving/drawbench_generation/outputs/fid_scores_by_prompt_drawbench_sd3_custom.json"