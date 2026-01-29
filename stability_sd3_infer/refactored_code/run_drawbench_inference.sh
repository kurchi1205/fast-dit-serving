PROMPT_JSON="/home/fast-dit-serving/drawbench_generation/drawbench_prompts.json"
MODEL_PATH="/home/fast-dit-serving/sd3_model/sd3_medium.safetensors"
MODEL_FOLDER="/home/fast-dit-serving/sd3_model"
OUTPUT_DIR="/home/fast-dit-serving/assets/drawbench_sd3"
WIDTH=1024
HEIGHT=1024
STEPS=50
SEED=50


echo "🚀 Starting batch inference..."

python batch_inference_drawbench.py \
    --prompt_json_path "$PROMPT_JSON" \
    --model_path "$MODEL_PATH" \
    --model_folder "$MODEL_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --steps "$STEPS" \
    --seed "$SEED" \

echo "All prompts processed. Images saved in $OUTPUT_DIR."