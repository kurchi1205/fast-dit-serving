python infer_diffuser_sd3.py \
    --prompt_json_path ../drawbench_prompts.json \
    --output_dir /home/fast-dit-serving/assets/drawbench_sd3_diffuser \
    --num_inference_steps 50 \
    --compile False \
    --seed 50
