import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("shunk031/DrawBench", split="test")

# Build prompt JSON
prompts_json = {}

for i, item in enumerate(dataset):
    key = f"p2_prompt_{i}"
    prompts_json[key] = item["prompts"]

# Example: inspect first few
for k in list(prompts_json.keys())[:5]:
    print(k, ":", prompts_json[k])

with open("drawbench_prompts.json", "w") as f:
    json.dump(prompts_json, f)