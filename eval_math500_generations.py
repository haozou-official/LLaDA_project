import json
import re
from tqdm import tqdm

def clean_text(text):
    text = text.lower().replace(' ', '')
    text = re.sub(r'[^a-zA-Z0-9/\\]', '', text)  # keep only alphanum and /, \ (for LaTeX)
    return text

def eval_math500_generations(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    match_count = 0

    for line in tqdm(lines, desc="Evaluating MATH500 generations"):
        data = json.loads(line)
        generated = data['full_generation']
        ground_truth = data['ground_truth_final']

        generated_clean = clean_text(generated)
        ground_truth_clean = clean_text(ground_truth)

        if ground_truth_clean in generated_clean:
            match_count += 1

    accuracy = match_count / total
    print(f"\n===== MATH500 Generation Evaluation =====")
    print(f"Total Samples: {total}")
    print(f"Exact Match (Ground Truth Exists in Generation): {match_count}/{total}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    file_path = "./results/math500_generation_instruct_20250415_045358.jsonl"
    eval_math500_generations(file_path)
