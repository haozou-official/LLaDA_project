import torch
import json
import numpy as np
import ast
import os
import re
import time
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from tqdm import tqdm
from generate import generate, generate_ar
from get_log_likelihood import get_log_likelihood, forward_process
from collections import defaultdict

# For MT task
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')

# For Sudoku task
from sudoku_code import data
from sudoku_code import evaluater

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script for LLaDA on multiple tasks")
    
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "piqa", "humaneval", "math", "mt", "sudoku"], help="Dataset to evaluate (e.g., gsm8k)")
    parser.add_argument("--model", type=str, default="instruct", choices=["base", "instruct"], help="Model type to use (base or instruct)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    
    parser.add_argument("--steps", type=int, default=256, help="Number of generation steps")
    parser.add_argument("--gen_length", type=int, default=256, help="Generated answer length")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for iterative remasking")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random", "ar"], help="Remasking strategy")
    
    parser.add_argument("--mc_num", type=int, default=128, help="Monte Carlo estimation times for log-likelihood computation")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="Unsupervised classifier-free guidance scale")

    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    return parser.parse_args()

def safe_execute(code, entry_point, test_case):
    try:
        local_env = {}
        exec(code, {}, local_env)  

        if entry_point not in local_env:
            return False

        exec(test_case, {}, local_env)  
        return True
    
    except Exception:
        return False

def clean_code_output(generated: str, indent: int = 4) -> str:
    import textwrap
    stop_tokens = ["###", "```", "if __name__ == \"__main__\"", "Explanation", "Usage"]
    for token in stop_tokens:
        idx = generated.find(token)
        if idx != -1:
            generated = generated[:idx]

    return generated

def fix_flat_python_block(code: str) -> str:
    """
    Naively adds nested indentation based on common Python keywords.
    Only use this if model output is totally flat.
    """
    lines = code.strip().splitlines()
    fixed = []
    indent = " " * 4
    level = 0

    for line in lines:
        line = line.strip()
        if any(line.startswith(k) for k in ["for ", "if ", "while ", "def "]):
            fixed.append(indent * level + line)
            level += 1
        elif line.startswith("return") or line.startswith("else"):
            fixed.append(indent * level + line)
            if level > 0:
                level -= 1
        else:
            fixed.append(indent * level + line)

    return "\n".join(fixed)

def postprocess_code(generated: str, indent: int = 4) -> str:
    """
    Cleans, fixes nesting, and indents all lines under the function.
    """
    stop_tokens = ["###", "```", "Explanation", "if __name__"]
    for token in stop_tokens:
        idx = generated.find(token)
        if idx != -1:
            generated = generated[:idx]

    # Flatten, then fix internal nesting
    flat = generated.strip()
    nested = fix_flat_python_block(flat)

    # Finally indent the whole thing to go under `def`
    prefix = " " * indent
    return "\n".join(
        prefix + line if line.strip() else "" for line in nested.splitlines()
    )

def load_4shot_examples(dataset, num_shots=4, seed=42):
    train_list = list(dataset)
    random.seed(seed)
    few_shot_examples = random.sample(train_list, num_shots)
    formatted_examples = []
    for example in few_shot_examples:
        formatted_examples.append(f"(Question) {example['question']}\n(Solution) {example['answer']}")
    return "\n\n".join(formatted_examples)

def load_4shot_examples_math500(dataset, num_shots=4):
    train_list = list(dataset)
    random.seed(args.seed)
    few_shot_examples = random.sample(train_list, num_shots)
    formatted_examples = []
    for example in few_shot_examples:
        formatted_examples.append(f"(Question) {example['problem']}\n(Solution) {example['answer']}")
    return "\n\n".join(formatted_examples)

def split_generation(generated_text, task):
    if task == "gsm8k":
        split_text = re.split(r"\nQ|Question", generated_text, maxsplit=1)
        relevant_text = split_text[0]

    return relevant_text

def extract_answer(generated_text, task):
    def clean_math_answer(ans):
        ans = ans.strip()

        # Convert \frac{a}{b} to a/b
        ans = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", ans)

        # Convert \sqrt{a} to sqrt(a)
        ans = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", ans)

        # Remove LaTeX wrappers like \boxed{}, \text{}, etc.
        ans = re.sub(r"\\(boxed|text|mathrm|begin|end|left|right)\{([^}]*)\}", r"\2", ans)

        # Remove inline LaTeX markers $, \( \), \[ \]
        ans = re.sub(r"\\[()\[\]]", "", ans)
        ans = ans.strip("$")

        # Strip surrounding parentheses
        ans = re.sub(r"^[\(\[]*(.*?)[\)\]]*$", r"\1", ans)

        # Remove remaining LaTeX commands like \pi, \cdot, etc.
        ans = re.sub(r"\\[a-zA-Z]+\s*", "", ans)

        # Remove all non-alphanumeric characters except / and .
        ans = re.sub(r"[^a-zA-Z0-9/\.]", "", ans)

        return ans.lower()

    if task == "gsm8k":
        numbers = re.findall(r"\d+", generated_text)
        return numbers[-1] if numbers else "unknown"

    elif task == "math":
        return clean_math_answer(generated_text)

    return generated_text.strip().lower()

def save_results(task, results, accuracy, model_mode):
    os.makedirs("results", exist_ok=True)  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    generation_file = f"results/{task}_generation_{model_mode}_{timestamp}.jsonl"
    with open(generation_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    scores_file = f"results/{task}_scores_{model_mode}_{timestamp}.json"
    with open(scores_file, "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=4)

    print(f"Results saved:\n- Log-likelihoods: {generation_file}\n- Scores: {scores_file}")

def evaluate_gsm8k_ar(model, tokenizer, dataset, args, device):   
    start_time = time.time()
    train_dataset = load_dataset("gsm8k", "main", split="train")  # Load training set for examples
     
    total_correct = 0
    total_questions = 0
    results = []
    
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size  

    system_prompt = (
        "prefix:  Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly."
    )

    for i in range(num_batches):
        batch_start_time = time.time()
        
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(dataset))

        batch_data = dataset[start_idx:end_idx]
        batch_data = [dict(zip(batch_data.keys(), values)) for values in zip(*batch_data.values())]

        questions = [item["question"] for item in batch_data]
        ground_truths = [item["answer"] for item in batch_data]

        correct = 0
        generated_answers = []

        for q, gt in zip(questions, ground_truths):
            # Load 4-shot examples
            few_shot_prompt = load_4shot_examples(train_dataset, seed=args.seed)
            
            final_prompt = (
                #f"{system_prompt}\n\n"
                f"{few_shot_prompt}\n\n"
                f"Question:\n{q}\nAnswer:\nLet's think step by step.\n"
            )

            input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)

            generated_output = generate_ar(
                model=model,
                prompt=input_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking
            )

            generated_text = tokenizer.decode(
                generated_output[:, input_ids.shape[1]:].squeeze(),
                skip_special_tokens=True
            )

            relevant_text = split_generation(generated_text, "gsm8k")
            predicted_answer = extract_answer(relevant_text, "gsm8k")
            gt_final = extract_answer(gt, "gsm8k")

            is_correct = (predicted_answer == gt_final)
            if is_correct:
                correct += 1

            generated_answers.append(predicted_answer)

            results.append({
                "question": q,
                #"4shot_prompt": few_shot_prompt,
                "full_generation": generated_text,
                "relevant_generation": relevant_text,
                "ground_truth": gt,
                "generated_answer": predicted_answer,
                "ground_truth_final": gt_final,
                "correct": is_correct
            })

        total_correct += correct
        total_questions += len(batch_data)

        batch_time = time.time() - batch_start_time 
        print(f"Batch {i+1}/{num_batches}: Accuracy {correct}/{len(batch_data)} | Batch Time: {batch_time:.2f} sec")

    final_acc = total_correct / total_questions
    total_time = time.time() - start_time
    print(f"Final Accuracy for GSM8K: {final_acc:.4f}")
    print(f"Total Execution Time: {total_time:.2f} sec")

    save_results("gsm8k", results, final_acc, args.model)

def evaluate_math_ar(model, tokenizer, dataset, args, device):   
    start_time = time.time()
     
    total_correct = 0
    total_questions = 0
    results = []
    
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size  

    system_prompt = (
        "prefix:  Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly."
    )

    for i in tqdm(range(num_batches), desc="Evaluating MATH500", unit="batch"):
        batch_start_time = time.time()
        
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(dataset))

        batch_data = dataset[start_idx:end_idx]
        batch_data = [dict(zip(batch_data.keys(), values)) for values in zip(*batch_data.values())]

        questions = [item["problem"] for item in batch_data]
        ground_truths = [item["answer"] for item in batch_data]

        correct = 0
        generated_answers = []

        for q, gt in zip(questions, ground_truths):
            final_prompt = f"Question: {q}\nAnswer: "

            input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)

            generated_output = generate_ar(
                model=model,
                prompt=input_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking
            )

            generated_text = tokenizer.decode(
                generated_output[:, input_ids.shape[1]:].squeeze(),
                skip_special_tokens=True
            )

            predicted_answer = extract_answer(generated_text, "math")
            gt_final = extract_answer(gt, "math")
            is_correct = (predicted_answer == gt_final)

            if is_correct:
                correct += 1

            generated_answers.append(predicted_answer)

            results.append({
                "question": q,
                "full_generation": generated_text,
                "ground_truth": gt,
                "generated_answer": predicted_answer,
                "ground_truth_final": gt_final,
                "correct": is_correct
            })

        total_correct += correct
        total_questions += len(batch_data)

        batch_time = time.time() - batch_start_time 
        print(f"Batch {i+1}/{num_batches}: Accuracy {correct}/{len(batch_data)} | Batch Time: {batch_time:.2f} sec")

    final_acc = total_correct / total_questions
    total_time = time.time() - start_time
    print(f"Final Accuracy for MATH500: {final_acc:.4f}")
    print(f"Total Execution Time: {total_time:.2f} sec")

    save_results("math500", results, final_acc, args.model)

def evaluate_humaneval_ar(model, tokenizer, args, device):
    from human_eval.data import read_problems, write_jsonl
    start_time = time.time()
    problems = read_problems() 

    samples = []

    for task_id, problem in problems.items():
        prompt = problem["prompt"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            generated_output = generate_ar(
                model=model,
                prompt=input_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking
            )

        generated_text = tokenizer.decode(
            generated_output[:, input_ids.shape[1]:].squeeze(),
            skip_special_tokens=True
        )
        
        cleaned_body = clean_code_output(generated_text)
        full_code = prompt + cleaned_body
        
        samples.append({
            "task_id": task_id,
            "completion": full_code
        })

    save_path = os.path.join("./results/humaneval/", "samples.jsonl")
    write_jsonl(save_path, samples)

    total_time = time.time() - start_time
    print(f"HumanEval completions saved to: {save_path}")
    print(f"Total Generation Time: {total_time:.2f} sec")
    print(f"Run `evaluate_functional_correctness {save_path}` to compute pass@k.")

def evaluate_mt_ar(model, tokenizer, dataset, args, device):
    start_time = time.time()

    total_bleu = 0.0
    total_meteor = 0.0
    total_samples = 0
    results = []

    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches), desc="Evaluating MT (en→fr)", unit="batch"):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(dataset))
        batch_data = dataset[start_idx:end_idx]
        batch_data = [dict(zip(batch_data.keys(), values)) for values in zip(*batch_data.values())]

        prompts = [
            f"Translate to French:\nEnglish: {sample['en']}\nFrench: "
            for sample in batch_data
        ]

        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        with torch.no_grad():
            generated_outputs = generate_ar(
                model=model,
                prompt=input_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking
            )

        generated_texts = tokenizer.batch_decode(
            generated_outputs[:, input_ids.shape[1]:], skip_special_tokens=True
        )

        for sample, generated_text in zip(batch_data, generated_texts):
            source_sentence = sample["en"]
            ground_truth = sample["fr"]
            generated_text = generated_text.strip()

            bleu = sentence_bleu(
                [ground_truth.split()],
                generated_text.split(),
                smoothing_function=SmoothingFunction().method1
            )
            meteor = meteor_score([ground_truth.split()], generated_text.split())

            total_bleu += bleu
            total_meteor += meteor
            total_samples += 1

            results.append({
                "source_en": source_sentence,
                "ground_truth_fr": ground_truth,
                "generated_fr": generated_text,
                "bleu": bleu,
                "meteor": meteor
            })

        batch_size = len(batch_data)
        print(f"[Batch {i+1}/{num_batches}] Avg BLEU: {batch_bleu / batch_size:.4f}, Avg METEOR: {batch_meteor / batch_size:.4f}")

    avg_bleu = total_bleu / total_samples
    avg_meteor = total_meteor / total_samples
    total_time = time.time() - start_time

    print(f"\n===== Machine Translation Results =====")
    print(f"Total Samples: {total_samples}")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average METEOR: {avg_meteor:.4f}")
    print(f"Total Evaluation Time: {total_time:.2f} sec")

    os.makedirs("results_mt", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results_mt/mt_translation_{args.model}_{timestamp}.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    with open(f"results_mt/mt_scores_{args.model}_{timestamp}.json", "w") as f:
        json.dump({
            "avg_bleu": avg_bleu,
            "avg_meteor": avg_meteor,
            "total_samples": total_samples
        }, f, indent=4)

def evaluate_sudoku_ar(model, tokenizer, args, device):
    max_eval_samples = 500

    class Config:
        test_puzzle_path = "./sudoku_code/sudoku-test-data.npy"
        seq_order = "random"
        seq_len = 243
        block_size = 81
        minibatch_size = args.batch_size

    config = Config()
    eval_data_iter = data.create_iter(config, config.minibatch_size, train=False)

    total_correct_puzzles = 0
    total_tokens = 0
    correct_tokens = 0
    total_samples = 0
    total_time = 0
    all_results = []

    os.makedirs("results_sudoku", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results_sudoku/sudoku_generations_{args.model}_{timestamp}.jsonl"
    score_file = f"results_sudoku/sudoku_scores_{args.model}_{timestamp}.json"

    for i, batch_data in enumerate(tqdm(eval_data_iter)):
        batch_start_time = time.time()

        input_seq, target_grid, start_index = batch_data
        input_seq = torch.tensor(input_seq).to(device)
        target_grid = torch.tensor(target_grid).to(device)
        start_index = torch.tensor(start_index).to(device)
        input_seq = torch.nn.functional.pad(input_seq, (0, config.seq_len - input_seq.shape[1]), value=0)

        with torch.no_grad():
            generated = generate_ar(
                model=model,
                prompt=input_seq,
                steps=args.steps,
                gen_length=config.seq_len,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking
            )

        generated = generated.cpu().numpy()
        target_grid = target_grid.cpu().numpy()
        batch_size = generated.shape[0]

        batch_token_correct = 0
        batch_token_total = 0
        batch_puzzle_correct = 0
        batch_results = []

        for j in range(batch_size):
            gen = generated[j]
            gt = target_grid[j]

            pred_board = np.full((9, 9), -1)
            gt_board = np.array([[gt[r * 9 + c] for c in range(9)] for r in range(9)])

            correct_triplet_count = 0
            total_triplet_count = 0
            triplets = []

            for k in range(0, len(gen), 3):
                row, col, val = int(gen[k]), int(gen[k + 1]), int(gen[k + 2])
                if not (0 <= row < 9 and 0 <= col < 9 and 1 <= val <= 9):
                    continue  # Skip invalid triplets

                pred_board[row, col] = val
                triplets.append((row, col, val))

                gt_val = gt_board[row, col]
                if val == gt_val:
                    correct_triplet_count += 1
                total_triplet_count += 1

            is_puzzle_correct = np.array_equal(pred_board, gt_board)

            batch_token_correct += correct_triplet_count
            batch_token_total += total_triplet_count
            if is_puzzle_correct:
                batch_puzzle_correct += 1

            batch_results.append({
                "index": total_samples + j,
                "generated_triplets": triplets,
                "correct_tokens": correct_triplet_count,
                "total_tokens": total_triplet_count,
                "is_valid_solution": is_puzzle_correct,
                "ground_truth": [(r, c, int(gt_board[r, c])) for r in range(9) for c in range(9)]
            })

        batch_time = time.time() - batch_start_time
        print(f"[Batch {i+1}] Token Acc: {batch_token_correct / max(1, batch_token_total):.4f} | "
              f"Puzzle Acc: {batch_puzzle_correct / batch_size:.4f} | "
              f"Batch Time: {batch_time:.2f} sec")

        correct_tokens += batch_token_correct
        print(f"correct_tokens: {correct_tokens}")
        total_tokens += batch_token_total
        total_correct_puzzles += batch_puzzle_correct
        total_samples += batch_size
        total_time += batch_time
        all_results.extend(batch_results)

        if total_samples >= max_eval_samples:
            break

    avg_token_acc = correct_tokens / max(1, total_tokens)
    avg_puzzle_acc = total_correct_puzzles / max(1, total_samples)

    print(f"\n===== Sudoku Evaluation Results =====")
    print(f"Total Samples: {total_samples}")
    print(f"Token Accuracy: {avg_token_acc:.4f}")
    print(f"Full Puzzle Accuracy: {avg_puzzle_acc:.4f}")
    print(f"Total Evaluation Time: {total_time:.2f} sec")

    with open(result_file, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    with open(score_file, "w") as f:
        json.dump({
            "token_accuracy": avg_token_acc,
            "puzzle_accuracy": avg_puzzle_acc,
            "total_samples": total_samples,
            "correct_tokens": correct_tokens,
            "total_tokens": total_tokens,
            "total_correct_puzzles": total_correct_puzzles,
            "total_time_sec": total_time
        }, f, indent=4)

    print(f"Saved results to {result_file}")
    print(f"Saved scores to {score_file}")

def evaluate():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "GSAI-ML/LLaDA-8B-Base" if args.model == "base" else "GSAI-ML/LLaDA-8B-Instruct"

    print(f"Loading {model_name} for task: {args.task}...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir="/mnt/swordfish-pool2/hz2999/hf_cache/"  
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if args.task == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        evaluate_gsm8k_ar(model, tokenizer, dataset, args, device) 
    elif args.task == "math":
        dataset = load_dataset("di-zhang-fdu/MATH500", split="test")
        evaluate_math_ar(model, tokenizer, dataset, args, device)  
    elif args.task == "humaneval":
        evaluate_humaneval_ar(model, tokenizer, args, device)
    elif args.task == "mt":
        dataset = load_dataset("wmt14", "fr-en", split="test")
        evaluate_mt_ar(model, tokenizer, dataset, args, device)
    elif args.task == "sudoku":
        evaluate_sudoku_ar(model, tokenizer, args, device)
    else:
        raise NotImplementedError(f"Dataset loading not implemented for task: {args.task}")

if __name__ == "__main__":
    evaluate()
