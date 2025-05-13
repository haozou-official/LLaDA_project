#!/bin/bash

# Low-confidence remasking

# Machine Translation
python evaluation.py --task mt --model instruct --gen_length 256 --block_length 4 --steps 256

# Math
python evaluation.py --task math --model instruct --gen_length 256 --block_length 256 --steps 256

# Sudoku
python evaluation.py --task sudoku --model instruct --gen_length 243 --block_length 81 --steps 243

# HumanEval (Code Generation)
python evaluation.py --task humaneval --model instruct --gen_length 512 --block_length 512 --steps 512

# GSM8K with multiple seeds
for seed in {1234,2023,2024,2025,2026}; do
    python evaluation.py --task gsm8k --model instruct --gen_length 256 --block_length 8 --seed $seed
done


# AR Decoding

# Machine Translation
python evaluation_ar.py --task mt --model instruct --gen_length 256 --block_length 4 --steps 256

# Math
python evaluation_ar.py --task math --model instruct --gen_length 256 --block_length 256 --steps 256

# Sudoku
python evaluation_ar.py --task sudoku --model instruct --gen_length 243 --block_length 81 --steps 243

# HumanEval (Code Generation)
python evaluation_ar.py --task humaneval --model instruct --gen_length 512 --block_length 512 --steps 512

# GSM8K with multiple seeds
for seed in {1234,2023,2024,2025,2026}; do
    python evaluation_ar.py --task gsm8k --model instruct --gen_length 256 --block_length 8 --seed $seed
done
