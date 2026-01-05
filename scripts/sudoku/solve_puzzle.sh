#!/bin/bash

# Usage: bash scripts/sudoku/solve_puzzle.sh <checkpoint_dir>
# Example: bash scripts/sudoku/solve_puzzle.sh output/sudoku/mdm-alpha0.25-gamma1-bs1024-lr1e-3-ep300-T20-20260101-124145

checkpoint_dir=${1:-"output/sudoku/mdm-alpha0.25-gamma1-bs1024-lr1e-3-ep300-T20-20260101-124145"}

output_dir=${checkpoint_dir}/my_puzzle
mkdir -p $output_dir

echo "Solving puzzle with model from: $checkpoint_dir"

CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset my_puzzle \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $output_dir \
    --checkpoint_dir $checkpoint_dir \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding True \
    --per_device_eval_batch_size 1

echo ""
echo "Solution saved to: $output_dir/generated_predictions.jsonl"
echo ""
cat $output_dir/generated_predictions.jsonl | python3 -m json.tool
