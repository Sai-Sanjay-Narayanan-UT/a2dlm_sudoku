#!/bin/bash
# A2DLM Test Script for Sudoku
# Runs inference on test set using a trained checkpoint

# Set your checkpoint directory here
checkpoint_dir=/home/sn32298/a2dlm_sudoku/output/sudoku/mdm-alpha0.25-gamma1-bs1024-lr1e-3-ep300-T20-20260104-093832

# Test dataset
dataset=sudoku_test_large
topk_decoding=True
diffusion_steps=5
# Create output directory
mkdir -p $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps

# Run inference
CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --per_device_eval_batch_size 32 \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps $diffusion_steps \
    --output_dir $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps \
    --checkpoint_dir $checkpoint_dir  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/eval-TopK$topk_decoding.log

echo "Test inference complete!"
echo "Results saved to: $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/"
echo "Check accuracy in: $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/all_results.json"
echo "Check predictions in: $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/generated_predictions.jsonl"

