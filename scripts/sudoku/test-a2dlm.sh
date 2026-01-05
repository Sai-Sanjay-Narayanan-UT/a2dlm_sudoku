#!/bin/bash
# A2DLM Test Script for Sudoku
# Runs inference on test set using a trained checkpoint

# Set your checkpoint directory here
checkpoint_dir=/home/sn32298/a2dlm_sudoku/output/sudoku/a2dlm-simpler-kappa0.5-sigma0.0-bs1024-lr1e-3-ep300-T20-20260104-135210

# Test dataset
dataset=sudoku_test_large
topk_decoding=True
diffusion_steps=5
# Create output directory
mkdir -p $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps

# Run inference
CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage mdm \
    --model_name_or_path model_config_a2dlm \
    --do_predict \
    --dataset $dataset \
    --checkpoint_dir $checkpoint_dir \
    --output_dir $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps \
    --per_device_eval_batch_size 32 \
    --cutoff_len 164 \
    --remove_unused_columns False \
    --diffusion_steps $diffusion_steps \
    --kappa 0.5 \
    --sigma_noise 0.0 \
    --topk_decoding $topk_decoding \
    --decoding_strategy stochastic0.5-linear \
    > $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/test.log

echo "Test inference complete!"
echo "Results saved to: $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/"
echo "Check accuracy in: $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/all_results.json"
echo "Check predictions in: $checkpoint_dir/$dataset/$topk_decoding/diffusion_steps_$diffusion_steps/generated_predictions.jsonl"

