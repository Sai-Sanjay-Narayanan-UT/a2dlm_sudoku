#!/bin/bash
# A2DLM Training Script for Sudoku
# Uses augmented vocabulary (63 tokens) with importance-based masking

# export WANDB_DISABLED=true
export WANDB_MODE=online

exp=output/sudoku/a2dlm-simpler-kappa0.5-sigma0.0-bs1024-lr1e-3-ep300-T20-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 4 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_a2dlm \
    --do_train \
    --dataset sudoku_train \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --plot_loss \
    --run_name a2dlm_simpler_sudoku_full_run_topkdecoding-True \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --kappa 0.5 \
    --sigma_noise 0.0 \
    --topk_decoding True \
    > $exp/train.log

# Inference on test set
for dataset in sudoku_test
do
topk_decoding=True
mkdir $exp/$dataset
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm \
    --model_name_or_path model_config_a2dlm \
    --do_predict \
    --dataset $dataset \
    --checkpoint_dir $exp \
    --output_dir $exp/$dataset \
    --per_device_eval_batch_size 32 \
    --cutoff_len 164 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --kappa 0.5 \
    --sigma_noise 0.0 \
    --topk_decoding True \
    --decoding_strategy stochastic0.5-linear \
    > $exp/${dataset}/eval-TopK$topk_decoding.log
done
