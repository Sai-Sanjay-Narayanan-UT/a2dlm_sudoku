#!/bin/bash
# SLURM job script for training A2DLM on Sudoku dataset on Vista supercomputer

#SBATCH -J a2dlm-sudoku              # Job name
#SBATCH -o vista_scripts/sudoku/logs/%x_%j.out    # log file (out & err)
#SBATCH -p gh                  # Partition/queue (adjust based on Vista's partition names)
#SBATCH -N 32                         # Number of nodes
#SBATCH -t 48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --open-mode=append           # Do not overwrite logs

# Source user shell configuration
source ~/.bashrc

# Load required modules (adjust module names based on Vista's available modules)
module load gcc/13
module load cuda/12.4
module load nvidia_math

# Activate conda environment
conda activate diffusion

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO

# WandB configuration (set your API key)
export WANDB_API_KEY="bde64e3456e67401c2909df1f0a5d753442df69d"  
export WANDB_MODE="online"
#export WANDB_PROJECT="a2dlm_sudoku"

# Move to project directory
cd /path/to/a2dlm_sudoku  # Update this path to your actual project location

# Create log directory if it doesn't exist
mkdir -p vista_scripts/sudoku/logs

# Training configuration
DATASET="sudoku_train"
MODEL_NAME="a2dlm-kappa0.5-sigma0.0-bs1024-lr1e-3-ep300-T20"
OUTPUT_DIR="output/sudoku/${MODEL_NAME}-vista-$(date +%Y%m%d-%H%M%S)"
CHECKPOINT_DIR=""  # Set to existing checkpoint path if resuming

# Launch distributed training with accelerate
srun accelerate launch \
    --multi_gpu \
    --num_machines $SLURM_JOB_NUM_NODES \
    --num_processes $SLURM_NTASKS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    --dynamo_backend no \
    src/train_bash.py \
    --stage mdm \
    --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_a2dlm \
    --do_train \
    --dataset ${DATASET} \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 300 \
    --plot_loss \
    --run_name a2dlm_sudoku_vista \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 3 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --kappa 0.5 \
    --sigma_noise 0.0 \
    --topk_decoding True \
    --decoding_strategy stochastic0.5-linear \
    ${CHECKPOINT_DIR:+--resume_from_checkpoint "$CHECKPOINT_DIR"}

echo "Training completed at $(date)"
