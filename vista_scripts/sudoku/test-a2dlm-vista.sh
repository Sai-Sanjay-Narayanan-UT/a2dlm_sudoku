#!/bin/bash
# SLURM job script for testing A2DLM on Sudoku dataset on Vista supercomputer

#SBATCH -J a2dlm-test                # Job name
#SBATCH -o vista_scripts/sudoku/logs/%x_%j.out    # log file (out & err)
#SBATCH -p gpu-a100                  # Partition/queue
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks (1 GPU for inference)
#SBATCH -t 4:00:00                   # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --mem=64G                    # Memory per node

# Source user shell configuration
source ~/.bashrc

# Load required modules
module purge
module load gcc/11.2.0
module load cuda/12.1
module load python/3.9

# Activate conda environment
conda activate diffusion

# WandB configuration
export WANDB_MODE="offline"

# Move to project directory
cd /path/to/a2dlm_sudoku  # Update this path

# Create log directory
mkdir -p vista_scripts/sudoku/logs

# Set checkpoint directory (update with your trained model path)
CHECKPOINT_DIR="output/sudoku/a2dlm-kappa0.5-sigma0.0-bs1024-lr1e-3-ep300-T20-vista-YYYYMMDD-HHMMSS"

# Run inference on test set
python src/train_bash.py \
    --stage mdm \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --do_predict \
    --dataset sudoku_test \
    --finetuning_type freeze \
    --output_dir ${CHECKPOINT_DIR}/sudoku_test \
    --overwrite_cache \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --bf16 \
    --ddp_find_unused_parameters False \
    --diffusion_steps 20 \
    --remove_unused_columns False \
    --topk_decoding True \
    --decoding_strategy stochastic0.5-linear \
    --kappa 0.5 \
    --sigma_noise 0.0

echo "Testing completed at $(date)"
echo "Results saved to: ${CHECKPOINT_DIR}/sudoku_test/"
