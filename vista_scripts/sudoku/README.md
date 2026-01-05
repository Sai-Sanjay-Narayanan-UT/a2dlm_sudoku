# Vista Supercomputer SLURM Scripts for Sudoku

This directory contains SLURM job scripts for running A2DLM and MDLM training/testing on the Vista supercomputer.

## Setup Instructions

### 1. Update Paths and Modules

Before running any scripts, you need to update:

1. **Project path**: Replace `/path/to/a2dlm_sudoku` with your actual project directory
2. **Module names**: Adjust module names based on Vista's available modules:
   ```bash
   module avail  # Check available modules on Vista
   ```
3. **WandB API key**: Replace `your_wandb_api_key_here` with your actual key
4. **Partition name**: Update `gpu-a100` to match Vista's GPU partition name

### 2. Create Log Directory

```bash
mkdir -p vista_scripts/sudoku/logs
```

### 3. Make Scripts Executable

```bash
chmod +x vista_scripts/sudoku/*.sh
```

## Usage

### Training A2DLM

```bash
sbatch vista_scripts/sudoku/train-a2dlm-vista.sh
```

### Training MDLM

```bash
sbatch vista_scripts/sudoku/train-mdlm-vista.sh
```

### Testing A2DLM

1. Update `CHECKPOINT_DIR` in `test-a2dlm-vista.sh` with your trained model path
2. Run:
```bash
sbatch vista_scripts/sudoku/test-a2dlm-vista.sh
```

### Testing MDLM

1. Update `CHECKPOINT_DIR` in `test-mdlm-vista.sh` with your trained model path
2. Run:
```bash
sbatch vista_scripts/sudoku/test-mdlm-vista.sh
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# View logs (while job is running)
tail -f vista_scripts/sudoku/logs/<job_name>_<job_id>.out

# Cancel a job
scancel <job_id>
```

## Resource Configuration

### Training Scripts
- **Nodes**: 1
- **GPUs**: 4 (A100 recommended)
- **CPUs per task**: 8
- **Memory**: 256GB
- **Time limit**: 48 hours
- **Batch size**: 128 per GPU × 4 GPUs = 512 effective batch size

### Testing Scripts
- **Nodes**: 1
- **GPUs**: 1
- **CPUs per task**: 8
- **Memory**: 64GB
- **Time limit**: 4 hours

## Adjusting Resources

To adjust resources, modify the `#SBATCH` directives:

```bash
#SBATCH -n 8              # Use 8 GPUs instead of 4
#SBATCH --gres=gpu:8      # Request 8 GPUs
#SBATCH --mem=512G        # Increase memory to 512GB
#SBATCH -t 72:00:00       # Increase time to 72 hours
```

Remember to also adjust `per_device_train_batch_size` and `gradient_accumulation_steps` accordingly.

## Troubleshooting

### Module Load Errors
If you get module errors, check Vista's documentation for correct module names:
```bash
module spider cuda
module spider gcc
```

### Out of Memory
Reduce batch size or enable gradient checkpointing:
```bash
--per_device_train_batch_size 64
--gradient_accumulation_steps 4
--gradient_checkpointing
```

### Checkpoint Issues
To resume from a checkpoint, set the `CHECKPOINT_DIR` variable:
```bash
CHECKPOINT_DIR="output/sudoku/your-checkpoint-dir"
```

## Notes

- Logs are saved to `vista_scripts/sudoku/logs/`
- Model checkpoints are saved to `output/sudoku/`
- Training uses bf16 precision by default for A100 GPUs
- Distributed training uses NCCL backend
