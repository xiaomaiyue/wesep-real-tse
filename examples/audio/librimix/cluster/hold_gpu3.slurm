#!/bin/bash
# Placeholder to hold 3 GPUs on gpu2node for the next experiment (R6/full).
# Time-boxed so it cannot idle-hold forever. Cancel with: scancel <jobid>
#SBATCH --job-name=tse_HOLD3
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

echo "HOLD start $(date) on $(hostname)"
nvidia-smi -L
echo "holding 3 GPUs for up to 90 min; scancel this job before launching the real run"
sleep 5400
echo "HOLD auto-released $(date)"
