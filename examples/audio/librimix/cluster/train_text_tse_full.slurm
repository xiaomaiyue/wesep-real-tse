#!/bin/bash
# Full-data run: text-guided TSE on the complete 13900-mixture train-100 set.
# Separate exp_dir so it never clashes with the 2000-subset job.
# Submit from login01:   sbatch train_text_tse_full.slurm
#SBATCH --job-name=tse_text_full
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "host=$(hostname)  start=$(date)"
nvidia-smi -L

export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())"

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/BSRNN_TEXT_FULL"
cd "${RECIPE}"

export OMP_NUM_THREADS=8
echo "[ckpt] launching torchrun (2 GPU DDP) on train-100-full"

# train on full 13900 set; validate on dev. Same model/cue/fusion as the subset run.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  wesep/bin/train.py \
  --config        confs/tse_bsrnn_text.yaml \
  --exp_dir       "${EXP}" \
  --gpus          "[0,1]" \
  --num_epochs    60 \
  --save_epoch_interval 5 \
  --num_avg       10 \
  --data_type     raw \
  --train_data    "${DATA}/train-100-full/raw.list" \
  --train_cues    "${DATA}/train-100-full/cues_text_oracle.yaml" \
  --train_samples "${DATA}/train-100-full/samples.jsonl" \
  --val_data      "${DATA}/dev/raw.list" \
  --val_cues      "${DATA}/dev/cues_text_oracle.yaml" \
  --val_samples   "${DATA}/dev/samples.jsonl"

echo "done=$(date)  ckpts in ${EXP}/models"
