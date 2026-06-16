#!/bin/bash
# Anti-overfit retrain (diagnostic A): regularized config on the full 13900-mix
# train set. Submit pinned to gpu01 (cu130 driver):
#   sbatch --nodelist=gpu01 train_text_reg.slurm
#SBATCH --job-name=tse_reg
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
EXP="${RECIPE}/exp/BSRNN_TEXT_REG"
cd "${RECIPE}"

export OMP_NUM_THREADS=8
echo "[ckpt] launching torchrun (2 GPU DDP) — regularized config, 20 epochs"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  wesep/bin/train.py \
  --config        confs/tse_bsrnn_text_reg.yaml \
  --exp_dir       "${EXP}" \
  --gpus          "[0,1]" \
  --num_avg       10 \
  --data_type     raw \
  --train_data    "${DATA}/train-100-full/raw.list" \
  --train_cues    "${DATA}/train-100-full/cues_text_oracle.yaml" \
  --train_samples "${DATA}/train-100-full/samples.jsonl" \
  --val_data      "${DATA}/dev/raw.list" \
  --val_cues      "${DATA}/dev/cues_text_oracle.yaml" \
  --val_samples   "${DATA}/dev/samples.jsonl"

echo "done=$(date)  ckpts in ${EXP}/models"
