#!/bin/bash
# Route B: 2-speaker PIT BSRNN separator, warm-started from spk_emb_100.
#   sbatch train_sep_2spk_warmstart.slurm
#SBATCH --job-name=tse_sep2_r9
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "host=$(hostname)  start=$(date)"
nvidia-smi -L

export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/SEP_BSRNN_2SPK_WARMSTART_R9"
INIT="${RECIPE}/pretrained/spk_emb_100/avg_model.pt"
cd "${RECIPE}"

export OMP_NUM_THREADS=6
echo "[sep] 3-GPU PIT separation, warm-start=${INIT}"

torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  wesep/bin/train_sep.py \
  --samples "${DATA}/train-100-full/samples.jsonl" \
  --exp     "${EXP}" \
  --init    "${INIT}" \
  --epochs  30 \
  --bs      3 \
  --lr      0.0005 \
  --chunk   96000 \
  --iters_per_epoch 2000

echo "done=$(date)  ckpts in ${EXP}/models"
