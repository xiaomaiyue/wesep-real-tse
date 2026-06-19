#!/bin/bash
# Probe token cross-attention text TSE at different audio chunk lengths.
# Submit from login01:
#   sbatch probe_xattn_chunklen.slurm
#
# Array mapping:
#   0 -> 1.5s, batch 4
#   1 -> 4s,   batch 2
#   2 -> 6s,   batch 1
# Each task trains 5 short epochs with sample_num_per_epoch=256.
# This is a memory/trend probe, not the final training recipe.
#SBATCH --job-name=tse_xattn_chunk
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%A_%a.out

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
cd "${RECIPE}"

LABELS=("15" "4s" "6s")
CONFIGS=(
  "confs/tse_bsrnn_text_xattn_probe_15.yaml"
  "confs/tse_bsrnn_text_xattn_probe_4s.yaml"
  "confs/tse_bsrnn_text_xattn_probe_6s.yaml"
)
EXPS=(
  "${RECIPE}/exp/BSRNN_TEXT_XATTN_PROBE_15"
  "${RECIPE}/exp/BSRNN_TEXT_XATTN_PROBE_4S"
  "${RECIPE}/exp/BSRNN_TEXT_XATTN_PROBE_6S"
)

idx="${SLURM_ARRAY_TASK_ID}"
LABEL="${LABELS[$idx]}"
CONFIG="${CONFIGS[$idx]}"
EXP="${EXPS[$idx]}"

PROBE_DIR="${RECIPE}/exp/probe_lists"
mkdir -p "${PROBE_DIR}"
head -n 128 "${DATA}/dev/samples.jsonl" > "${PROBE_DIR}/dev_128.samples.jsonl"

export OMP_NUM_THREADS=4
echo "[probe] label=${LABEL} config=${CONFIG} exp=${EXP}"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  wesep/bin/train.py \
  --config        "${CONFIG}" \
  --exp_dir       "${EXP}" \
  --gpus          "[0]" \
  --num_epochs    5 \
  --save_epoch_interval 1 \
  --num_avg       2 \
  --data_type     raw \
  --train_data    "${DATA}/train-100-full/raw.list" \
  --train_cues    "${DATA}/train-100-full/cues_text_oracle_token.yaml" \
  --train_samples "${DATA}/train-100-full/samples.jsonl" \
  --val_data      "${DATA}/dev/raw.list" \
  --val_cues      "${DATA}/dev/cues_text_oracle_token.yaml" \
  --val_samples   "${PROBE_DIR}/dev_128.samples.jsonl"

echo "[probe] done label=${LABEL} date=$(date)"
echo "[probe] ckpts in ${EXP}/models"
