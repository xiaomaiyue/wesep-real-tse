#!/bin/bash
# B: online dynamic-mixing run. 27800 single-speaker utterances randomly paired
# + remixed every epoch (effectively unlimited mixtures). Val = online-mixed dev
# (held-out speakers). Submit pinned to gpu01 (cu130 driver):
#   sbatch --nodelist=gpu01 train_text_online.slurm
#SBATCH --job-name=tse_online
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
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
D="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/BSRNN_TEXT_ONLINE"
cd "${RECIPE}"

export OMP_NUM_THREADS=8
echo "[ckpt] launching torchrun (4 GPU DDP) — online dynamic mixing, 40 epochs"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  wesep/bin/train.py \
  --config        confs/tse_bsrnn_text_online.yaml \
  --exp_dir       "${EXP}" \
  --gpus          "[0,1,2,3]" \
  --num_avg       10 \
  --data_type     raw \
  --train_data    "${D}/train-100-full/samples_single.jsonl" \
  --train_cues    "${D}/train-100-full/cues_text_oracle_online.yaml" \
  --train_samples "${D}/train-100-full/samples_single.jsonl" \
  --val_data      "${D}/dev/samples_single.jsonl" \
  --val_cues      "${D}/dev/cues_text_oracle_online.yaml" \
  --val_samples   "${D}/dev/samples_single.jsonl"

echo "done=$(date)  ckpts in ${EXP}/models"
