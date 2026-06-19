#!/bin/bash
# Route B separator, FULL 150-epoch run, resumed from the latest checkpoint
# with exponential LR decay.  sbatch train_sep_2spk_150.slurm
#SBATCH --job-name=tse_sep2_150
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

LATEST=$(ls -t ${EXP}/models/ckpt_*.pt 2>/dev/null | head -1 || true)
export OMP_NUM_THREADS=6
if [ -n "${LATEST}" ]; then
  echo "[sep-150] resuming from ${LATEST}"
  RESUME_ARG="--resume ${LATEST}"
  INIT_ARG=""
else
  echo "[sep-150] fresh start, warm-start ${INIT}"
  RESUME_ARG=""
  INIT_ARG="--init ${INIT}"
fi

torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  wesep/bin/train_sep.py \
  --samples "${DATA}/train-100-full/samples.jsonl" \
  --exp     "${EXP}" \
  ${INIT_ARG} ${RESUME_ARG} \
  --epochs  150 \
  --bs      3 \
  --lr      0.0005 \
  --final_lr 0.000025 \
  --chunk   96000 \
  --iters_per_epoch 2000

echo "done=$(date)  ckpts in ${EXP}/models"
