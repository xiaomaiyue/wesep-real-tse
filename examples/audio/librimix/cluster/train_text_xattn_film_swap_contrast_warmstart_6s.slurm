#!/bin/bash
# R4: 6s token cross-attn + time-wise FiLM + swap/contrast loss,
# warm-started from the REAL-TSE spk_emb_100 BSRNN backbone.
# Download the checkpoint on the LOGIN node first (compute nodes have no net):
#   cd examples/audio/librimix && mkdir -p pretrained && \
#   gdown 1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC -O /tmp/realtse_ckpt.zip && \
#   unzip -o /tmp/realtse_ckpt.zip 'spk_emb_100/*' -d pretrained/
# then:  sbatch train_text_xattn_film_swap_contrast_warmstart_6s.slurm
#SBATCH --job-name=tse_xattn_film_r4ws
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:4
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())"

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/BSRNN_TEXT_XATTN_FILM_SWAP_CONTRAST_WARMSTART_6S_PROBE_R4"
PRETRAIN="${RECIPE}/pretrained/spk_emb_100/avg_model.pt"
cd "${RECIPE}"

if [ ! -f "${PRETRAIN}" ]; then
  echo "missing warm-start checkpoint: ${PRETRAIN}" >&2
  echo "download it on the login node (see header of this script)." >&2
  exit 2
fi

export OMP_NUM_THREADS=6
echo "[train] 4-GPU DDP: 6s xattn + FiLM(scale=1.0) + swap/contrast, warm-start=${PRETRAIN}"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  wesep/bin/train.py \
  --config        confs/tse_bsrnn_text_xattn_film_swap_contrast_warmstart_6s.yaml \
  --exp_dir       "${EXP}" \
  --gpus          "[0,1,2,3]" \
  --num_epochs    5 \
  --save_epoch_interval 1 \
  --num_avg       2 \
  --data_type     raw \
  --train_data    "${DATA}/train-100-full/raw.list" \
  --train_cues    "${DATA}/train-100-full/cues_text_oracle_token.yaml" \
  --train_samples "${DATA}/train-100-full/samples.jsonl" \
  --val_data      "${DATA}/dev/raw.list" \
  --val_cues      "${DATA}/dev/cues_text_oracle_token.yaml" \
  --val_samples   "${DATA}/dev/samples.jsonl"

echo "done=$(date)  ckpts in ${EXP}/models"
