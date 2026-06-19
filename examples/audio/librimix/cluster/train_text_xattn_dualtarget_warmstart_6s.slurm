#!/bin/bash
# R5: 6s warm-started BSRNN (joint, unfrozen) + time-wise FiLM +
# dual-target direct supervision.  sbatch train_text_xattn_dualtarget_warmstart_6s.slurm
#SBATCH --job-name=tse_xattn_dualtgt_r5
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())"

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/BSRNN_TEXT_XATTN_DUALTARGET_WARMSTART_6S_PROBE_R5"
PRETRAIN="${RECIPE}/pretrained/spk_emb_100/avg_model.pt"
cd "${RECIPE}"

if [ ! -f "${PRETRAIN}" ]; then
  echo "missing warm-start checkpoint: ${PRETRAIN}" >&2
  exit 2
fi

export OMP_NUM_THREADS=6
echo "[train] 3-GPU DDP: 6s xattn + FiLM(1.0) + dual-target, joint warm-start=${PRETRAIN}"

torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  wesep/bin/train.py \
  --config        confs/tse_bsrnn_text_xattn_dualtarget_warmstart_6s.yaml \
  --exp_dir       "${EXP}" \
  --gpus          "[0,1,2]" \
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
