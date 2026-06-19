#!/bin/bash
# Swap eval for chunk-length probe checkpoints.
# Submit from login01:
#   sbatch eval_swap_xattn_probe_chunklen.slurm
#SBATCH --job-name=tse_eval_chunk
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%A_%a.out

set -uo pipefail
echo "host=$(hostname)  start=$(date)"
nvidia-smi -L

export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1

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
CKPT="${EXP}/models/checkpoint_5.pt"

echo "[eval] label=${LABEL} config=${CONFIG} ckpt=${CKPT}"
if [ ! -f "${CKPT}" ]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 2
fi

run_eval () {
  local tag="$1"
  local cues="$2"
  echo ""
  echo "############ EVAL label=${LABEL} tag=${tag} cues=${cues} ############"
  python wesep/bin/infer.py \
    --config       "${CONFIG}" \
    --fs           16k \
    --gpus         0 \
    --exp_dir      "${EXP}" \
    --data_type    raw \
    --checkpoint   "${CKPT}" \
    --test_data    "${DATA}/test/raw.list" \
    --test_cues    "${DATA}/test/${cues}" \
    --test_samples "${DATA}/test/samples.jsonl" \
    --save_wav     False
  echo "---- latest metrics for ${tag} ----"
  tail -n 8 "${EXP}/infer.log"
  echo "############ END label=${LABEL} tag=${tag} ############"
}

run_eval "ORACLE"     cues_text_oracle_token.yaml
run_eval "INTERFERER" cues_text_interferer_token.yaml

echo "done=$(date)"
