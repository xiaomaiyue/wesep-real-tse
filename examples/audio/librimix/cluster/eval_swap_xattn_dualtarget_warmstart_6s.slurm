#!/bin/bash
# Swap eval for R5: warm-started 6s dual-target probe.
#   sbatch eval_swap_xattn_dualtarget_warmstart_6s.slurm
#SBATCH --job-name=tse_eval_dualtgt_r5
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "host=$(hostname)  start=$(date)"
nvidia-smi -L

export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/BSRNN_TEXT_XATTN_DUALTARGET_WARMSTART_6S_PROBE_R5"
CONFIG="confs/tse_bsrnn_text_xattn_dualtarget_warmstart_6s.yaml"
CKPT="${EXP}/models/checkpoint_5.pt"
cd "${RECIPE}"

echo "[eval] config=${CONFIG} ckpt=${CKPT}"
if [ ! -f "${CKPT}" ]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 2
fi

run_eval () {
  local tag="$1"
  local cues="$2"
  echo ""
  echo "############ EVAL tag=${tag} cues=${cues} ############"
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
  echo "############ END tag=${tag} ############"
}

run_eval "ORACLE"     cues_text_oracle_token.yaml
run_eval "INTERFERER" cues_text_interferer_token.yaml

echo "done=$(date)"
