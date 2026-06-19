#!/bin/bash
# Auto final eval: runs after the 150-epoch separator finishes. Full test set,
# on GPU, with the oracle-vs-interferer text-routing ablation.
#   sbatch --dependency=afterok:<train_jobid> eval_routeB_full.slurm
#SBATCH --job-name=tse_routeB_eval
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "host=$(hostname)  start=$(date)"
export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONUNBUFFERED=1

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/SEP_BSRNN_2SPK_WARMSTART_R9"
cd "${RECIPE}"

CK=$(ls -1t ${EXP}/models/ckpt_*.pt 2>/dev/null | head -1 || true)
echo "[routeB-eval] checkpoint=${CK}  (full test set, GPU)"

python wesep/bin/eval_routeB.py \
  --sep_ckpt "${CK}" \
  --samples "${DATA}/test/samples.jsonl" \
  --meta    "${ROOT}/libri_data/text_cues/test.text_meta.jsonl" \
  --n 100000 --device 0

echo "done=$(date)"
