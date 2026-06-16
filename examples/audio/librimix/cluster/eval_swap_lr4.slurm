#!/bin/bash
# Swap test for the Exp1 cross-attn model (token text cue).
#   oracle text  -> should extract the target (positive SI-SDRi)
#   interferer text -> if text is truly used, extracts the WRONG speaker
# torch is cu124 now -> runs on any node, no nodelist pin needed.
# Submit: sbatch eval_swap_xattn.slurm   (override CKPT_EPOCH via --export)
#SBATCH --job-name=tse_eval_lr4
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -uo pipefail
echo "host=$(hostname)  start=$(date)"
export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
EXP="${RECIPE}/exp/BSRNN_TEXT_XATTN_LR4"
CKPT_EPOCH="${CKPT_EPOCH:-5}"
CKPT="${EXP}/models/checkpoint_${CKPT_EPOCH}.pt"
cd "${RECIPE}"
echo "ckpt=${CKPT}"

run_eval () {
  local tag="$1"; local cues="$2"
  echo ""
  echo "############ EVAL [$tag] cues=$cues ############"
  python wesep/bin/infer.py \
    --config       confs/tse_bsrnn_text_xattn_lr4.yaml \
    --fs           16k \
    --gpus         0 \
    --exp_dir      "${EXP}" \
    --data_type    raw \
    --checkpoint   "${CKPT}" \
    --test_data    "${DATA}/test/raw.list" \
    --test_cues    "${DATA}/test/${cues}" \
    --test_samples "${DATA}/test/samples.jsonl" \
    --save_wav     False
  echo "############ END [$tag] ############"
}

run_eval "ORACLE"     cues_text_oracle_token.yaml
run_eval "INTERFERER" cues_text_interferer_token.yaml

echo "done=$(date)"
