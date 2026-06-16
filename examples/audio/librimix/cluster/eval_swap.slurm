#!/bin/bash
# Evaluate a checkpoint on the test set under two text cues (swap test):
#   oracle text  -> should give positive SI-SDRi (real extraction)
#   interferer text -> if the model truly uses text, it extracts the WRONG
#                      speaker, so SI-SDRi collapses / flips.
# Pin to gpu01 (cu130-capable driver).  Submit: sbatch --nodelist=gpu01 eval_swap.slurm
#SBATCH --job-name=tse_eval
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
CKPT="${RECIPE}/exp/BSRNN_TEXT_FULL/models/checkpoint_5.pt"
cd "${RECIPE}"

run_eval () {
  local tag="$1"; local cues="$2"
  echo ""
  echo "############ EVAL [$tag] cues=$cues ############"
  python wesep/bin/infer.py \
    --config       confs/tse_bsrnn_text.yaml \
    --fs           16k \
    --gpus         0 \
    --exp_dir      "${RECIPE}/exp/BSRNN_TEXT_FULL" \
    --data_type    raw \
    --checkpoint   "${CKPT}" \
    --test_data    "${DATA}/test/raw.list" \
    --test_cues    "${DATA}/test/${cues}" \
    --test_samples "${DATA}/test/samples.jsonl" \
    --save_wav     False
  echo "############ END [$tag] ############"
}

run_eval "ORACLE"     cues_text_oracle.yaml
run_eval "INTERFERER" cues_text_interferer.yaml

echo "done=$(date)"
