#!/bin/bash
# Diagnostic D: does the text mechanism work on TRAIN but fail on TEST
# (= pure generalization gap)?  Eval the most train-fit checkpoint (60) on a
# 200-mixture subset under 3 conditions:
#   train-oracle      -> expect high SI-SDRi (model extracts on seen data)
#   train-interferer  -> expect strongly negative (swap flip = uses text)
#   test-oracle       -> expect ~0 (collapses on unseen data)
# Pin gpu01.  Submit: sbatch --nodelist=gpu01 eval_D.slurm
#SBATCH --job-name=tse_evalD
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
CKPT="${RECIPE}/exp/BSRNN_TEXT_FULL/models/checkpoint_60.pt"
cd "${RECIPE}"

run_eval () {
  local tag="$1"; local samples="$2"; local cues="$3"
  echo ""
  echo "############ EVAL [$tag] ############"
  python wesep/bin/infer.py \
    --config       confs/tse_bsrnn_text.yaml \
    --fs           16k \
    --gpus         0 \
    --exp_dir      "${RECIPE}/exp/BSRNN_TEXT_FULL" \
    --data_type    raw \
    --checkpoint   "${CKPT}" \
    --test_data    "${samples}" \
    --test_cues    "${cues}" \
    --test_samples "${samples}" \
    --save_wav     False
  echo "############ END [$tag] ############"
}

run_eval "TRAIN-ORACLE"     "${DATA}/train-100-full/samples_d200.jsonl" "${DATA}/train-100-full/cues_text_oracle.yaml"
run_eval "TRAIN-INTERFERER" "${DATA}/train-100-full/samples_d200.jsonl" "${DATA}/train-100-full/cues_text_interferer.yaml"
run_eval "TEST-ORACLE"      "${DATA}/test/samples_d200.jsonl"           "${DATA}/test/cues_text_oracle.yaml"

echo "done=$(date)"
