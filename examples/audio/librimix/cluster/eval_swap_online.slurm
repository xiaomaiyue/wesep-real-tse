#!/bin/bash
# Swap test on the fixed TEST set for the online run B (checkpoint_4 = best val).
# Model arch is identical to tse_bsrnn_text.yaml (num_repeat 6); use that conf so
# infer runs the OFFLINE 2-speaker test path (online_mix off). cu124 torch works
# on any node now.  Submit: sbatch --nodelist=gpu02 eval_swap_online.slurm
#SBATCH --job-name=tse_eval_online
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
CKPT="${RECIPE}/exp/BSRNN_TEXT_ONLINE/models/checkpoint_4.pt"
cd "${RECIPE}"

run_eval () {
  local tag="$1"; local cues="$2"
  echo ""
  echo "############ EVAL [$tag] ############"
  python wesep/bin/infer.py \
    --config       confs/tse_bsrnn_text.yaml \
    --fs           16k \
    --gpus         0 \
    --exp_dir      "${RECIPE}/exp/BSRNN_TEXT_ONLINE" \
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
