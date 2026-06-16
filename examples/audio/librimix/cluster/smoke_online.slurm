#!/bin/bash
# Smoke test for the online dynamic-mixing + text-cue path (never run before).
# Tiny subset, 1 GPU, 2 short epochs, frequent logging. Validates:
#   - online_mix pipeline (single-spk -> sample_speaker_group -> snr_mixer)
#   - utt-level text cue attaches (no KeyError on utt_spk*)
#   - val also mixed (dataset.py change) + model forward + loss decreases
# Submit: sbatch --nodelist=gpu01 smoke_online.slurm
#SBATCH --job-name=smoke_online
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
D="${ROOT}/libri_data/Libri2Mix/wav16k/min"
cd "${RECIPE}"

# tiny subsets
head -400 "${D}/train-100-full/samples_single.jsonl" > "${D}/train-100-full/samples_single_smoke.jsonl"
head -100 "${D}/dev/samples_single.jsonl"            > "${D}/dev/samples_single_smoke.jsonl"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  wesep/bin/train.py \
  --config        confs/tse_bsrnn_text_online.yaml \
  --exp_dir       "${RECIPE}/exp/SMOKE_ONLINE" \
  --gpus          "[0]" \
  --num_epochs    2 \
  --log_batch_interval 20 \
  --num_avg       1 \
  --data_type     raw \
  --train_data    "${D}/train-100-full/samples_single_smoke.jsonl" \
  --train_cues    "${D}/train-100-full/cues_text_oracle_online.yaml" \
  --train_samples "${D}/train-100-full/samples_single_smoke.jsonl" \
  --val_data      "${D}/dev/samples_single_smoke.jsonl" \
  --val_cues      "${D}/dev/cues_text_oracle_online.yaml" \
  --val_samples   "${D}/dev/samples_single_smoke.jsonl"

echo "done=$(date)"
