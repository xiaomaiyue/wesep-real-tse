#!/bin/bash
# Evaluate the official REAL-TSE speaker-embedding BSRNN checkpoint on
# Libri2Mix test oracle audio-enrollment cues. This script does not train.
# Submit from this directory with:
#   sbatch eval_official_spk_emb_test_oracle.slurm
#SBATCH --job-name=tse_official_spk
#SBATCH --partition=gpu2node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail

echo "host=$(hostname)  start=$(date)"
export PATH="${HOME}/.conda/envs/wesep/bin:${PATH}"
export PYTHONIOENCODING=UTF-8
export PYTHONUNBUFFERED=1

ROOT="${HOME}/tse1"
RECIPE="${ROOT}/wesep-real-tse/examples/audio/librimix"
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"
CKPT="${ROOT}/real-tse-ckpt/spk_emb_100/avg_model.pt"
SRC_CFG="${ROOT}/real-tse-ckpt/spk_emb_100/config.yaml"
EXP_NAME="${EXP_NAME:-OFFICIAL_SPK_EMB_100_TEST_ORACLE}"
EXP="${RECIPE}/exp/${EXP_NAME}"
EVAL_CFG="${EXP}/config_no_pretrained.yaml"

mkdir -p "${EXP}"

python - "${SRC_CFG}" "${EVAL_CFG}" <<'PY'
import sys
import yaml

src_cfg, dst_cfg = sys.argv[1:]


def disable_pretrained(value):
    if isinstance(value, dict):
        return {
            key: None if key == "pretrained" else disable_pretrained(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [disable_pretrained(item) for item in value]
    return value


with open(src_cfg, "r", encoding="utf-8") as fin:
    cfg = yaml.load(fin, Loader=yaml.FullLoader)

cfg = disable_pretrained(cfg)

with open(dst_cfg, "w", encoding="utf-8") as fout:
    yaml.safe_dump(cfg, fout, sort_keys=False)
PY

cd "${RECIPE}"

python wesep/bin/infer.py \
  --config "${EVAL_CFG}" \
  --fs 16k \
  --gpus 0 \
  --exp_dir "${EXP}" \
  --data_type raw \
  --checkpoint "${CKPT}" \
  --test_data "${DATA}/test/samples.jsonl" \
  --test_cues "${DATA}/test/cues_spk_text_oracle.yaml" \
  --test_samples "${DATA}/test/samples.jsonl" \
  --save_wav False

echo "done=$(date)"
