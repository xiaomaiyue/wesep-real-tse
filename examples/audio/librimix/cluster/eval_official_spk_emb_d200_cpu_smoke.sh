#!/bin/bash
# Quick CPU smoke test for the official speaker-embedding checkpoint.
# It evaluates the 200-mixture test subset only; the full-test Slurm job is
# eval_official_spk_emb_test_oracle.slurm.

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
EXP="${RECIPE}/exp/OFFICIAL_SPK_EMB_100_D200_CPU_SMOKE"
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
  --gpus -1 \
  --exp_dir "${EXP}" \
  --data_type raw \
  --checkpoint "${CKPT}" \
  --test_data "${DATA}/test/samples_d200.jsonl" \
  --test_cues "${DATA}/test/cues_spk_text_oracle.yaml" \
  --test_samples "${DATA}/test/samples_d200.jsonl" \
  --save_wav False

echo "done=$(date)"
