#!/bin/bash
# Run this ON login01 (ssh <user>@10.26.1.75) AFTER transfer_from_mac.sh finished.
# It (1) rewrites the Mac absolute paths inside the data index/cue files to the
# cluster paths, (2) creates the raw.list the recipe expects, (3) builds the
# conda env and installs deps (CUDA torch is installed explicitly).
set -euo pipefail

ROOT="${HOME}/tse1"
OLD="/Users/xiaomaiyue/Downloads/slr/tse1"   # Mac prefix baked into samples.jsonl / cue yamls
DATA="${ROOT}/libri_data/Libri2Mix/wav16k/min"

echo "[1/4] rewrite absolute paths  ${OLD} -> ${ROOT}"
# only text files (samples.jsonl, cues_*.yaml, cues/*.json) contain the prefix;
# embeddings.npz is binary and resource jsons hold no paths. grep -F skips binaries.
mapfile -t FILES < <(grep -rlF "${OLD}" "${ROOT}/libri_data" || true)
for f in "${FILES[@]}"; do
  sed -i "s|${OLD}|${ROOT}|g" "$f"
done
echo "      rewrote ${#FILES[@]} files"

echo "[2/4] create raw.list (recipe uses --data_type raw)"
for split in train-100 dev test; do
  d="${DATA}/${split}"
  [ -f "${d}/samples.jsonl" ] && ln -sf samples.jsonl "${d}/raw.list" && echo "      ${split}/raw.list"
done

echo "[3/4] conda env 'wesep' (python 3.10)"
module load anaconda3/latest 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda create -y -n wesep python=3.10
conda activate wesep

echo "[4/4] install deps"
PIP="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"
# CUDA torch FIRST (requirements.txt does not pin torch; setup.py would pull CPU build)
$PIP torch torchaudio
# project deps; pesq/s3prl/whisper are only needed for SCORING, ok if they warn
$PIP -r "${ROOT}/wesep-real-tse/requirements.txt" || echo "WARN: some optional deps failed (fine for training)"
# editable install of wesep, no-deps so it won't re-resolve / downgrade torch
( cd "${ROOT}/wesep-real-tse" && pip install -e . --no-deps )

echo
echo ">>> env ready. NOTE: torch.cuda.is_available() is False on login01 (no GPU here)."
echo ">>> verify CUDA inside a GPU job:"
echo "    srun --partition=gpu2node --gres=gpu:1 --pty python -c 'import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
