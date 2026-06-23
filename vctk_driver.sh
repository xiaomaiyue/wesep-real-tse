#!/bin/bash
# End-to-end driver (run on login node): wait for VCTK download, extract,
# build accent mixtures, then submit the GPU eval job.
set -e
cd ~/tse1
LOG=~/tse1/vctk_driver.log
exec > >(tee -a "$LOG") 2>&1
echo "=== driver start $(date) ==="

# 1) wait for the wget to finish
while pgrep -f "wget.*DS_10283" >/dev/null; do sleep 60; done
echo "download finished, size: $(ls -la vctk_dl/DS_10283_3443.zip | awk '{print $5}')"

# 2) extract (idempotent)
ROOT=~/tse1/VCTK-Corpus-0.92
if [ ! -d "$ROOT/txt" ]; then
    mkdir -p "$ROOT"
    echo "unzip..."
    unzip -q -o vctk_dl/DS_10283_3443.zip -d "$ROOT"
    # some VCTK zips nest one level
    [ -d "$ROOT/VCTK-Corpus-0.92" ] && ROOT="$ROOT/VCTK-Corpus-0.92"
fi
[ -d "$ROOT/txt" ] || ROOT=$(dirname "$(find ~/tse1/VCTK-Corpus-0.92 -name speaker-info.txt | head -1)")
echo "VCTK root: $ROOT"

# 3) build per-accent 2-spk mixtures
PY=~/.conda/envs/wesep/bin/python
$PY wesep-real-tse/wesep/bin/build_vctk_mix.py \
    --vctk_root "$ROOT" --out ~/tse1/libri_data/vctk_mix \
    --pairs_per_accent 100 --seed 0

# 4) submit GPU eval
cd ~/tse1
sbatch -p gpu3node --gres=gpu:1 --mem=16G -c 4 -t 01:30:00 \
    -J vctk_robust -o ~/tse1/vctk_robust.out \
    --wrap "bash ~/tse1/run_vctk_robust.sh"
echo "=== eval submitted; results -> ~/tse1/vctk_robust.out ==="
squeue -u maiyue
