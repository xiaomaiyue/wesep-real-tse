#!/bin/bash
# Accent / unseen-speaker robustness eval for the route-B system.
# SEP_LC separator + MATCHER_WHAM matcher, clean + 0 dB WHAM, per VCTK accent,
# with the in-domain LibriMix test set as reference.
set -e
cd ~/tse1/wesep-real-tse/examples/audio/librimix
PY=~/.conda/envs/wesep/bin/python
SEP=exp/SEP_LC/models/${SEP_CKPT:-ckpt_100.pt}
MAT=exp/MATCHER_WHAM/models/${MAT_CKPT:-matcher_80.pt}
WHAM=~/tse1/wham/wham_noise/tt
VROOT=~/tse1/libri_data/vctk_mix
META=$VROOT/text_meta.jsonl
DEV=${DEV:-0}
N=${N:-200}
NR=${NR:-8}
EC="$PY ../../../wesep/bin/eval_compare.py --sep_ckpt $SEP --matcher $MAT \
    --device $DEV --num_repeat $NR --meta $META --n $N"

echo "##### SEP=$SEP  MAT=$MAT"
echo

# In-domain LibriMix baseline (reference)
LS=~/tse1/libri_data/Libri2Mix/wav16k/min/test/samples.jsonl
LM=~/tse1/libri_data/text_cues/test.text_meta.jsonl
echo "===== BASELINE  LibriMix-test (in-domain, unseen US-English spk) ====="
$PY ../../../wesep/bin/eval_compare.py --sep_ckpt $SEP --matcher $MAT \
    --device $DEV --num_repeat $NR --meta $LM --samples $LS --n $N
$PY ../../../wesep/bin/eval_compare.py --sep_ckpt $SEP --matcher $MAT \
    --device $DEV --num_repeat $NR --meta $LM --samples $LS --n $N --snr 0 --noise_dir $WHAM
echo

# Per-accent VCTK (unseen speakers + accent shift)
for d in $VROOT/*/; do
    acc=$(basename "$d")
    [ -f "$d/samples.jsonl" ] || continue
    echo "===== VCTK accent: $acc ====="
    $EC --samples "$d/samples.jsonl"
    $EC --samples "$d/samples.jsonl" --snr 0 --noise_dir $WHAM
    echo
done
