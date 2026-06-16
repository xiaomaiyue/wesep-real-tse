#!/bin/bash
# One-shot text-cue data preparation (Week 2-3), reproducible on laptop/cluster.
#
# Per split: transcript metadata -> cue resources (oracle/interferer/random)
# Then: one shared embedding cache per text encoder.
#
# Usage:
#   ./local/prepare_text_data.sh \
#     --librispeech_dir /path/LibriSpeech \
#     --libri2mix_meta  /path/LibriMix/metadata/Libri2Mix \
#     --out_dir         /path/text_cues \
#     --splits          "train-100 dev test" \
#     --encoders        "minilm"
set -e

librispeech_dir=
libri2mix_meta=
out_dir=
splits="train-100 dev test"
encoders="minilm"
python_bin=${PYTHON_BIN:-python}

. tools/parse_options.sh 2>/dev/null || {
  # standalone arg parsing when run outside the recipe dir
  while [ $# -gt 0 ]; do
    case "$1" in
      --librispeech_dir) librispeech_dir=$2; shift 2;;
      --libri2mix_meta)  libri2mix_meta=$2;  shift 2;;
      --out_dir)         out_dir=$2;         shift 2;;
      --splits)          splits=$2;          shift 2;;
      --encoders)        encoders=$2;        shift 2;;
      *) echo "unknown option $1"; exit 1;;
    esac
  done
}

[ -z "$librispeech_dir" ] && echo "--librispeech_dir required" && exit 1
[ -z "$libri2mix_meta" ] && echo "--libri2mix_meta required" && exit 1
[ -z "$out_dir" ] && echo "--out_dir required" && exit 1

local_dir=$(dirname "$0")

declare -A SPLIT2CSV=(
  ["train-100"]="libri2mix_train-clean-100.csv"
  ["train-360"]="libri2mix_train-clean-360.csv"
  ["dev"]="libri2mix_dev-clean.csv"
  ["test"]="libri2mix_test-clean.csv"
)

echo "=== Stage 1: transcript metadata per split ==="
for split in $splits; do
  csv=${SPLIT2CSV[$split]}
  [ -z "$csv" ] && echo "unknown split: $split" && exit 1
  $python_bin "$local_dir/build_text_metadata.py" \
    --librispeech_dir "$librispeech_dir" \
    --libri2mix_csv "$libri2mix_meta/$csv" \
    --split_name "$split" \
    --out_dir "$out_dir"
done

echo "=== Stage 2: cue resources (oracle / interferer / random) ==="
for split in $splits; do
  $python_bin "$local_dir/build_text_cue_resources.py" \
    --text_meta "$out_dir/${split}.text_meta.jsonl" \
    --out_dir "$out_dir" \
    --split_name "$split"
done

echo "=== Stage 3: embedding caches ==="
for enc in $encoders; do
  $python_bin "$local_dir/extract_text_embeddings.py" \
    --transcripts "$out_dir/transcripts.json" \
    --encoder "$enc" \
    --out_dir "$out_dir"
done

echo "=== Done. Artifacts under $out_dir ==="
