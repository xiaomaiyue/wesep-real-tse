#!/usr/bin/env python3
"""Build text-cue metadata for Libri2Mix TSE (Stage 1, Week 2).

For each Libri2Mix mixture we already know *exactly* which two LibriSpeech
utterances were mixed (from the official Libri2Mix metadata CSV). Because
Libri2Mix is built from LibriSpeech, every source utterance has a
human-verified ground-truth transcript in LibriSpeech's ``*.trans.txt``
files. So building the text cue is a pure lookup -- **no ASR needed**.

This script produces three artifacts under ``--out_dir``:

  1. ``transcripts.json``    utt_id -> {"transcript": str, "n_words": int}
                             (the "oracle transcript" cue resource)
  2. ``<split>.text_meta.jsonl``
                             one line per mixture, listing the target and
                             interferer transcript for *both* orientations
                             (target = s1, or target = s2)
  3. ``<split>.text_cues.csv``
                             flat, human-readable table (2 rows per mixture)

The ``key`` / ``spk`` fields mirror ``scan_librimix.py`` so this metadata
lines up 1:1 with the separator's ``samples.jsonl``.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path


def index_transcripts(librispeech_dir: Path):
    """Walk LibriSpeech and index every utterance's transcript.

    Returns utt_id -> transcript (UPPERCASE, as shipped by LibriSpeech).
    followlinks=True so a symlinked split (e.g. test-clean) is included.
    """
    utt2text = {}
    n_files = 0
    for root, _dirs, files in os.walk(librispeech_dir, followlinks=True):
        for fn in files:
            if not fn.endswith(".trans.txt"):
                continue
            n_files += 1
            with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # line: "1578-6379-0038 AND HE LOOKED ABOUT HIM"
                    utt_id, _, text = line.partition(" ")
                    utt2text[utt_id] = text
    print(f"[index] scanned {n_files} .trans.txt files, "
          f"{len(utt2text)} utterances")
    return utt2text


def utt_id_from_source_path(source_path: str) -> str:
    """'train-clean-100/1578/6379/1578-6379-0038.flac' -> '1578-6379-0038'."""
    return Path(source_path).stem


def n_words(text: str) -> int:
    return len(text.split())


def build(csv_path: Path, utt2text: dict, split_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / f"{split_name}.text_meta.jsonl"
    flat_path = out_dir / f"{split_name}.text_cues.csv"
    used_transcripts = {}

    n_rows = 0
    n_missing = 0
    missing_examples = []

    with open(csv_path, "r", encoding="utf-8") as cf, \
            meta_path.open("w", encoding="utf-8") as mf, \
            flat_path.open("w", encoding="utf-8", newline="") as ff:
        reader = csv.DictReader(cf)
        writer = csv.writer(ff)
        writer.writerow([
            "mix_key", "target_spk", "target_utt", "interferer_spk",
            "interferer_utt", "n_words", "target_transcript"
        ])

        for row in reader:
            mix_key = row["mixture_ID"]
            u1 = utt_id_from_source_path(row["source_1_path"])
            u2 = utt_id_from_source_path(row["source_2_path"])
            spk1, spk2 = u1.split("-")[0], u2.split("-")[0]

            t1 = utt2text.get(u1)
            t2 = utt2text.get(u2)
            for uid, t in ((u1, t1), (u2, t2)):
                if t is None:
                    n_missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(uid)
                else:
                    used_transcripts[uid] = t
            if t1 is None or t2 is None:
                # skip mixtures we cannot fully resolve (should be 0 when the
                # matching LibriSpeech split is downloaded)
                continue

            # one mixture -> two target orientations
            targets = []
            for (tu, ts, tx), (iu, is_, ix) in (
                ((u1, spk1, t1), (u2, spk2, t2)),
                ((u2, spk2, t2), (u1, spk1, t1)),
            ):
                targets.append({
                    "target_spk": ts,
                    "target_utt": tu,
                    "target_transcript": tx,
                    "interferer_spk": is_,
                    "interferer_utt": iu,
                    "interferer_transcript": ix,
                })
                writer.writerow([mix_key, ts, tu, is_, iu, n_words(tx), tx])

            mf.write(json.dumps({
                "key": mix_key,
                "spk": [spk1, spk2],
                "src_text": {u1: t1, u2: t2},
                "targets": targets,
            }, ensure_ascii=False) + "\n")
            n_rows += 1

    # write the per-utterance oracle transcript resource
    tjson_path = out_dir / "transcripts.json"
    # merge with any existing transcripts.json so multiple splits accumulate
    merged = {}
    if tjson_path.exists():
        with tjson_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
    for uid, text in used_transcripts.items():
        merged[uid] = {"transcript": text, "n_words": n_words(text)}
    with tjson_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=0)

    print(f"[{split_name}] mixtures written : {n_rows}")
    print(f"[{split_name}] unique utterances: {len(used_transcripts)}")
    print(f"[{split_name}] missing lookups  : {n_missing}"
          + (f"  e.g. {missing_examples}" if missing_examples else ""))
    print(f"[{split_name}] -> {meta_path}")
    print(f"[{split_name}] -> {flat_path}")
    print(f"[{split_name}] -> {tjson_path} (total {len(merged)} utts)")
    return n_rows, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech_dir", required=True,
                    help="parent dir containing train-clean-100/, dev-clean/, "
                         "test-clean/ ...")
    ap.add_argument("--libri2mix_csv", required=True,
                    help="official Libri2Mix metadata CSV for one split")
    ap.add_argument("--split_name", required=True,
                    help="output name, e.g. train-100 / dev / test")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    utt2text = index_transcripts(Path(args.librispeech_dir))
    build(Path(args.libri2mix_csv), utt2text, args.split_name,
          Path(args.out_dir))


if __name__ == "__main__":
    main()
