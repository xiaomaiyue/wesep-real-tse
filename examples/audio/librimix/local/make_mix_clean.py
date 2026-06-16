#!/usr/bin/env python3
"""Generate Libri2Mix *mix_clean* (+ s1/s2) for a subset, locally.

Faithful reproduction of the official LibriMix ``mix_clean`` pipeline
(``create_librimix_from_metadata.py``) but:

  * only mix_clean is produced -> **no WHAM noise needed**
  * a ``--limit`` lets us generate a small subset quickly on a laptop
  * output layout matches what wesep's ``scan_librimix.py`` expects:

        {out_root}/Libri2Mix/wav{fs}/{mode}/{split}/
            mix_clean/<mixture_ID>.wav
            s1/<mixture_ID>.wav
            s2/<mixture_ID>.wav

Algorithm per mixture (identical to LibriMix):
    source_i  = read(librispeech_dir / source_i_path)        # 16 kHz float32
    source_i *= source_i_gain                                # loudness gain
    source_i  = resample_poly(source_i, fs, 16000)           # no-op at 16 kHz
    truncate all sources to the shortest length (mode=min)
    s1, s2    = the two transformed sources
    mix_clean = s1 + s2
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16000  # LibriSpeech native sample rate


def gen_one(librispeech_dir, row, fs, mode):
    paths = [row["source_1_path"], row["source_2_path"]]
    gains = [float(row["source_1_gain"]), float(row["source_2_gain"])]
    srcs = []
    for p, g in zip(paths, gains):
        x, sr = sf.read(os.path.join(librispeech_dir, p), dtype="float32")
        assert sr == RATE, f"expected {RATE} Hz, got {sr} for {p}"
        x = x * g
        if fs != RATE:
            x = resample_poly(x, fs, RATE)
        srcs.append(x)
    # fit lengths
    if mode == "min":
        L = min(len(s) for s in srcs)
        srcs = [s[:L] for s in srcs]
    else:  # max
        L = max(len(s) for s in srcs)
        srcs = [np.pad(s, (0, L - len(s))) for s in srcs]
    mix = srcs[0] + srcs[1]
    return srcs[0], srcs[1], mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech_dir", required=True)
    ap.add_argument("--libri2mix_csv", required=True)
    ap.add_argument("--split_name", required=True,
                    help="output dir name, e.g. train-100 / dev / test")
    ap.add_argument("--out_root", required=True,
                    help="a 'Libri2Mix' dir is created under here")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--mode", choices=["min", "max"], default="min")
    ap.add_argument("--limit", type=int, default=None,
                    help="only generate the first N mixtures")
    args = ap.parse_args()

    fs_tag = f"{args.fs // 1000}k"
    base = (Path(args.out_root) / "Libri2Mix" / f"wav{fs_tag}" /
            args.mode / args.split_name)
    dirs = {d: base / d for d in ("mix_clean", "s1", "s2")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with open(args.libri2mix_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[:args.limit]

    n = 0
    for row in rows:
        mix_id = row["mixture_ID"]
        s1, s2, mix = gen_one(args.librispeech_dir, row, args.fs, args.mode)
        sf.write(str(dirs["s1"] / f"{mix_id}.wav"), s1, args.fs)
        sf.write(str(dirs["s2"] / f"{mix_id}.wav"), s2, args.fs)
        sf.write(str(dirs["mix_clean"] / f"{mix_id}.wav"), mix, args.fs)
        n += 1
        if n % 500 == 0:
            print(f"  ... {n}/{len(rows)}")

    print(f"[{args.split_name}] generated {n} mixtures -> {base}")


if __name__ == "__main__":
    main()
