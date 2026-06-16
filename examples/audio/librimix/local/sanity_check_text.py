#!/usr/bin/env python3
"""Week-2 data-loading sanity check for text-guided Libri2Mix TSE.

Joins the three artifacts produced for a split and verifies they line up:

  * samples.jsonl            (audio index, from scan_librimix.py)
  * <split>.text_meta.jsonl  (text cues, from build_text_metadata.py)
  * the actual wav files      (mix_clean / s1 / s2)

Checks:
  1. every audio sample has a matching text-meta entry (join on `key`)
  2. mix / s1 / s2 wavs exist, load, share sample rate & length
  3. mix == s1 + s2 within PCM quantization error
  4. each target speaker has a non-empty transcript
Then prints a few "training sample cards" (what the model will actually see).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_jsonl(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["key"]] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_jsonl", required=True)
    ap.add_argument("--text_meta", required=True)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    samples = load_jsonl(args.samples_jsonl)
    text = load_jsonl(args.text_meta)
    print(f"audio samples : {len(samples)}")
    print(f"text-meta keys: {len(text)}")

    keys = list(samples)
    missing_text = [k for k in keys if k not in text]
    print(f"audio without text-meta: {len(missing_text)}"
          + (f"  e.g. {missing_text[:3]}" if missing_text else "  OK"))

    n_checked = 0
    n_audio_ok = 0
    n_mix_ok = 0
    n_text_ok = 0
    for k in keys:
        s = samples[k]
        mix_p = s["mix"]["default"][0]
        spk = s["spk"]
        s1_p = s["src"][spk[0]][0]
        s2_p = s["src"][spk[1]][0]
        try:
            mix, sr = sf.read(mix_p)
            a1, sr1 = sf.read(s1_p)
            a2, sr2 = sf.read(s2_p)
        except Exception as e:
            print(f"  [audio FAIL] {k}: {e}")
            continue
        n_checked += 1
        if sr == sr1 == sr2 and len(mix) == len(a1) == len(a2):
            n_audio_ok += 1
        if np.max(np.abs(mix - (a1 + a2))) < 2e-4:
            n_mix_ok += 1
        tm = text.get(k)
        if tm:
            txts = [t["target_transcript"].strip() for t in tm["targets"]]
            if all(len(t) > 0 for t in txts):
                n_text_ok += 1

    print(f"\nchecked {n_checked} samples:")
    print(f"  sr & length consistent : {n_audio_ok}/{n_checked}")
    print(f"  mix == s1+s2           : {n_mix_ok}/{n_checked}")
    print(f"  target transcripts ok  : {n_text_ok}/{n_checked}")

    print("\n=== sample cards ===")
    for k in keys[:args.show]:
        tm = text[k]
        mix, sr = sf.read(samples[k]["mix"]["default"][0])
        print(f"\nmix: {k}   ({len(mix)/sr:.2f}s)")
        for t in tm["targets"]:
            print(f"  TARGET spk {t['target_spk']} [{t['target_utt']}]")
            print(f"     text : {t['target_transcript']}")
            print(f"     vs interferer spk {t['interferer_spk']}: "
                  f"{t['interferer_transcript'][:60]}...")


if __name__ == "__main__":
    main()
