"""Build 2-speaker mixtures from VCTK for accent / unseen-speaker robustness eval.

VCTK speakers are disjoint from LibriSpeech (=> unseen speakers) and carry
accent labels (=> accent axis). Output matches the schema eval_compare.py wants:
  - <out>/<accent>/samples.jsonl   : {key, spk:[id1,id2], mix.default, src{id:..}}
  - <out>/text_meta.jsonl          : {key, spk, src_text{utt:TEXT}}  (all accents)
  - <out>/accent_map.json          : accent -> list of sample keys (for breakdown)

Utterance ids use 'pXXX-NNN' so eval_compare's utt.split('-')[0]==spk holds.

  python build_vctk_mix.py --vctk_root ~/tse1/VCTK-Corpus-0.92 \
      --out ~/tse1/libri_data/vctk_mix --pairs_per_accent 80 --seed 0
"""
import glob
import json
import os
import random
import re

import fire
import torch
import torchaudio

VOCAB = set(" 'ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def clean_text(s):
    s = s.upper()
    return "".join(c for c in s if c in VOCAB).strip()


def parse_speakers(root):
    """speaker-info.txt -> {spk: accent}. Columns: ID AGE GENDER ACCENT REGION."""
    info = os.path.join(root, "speaker-info.txt")
    spk2acc = {}
    with open(info) as f:
        for line in f.readlines()[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            sid, acc = parts[0], parts[3]
            spk2acc["p" + sid if not sid.startswith("p") else sid] = acc
    return spk2acc


def find_wav_dir(root):
    for cand in ("wav48_silence_trimmed", "wav48", "wav16"):
        d = os.path.join(root, cand)
        if os.path.isdir(d):
            return d
    raise SystemExit("no wav dir under %s" % root)


def utts_for(wav_dir, txt_dir, spk):
    """Return [(utt_id, flac_path, text)] for a speaker that have a transcript."""
    out = []
    sd = os.path.join(wav_dir, spk)
    if not os.path.isdir(sd):
        return out
    for p in sorted(glob.glob(os.path.join(sd, "*mic1.flac"))
                    or glob.glob(os.path.join(sd, "*.flac"))
                    or glob.glob(os.path.join(sd, "*.wav"))):
        base = os.path.basename(p)
        m = re.match(r"(p?\d+)_(\d+)", base)
        if not m:
            continue
        num = m.group(2)
        tp = os.path.join(txt_dir, spk, "%s_%s.txt" % (spk, num))
        if not os.path.exists(tp):
            continue
        with open(tp) as f:
            txt = clean_text(f.read())
        if len(txt) < 5:
            continue
        out.append(("%s-%s" % (spk, num), p, txt))
    return out


def load_16k(path):
    w, sr = torchaudio.load(path)
    w = w.mean(0)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    # trim leading/trailing near-silence so VCTK pauses don't dominate
    e = w.abs()
    idx = (e > e.max() * 0.01).nonzero()
    if idx.numel():
        w = w[idx[0, 0]:idx[-1, 0] + 1]
    return w


def main(vctk_root, out, pairs_per_accent=80, seed=0, min_spk=2,
         max_accents=8, sr=16000):
    random.seed(seed)
    torch.manual_seed(seed)
    wav_dir = find_wav_dir(vctk_root)
    txt_dir = os.path.join(vctk_root, "txt")
    spk2acc = parse_speakers(vctk_root)

    # group speakers (that actually have audio) by accent
    by_acc = {}
    for spk, acc in spk2acc.items():
        if os.path.isdir(os.path.join(wav_dir, spk)):
            by_acc.setdefault(acc, []).append(spk)
    by_acc = {a: s for a, s in by_acc.items() if len(s) >= min_spk}
    # rank accents by #speakers, keep the largest groups
    accents = sorted(by_acc, key=lambda a: -len(by_acc[a]))[:max_accents]
    print("accents kept:", {a: len(by_acc[a]) for a in accents})

    os.makedirs(out, exist_ok=True)
    text_meta = open(os.path.join(out, "text_meta.jsonl"), "w")
    accent_map = {}
    seen_meta = set()

    for acc in accents:
        spks = by_acc[acc]
        # cache utterance lists lazily
        cache = {}

        def get(s):
            if s not in cache:
                cache[s] = utts_for(wav_dir, txt_dir, s)
            return cache[s]

        adir = os.path.join(out, acc)
        os.makedirs(os.path.join(adir, "mix_clean"), exist_ok=True)
        os.makedirs(os.path.join(adir, "s1"), exist_ok=True)
        os.makedirs(os.path.join(adir, "s2"), exist_ok=True)
        sf = open(os.path.join(adir, "samples.jsonl"), "w")
        keys = []
        tries = 0
        made = 0
        while made < pairs_per_accent and tries < pairs_per_accent * 20:
            tries += 1
            s1, s2 = random.sample(spks, 2)
            u1, u2 = get(s1), get(s2)
            if not u1 or not u2:
                continue
            a1, a2 = random.choice(u1), random.choice(u2)
            id1, p1, t1 = a1
            id2, p2, t2 = a2
            w1, w2 = load_16k(p1), load_16k(p2)
            L = min(w1.numel(), w2.numel())
            if L < sr:  # need >=1s
                continue
            w1, w2 = w1[:L], w2[:L]
            # mix at 0 dB relative (equal source power)
            p1p = w1.pow(2).mean().clamp_min(1e-8)
            p2p = w2.pow(2).mean().clamp_min(1e-8)
            w2 = w2 * (p1p / p2p).sqrt()
            mix = w1 + w2
            peak = mix.abs().max().clamp_min(1e-8)
            if peak > 1.0:
                mix, w1, w2 = mix / peak, w1 / peak, w2 / peak
            key = "%s_%s" % (id1, id2)
            mp = os.path.join(adir, "mix_clean", key + ".wav")
            s1p = os.path.join(adir, "s1", key + ".wav")
            s2p = os.path.join(adir, "s2", key + ".wav")
            torchaudio.save(mp, mix.unsqueeze(0), sr)
            torchaudio.save(s1p, w1.unsqueeze(0), sr)
            torchaudio.save(s2p, w2.unsqueeze(0), sr)
            sf.write(json.dumps({
                "key": key, "spk": [s1, s2],
                "mix": {"default": [mp]},
                "src": {s1: [s1p], s2: [s2p]},
            }) + "\n")
            if key not in seen_meta:
                seen_meta.add(key)
                text_meta.write(json.dumps({
                    "key": key, "spk": [s1, s2],
                    "src_text": {id1: t1, id2: t2},
                }) + "\n")
            keys.append(key)
            made += 1
        sf.close()
        accent_map[acc] = keys
        print("%-12s pairs=%d" % (acc, made))

    text_meta.close()
    json.dump(accent_map, open(os.path.join(out, "accent_map.json"), "w"),
              indent=0)
    print("done ->", out)


if __name__ == "__main__":
    fire.Fire(main)
