#!/usr/bin/env python3
"""Build single-speaker data + utt-level text resource for online dynamic mixing.

The existing Libri2Mix samples.jsonl already gives us, per mixture, the two
separated single-speaker sources (s1/s2) and their utterance ids (the mixture
key is "<utt1>_<utt2>"). We deduplicate them into a pool of single-speaker
utterances that `online_mix` (sample_speaker_group + snr_mixer) randomly pairs
and remixes each epoch -> effectively unlimited training mixtures.

Outputs (next to the input samples.jsonl):
  <split>/samples_single.jsonl   single-spk pool: {key:utt_id, spk:[id], src:{id:[wav]}}
  <split>/cues_text_oracle_online.yaml  text cue, key=utt_id, identity resource
and a shared:
  text_cues/utt_identity.json    {utt_id: {utt_id, emb_key: utt_id}}  (npz-verified)
"""
import argparse
import json
import os

import numpy as np


def build_split(samples_jsonl, npz_keys):
    """Yield deduped single-speaker entries; collect utt ids actually emitted."""
    seen = set()
    out = []
    emitted = set()
    skipped = 0
    with open(samples_jsonl) as f:
        for line in f:
            s = json.loads(line)
            key = s["key"]
            spks = s["spk"]            # [spkA, spkB]  (A->s1, B->s2)
            parts = key.split("_")
            if len(parts) != 2 or len(spks) != 2:
                skipped += 1
                continue
            for utt_id, spk in zip(parts, spks):
                if utt_id.split("-")[0] != spk:
                    # sanity: utt_id prefix must equal its speaker id
                    skipped += 1
                    continue
                if utt_id in seen:
                    continue
                seen.add(utt_id)
                if utt_id not in npz_keys:
                    skipped += 1            # no text embedding -> drop
                    continue
                wav = s["src"][spk][0]
                out.append({"key": utt_id, "spk": [spk], "src": {spk: [wav]}})
                emitted.add(utt_id)
    return out, emitted, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="...Libri2Mix/wav16k/min")
    ap.add_argument("--text_cues_dir", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--splits", nargs="+",
                    default=["train-100-full", "dev"])
    args = ap.parse_args()

    npz = np.load(args.npz)
    npz_keys = set(npz.files)
    print(f"npz has {len(npz_keys)} embeddings")

    all_utts = set()
    for split in args.splits:
        sj = os.path.join(args.data_root, split, "samples.jsonl")
        entries, emitted, skipped = build_split(sj, npz_keys)
        out_jsonl = os.path.join(args.data_root, split, "samples_single.jsonl")
        with open(out_jsonl, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        all_utts |= emitted
        print(f"[{split}] {len(entries)} single-spk utts -> {out_jsonl} "
              f"(skipped {skipped})")

    # shared identity text resource (emb_key == utt_id)
    res = {u: {"utt_id": u, "emb_key": u} for u in sorted(all_utts)}
    res_path = os.path.join(args.text_cues_dir, "utt_identity.json")
    with open(res_path, "w") as f:
        json.dump(res, f)
    print(f"identity resource: {len(res)} utts -> {res_path}")

    # per-split cue yaml (utt_id keying, identity resource, same npz)
    for split in args.splits:
        yml = os.path.join(args.data_root, split, "cues_text_oracle_online.yaml")
        with open(yml, "w") as f:
            f.write(
                "cues:\n"
                "  text:\n"
                "    type: embedding\n"
                "    guaranteed: true\n"
                "    scope: speaker\n"
                "    policy:\n"
                "      type: fixed\n"
                "      key: utt_id\n"
                f"      resource: {res_path}\n"
                f"      embeddings: {args.npz}\n")
        print(f"[{split}] cue yaml -> {yml}")


if __name__ == "__main__":
    main()
