#!/usr/bin/env python3
"""Build text-cue resource JSONs for one split. **The ablation switchboard.**

A resource maps  "mix_key::target_spk" -> {"utt_id", "emb_key"}  where
``emb_key`` indexes embeddings.npz. All cue conditions share the same
embedding cache; an ablation is just *a different mapping file* referenced
from cues.yaml -- zero code change:

  oracle      emb_key = the target's own transcript      (upper bound)
  interferer  emb_key = the *other* speaker's transcript (cue-following test:
              if the model truly uses text, it should switch target)
  random      emb_key = a transcript from an unrelated mixture
              (over-trust test: does wrong semantics hurt?)
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_meta", required=True,
                    help="<split>.text_meta.jsonl from build_text_metadata.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with open(args.text_meta, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    all_utts = sorted({t["target_utt"] for r in rows for t in r["targets"]})

    oracle, interferer, rand = {}, {}, {}
    for r in rows:
        for t in r["targets"]:
            k = f"{r['key']}::{t['target_spk']}"
            oracle[k] = {"utt_id": t["target_utt"],
                         "emb_key": t["target_utt"]}
            interferer[k] = {"utt_id": t["target_utt"],
                             "emb_key": t["interferer_utt"]}
            # random text: any utterance not in this mixture
            while True:
                ru = rng.choice(all_utts)
                if ru != t["target_utt"] and ru != t["interferer_utt"]:
                    break
            rand[k] = {"utt_id": t["target_utt"], "emb_key": ru}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, res in (("oracle", oracle), ("interferer", interferer),
                      ("random", rand)):
        p = out / f"{args.split_name}.text_{name}.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=0)
        print(f"[{args.split_name}] {name:<10} {len(res):>6} keys -> {p}")


if __name__ == "__main__":
    main()
