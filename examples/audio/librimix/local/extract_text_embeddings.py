#!/usr/bin/env python3
"""Week-3: encode transcripts with a frozen text encoder and cache to disk.

Design notes (research-motivated):
  * The encoder is a *config axis*, not a hard-coded choice. Each run writes
    to its own subdir ``emb/{tag}/`` so encoder ablations (BERT vs MiniLM vs
    ...) share the exact same downstream pipeline.
  * Pooling is fixed to mean-pooling by default for *all* encoders so the
    encoder comparison is not confounded by pooling differences.
  * Embeddings are L2-normalized (recorded in meta.json) to keep the scale
    stable for multiply/FiLM fusion.

Input : transcripts.json   (utt_id -> {"transcript": str, ...})
Output: emb/{tag}/embeddings.npz   (utt_id -> float32 vector)
        emb/{tag}/meta.json        (model name, dim, pooling, norm)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# encoder registry: tag -> (hf_model_name, expected_dim)
ENCODERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "bert": ("bert-base-uncased", 768),
    "distilbert": ("distilbert-base-uncased", 768),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 768),
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode(texts, tokenizer, model, device, pooling="mean", batch_size=64,
           max_length=128):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state  # (b, L, H)
        if pooling == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif pooling == "cls":
            emb = hidden[:, 0]
        else:
            raise ValueError(pooling)
        out.append(emb.float().cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  ... {i + len(chunk)}/{len(texts)}")
    return torch.cat(out, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True,
                    help="transcripts.json from build_text_metadata.py")
    ap.add_argument("--encoder", default="minilm", choices=list(ENCODERS))
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--no_l2norm", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", required=True,
                    help="emb/{encoder} is created under here")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    model_name, dim = ENCODERS[args.encoder]
    device = pick_device()
    print(f"[enc] {args.encoder} = {model_name} on {device}")

    with open(args.transcripts, "r", encoding="utf-8") as f:
        tr = json.load(f)
    utt_ids = sorted(tr.keys())
    # LibriSpeech text is all-caps; uncased models lowercase anyway.
    texts = [tr[u]["transcript"] for u in utt_ids]
    print(f"[enc] {len(texts)} transcripts")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    t0 = time.time()
    embs = encode(texts, tokenizer, model, device,
                  pooling=args.pooling, batch_size=args.batch_size)
    assert embs.shape == (len(utt_ids), dim), embs.shape
    if not args.no_l2norm:
        embs = embs / np.clip(
            np.linalg.norm(embs, axis=1, keepdims=True), 1e-9, None)
    print(f"[enc] done in {time.time() - t0:.1f}s, shape={embs.shape}")

    out = Path(args.out_dir) / "emb" / args.encoder
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz",
             **{u: embs[i].astype(np.float32) for i, u in enumerate(utt_ids)})
    with (out / "meta.json").open("w") as f:
        json.dump({
            "encoder_tag": args.encoder,
            "model_name": model_name,
            "dim": dim,
            "pooling": args.pooling,
            "l2norm": not args.no_l2norm,
            "num_utts": len(utt_ids),
        }, f, indent=2)
    print(f"[enc] -> {out}/embeddings.npz")
    print(f"[enc] -> {out}/meta.json")


if __name__ == "__main__":
    main()
