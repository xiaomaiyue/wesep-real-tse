#!/usr/bin/env python3
"""Week-4 smoke test: first text-guided TSE training run (local, tiny).

Goal is NOT a good model -- it is to prove the full pipeline end to end:
  samples.jsonl -> cue layer (text embedding npz) -> collate -> BSRNN+textemb
  -> SI-SDR loss -> backward -> loss decreases when overfitting a tiny set.

Uses the real wesep Dataset / collate / model classes (no mocks), but a
shrunken separator and a handful of mixtures so it runs on a laptop.
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (BASE_COLLECT_KEYS, build_collect_keys,
                                   tse_collate_fn)
from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK
from wesep.utils.file_utils import load_yaml
from wesep.utils.losses import parse_loss


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # stft/istft path must work; probe it
        try:
            x = torch.randn(1, 16000, device="mps")
            w = torch.hann_window(512, device="mps")
            s = torch.stft(x, 512, 128, window=w, return_complex=True)
            torch.istft(s, 512, 128, window=w)
            return torch.device("mps")
        except Exception as e:
            print(f"[smoke] MPS stft probe failed ({e}); falling back to CPU")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="samples.jsonl")
    ap.add_argument("--cues", required=True, help="cues yaml for this subset")
    ap.add_argument("--text_dim", type=int, default=384)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chunk_len", type=int, default=32000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[smoke] device = {device}")

    # ---- dataset configs (mirrors dataset_args, shrunk) ----
    ds_conf = {
        "resample_rate": 16000,
        "shuffle": True,
        "shuffle_args": {"shuffle_size": 64},
        "chunk_len": args.chunk_len,
        "noise_prob": 0,
        "cues": {
            "audio": {"use": False, "required": False},
            "text": {"use": True, "required": True},
        },
    }

    dataset = Dataset(
        "raw",
        args.samples,
        ds_conf,
        state="train",
        repeat_dataset=True,  # loop the tiny set
        cues_yaml=args.cues,
    )
    cues_conf = load_yaml(args.cues)
    collect_keys = build_collect_keys(cues_conf, ds_conf, BASE_COLLECT_KEYS)
    print(f"[smoke] collect keys: {list(collect_keys)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=lambda b: tse_collate_fn(b, collect_keys),
    )

    # ---- tiny text-only model ----
    cfg = {
        "separator": dict(sr=16000, win=512, stride=128, feature_dim=64,
                          num_repeat=2, causal=False, nspk=1),
        "speaker": {
            "features": {
                "spkemb": {"enabled": False},
                "textemb": {"enabled": True, "text_dim": args.text_dim,
                            "proj_dim": 192, "fusion": "multiply"},
            },
        },
    }
    model = TSE_BSRNN_SPK(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[smoke] model params: {n_param:.2f}M")

    criterion = parse_loss("SISDR")[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    losses = []
    t0 = time.time()
    it = iter(loader)
    for step in range(args.steps):
        batch = next(it)
        mix = batch["wav_mix"].float().to(device)
        target = batch["wav_target"].float().to(device)
        text = batch["text_aux"].float().to(device)

        est = model(mix, [text])  # (B, 1, T)
        loss = criterion(est, target).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        losses.append(loss.item())
        if step % 5 == 0 or step == args.steps - 1:
            print(f"  step {step:>3}  SISDR-loss {loss.item():8.3f}  "
                  f"({time.time() - t0:.0f}s)")

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    print(f"\n[smoke] mean loss first5={first:.3f}  last5={last:.3f}")
    if last < first:
        print("[smoke] PASS: loss decreased -> text-guided pipeline trains")
    else:
        print("[smoke] WARN: loss did not decrease; check pipeline")


if __name__ == "__main__":
    main()
