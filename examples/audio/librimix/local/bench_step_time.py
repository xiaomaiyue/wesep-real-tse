#!/usr/bin/env python3
"""Measure real training step time for the full-size text-TSE model.

Run this FIRST on any new machine (laptop / Colab / 4090 cluster) to get the
true s/step, then total training time = steps_per_epoch x s/step x epochs.

  steps_per_epoch = n_mixtures * 2 / batch_size      (each mix -> 2 samples)

Examples:
  python local/bench_step_time.py --device cuda --batch 8
  python local/bench_step_time.py --device cpu  --batch 4 --chunk 32000
"""
import argparse
import time

import torch

from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8,
                    help="expanded samples per step (mixtures x 2)")
    ap.add_argument("--chunk", type=int, default=48000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--num_repeat", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = {
        "separator": dict(sr=16000, win=512, stride=128,
                          feature_dim=args.feature_dim,
                          num_repeat=args.num_repeat, causal=False, nspk=1),
        "speaker": {"features": {
            "spkemb": {"enabled": False},
            "textemb": {"enabled": True, "text_dim": 384, "proj_dim": 192,
                        "fusion": "multiply"}}},
    }
    model = TSE_BSRNN_SPK(cfg).to(dev)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {n:.1f}M params | {args.feature_dim}x{args.num_repeat} | "
          f"batch {args.batch} x {args.chunk/16000:.0f}s | amp={args.amp}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    B, T = args.batch, args.chunk
    mix = torch.randn(B, 1, T, device=dev)
    txt = torch.randn(B, 384, device=dev)
    tgt = torch.randn(B, 1, T, device=dev)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    def step():
        with torch.autocast(device_type=dev.type, enabled=args.amp):
            out = model(mix, [txt])
            loss = ((out - tgt) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    for _ in range(3):  # warmup
        step()
    sync()
    t0 = time.time()
    for _ in range(args.steps):
        step()
    sync()
    dt = (time.time() - t0) / args.steps

    print(f"=> {dt:.3f} s/step   ({dt / (B * T / 16000):.4f} s per sample-sec)")
    for n_mix, name in ((2000, "subset-2k"), (13900, "train-100 full")):
        spe = n_mix * 2 / B
        print(f"   {name:<15} {spe:6.0f} steps/epoch | "
              f"epoch {spe * dt / 60:6.1f} min | "
              f"150 epochs {spe * dt * 150 / 3600:6.1f} h")
    if dev.type == "cuda":
        print(f"   VRAM peak: {torch.cuda.max_memory_allocated()/2**30:.1f} GB")


if __name__ == "__main__":
    main()
