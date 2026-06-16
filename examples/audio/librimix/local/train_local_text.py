#!/usr/bin/env python3
"""Local small-scale text-TSE training for Apple Silicon (MPS) / CPU.

Trains the FULL-SIZE BSRNN+textemb (same architecture as the cluster config)
on a Libri2Mix subset, with the memory recipe that survives 16GB unified
memory: micro-batch of 1 mixture (=2 expanded samples) x 2s chunks +
gradient accumulation.

Features: per-epoch validation (SI-SDR + SI-SDRi vs mixture), best/latest
checkpoints, auto-resume, CSV log, MPS OOM self-healing.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (BASE_COLLECT_KEYS, build_collect_keys,
                                   tse_collate_fn)
from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK
from wesep.utils.file_utils import load_yaml
from wesep.utils.losses import parse_loss


def build_loader(samples, cues, ds_conf, state, batch_mix):
    dataset = Dataset("raw", samples, ds_conf, state=state,
                      repeat_dataset=False, cues_yaml=cues)
    cues_conf = load_yaml(cues)
    keys = build_collect_keys(cues_conf, ds_conf, BASE_COLLECT_KEYS)
    loader = DataLoader(dataset, batch_size=batch_mix, num_workers=0,
                        collate_fn=lambda b: tse_collate_fn(b, keys))
    return dataset, loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_samples", required=True)
    ap.add_argument("--train_cues", required=True)
    ap.add_argument("--val_samples", required=True)
    ap.add_argument("--val_cues", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_mix", type=int, default=1,
                    help="mixtures per micro-step (x2 = samples)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32000)
    ap.add_argument("--val_mixtures", type=int, default=100)
    ap.add_argument("--val_cap_sec", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_sched", action="store_true",
                    help="constant LR (disable ReduceLROnPlateau) -- use for "
                         "overfit probes so LR decay can't confound the result")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--num_repeat", type=int, default=6)
    ap.add_argument("--text_dim", type=int, default=384)
    ap.add_argument("--fusion", default="multiply")
    args = ap.parse_args()

    dev = torch.device(args.device)
    exp = Path(args.exp_dir)
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "args.json").write_text(json.dumps(vars(args), indent=2))

    ds_conf = {
        "resample_rate": 16000,
        "shuffle": True,
        "shuffle_args": {"shuffle_size": 1500},
        "chunk_len": args.chunk,
        "noise_prob": 0,
        "cues": {"audio": {"use": False, "required": False},
                 "text": {"use": True, "required": True}},
    }
    val_conf = dict(ds_conf, shuffle=False, whole_utt=True)

    train_ds, train_loader = build_loader(
        args.train_samples, args.train_cues, ds_conf, "train", args.batch_mix)
    _, val_loader = build_loader(
        args.val_samples, args.val_cues, val_conf, "val", 1)

    cfg = {"separator": dict(sr=16000, win=512, stride=128,
                             feature_dim=args.feature_dim,
                             num_repeat=args.num_repeat, causal=False, nspk=1),
           "speaker": {"features": {
               "spkemb": {"enabled": False},
               "textemb": {"enabled": True, "text_dim": args.text_dim,
                           "proj_dim": 192, "fusion": args.fusion}}}}
    model = TSE_BSRNN_SPK(cfg).to(dev)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    criterion = parse_loss("SISDR")[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3)

    # ---- resume ----
    start_ep, best_val = 1, float("-inf")
    latest = exp / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        best_val = ck["best_val"]
        print(f"[resume] from epoch {ck['epoch']} (best val {best_val:.2f})")

    log_path = exp / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_sisdr", "val_sisdri",
                 "lr", "minutes", "oom_skips"])

    cap = int(args.val_cap_sec * 16000)

    @torch.no_grad()
    def validate():
        model.eval()
        tot_est, tot_mix, n = 0.0, 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= args.val_mixtures:
                break
            mix = batch["wav_mix"].float().to(dev)[..., :cap]
            tgt = batch["wav_target"].float().to(dev)[..., :cap]
            txt = batch["text_aux"].float().to(dev)
            est = model(mix, [txt])
            # istft trims to a multiple of the hop -> align all lengths
            L = min(est.shape[-1], tgt.shape[-1])
            est, tgt, mix = est[..., :L], tgt[..., :L], mix[..., :L]
            tot_est += -criterion(est, tgt).item()   # SI-SDR of estimate
            tot_mix += -criterion(mix, tgt).item()   # SI-SDR of raw mixture
            n += 1
        model.train()
        return tot_est / n, (tot_est - tot_mix) / n

    print(f"[train] {n_param:.1f}M params on {dev} | micro-batch "
          f"{args.batch_mix * 2} samples x {args.chunk / 16000:.0f}s "
          f"x accum {args.accum} | epochs {start_ep}..{args.epochs}")

    for ep in range(start_ep, args.epochs + 1):
        train_ds.set_epoch(ep)
        t0 = time.time()
        run_loss, n_steps, oom_skips = 0.0, 0, 0
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            try:
                mix = batch["wav_mix"].float().to(dev)
                tgt = batch["wav_target"].float().to(dev)
                txt = batch["text_aux"].float().to(dev)
                loss = criterion(model(mix, [txt]), tgt).mean() / args.accum
                loss.backward()
                if (i + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad()
                run_loss += loss.item() * args.accum
                n_steps += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_skips += 1
                    opt.zero_grad()
                    if dev.type == "mps":
                        torch.mps.empty_cache()
                    continue
                raise
            if n_steps % 400 == 0 and n_steps > 0:
                print(f"  ep{ep} step {n_steps} loss "
                      f"{run_loss / n_steps:7.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()

        # save BEFORE validation: a val crash must never lose training
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": ep,
                 "best_val": best_val, "cfg": cfg}
        torch.save(state, latest)

        try:
            val_sisdr, val_sisdri = validate()
        except Exception as e:
            print(f"        [val failed: {str(e)[:90]}] continuing", flush=True)
            val_sisdr, val_sisdri = float("nan"), float("nan")
        if not args.no_sched and val_sisdr == val_sisdr:  # not NaN
            sched.step(val_sisdr)
        mins = (time.time() - t0) / 60
        lr_now = opt.param_groups[0]["lr"]
        tr_loss = run_loss / max(n_steps, 1)
        print(f"[ep {ep:>3}] train_loss {tr_loss:7.3f} | "
              f"val SI-SDR {val_sisdr:6.2f} dB | SI-SDRi {val_sisdri:6.2f} dB"
              f" | lr {lr_now:.1e} | {mins:.1f} min | oom_skip {oom_skips}",
              flush=True)
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [ep, f"{tr_loss:.4f}", f"{val_sisdr:.3f}",
                 f"{val_sisdri:.3f}", f"{lr_now:.2e}", f"{mins:.1f}",
                 oom_skips])

        if val_sisdr == val_sisdr and val_sisdr > best_val:
            best_val = val_sisdr
            state["best_val"] = best_val
            torch.save(state, exp / "best.pt")
            print(f"        new best ({best_val:.2f} dB) -> best.pt",
                  flush=True)

    print(f"[done] best val SI-SDR {best_val:.2f} dB | ckpt: {exp}/best.pt")


if __name__ == "__main__":
    main()
