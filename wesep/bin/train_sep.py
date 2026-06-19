"""Standalone 2-speaker PIT separation trainer (route B).

Uses the in-repo BSRNN(nspk=2) separator, warm-started from the spk_emb_100
TSE checkpoint (sep_model.* backbone transfers; the nspk-dependent output head
is re-initialised). No cue -> no passthrough hedge (PIT demands two clean
speakers). Pure separation; the text cue is applied later by ASR routing.

  torchrun --nproc_per_node=N wesep/bin/train_sep.py \
    --samples <train samples.jsonl> --exp <exp_dir> \
    --init <spk_emb_100/avg_model.pt> --epochs 40
"""
import argparse
import json
import os
import random

import torch
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from wesep.modules.separator.bsrnn import BSRNN


class SepData(Dataset):
    def __init__(self, samples_jsonl, chunk=96000, sr=16000):
        self.items = []
        with open(samples_jsonl) as f:
            for line in f:
                d = json.loads(line)
                spks = d["spk"]
                if len(spks) < 2:
                    continue
                self.items.append((d["mix"]["default"][0],
                                   d["src"][spks[0]][0],
                                   d["src"][spks[1]][0]))
        self.chunk = chunk
        self.sr = sr

    def __len__(self):
        return len(self.items)

    def _load(self, p):
        w, sr = torchaudio.load(p)
        if sr != self.sr:
            w = torchaudio.functional.resample(w, sr, self.sr)
        return w.mean(0)

    def __getitem__(self, i):
        mix, s1, s2 = self.items[i]
        m, a, b = self._load(mix), self._load(s1), self._load(s2)
        L = min(len(m), len(a), len(b))
        m, a, b = m[:L], a[:L], b[:L]
        if L >= self.chunk:
            st = random.randint(0, L - self.chunk)
            sl = slice(st, st + self.chunk)
            m, a, b = m[sl], a[sl], b[sl]
        else:
            pad = self.chunk - L
            m = torch.nn.functional.pad(m, (0, pad))
            a = torch.nn.functional.pad(a, (0, pad))
            b = torch.nn.functional.pad(b, (0, pad))
        return m, torch.stack([a, b], 0)


def _sisnr(est, ref, eps=1e-8):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    s = ((est * ref).sum(-1, keepdim=True) * ref /
         (ref.pow(2).sum(-1, keepdim=True) + eps))
    n = est - s
    return 10 * torch.log10(s.pow(2).sum(-1) / (n.pow(2).sum(-1) + eps) + eps)


def pit_loss(est, ref):
    # est, ref: (B, 2, T) -> negative best-permutation mean SI-SNR
    a = _sisnr(est[:, 0], ref[:, 0]) + _sisnr(est[:, 1], ref[:, 1])
    b = _sisnr(est[:, 0], ref[:, 1]) + _sisnr(est[:, 1], ref[:, 0])
    return -torch.maximum(a, b).mean() / 2


def warmstart(model, ckpt, rank):
    st = torch.load(ckpt, map_location="cpu")["models"][0]
    sd = {k[len("sep_model."):]: v
          for k, v in st.items() if k.startswith("sep_model.")}
    msd = model.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    if rank == 0:
        print("[warmstart] loaded {}/{} (skipped {})".format(
            len(keep), len(msd), len(msd) - len(keep)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=96000)
    ap.add_argument("--iters_per_epoch", type=int, default=600)
    ap.add_argument("--final_lr", type=float, default=2.5e-5)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    model = BSRNN(sr=16000, win=512, stride=128, feature_dim=128,
                  num_repeat=6, causal=False, nspk=2, spec_dim=2).to(dev)
    if args.init and not args.resume:
        warmstart(model, args.init, rank)
    model = DDP(model, device_ids=[local])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    if args.resume:
        import re as _re
        ck = torch.load(args.resume, map_location="cpu")
        model.module.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        m = _re.search(r"ckpt_(\d+)", args.resume)
        start_epoch = ck.get("epoch", int(m.group(1)) if m else 0) + 1
        if rank == 0:
            print("[resume] from {} -> start epoch {}".format(
                args.resume, start_epoch), flush=True)

    ds = SepData(args.samples, chunk=args.chunk)
    samp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=samp, num_workers=4,
                    drop_last=True, pin_memory=True)
    if rank == 0:
        os.makedirs(args.exp + "/models", exist_ok=True)
        print("[data] {} mixtures | world={} bs={}".format(
            len(ds), world, args.bs), flush=True)

    for ep in range(start_epoch, args.epochs + 1):
        if args.epochs > 1:
            lr_ep = args.lr * (args.final_lr / args.lr) ** (
                (ep - 1) / (args.epochs - 1))
            for g in opt.param_groups:
                g["lr"] = lr_ep
        samp.set_epoch(ep)
        model.train()
        tot, n = 0.0, 0
        for it, (mix, ref) in enumerate(dl):
            mix, ref = mix.to(dev), ref.to(dev)
            with torch.cuda.amp.autocast():
                est = model(mix)
                L = min(est.shape[-1], ref.shape[-1])
                loss = pit_loss(est[..., :L], ref[..., :L])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
            if rank == 0 and (it + 1) % 50 == 0:
                print("ep{} it{} loss(-SISNR)={:.4f}".format(
                    ep, it + 1, tot / n), flush=True)
            if (it + 1) >= args.iters_per_epoch:
                break
        if rank == 0:
            print("=== epoch {} mean loss(-SISNR)={:.4f} ===".format(
                ep, tot / n), flush=True)
            torch.save({"model": model.module.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep},
                       "{}/models/ckpt_{}.pt".format(args.exp, ep))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
