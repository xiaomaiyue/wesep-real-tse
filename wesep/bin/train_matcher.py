"""自训「语音内容 ↔ 文本」对比匹配器(无 ASR 路由用)。

音频侧: 冻结 wav2vec2 -> mean-pool -> 投影。
文本侧: 字符级 embedding -> BiGRU -> 投影(从零训,推理时不需外部文本模型)。
损失:   InfoNCE(batch 内 audio_i 与 text_i 配对,其余为负例)。
训练对: (干净源音频, 其转录文本),取自 train-100。

  torchrun --nproc_per_node=N wesep/bin/train_matcher.py \
    --samples <train samples.jsonl> --meta <train.text_meta.jsonl> \
    --exp <exp_dir> --epochs 60
"""
import argparse
import glob
import json
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

VOCAB = " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C2I = {c: i + 1 for i, c in enumerate(VOCAB)}        # 0 = pad


def encode_text(s, maxlen=240):
    s = s.upper()
    return [C2I[c] for c in s if c in C2I][:maxlen]


def _mix_at_snr(clean, noise, snr_db):
    cp = clean.pow(2).mean()
    npow = noise.pow(2).mean() + 1e-8
    g = (cp / (10 ** (snr_db / 10.0)) / npow).sqrt()
    return clean + noise * g


def add_noise(w, snr_db):
    return _mix_at_snr(w, torch.randn_like(w), snr_db)


class ATData(Dataset):
    def __init__(self, samples, meta, chunk=64000, sr=16000,
                 noise_prob=0.0, snr_min=0.0, snr_max=20.0, noise_dir=None,
                 gauss_prob=0.0):
        self.noise_prob = noise_prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.gauss_prob = gauss_prob          # when noising, use Gaussian (not
        #                                       WHAM) with this prob -> mix types
        self.noise_paths = (sorted(glob.glob(
            os.path.join(noise_dir, "**", "*.wav"), recursive=True))
            if noise_dir else [])
        text = {}
        with open(meta) as f:
            for line in f:
                d = json.loads(line)
                for u, t in d.get("src_text", {}).items():
                    text[u] = t
        self.items = []
        with open(samples) as f:
            for line in f:
                d = json.loads(line)
                for s in d["spk"]:
                    utt = next((u for u in text
                                if u.split("-")[0] == s and u in d["key"]),
                               None)
                    if utt and s in d["src"]:
                        self.items.append((d["src"][s][0], text[utt]))
        self.chunk = chunk
        self.sr = sr

    def __len__(self):
        return len(self.items)

    def _real_noise(self, length):
        n, sr = torchaudio.load(random.choice(self.noise_paths))
        if sr != self.sr:
            n = torchaudio.functional.resample(n, sr, self.sr)
        n = n.mean(0)
        if len(n) >= length:
            st = random.randint(0, len(n) - length)
            n = n[st:st + length]
        else:
            n = n.repeat(length // len(n) + 1)[:length]
        return n

    def __getitem__(self, i):
        p, t = self.items[i]
        w, sr = torchaudio.load(p)
        if sr != self.sr:
            w = torchaudio.functional.resample(w, sr, self.sr)
        w = w.mean(0)
        if len(w) >= self.chunk:
            st = random.randint(0, len(w) - self.chunk)
            w = w[st:st + self.chunk]
        else:
            w = F.pad(w, (0, self.chunk - len(w)))
        if self.noise_prob > 0 and random.random() < self.noise_prob:
            snr = random.uniform(self.snr_min, self.snr_max)
            if self.noise_paths and random.random() >= self.gauss_prob:
                w = _mix_at_snr(w, self._real_noise(len(w)), snr)
            else:
                w = add_noise(w, snr)
        return w, torch.tensor(encode_text(t), dtype=torch.long)


def collate(batch):
    ws = torch.stack([b[0] for b in batch])
    maxl = max(len(b[1]) for b in batch)
    ids = torch.zeros(len(batch), maxl, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b[1])] = b[1]
    return ws, ids


class AudioEnc(nn.Module):
    def __init__(self, w2v, d=256):
        super().__init__()
        self.w2v = w2v
        self.proj = nn.Sequential(nn.Linear(768, 512), nn.GELU(),
                                  nn.Linear(512, d))

    def forward(self, wav):
        with torch.no_grad():
            f = self.w2v.extract_features(wav)[0][-1]    # (B,T,768)
        return self.proj(f.mean(1))


class TextEnc(nn.Module):
    def __init__(self, vocab=len(VOCAB) + 1, d=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, 256, padding_idx=0)
        self.gru = nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.GELU(),
                                  nn.Linear(512, d))

    def forward(self, ids):
        out, _ = self.gru(self.emb(ids))
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--chunk", type=int, default=64000)
    ap.add_argument("--iters_per_epoch", type=int, default=800)
    ap.add_argument("--noise_prob", type=float, default=0.0)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--noise_dir", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(dev).eval()
    for p in w2v.parameters():
        p.requires_grad_(False)
    aenc = AudioEnc(w2v).to(dev)
    tenc = TextEnc().to(dev)
    TEMP = 0.07                                   # fixed temperature (stable)
    aenc = DDP(aenc, device_ids=[local])
    tenc = DDP(tenc, device_ids=[local])
    params = (list(aenc.module.proj.parameters()) +
              list(tenc.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    ds = ATData(args.samples, args.meta, chunk=args.chunk,
                noise_prob=args.noise_prob, snr_min=args.snr_min,
                snr_max=args.snr_max, noise_dir=args.noise_dir)
    samp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=samp, num_workers=4,
                    drop_last=True, pin_memory=True, collate_fn=collate)
    if rank == 0:
        os.makedirs(args.exp + "/models", exist_ok=True)
        print("[data] {} utt | world={} bs={}".format(len(ds), world, args.bs),
              flush=True)

    for ep in range(1, args.epochs + 1):
        samp.set_epoch(ep)
        aenc.train(); tenc.train()
        tot = acc = n = 0
        for it, (wav, ids) in enumerate(dl):
            wav, ids = wav.to(dev), ids.to(dev)
            with torch.cuda.amp.autocast():
                a = aenc(wav)
                t = tenc(ids)
            # contrastive in fp32 for stability
            a = F.normalize(a.float(), dim=-1)
            t = F.normalize(t.float(), dim=-1)
            logit = (a @ t.T) / TEMP
            lab = torch.arange(a.size(0), device=dev)
            loss = (F.cross_entropy(logit, lab) +
                    F.cross_entropy(logit.T, lab)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            acc += (logit.argmax(1) == lab).float().mean().item()
            n += 1
            if rank == 0 and (it + 1) % 50 == 0:
                print("ep{} it{} loss={:.4f} batch_acc={:.3f}".format(
                    ep, it + 1, tot / n, acc / n), flush=True)
            if (it + 1) >= args.iters_per_epoch:
                break
        if rank == 0:
            print("=== epoch {} loss={:.4f} batch_acc={:.3f} ===".format(
                ep, tot / n, acc / n), flush=True)
            torch.save({"audio_proj": aenc.module.proj.state_dict(),
                        "text_enc": tenc.module.state_dict(),
                        "temp": TEMP, "epoch": ep},
                       "{}/models/matcher_{}.pt".format(args.exp, ep))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
