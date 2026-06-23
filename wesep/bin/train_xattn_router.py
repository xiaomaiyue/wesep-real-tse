"""Train a text/audio cross-attention router for route-B TSE.

This is an experimental side path. It does not modify the separator or the
existing matcher. The router sees two candidate utterances and one cue text,
then learns which candidate matches the cue.

Training pairs are built from clean sources in samples.jsonl:
  candidate A = target source, candidate B = other source, text = target text.
The candidate order is randomly swapped, so the task is a 2-way CE loss.

  torchrun --nproc_per_node=N wesep/bin/train_xattn_router.py \
    --samples <train samples.jsonl> --meta <train.text_meta.jsonl> \
    --exp <exp_dir> --epochs 40
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from wesep.bin.train_matcher import VOCAB, encode_text


def _mix_at_snr(clean, noise, snr_db):
    cp = clean.pow(2).mean()
    npow = noise.pow(2).mean() + 1e-8
    g = (cp / (10 ** (snr_db / 10.0)) / npow).sqrt()
    return clean + noise * g


def _build_spk2utts(text):
    spk2utts = {}
    for utt in text:
        spk2utts.setdefault(utt.split("-")[0], []).append(utt)
    return spk2utts


def _find_utt(spk2utts, spk, key):
    return next((u for u in spk2utts.get(spk, []) if u in key), None)


class RouterPairData(Dataset):
    def __init__(self, samples, meta, chunk=64000, sr=16000,
                 noise_prob=0.0, snr_min=0.0, snr_max=20.0, noise_dir=None):
        self.chunk = chunk
        self.sr = sr
        self.noise_prob = noise_prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_paths = (sorted(glob.glob(
            os.path.join(noise_dir, "**", "*.wav"), recursive=True))
            if noise_dir else [])

        text = {}
        with open(meta) as f:
            for line in f:
                d = json.loads(line)
                for u, t in d.get("src_text", {}).items():
                    text[u] = t
        spk2utts = _build_spk2utts(text)

        self.items = []
        with open(samples) as f:
            for line in f:
                d = json.loads(line)
                spks = d.get("spk", [])
                if len(spks) < 2:
                    continue
                utts = {s: _find_utt(spk2utts, s, d["key"]) for s in spks}
                utts = {s: u for s, u in utts.items() if u and s in d["src"]}
                if len(utts) < 2:
                    continue
                for tgt in utts:
                    other = next(s for s in utts if s != tgt)
                    self.items.append((d["src"][tgt][0],
                                       d["src"][other][0],
                                       text[utts[tgt]]))

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(0)
        if len(wav) >= self.chunk:
            st = random.randint(0, len(wav) - self.chunk)
            wav = wav[st:st + self.chunk]
        else:
            wav = F.pad(wav, (0, self.chunk - len(wav)))
        return wav

    def _real_noise(self, length):
        wav, sr = torchaudio.load(random.choice(self.noise_paths))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(0)
        if len(wav) >= length:
            st = random.randint(0, len(wav) - length)
            wav = wav[st:st + length]
        else:
            wav = wav.repeat(length // len(wav) + 1)[:length]
        return wav

    def _maybe_noise(self, wav):
        if self.noise_prob <= 0 or random.random() >= self.noise_prob:
            return wav
        snr = random.uniform(self.snr_min, self.snr_max)
        noise = (self._real_noise(len(wav)) if self.noise_paths
                 else torch.randn_like(wav))
        return _mix_at_snr(wav, noise, snr)

    def __getitem__(self, idx):
        pos_path, neg_path, text = self.items[idx]
        pos = self._maybe_noise(self._load(pos_path))
        neg = self._maybe_noise(self._load(neg_path))
        label = 0
        if random.random() < 0.5:
            pos, neg = neg, pos
            label = 1
        return torch.stack([pos, neg], 0), torch.tensor(
            encode_text(text), dtype=torch.long), label


def collate(batch):
    wav = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[2] for x in batch], dtype=torch.long)
    max_len = max(len(x[1]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (_, text_ids, _) in enumerate(batch):
        ids[i, :len(text_ids)] = text_ids
    return wav, ids, labels


class TokenTextEnc(nn.Module):
    def __init__(self, vocab=len(VOCAB) + 1, d=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, 256, padding_idx=0)
        self.gru = nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(512, d)

    def forward(self, ids):
        out, _ = self.gru(self.emb(ids))
        return self.proj(out), ids != 0


class XAttnRouter(nn.Module):
    def __init__(self, w2v, d=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.w2v = w2v
        self.audio_proj = nn.Sequential(nn.Linear(768, 512), nn.GELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(512, d))
        self.text_enc = TokenTextEnc(d=d)
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.scorer = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d, 1))

    def _audio_frames(self, wav):
        with torch.no_grad():
            frames = self.w2v.extract_features(wav)[0][-1]
        return self.audio_proj(frames)

    def score(self, wav, ids):
        audio = self._audio_frames(wav)
        text, mask = self.text_enc(ids)
        ctx, _ = self.attn(text, audio, audio, need_weights=False)
        ctx = self.norm(ctx + text)
        m = mask.unsqueeze(-1).float()
        text_pool = (text * m).sum(1) / m.sum(1).clamp(min=1)
        ctx_pool = (ctx * m).sum(1) / m.sum(1).clamp(min=1)
        audio_pool = audio.mean(1)
        feat = torch.cat([ctx_pool, text_pool, audio_pool], -1)
        return self.scorer(feat).squeeze(-1)

    def forward(self, candidates, ids):
        bsz, ncand, nsamp = candidates.shape
        wav = candidates.reshape(bsz * ncand, nsamp)
        text = ids[:, None, :].expand(bsz, ncand, ids.size(1))
        scores = self.score(wav, text.reshape(bsz * ncand, ids.size(1)))
        return scores.view(bsz, ncand)


def _trainable_state_dict(model):
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith("w2v.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--chunk", type=int, default=64000)
    ap.add_argument("--iters_per_epoch", type=int, default=800)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
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
    model = XAttnRouter(w2v, d=args.d_model, num_heads=args.num_heads,
                        dropout=args.dropout).to(dev)
    model = DDP(model, device_ids=[local])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    ds = RouterPairData(args.samples, args.meta, chunk=args.chunk,
                        noise_prob=args.noise_prob, snr_min=args.snr_min,
                        snr_max=args.snr_max, noise_dir=args.noise_dir)
    samp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=samp, num_workers=4,
                    drop_last=True, pin_memory=True, collate_fn=collate)

    if rank == 0:
        os.makedirs(args.exp + "/models", exist_ok=True)
        nparam = sum(p.numel() for p in model.module.parameters()
                     if p.requires_grad)
        print("[data] {} router pairs | world={} bs={} | params={:.2f}M".format(
            len(ds), world, args.bs, nparam / 1e6), flush=True)

    for ep in range(1, args.epochs + 1):
        samp.set_epoch(ep)
        model.train()
        tot = acc = n = 0
        for it, (cand, ids, lab) in enumerate(dl):
            cand, ids, lab = cand.to(dev), ids.to(dev), lab.to(dev)
            with torch.cuda.amp.autocast():
                logits = model(cand, ids)
                loss = F.cross_entropy(logits.float(), lab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            acc += (logits.argmax(1) == lab).float().mean().item()
            n += 1
            if rank == 0 and (it + 1) % 50 == 0:
                print("ep{} it{} loss={:.4f} acc={:.3f}".format(
                    ep, it + 1, tot / n, acc / n), flush=True)
            if it + 1 >= args.iters_per_epoch:
                break
        if rank == 0:
            print("=== epoch {} loss={:.4f} acc={:.3f} ===".format(
                ep, tot / n, acc / n), flush=True)
            torch.save({"kind": "xattn_router",
                        "router": _trainable_state_dict(model.module),
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                        "epoch": ep},
                       "{}/models/xattn_router_{}.pt".format(args.exp, ep))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
