"""Train a cross-attention router on frozen separator outputs.

This is the stronger route-B router experiment:
  mixture -> frozen SEP_LC -> two estimated streams
  estimated streams + cue text -> x-attn router -> pick target stream

Labels are generated online with oracle SI-SNR against the clean target source.
Optionally initialize the audio/text encoders from a trained MATCHER_WHAM
checkpoint, while the cross-attention and scorer stay random.

  torchrun --nproc_per_node=N wesep/bin/train_xattn_router_sep.py \
    --sep_ckpt <SEP_LC/models/ckpt_100.pt> \
    --matcher_init <MATCHER_WHAM/models/matcher_80.pt> \
    --router_init <optional previous xattn_router_sep checkpoint> \
    --samples <train samples.jsonl> --meta <train-100.text_meta.jsonl> \
    --exp <exp_dir> --epochs 10 --num_repeat 8
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
from wesep.modules.separator.bsrnn import BSRNN


def _build_spk2utts(text):
    spk2utts = {}
    for utt in text:
        spk2utts.setdefault(utt.split("-")[0], []).append(utt)
    return spk2utts


def _find_utt(spk2utts, spk, key):
    return next((u for u in spk2utts.get(spk, []) if u in key), None)


def _mix_at_snr(clean, noise, snr_db):
    cp = clean.pow(2).mean()
    npow = noise.pow(2).mean() + 1e-8
    g = (cp / (10 ** (snr_db / 10.0)) / npow).sqrt()
    return clean + noise * g


def _sisnr(est, ref, eps=1e-8):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    s = ((est * ref).sum(-1, keepdim=True) * ref /
         (ref.pow(2).sum(-1, keepdim=True) + eps))
    n = est - s
    return 10 * torch.log10(s.pow(2).sum(-1) /
                            (n.pow(2).sum(-1) + eps) + eps)


class SepRouterData(Dataset):
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
                for utt, value in d.get("src_text", {}).items():
                    text[utt] = value
        spk2utts = _build_spk2utts(text)

        self.items = []
        with open(samples) as f:
            for line in f:
                d = json.loads(line)
                spks = d.get("spk", [])
                if len(spks) < 2:
                    continue
                for spk in spks:
                    utt = _find_utt(spk2utts, spk, d["key"])
                    if utt and spk in d["src"]:
                        self.items.append((d["mix"]["default"][0],
                                           d["src"][spk][0],
                                           text[utt]))

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.mean(0)

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

    def __getitem__(self, idx):
        mix_path, ref_path, text = self.items[idx]
        mix = self._load(mix_path)
        ref = self._load(ref_path)
        length = min(len(mix), len(ref))
        mix, ref = mix[:length], ref[:length]
        if length >= self.chunk:
            st = random.randint(0, length - self.chunk)
            mix = mix[st:st + self.chunk]
            ref = ref[st:st + self.chunk]
        else:
            pad = self.chunk - length
            mix = F.pad(mix, (0, pad))
            ref = F.pad(ref, (0, pad))
        if self.noise_prob > 0 and random.random() < self.noise_prob:
            snr = random.uniform(self.snr_min, self.snr_max)
            noise = (self._real_noise(len(mix)) if self.noise_paths
                     else torch.randn_like(mix))
            mix = _mix_at_snr(mix, noise, snr)
        return mix, ref, torch.tensor(encode_text(text), dtype=torch.long)


def collate(batch):
    mix = torch.stack([x[0] for x in batch])
    ref = torch.stack([x[1] for x in batch])
    max_len = max(len(x[2]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (_, _, text_ids) in enumerate(batch):
        ids[i, :len(text_ids)] = text_ids
    return mix, ref, ids


class MatcherTokenTextEnc(nn.Module):
    def __init__(self, vocab=len(VOCAB) + 1, d=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, 256, padding_idx=0)
        self.gru = nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.GELU(),
                                  nn.Linear(512, d))

    def forward(self, ids):
        out, _ = self.gru(self.emb(ids))
        return self.proj(out), ids != 0


class SepXAttnRouter(nn.Module):
    def __init__(self, w2v, d=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.w2v = w2v
        self.audio_proj = nn.Sequential(nn.Linear(768, 512), nn.GELU(),
                                        nn.Linear(512, d))
        self.text_enc = MatcherTokenTextEnc(d=d)
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.scorer = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d, 1))
        self.base_scale = nn.Parameter(torch.tensor(10.0))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def load_matcher_init(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.audio_proj.load_state_dict(ckpt["audio_proj"])
        self.text_enc.load_state_dict(ckpt["text_enc"])

    def load_router_init(self, path):
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt.get("kind") == "xattn_router_sep", (
            "not a separator-output xattn router checkpoint")
        self.load_state_dict(ckpt["router"], strict=False)

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
        base = (F.normalize(audio_pool.float(), dim=-1) *
                F.normalize(text_pool.float(), dim=-1)).sum(-1)
        residual = self.scorer(torch.cat([ctx_pool, text_pool, audio_pool], -1)
                               ).squeeze(-1)
        return self.base_scale * base + residual

    def forward(self, candidates, ids):
        bsz, ncand, nsamp = candidates.shape
        wav = candidates.reshape(bsz * ncand, nsamp)
        text = ids[:, None, :].expand(bsz, ncand, ids.size(1))
        scores = self.score(wav, text.reshape(bsz * ncand, ids.size(1)))
        return scores.view(bsz, ncand)


def _load_sep(path, feature_dim, num_repeat, dev):
    sep = BSRNN(sr=16000, win=512, stride=128, feature_dim=feature_dim,
                num_repeat=num_repeat, causal=False, nspk=2,
                spec_dim=2).to(dev).eval()
    ckpt = torch.load(path, map_location="cpu")
    sep.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    for p in sep.parameters():
        p.requires_grad_(False)
    return sep


def _save_router(path, model, args, epoch):
    torch.save({"kind": "xattn_router_sep",
                "router": {k: v for k, v in model.state_dict().items()
                           if not k.startswith("w2v.")},
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "noise_prob": args.noise_prob,
                "snr_min": args.snr_min,
                "snr_max": args.snr_max,
                "epoch": epoch},
               path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sep_ckpt", required=True)
    ap.add_argument("--matcher_init", default=None)
    ap.add_argument("--router_init", default=None)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=64000)
    ap.add_argument("--iters_per_epoch", type=int, default=800)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--num_repeat", type=int, default=8)
    ap.add_argument("--freeze_encoders", action="store_true")
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

    sep = _load_sep(args.sep_ckpt, args.feature_dim, args.num_repeat, dev)
    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(dev).eval()
    for p in w2v.parameters():
        p.requires_grad_(False)
    router = SepXAttnRouter(w2v, d=args.d_model, num_heads=args.num_heads,
                            dropout=args.dropout).to(dev)
    if args.matcher_init:
        router.load_matcher_init(args.matcher_init)
        if rank == 0:
            print("[init] loaded matcher init from {}".format(
                args.matcher_init), flush=True)
    if args.router_init:
        router.load_router_init(args.router_init)
        if rank == 0:
            print("[init] loaded router init from {}".format(
                args.router_init), flush=True)
    if args.freeze_encoders:
        for p in router.audio_proj.parameters():
            p.requires_grad_(False)
        for p in router.text_enc.parameters():
            p.requires_grad_(False)

    router = DDP(router, device_ids=[local])
    params = [p for p in router.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    ds = SepRouterData(args.samples, args.meta, chunk=args.chunk,
                       noise_prob=args.noise_prob, snr_min=args.snr_min,
                       snr_max=args.snr_max, noise_dir=args.noise_dir)
    samp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=samp, num_workers=4,
                    drop_last=True, pin_memory=True, collate_fn=collate)

    if rank == 0:
        os.makedirs(args.exp + "/models", exist_ok=True)
        nparam = sum(p.numel() for p in router.module.parameters()
                     if p.requires_grad)
        print("[data] {} target pairs | world={} bs={} | params={:.2f}M".format(
            len(ds), world, args.bs, nparam / 1e6), flush=True)
        print("[noise] prob={} snr=[{}, {}] noise_dir={}".format(
            args.noise_prob, args.snr_min, args.snr_max, args.noise_dir),
            flush=True)

    for ep in range(1, args.epochs + 1):
        samp.set_epoch(ep)
        router.train()
        tot = acc = margin = n = 0
        for it, (mix, ref, ids) in enumerate(dl):
            mix, ref, ids = mix.to(dev), ref.to(dev), ids.to(dev)
            with torch.no_grad():
                est = sep(mix)
                length = min(est.shape[-1], ref.shape[-1])
                est, ref = est[..., :length], ref[..., :length]
                s0 = _sisnr(est[:, 0], ref)
                s1 = _sisnr(est[:, 1], ref)
                labels = (s1 > s0).long()
                oracle_margin = (s0 - s1).abs().mean().item()
            with torch.cuda.amp.autocast():
                logits = router(est, ids)
                loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            acc += (logits.argmax(1) == labels).float().mean().item()
            margin += oracle_margin
            n += 1
            if rank == 0 and (it + 1) % 50 == 0:
                print("ep{} it{} loss={:.4f} acc={:.3f} oracle_gap={:.2f}"
                      .format(ep, it + 1, tot / n, acc / n, margin / n),
                      flush=True)
            if it + 1 >= args.iters_per_epoch:
                break
        if rank == 0:
            print("=== epoch {} loss={:.4f} acc={:.3f} oracle_gap={:.2f} ==="
                  .format(ep, tot / n, acc / n, margin / n), flush=True)
            _save_router("{}/models/xattn_router_sep_{}.pt".format(
                args.exp, ep), router.module, args, ep)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
