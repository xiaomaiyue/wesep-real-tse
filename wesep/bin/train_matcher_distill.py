"""蒸馏一个「无 wav2vec2」的轻量音频编码器,替代匹配器里 ~95M 的 wav2vec2。

老师: 训好的匹配器 (MATCHER_WHAM) 的音频侧 = 冻结 wav2vec2 -> proj -> 256d。
学生: log-mel -> 小型 1D-CNN -> 256d (~0.7M 参数, CPU 友好, 不依赖 wav2vec2)。
文本侧: 直接复用并冻结老师的 TextEnc, 让学生落在同一嵌入空间, 可直接配 eval。
损失:  (1 - cos(student, teacher))            # 特征蒸馏, 贴老师
       + lambda * InfoNCE(student_audio, text) # 任务对齐到冻结文本空间
只训学生 (+ 可选微调). 保存格式带 kind=student_cnn, eval_compare 自动识别。

  torchrun --nproc_per_node=N wesep/bin/train_matcher_distill.py \
    --teacher <MATCHER_WHAM/models/matcher_80.pt> \
    --samples <train samples.jsonl> --meta <train.text_meta.jsonl> \
    --exp <exp_dir> --epochs 60 \
    --noise_prob 0.8 --snr_min 0 --snr_max 15 --noise_dir <wham>
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from wesep.bin.train_matcher import ATData, collate, AudioEnc, TextEnc


class StudentAudioEnc(nn.Module):
    """log-mel + 小型 1D-CNN -> d 维嵌入。无 wav2vec2，可 CPU 实时。"""

    def __init__(self, d=256, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=n_mels)

        def blk(ci, co, s):
            # GroupNorm (per-sample, over channels): identical train/eval,
            # safe at batch=1 (eval routes one separated source at a time).
            # BatchNorm collapses / NaNs here — see diag_student.py.
            return nn.Sequential(
                nn.Conv1d(ci, co, 5, stride=s, padding=2),
                nn.GroupNorm(min(16, co), co), nn.GELU())

        self.net = nn.Sequential(
            blk(n_mels, 128, 2), blk(128, 128, 2),
            blk(128, 256, 2), blk(256, 256, 2))
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, d))

    def forward(self, wav):
        m = self.mel(wav)                                   # (B, n_mels, T)
        m = torch.log(m + 1e-6)
        m = (m - m.mean((1, 2), keepdim=True)) / (
            m.std((1, 2), keepdim=True) + 1e-5)
        h = self.net(m)                                     # (B, 256, T')
        return self.proj(h.mean(-1))                        # (B, d)


class WhisperStudentEnc(nn.Module):
    """Frozen pretrained Whisper-tiny encoder (~8M) + trainable proj head.
    Pretrained on 680k h of diverse/noisy audio -> robust across noise types,
    which the from-scratch CNN lacked. Only proj trains (encoder frozen)."""

    def __init__(self, whisper_path, d=256, freeze=True):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor
        self.enc = WhisperModel.from_pretrained(whisper_path).encoder
        self.frozen = freeze
        if freeze:
            self.enc.eval()
            for p in self.enc.parameters():
                p.requires_grad_(False)
        fe = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.n_fft, self.hop, self.n_frames = fe.n_fft, fe.hop_length, fe.nb_max_frames
        mf = torch.tensor(fe.mel_filters, dtype=torch.float32)   # (n_fft/2+1, 80)
        self.register_buffer("mel_filters", mf)
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.proj = nn.Sequential(nn.Linear(self.enc.config.d_model, 512),
                                  nn.GELU(), nn.Linear(512, d))

    def _logmel(self, wav):                                  # Whisper's log-mel
        stft = torch.stft(wav, self.n_fft, self.hop, window=self.window.to(wav),
                          return_complex=True)
        mag = stft[..., :-1].abs() ** 2                      # (B, F, T)
        mel = self.mel_filters.to(wav).T @ mag               # (B, 80, T)
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.amax((-2, -1), keepdim=True) - 8.0)
        log = (log + 4.0) / 4.0
        T = log.shape[-1]
        if T < self.n_frames:
            log = F.pad(log, (0, self.n_frames - T))
        else:
            log = log[..., :self.n_frames]
        return log

    def forward(self, wav):
        feats = self._logmel(wav)                            # (B, 80, 3000)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            h = self.enc(feats).last_hidden_state            # (B, 1500, d_model)
        return self.proj(h.mean(1))                          # (B, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--chunk", type=int, default=64000)
    ap.add_argument("--iters_per_epoch", type=int, default=800)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--noise_prob", type=float, default=0.0)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--noise_dir", default=None)
    ap.add_argument("--gauss_prob", type=float, default=0.0,
                    help="when noising, fraction that uses Gaussian instead of "
                         "the real (WHAM) noise -> mixes noise types")
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "whisper"])
    ap.add_argument("--whisper_path", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    # ---- teacher (frozen): wav2vec2 audio enc + text enc from MATCHER_WHAM ----
    tk = torch.load(args.teacher, map_location="cpu")
    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(dev).eval()
    for p in w2v.parameters():
        p.requires_grad_(False)
    teacher = AudioEnc(w2v).to(dev).eval()
    teacher.proj.load_state_dict(tk["audio_proj"])
    tenc = TextEnc().to(dev).eval()
    tenc.load_state_dict(tk["text_enc"])
    for m in (teacher, tenc):
        for p in m.parameters():
            p.requires_grad_(False)
    TEMP = tk.get("temp", 0.07)

    # ---- student (trained) ----
    if args.backbone == "whisper":
        assert args.whisper_path, "--whisper_path required for whisper backbone"
        student = WhisperStudentEnc(args.whisper_path).to(dev)
    else:
        student = StudentAudioEnc().to(dev)
    student = DDP(student, device_ids=[local])
    train_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    ds = ATData(args.samples, args.meta, chunk=args.chunk,
                noise_prob=args.noise_prob, snr_min=args.snr_min,
                snr_max=args.snr_max, noise_dir=args.noise_dir,
                gauss_prob=args.gauss_prob)
    samp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=samp, num_workers=4,
                    drop_last=True, pin_memory=True, collate_fn=collate)
    if rank == 0:
        os.makedirs(args.exp + "/models", exist_ok=True)
        tot = sum(p.numel() for p in student.parameters())
        trn = sum(p.numel() for p in train_params)
        print("[data] {} utt | world={} bs={} | backbone={} | params total="
              "{:.2f}M trainable={:.2f}M".format(
                  len(ds), world, args.bs, args.backbone, tot / 1e6, trn / 1e6),
              flush=True)

    for ep in range(1, args.epochs + 1):
        samp.set_epoch(ep)
        student.train()
        tot = dst = nce = acc = n = 0
        for it, (wav, ids) in enumerate(dl):
            wav, ids = wav.to(dev), ids.to(dev)
            with torch.no_grad():
                ta = F.normalize(teacher(wav).float(), dim=-1)
                tt = F.normalize(tenc(ids).float(), dim=-1)
            with torch.cuda.amp.autocast():
                sa = student(wav)
            sa = F.normalize(sa.float(), dim=-1)
            l_distill = (1 - (sa * ta).sum(-1)).mean()
            logit = (sa @ tt.T) / TEMP
            lab = torch.arange(sa.size(0), device=dev)
            l_nce = F.cross_entropy(logit, lab)
            loss = l_distill + args.lambda_nce * l_nce
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(train_params, 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); dst += l_distill.item(); nce += l_nce.item()
            acc += (logit.argmax(1) == lab).float().mean().item()
            n += 1
            if rank == 0 and (it + 1) % 50 == 0:
                print("ep{} it{} loss={:.4f} distill={:.4f} nce={:.4f} "
                      "batch_acc={:.3f}".format(ep, it + 1, tot / n, dst / n,
                                                nce / n, acc / n), flush=True)
            if (it + 1) >= args.iters_per_epoch:
                break
        if rank == 0:
            print("=== epoch {} loss={:.4f} distill={:.4f} nce={:.4f} "
                  "batch_acc={:.3f} ===".format(ep, tot / n, dst / n, nce / n,
                                                acc / n), flush=True)
            kind = ("student_whisper" if args.backbone == "whisper"
                    else "student_cnn")
            torch.save({"student": student.module.state_dict(),
                        "text_enc": tenc.state_dict(), "temp": TEMP,
                        "kind": kind, "whisper_path": args.whisper_path,
                        "epoch": ep},
                       "{}/models/student_{}.pt".format(args.exp, ep))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
