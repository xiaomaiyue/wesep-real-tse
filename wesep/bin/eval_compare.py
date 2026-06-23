"""同一批 test、同一个分离器，对比两种路由器(ASR+CER vs 自训匹配器)在
不同噪声 SNR 下的提取效果。可加噪声到指定 --snr(dB);不给则干净。
默认加高斯噪声;给 --noise_dir <真噪声 wav 目录>(如 WHAM)则混入真噪声。

  python wesep/bin/eval_compare.py --sep_ckpt <sep.pt> --matcher <matcher.pt> \
    --samples <test samples.jsonl> --meta <test.text_meta.jsonl> \
    --n 150 --device 0 --snr 5 --noise_dir /path/to/wham_noise
"""
import glob
import json
import os
import random

import fire
import torch
import torch.nn.functional as F
import torchaudio

from wesep.modules.separator.bsrnn import BSRNN
from wesep.bin.train_matcher import AudioEnc, TextEnc, encode_text
from wesep.utils.score import cal_SISNRi


def greedy(emission, labels):
    idx = emission.argmax(-1).tolist()
    out, prev = [], None
    for i in idx:
        if i != prev and i != 0:
            out.append(labels[i])
        prev = i
    return "".join(out).replace("|", " ").strip()


def cer(hyp, ref):
    if not ref:
        return 1.0 if hyp else 0.0
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i-1] != hyp[j-1]))
            prev = cur
    return dp[len(hyp)] / len(ref)


def add_noise(mix, snr_db, noise=None):
    if noise is None:
        noise = torch.randn_like(mix)
    mp = mix.pow(2).mean()
    npow = noise.pow(2).mean() + 1e-8
    target = mp / (10 ** (snr_db / 10.0))
    return mix + noise * (target / npow).sqrt()


def sample_noise(noise_files, length, dev):
    """从真噪声目录随机取一段，裁剪/平铺到 length。"""
    p = random.choice(noise_files)
    w, s = torchaudio.load(p)
    if s != 16000:
        w = torchaudio.functional.resample(w, s, 16000)
    w = w.mean(0).to(dev)
    if w.numel() < length:
        w = w.repeat((length // w.numel()) + 1)
    off = random.randint(0, w.numel() - length)
    return w[off:off + length]


def main(sep_ckpt, matcher, samples, meta, n=150, device="cpu", snr=None,
         noise_dir=None, seed=0, feature_dim=128, num_repeat=6):
    dev = torch.device(device if device == "cpu" else "cuda:" + str(device))
    random.seed(seed)
    noise_files = None
    if noise_dir is not None:
        noise_files = sorted(glob.glob(os.path.join(noise_dir, "**", "*.wav"),
                                       recursive=True))
        assert noise_files, "no wav under %s" % noise_dir
    sep = BSRNN(sr=16000, win=512, stride=128, feature_dim=feature_dim,
                num_repeat=num_repeat, causal=False, nspk=2,
                spec_dim=2).to(dev).eval()
    sd = torch.load(sep_ckpt, map_location="cpu")
    sep.load_state_dict(sd["model"] if "model" in sd else sd)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v = bundle.get_model().to(dev).eval()
    labels = bundle.get_labels()
    mk = torch.load(matcher, map_location="cpu")
    if mk.get("kind") == "student_whisper":
        from wesep.bin.train_matcher_distill import WhisperStudentEnc
        aenc = WhisperStudentEnc(mk["whisper_path"]).to(dev).eval()
        aenc.load_state_dict(mk["student"])
    elif mk.get("kind") == "student_cnn":
        from wesep.bin.train_matcher_distill import StudentAudioEnc
        aenc = StudentAudioEnc().to(dev).eval()
        aenc.load_state_dict(mk["student"])
    else:
        aenc = AudioEnc(w2v).to(dev).eval()
        aenc.proj.load_state_dict(mk["audio_proj"])
    tenc = TextEnc().to(dev).eval(); tenc.load_state_dict(mk["text_enc"])

    text = {}
    with open(meta) as f:
        for line in f:
            d = json.loads(line)
            for u, t in d.get("src_text", {}).items():
                text[u] = t

    @torch.no_grad()
    def load(p):
        w, s = torchaudio.load(p)
        if s != 16000:
            w = torchaudio.functional.resample(w, s, 16000)
        return w.mean(0).to(dev)

    @torch.no_grad()
    def asr_text(wav):
        x = wav.unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return greedy(w2v(x)[0][0], labels)

    @torch.no_grad()
    def avec(wav):
        return F.normalize(aenc(wav.unsqueeze(0)).float(), dim=-1)[0]

    @torch.no_grad()
    def tvec(s):
        ids = torch.tensor([encode_text(s)], dtype=torch.long, device=dev)
        return F.normalize(tenc(ids).float(), dim=-1)[0]

    lines = [json.loads(x) for x in open(samples)]
    npair = 0
    a_ok = m_ok = 0
    a_sr = m_sr = o_sr = 0.0
    a_acc = m_acc = 0
    with torch.no_grad():
        for d in lines[:n]:
            spks = d["spk"]
            if len(spks) < 2:
                continue
            utts = {}
            for s in spks:
                u = next((u for u in text
                          if u.split("-")[0] == s and u in d["key"]), None)
                if u:
                    utts[s] = u
            if len(utts) < 2:
                continue
            mix = load(d["mix"]["default"][0])
            if snr is not None:
                ns = (sample_noise(noise_files, mix.numel(), dev)
                      if noise_files else None)
                mix = add_noise(mix, float(snr), ns)
            est = sep(mix.unsqueeze(0))[0]
            L = min(est.shape[-1], mix.shape[-1])
            mix_np = mix[:L].cpu().numpy()
            e0 = est[0, :L].cpu().numpy(); e1 = est[1, :L].cpu().numpy()
            hyp = [asr_text(est[0, :L]), asr_text(est[1, :L])]
            a = torch.stack([avec(est[0, :L]), avec(est[1, :L])])
            for tgt in spks:
                other = [s for s in spks if s != tgt][0]
                ref = load(d["src"][tgt][0])[:L].cpu().numpy()
                _, d0 = cal_SISNRi(e0, ref, mix_np)
                _, d1 = cal_SISNRi(e1, ref, mix_np)
                opick = 0 if d0 >= d1 else 1
                cue = text[utts[tgt]]
                # ASR routing
                ap = 0 if cer(hyp[0], cue) <= cer(hyp[1], cue) else 1
                # matcher routing
                mp = int((a @ tvec(cue)).argmax())
                npair += 1
                a_ok += (ap == opick); m_ok += (mp == opick)
                a_sr += [d0, d1][ap]; m_sr += [d0, d1][mp]; o_sr += [d0, d1][opick]
                a_acc += ([d0, d1][ap] > 1); m_acc += ([d0, d1][mp] > 1)
    M = max(npair, 1)
    nz = "WHAM" if noise_files else "gauss"
    tag = "clean" if snr is None else ("%s SNR=%.0fdB" % (nz, float(snr)))
    print("=== %s | pairs=%d | separator ceiling(oracle)=%.2f dB ===" %
          (tag, npair, o_sr / M))
    print("ASR     route: SI-SNRi=%.2f | accept=%.1f%% | routing_acc=%.1f%%" %
          (a_sr / M, 100.0 * a_acc / M, 100.0 * a_ok / M))
    print("MATCHER route: SI-SNRi=%.2f | accept=%.1f%% | routing_acc=%.1f%%" %
          (m_sr / M, 100.0 * m_acc / M, 100.0 * m_ok / M))


if __name__ == "__main__":
    fire.Fire(main)
