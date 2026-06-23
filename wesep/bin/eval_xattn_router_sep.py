"""Evaluate a separator-output-trained cross-attention router.

This pairs with train_xattn_router_sep.py. It evaluates the router on frozen
separator outputs and reports routed SI-SNRi, routing accuracy against oracle
assignment, oracle ceiling, and wrong-text ablation.

  python wesep/bin/eval_xattn_router_sep.py \
    --sep_ckpt <SEP_LC/models/ckpt_100.pt> \
    --router <XATTN_ROUTER_SEP/models/xattn_router_sep_10.pt> \
    --samples <test samples.jsonl> --meta <test.text_meta.jsonl> \
    --n 300 --device 0 --num_repeat 8
"""
import glob
import json
import os
import random

import fire
import torch
import torchaudio

from wesep.bin.train_matcher import encode_text
from wesep.bin.train_xattn_router_sep import SepXAttnRouter
from wesep.modules.separator.bsrnn import BSRNN
from wesep.utils.score import cal_SISNRi


def build_spk2utts(text):
    spk2utts = {}
    for utt in text:
        spk2utts.setdefault(utt.split("-")[0], []).append(utt)
    return spk2utts


def add_noise(mix, snr_db, noise=None):
    if noise is None:
        noise = torch.randn_like(mix)
    mp = mix.pow(2).mean()
    npow = noise.pow(2).mean() + 1e-8
    target = mp / (10 ** (snr_db / 10.0))
    return mix + noise * (target / npow).sqrt()


def sample_noise(noise_files, length, dev):
    path = random.choice(noise_files)
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0).to(dev)
    if wav.numel() < length:
        wav = wav.repeat(length // wav.numel() + 1)
    off = random.randint(0, wav.numel() - length)
    return wav[off:off + length]


def main(sep_ckpt, router, samples, meta, n=300, device="cpu", snr=None,
         noise_dir=None, seed=0, feature_dim=128, num_repeat=8):
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
    sep_sd = torch.load(sep_ckpt, map_location="cpu")
    sep.load_state_dict(sep_sd["model"] if "model" in sep_sd else sep_sd)

    ck = torch.load(router, map_location="cpu")
    assert ck.get("kind") == "xattn_router_sep", (
        "not a separator-output xattn router checkpoint")
    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(dev).eval()
    for p in w2v.parameters():
        p.requires_grad_(False)
    xr = SepXAttnRouter(w2v, d=ck.get("d_model", 256),
                        num_heads=ck.get("num_heads", 4),
                        dropout=ck.get("dropout", 0.1)).to(dev).eval()
    xr.load_state_dict(ck["router"], strict=False)

    text = {}
    with open(meta) as f:
        for line in f:
            d = json.loads(line)
            for utt, value in d.get("src_text", {}).items():
                text[utt] = value
    spk2utts = build_spk2utts(text)

    @torch.no_grad()
    def load(path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.mean(0).to(dev)

    @torch.no_grad()
    def pick_with_text(est, cue):
        ids = torch.tensor([encode_text(cue)], dtype=torch.long, device=dev)
        scores = xr(est[:2].unsqueeze(0), ids)[0]
        return int(scores.argmax())

    lines = [json.loads(x) for x in open(samples)]
    npair = ok = acc = 0
    routed = oracle = wrong = 0.0
    with torch.no_grad():
        for d in lines[:int(n)]:
            spks = d.get("spk", [])
            if len(spks) < 2:
                continue
            utts = {}
            for spk in spks:
                utt = next((u for u in spk2utts.get(spk, [])
                            if u in d["key"]), None)
                if utt and spk in d["src"]:
                    utts[spk] = utt
            if len(utts) < 2:
                continue

            mix = load(d["mix"]["default"][0])
            if snr is not None:
                ns = (sample_noise(noise_files, mix.numel(), dev)
                      if noise_files else None)
                mix = add_noise(mix, float(snr), ns)
            est = sep(mix.unsqueeze(0))[0]
            length = min(est.shape[-1], mix.shape[-1])
            est = est[:, :length]
            mix_np = mix[:length].cpu().numpy()
            e0 = est[0].cpu().numpy()
            e1 = est[1].cpu().numpy()

            for tgt in spks:
                if tgt not in utts:
                    continue
                other = next((s for s in spks if s != tgt and s in utts), None)
                if other is None:
                    continue
                ref = load(d["src"][tgt][0])[:length].cpu().numpy()
                _, d0 = cal_SISNRi(e0, ref, mix_np)
                _, d1 = cal_SISNRi(e1, ref, mix_np)
                opick = 0 if d0 >= d1 else 1
                pick = pick_with_text(est, text[utts[tgt]])
                pick_wrong = pick_with_text(est, text[utts[other]])
                dr = [d0, d1][pick]
                routed += dr
                oracle += [d0, d1][opick]
                wrong += [d0, d1][pick_wrong]
                ok += (pick == opick)
                acc += (dr > 1)
                npair += 1

    m = max(npair, 1)
    nz = "WHAM" if noise_files else "gauss"
    tag = "clean" if snr is None else ("%s SNR=%.0fdB" % (nz, float(snr)))
    print("=== %s | pairs=%d ===" % (tag, npair))
    print("XATTN-SEP route: SI-SNRi=%.3f | accept=%.1f%% | routing_acc=%.1f%%" %
          (routed / m, 100.0 * acc / m, 100.0 * ok / m))
    print("ORACLE ceiling: %.3f dB" % (oracle / m))
    print("ABLATION interferer-text: %.3f dB" % (wrong / m))


if __name__ == "__main__":
    fire.Fire(main)
