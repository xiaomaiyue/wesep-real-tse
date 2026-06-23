"""用自训匹配器做「无 ASR 路由」eval：分离两路 -> 各自过音频编码器、cue 过
文本编码器 -> 同一空间算相似度 -> 选人。报 ROUTED SI-SNRi + interferer 消融。

  python wesep/bin/eval_matcher.py --sep_ckpt <sep.pt> --matcher <matcher.pt> \
    --samples <test samples.jsonl> --meta <test.text_meta.jsonl> --n 200 --device 0
"""
import json
import fire
import torch
import torch.nn.functional as F
import torchaudio

from wesep.modules.separator.bsrnn import BSRNN
from wesep.bin.train_matcher import AudioEnc, TextEnc, encode_text
from wesep.utils.score import cal_SISNRi


def main(sep_ckpt, matcher, samples, meta, n=200, device="cpu"):
    dev = torch.device(device if device == "cpu" else "cuda:" + str(device))

    sep = BSRNN(sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
                causal=False, nspk=2, spec_dim=2).to(dev).eval()
    sd = torch.load(sep_ckpt, map_location="cpu")
    sep.load_state_dict(sd["model"] if "model" in sd else sd)

    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(dev).eval()
    mk = torch.load(matcher, map_location="cpu")
    aenc = AudioEnc(w2v).to(dev).eval()
    aenc.proj.load_state_dict(mk["audio_proj"])
    tenc = TextEnc().to(dev).eval()
    tenc.load_state_dict(mk["text_enc"])

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
    def avec(wav):
        return F.normalize(aenc(wav.unsqueeze(0)).float(), dim=-1)[0]

    @torch.no_grad()
    def tvec(s):
        ids = torch.tensor([encode_text(s)], dtype=torch.long, device=dev)
        return F.normalize(tenc(ids).float(), dim=-1)[0]

    lines = [json.loads(x) for x in open(samples)]
    ok = npair = acc = 0
    sr = so = si = 0.0
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
            est = sep(mix.unsqueeze(0))[0]
            L = min(est.shape[-1], mix.shape[-1])
            mix_np = mix[:L].cpu().numpy()
            a = torch.stack([avec(est[k, :L]) for k in range(2)])   # (2,D)
            for tgt in spks:
                other = [s for s in spks if s != tgt][0]
                ref = load(d["src"][tgt][0])[:L].cpu().numpy()
                e0 = est[0, :L].cpu().numpy(); e1 = est[1, :L].cpu().numpy()
                _, d0 = cal_SISNRi(e0, ref, mix_np)
                _, d1 = cal_SISNRi(e1, ref, mix_np)
                opick = 0 if d0 >= d1 else 1
                # matcher routing (oracle text) and ablation (interferer text)
                tt = tvec(text[utts[tgt]])
                ti = tvec(text[utts[other]])
                pick = int((a @ tt).argmax())
                pick_i = int((a @ ti).argmax())
                ok += (pick == opick); npair += 1
                sr += [d0, d1][pick]; so += [d0, d1][opick]
                si += [d0, d1][pick_i]
                acc += ([d0, d1][pick] > 1)
    m = max(npair, 1)
    print("pairs=%d" % npair)
    print("routing-vs-oracle agreement = %.1f%%" % (100.0 * ok / m))
    print("ROUTED(matcher) SI-SNRi = %.3f dB | accept>1dB = %.1f%%" %
          (sr / m, 100.0 * acc / m))
    print("ORACLE ceiling          = %.3f dB" % (so / m))
    print("ABLATION interferer-text= %.3f dB" % (si / m))


if __name__ == "__main__":
    fire.Fire(main)
