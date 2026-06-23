"""诊断蒸馏学生为何 eval 塌到 chance:隔离 BN/eval-path bug vs 域偏移。
对同一批干净源音频，分别在 train() 和 eval() 模式下算 student 音频嵌入，
与冻结文本嵌入做 batch 内匹配准确率；并打印嵌入的方差(检查是否恒定输出)。
"""
import json
import random
import sys

import torch
import torch.nn.functional as F
import torchaudio

from wesep.bin.train_matcher import TextEnc, encode_text
from wesep.bin.train_matcher_distill import StudentAudioEnc


def main(student_ckpt, samples, meta, n=64, chunk=64000):
    n, chunk = int(n), int(chunk)
    dev = torch.device("cpu")
    mk = torch.load(student_ckpt, map_location="cpu")
    stu = StudentAudioEnc().to(dev)
    stu.load_state_dict(mk["student"])
    tenc = TextEnc().to(dev).eval()
    tenc.load_state_dict(mk["text_enc"])

    text = {}
    with open(meta) as f:
        for line in f:
            d = json.loads(line)
            for u, t in d.get("src_text", {}).items():
                text[u] = t

    items = []
    for line in open(samples):
        d = json.loads(line)
        for s in d["spk"]:
            u = next((u for u in text
                      if u.split("-")[0] == s and u in d["key"]), None)
            if u and s in d["src"]:
                items.append((d["src"][s][0], text[u]))
    random.seed(0)
    random.shuffle(items)
    items = items[:n]

    wavs, ids = [], []
    for p, t in items:
        w, sr = torchaudio.load(p)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w.mean(0)
        if len(w) >= chunk:
            w = w[:chunk]
        else:
            w = F.pad(w, (0, chunk - len(w)))
        wavs.append(w)
        ids.append(torch.tensor(encode_text(t), dtype=torch.long))
    wav = torch.stack(wavs)
    maxl = max(len(x) for x in ids)
    idt = torch.zeros(len(ids), maxl, dtype=torch.long)
    for i, x in enumerate(ids):
        idt[i, :len(x)] = x

    with torch.no_grad():
        t = F.normalize(tenc(idt).float(), dim=-1)
        for mode in ("train", "eval"):
            getattr(stu, mode)()
            # batched (like training)
            a_b = F.normalize(stu(wav).float(), dim=-1)
            # per-sample (like eval_compare: one wav at a time)
            a_s = torch.stack([F.normalize(stu(wav[i:i+1]).float(), dim=-1)[0]
                               for i in range(len(wav))])
            for name, a in (("batched", a_b), ("persample", a_s)):
                logit = a @ t.T
                acc = (logit.argmax(1) ==
                       torch.arange(len(a))).float().mean().item()
                var = a.var(0).mean().item()      # 跨样本的嵌入方差
                print("[{:5s} | {:9s}] in-domain acc={:.3f}  emb_var={:.5f}"
                      .format(mode, name, acc, var), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
