"""极简本地推理：用训好的路线 B 分离器，把一段混音分成两路。
若提供干净参考(target/interferer)，会打印 SI-SNR 让你直观看到分离效果。

用法：
  python separate.py \
    --ckpt models/route_b_separator_ckpt150.pt \
    --mix  ~/LibriMix/mini_librimix_tse/mixture.wav \
    --ref_target ~/LibriMix/mini_librimix_tse/target_clean.wav \
    --ref_interferer ~/LibriMix/mini_librimix_tse/interferer_clean.wav \
    --out_dir /tmp/sep_out
"""
import argparse
import os

import soundfile as sf
import torch
import torchaudio

from wesep.modules.separator.bsrnn import BSRNN


def load_wav(path, sr=16000):
    w, s = sf.read(os.path.expanduser(path), dtype="float32", always_2d=True)
    w = torch.from_numpy(w).mean(1)  # mono (T,)
    if s != sr:
        w = torchaudio.functional.resample(w, s, sr)
    return w


def sisnr(est, ref, eps=1e-8):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = (est * ref).sum() * ref / (ref.pow(2).sum() + eps)
    n = est - a
    return float(10 * torch.log10(a.pow(2).sum() / (n.pow(2).sum() + eps) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--ref_target", default=None)
    ap.add_argument("--ref_interferer", default=None)
    ap.add_argument("--out_dir", default="sep_out")
    args = ap.parse_args()

    model = BSRNN(sr=16000, win=512, stride=128, feature_dim=128,
                  num_repeat=6, causal=False, nspk=2, spec_dim=2).eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd)

    mix = load_wav(args.mix)
    with torch.no_grad():
        est = model(mix.unsqueeze(0))[0]          # (2, T)
    L = est.shape[-1]

    os.makedirs(args.out_dir, exist_ok=True)
    for k in range(2):
        sf.write(os.path.join(args.out_dir, "est%d.wav" % (k + 1)),
                 est[k].cpu().numpy(), 16000)
    print("已输出两路到:", args.out_dir, "(est1.wav / est2.wav)")

    if args.ref_target and args.ref_interferer:
        t = load_wav(args.ref_target)[:L]
        i = load_wav(args.ref_interferer)[:L]
        m = mix[:L]
        print("\n--- 分离效果(SI-SNR, dB, 越高越像) ---")
        for k in range(2):
            e = est[k]
            print("est%d:  vs 目标=%.2f   vs 干扰=%.2f" %
                  (k + 1, sisnr(e, t), sisnr(e, i)))
        # 混音本身 vs 目标(作为基线参照)
        print("混音 vs 目标 = %.2f dB（基线；分离后应明显更高）" % sisnr(m, t))
        # 自动判断哪一路是目标
        best = max(range(2), key=lambda k: sisnr(est[k], t))
        print("\n=> est%d 是「目标说话人」那一路，SI-SNR=%.2f dB" %
              (best + 1, sisnr(est[best], t)))


if __name__ == "__main__":
    main()
