"""无 ASR 路由原型：分离两路后，用 CLAP(音频-文本对比模型)直接算
「每路音频」与「文本线索」的相似度来选人，全程不转录成文字。

依赖(本地装一次)：  pip install transformers
首次运行会自动下载 CLAP 权重(需联网，约几百 MB)。

用法：
  python extract_by_clap.py \
    --ckpt models/route_b_separator_ckpt150.pt \
    --mix  ~/LibriMix/mini_librimix_tse/mixture.wav \
    --text "BUT A WORD FURTHER CONCERNING THE EXPEDITION IN GENERAL" \
    --ref_target ~/LibriMix/mini_librimix_tse/target_clean.wav \
    --ref_interferer ~/LibriMix/mini_librimix_tse/interferer_clean.wav \
    --out_dir ~/LibriMix/mini_librimix_tse

注意：CLAP 偏「声音场景/音色」，对【自由描述】线索最在行；对【逐字转录】这种
纯词内容，CLAP 不一定比 ASR 路由强 —— 这本就是用来对比看效果的原型。
可以同时拿一句描述试，比如 --text "a clear male voice reading a sentence"。
"""
import argparse
import os

import soundfile as sf
import torch
import torchaudio

from wesep.modules.separator.bsrnn import BSRNN

CLAP_ID = "laion/clap-htsat-unfused"


def load_wav(path, sr=16000):
    w, s = sf.read(os.path.expanduser(path), dtype="float32", always_2d=True)
    w = torch.from_numpy(w).mean(1)
    if s != sr:
        w = torchaudio.functional.resample(w, s, sr)
    return w


def sisnr(est, ref, eps=1e-8):
    ref = ref - ref.mean(); est = est - est.mean()
    a = (est * ref).sum() * ref / (ref.pow(2).sum() + eps)
    n = est - a
    return float(10 * torch.log10(a.pow(2).sum() / (n.pow(2).sum() + eps) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref_target", default=None)
    ap.add_argument("--ref_interferer", default=None)
    ap.add_argument("--out_dir", default="sep_out")
    args = ap.parse_args()

    # 1) 分离
    sep = BSRNN(sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
                causal=False, nspk=2, spec_dim=2).eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    sep.load_state_dict(sd["model"] if "model" in sd else sd)
    mix = load_wav(args.mix)
    with torch.no_grad():
        est = sep(mix.unsqueeze(0))[0]            # (2, T) @16k
    L = est.shape[-1]

    # 2) CLAP 路由(音频/文本→同一空间→余弦)，无需转文字
    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(CLAP_ID).eval()
    proc = ClapProcessor.from_pretrained(CLAP_ID)
    audios = [torchaudio.functional.resample(est[k], 16000, 48000).cpu().numpy()
              for k in range(2)]                  # CLAP 用 48kHz
    with torch.no_grad():
        inputs = proc(text=[args.text], audio=audios, sampling_rate=48000,
                      return_tensors="pt", padding=True)
        out = clap(**inputs)
    sims = out.logits_per_audio.squeeze(-1)       # (2,) 音频->文本相似度
    pick = int(sims.argmax())

    os.makedirs(args.out_dir, exist_ok=True)
    sf.write(os.path.join(args.out_dir, "extracted_clap.wav"),
             est[pick].cpu().numpy(), 16000)

    print("--- CLAP 音频-文本相似度(越高越匹配) ---")
    print("est1: %.4f   est2: %.4f" % (float(sims[0]), float(sims[1])))
    print("=> 选中 est%d，已存 %s/extracted_clap.wav" % (pick + 1, args.out_dir))

    if args.ref_target and args.ref_interferer:
        t = load_wav(args.ref_target)[:L]
        i = load_wav(args.ref_interferer)[:L]
        print("\n选中那路 SI-SNR:  vs 目标 = %.2f   vs 干扰 = %.2f"
              % (sisnr(est[pick], t), sisnr(est[pick], i)))


if __name__ == "__main__":
    main()
