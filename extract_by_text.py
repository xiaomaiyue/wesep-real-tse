"""路线 B 完整本地 demo：给【文本】线索，从混音里提取目标说话人。
流程：分离两路 -> 冻结 wav2vec2 转录两路 -> 跟 cue 文本比 CER -> 选最像的那路。

线索两种给法（二选一）：
  --text "THE ACTUAL WORDS ..."     直接给目标在混音里说的那句文字
  --cue_wav <一段目标干净语音>       脚本用 ASR 把它转成文字当线索（懒人版，免打字）

用法示例：
  python extract_by_text.py \
    --ckpt models/route_b_separator_ckpt150.pt \
    --mix  ~/LibriMix/mini_librimix_tse/mixture.wav \
    --cue_wav ~/LibriMix/mini_librimix_tse/target_clean.wav \
    --ref_target ~/LibriMix/mini_librimix_tse/target_clean.wav \
    --ref_interferer ~/LibriMix/mini_librimix_tse/interferer_clean.wav \
    --out_dir /tmp/sep_out

证明是“文本”在起作用：把 --cue_wav 换成 interferer_clean.wav（或 --text 改成另一个人的话），
选出来的那路会变成另一个人 —— 说明选择完全由文本决定。
"""
import argparse
import os

import soundfile as sf
import torch
import torchaudio

from wesep.modules.separator.bsrnn import BSRNN

_BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H


def load_wav(path, sr=16000):
    w, s = sf.read(os.path.expanduser(path), dtype="float32", always_2d=True)
    w = torch.from_numpy(w).mean(1)          # (T,) mono
    if s != sr:
        w = torchaudio.functional.resample(w, s, sr)
    return w


def sisnr(est, ref, eps=1e-8):
    ref = ref - ref.mean(); est = est - est.mean()
    a = (est * ref).sum() * ref / (ref.pow(2).sum() + eps)
    n = est - a
    return float(10 * torch.log10(a.pow(2).sum() / (n.pow(2).sum() + eps) + eps))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--cue_wav", default=None)
    ap.add_argument("--ref_target", default=None)
    ap.add_argument("--ref_interferer", default=None)
    ap.add_argument("--out_dir", default="sep_out")
    args = ap.parse_args()

    sep = BSRNN(sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
                causal=False, nspk=2, spec_dim=2).eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    sep.load_state_dict(sd["model"] if "model" in sd else sd)

    asr = _BUNDLE.get_model().eval()
    labels = _BUNDLE.get_labels()

    @torch.no_grad()
    def transcribe(wav):
        x = wav.unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return greedy(asr(x)[0][0], labels)

    # 1) 确定 cue 文本
    if args.text:
        cue = args.text.upper()
    elif args.cue_wav:
        cue = transcribe(load_wav(args.cue_wav))
        print("[cue] 由 --cue_wav 转录得到的文本线索:\n   ", cue, "\n")
    else:
        raise SystemExit("必须给 --text 或 --cue_wav 之一")

    # 2) 分离
    mix = load_wav(args.mix)
    with torch.no_grad():
        est = sep(mix.unsqueeze(0))[0]            # (2, T)
    L = est.shape[-1]

    # 3) 转录两路 + 跟 cue 比 CER -> 选人
    hyp = [transcribe(est[k]) for k in range(2)]
    cers = [cer(hyp[k], cue) for k in range(2)]
    pick = 0 if cers[0] <= cers[1] else 1

    os.makedirs(args.out_dir, exist_ok=True)
    sf.write(os.path.join(args.out_dir, "extracted_target.wav"),
             est[pick].cpu().numpy(), 16000)
    for k in range(2):
        sf.write(os.path.join(args.out_dir, "est%d.wav" % (k+1)),
                 est[k].cpu().numpy(), 16000)

    print("--- 两路转录 vs 文本线索 ---")
    for k in range(2):
        print("est%d (CER=%.2f): %s" % (k+1, cers[k], hyp[k][:80]))
    print("\n=> 按文本选中 est%d 作为目标，已存 %s/extracted_target.wav"
          % (pick+1, args.out_dir))

    if args.ref_target and args.ref_interferer:
        t = load_wav(args.ref_target)[:L]
        i = load_wav(args.ref_interferer)[:L]
        print("\n--- 选中那路的 SI-SNR(dB) ---")
        print("   vs 目标 = %.2f   vs 干扰 = %.2f" %
              (sisnr(est[pick], t), sisnr(est[pick], i)))
        print("(若选对了，vs 目标 应明显高于 vs 干扰)")


if __name__ == "__main__":
    main()
