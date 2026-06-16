#!/usr/bin/env python3
"""Synthetic smoke test for the Exp1 time-resolved text cue (cross-attention).

No data / no MiniLM needed -- random tensors only. Checks the parts most likely
to be wrong when wiring a variable-length token sequence through:
  1. collate: variable-L (D, L) text -> padded (B, D, L)
  2. forward routing: (B, D, L) is treated as text (not waveform enrollment)
  3. output shape is sane and finite
  4. the text cue actually influences the output (different text -> different out)
  5. key_padding_mask correctness: padded tokens are ignored, i.e. forward on a
     zero-padded sequence == forward on the cropped (real-length) sequence
Run:  python local/smoke_xattn.py
"""
import numpy as np
import torch

from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK
from wesep.dataset.collate import BASE_COLLECT_KEYS, tse_collate_fn

torch.manual_seed(0)
np.random.seed(0)

D = 384
CFG = {
    "separator": {
        "sr": 16000, "win": 512, "stride": 128,
        "feature_dim": 128, "num_repeat": 2, "causal": False, "nspk": 1,
    },
    "speaker": {
        "features": {
            "textemb": {
                "enabled": True, "mode": "cross_attn", "text_dim": D,
                "atten_dim": 128, "num_heads": 4, "fusion": "multiply",
                "mix_dim": 128,
            }
        }
    },
}


def rand_text(L):
    """A fake (D, L) L2-normalized token sequence (per-token norm 1)."""
    x = np.random.randn(D, L).astype(np.float32)
    x /= np.clip(np.linalg.norm(x, axis=0, keepdims=True), 1e-9, None)
    return x


def main():
    model = TSE_BSRNN_SPK(CFG).eval()
    nband = model.sep_model.nband
    print(f"[smoke] model built, nband={nband}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ---- 1) collate: variable-length token sequences -> (B, D, L) ----
    collect = {k: dict(BASE_COLLECT_KEYS[k]) for k in
               ["wav_mix", "wav_target", "spk", "key", "num_speaker"]}
    collect["text_aux"] = dict(BASE_COLLECT_KEYS["text_aux"])
    collect["text_aux"]["required"] = True
    T = 16000
    batch = []
    Ls = [7, 13, 4]
    for i, L in enumerate(Ls):
        batch.append({
            "key": f"mix{i}", "num_speaker": 1, "spk1": f"s{i}",
            "wav_mix": np.random.randn(T).astype(np.float32),
            "wav_spk1": np.random.randn(T).astype(np.float32),
            "text_spk1": rand_text(L),
        })
    out = tse_collate_fn(batch, collect)
    txt = out["text_aux"]
    assert txt.shape == (3, D, max(Ls)), txt.shape
    # the short sequences must be zero-padded on the right (the mask relies on it)
    assert torch.allclose(txt[2, :, Ls[2]:], torch.zeros_like(txt[2, :, Ls[2]:]))
    assert not torch.allclose(txt[2, :, :Ls[2]], torch.zeros_like(txt[2, :, :Ls[2]]))
    print(f"[smoke] OK collate: text_aux {tuple(txt.shape)}, "
          f"short seq padded after L={Ls[2]}")

    # ---- 2/3) forward routing + output shape ----
    mix = out["wav_mix"].unsqueeze(1)              # (B, 1, T)
    with torch.no_grad():
        y = model(mix, [txt])
    assert torch.isfinite(y).all(), "output has NaN/Inf"
    assert y.shape[0] == 3 and y.shape[-1] == T, y.shape
    print(f"[smoke] OK forward: out {tuple(y.shape)}, finite")

    # ---- 4) the text cue must influence the output ----
    txt2 = txt.clone()
    txt2[:, :, :] = torch.from_numpy(
        np.stack([rand_text(max(Ls)) for _ in range(3)]))
    with torch.no_grad():
        y2 = model(mix, [txt2])
    diff = (y - y2).abs().mean().item()
    assert diff > 1e-6, f"output insensitive to text cue (diff={diff})"
    print(f"[smoke] OK text-sensitivity: mean|out-out2|={diff:.4e} (>0)")

    # ---- 5) key_padding_mask correctness: pad-invariant ----
    Lr = 6
    base = rand_text(Lr)                            # (D, Lr) real tokens
    padded = np.zeros((D, Lr + 9), np.float32)
    padded[:, :Lr] = base
    one_mix = mix[:1]
    with torch.no_grad():
        y_pad = model(one_mix, [torch.from_numpy(padded).unsqueeze(0)])
        y_crop = model(one_mix, [torch.from_numpy(base).unsqueeze(0)])
    mask_err = (y_pad - y_crop).abs().max().item()
    assert mask_err < 1e-4, f"padding changed output (max err={mask_err})"
    print(f"[smoke] OK mask: padded vs cropped max err={mask_err:.2e} (~0)")

    print("[smoke] ALL PASSED")


if __name__ == "__main__":
    main()
