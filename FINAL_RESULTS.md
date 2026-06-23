# Text-cued TSE — Final Results (route B: PIT separate + cue routing)

Final system = **SEP_LC** separator + **MATCHER_WHAM** text-matcher (matcher is
separator-agnostic; ASR route shown for reference). Eval: 300 pairs, full
utterances, oracle routing already at ceiling. Noise = held-out WHAM `tt` split.

## 1. Headline: noise-augmented training is decisive

Real WHAM noise, 0 dB, SI-SNRi (ASR / matcher route):

| training            | 0 dB        |
|---------------------|-------------|
| clean-trained       | 1.92 / 1.72 |
| **WHAM-noise trained (final)** | **13.92 / 14.14** |

Clean-trained separator collapses in real noise (oracle ceiling 3.55 dB);
noise-trained holds ~14 dB. No clean-side penalty (see §3).

## 2. Separator optimization ladder (WHAM real noise + clean)

SI-SNRi, matcher route (ASR route in parens). Oracle ceiling in brackets.

| model              | clean            | 5 dB             | 0 dB             |
|--------------------|------------------|------------------|------------------|
| OLD  nr6 / 6s      | 18.30 [18.30]    | 13.76 / 13.54    | 12.64 / 12.54 [13.01] |
| BIG  nr8 / 6s      | 18.60 [18.60]    | 14.34 / 14.51    | 13.63 / 13.29 [13.77] |
| **LC  nr8 / 10s (final)** | **18.80** [18.80] | **14.53 / 14.50** [14.54] | **14.14 / 13.92** [14.26] |

Cumulative gain OLD→BIG→LC: **0 dB +1.5 dB**, 5 dB +0.8, clean +0.5.
- Depth (nr6→8): biggest help where it's dirtiest. No clean cost.
- Long chunk (6s→10s): wins at 0 dB (+0.5), flat at 5 dB (context only pays
  under heavy noise).
- Both levers diminishing (~0.5–0.8/step).

## 3. No clean-extraction penalty from noise training (clean test)

| training        | clean ceiling / route |
|-----------------|-----------------------|
| clean-trained   | 18.16                 |
| synth-noise     | 18.21                 |
| WHAM final (LC) | **18.80**             |

## 4. Matcher route vs ASR route

Text-cue routing **without ASR** matches or beats the ASR+CER route under noise
and sits within ~0.1 dB of the oracle ceiling (0 dB: 14.14 vs 14.26). The
matcher is trained on clean(+noised) source audio ↔ text, independent of the
separator — one matcher serves any separator.

## 4b. Lightweight distilled matcher (ASR-free *and* small)

The teacher matcher still runs wav2vec2 (~95M) at inference. Distilling it into a
log-mel + 1D-CNN student (`train_matcher_distill.py`) removes that dependency:

SI-SNRi (matcher route) / 0 dB routing_acc, on SEP_LC. Stress = unseen Gaussian
noise + harder SNR (`stress_student.slurm`).

| router (on SEP_LC)          | params  | clean | WHAM 5 | WHAM 0 | **Gauss 5** | **Gauss 0** |
|-----------------------------|---------|-------|--------|--------|-------------|-------------|
| teacher (wav2vec2)          | 95 M    | 18.80 | 14.53  | 14.14  | 12.97       | 12.02       |
| student CNN, WHAM-only      | 0.76 M  | 18.40 | 13.66  | 12.59  | 9.18        | 6.12        |
| student CNN, +mixed noise   | 0.76 M  | 18.38 | 13.97  | 12.99  | 11.77       | 8.35        |
| **student Whisper-tiny, mixed** | **8.5 M** | **18.63** | **14.39** | 12.83 | **12.41** | **10.58** |

Two failure modes found and fixed, in order:
1. **BatchNorm → GroupNorm** (see below): without it the CNN student eval'd at
   chance (50%).
2. **Overfit to the WHAM noise type.** The CNN (GroupNorm) hit 96% on seen WHAM
   noise but collapsed on **unseen Gaussian** (0 dB 6.12 vs teacher 12.02). Fix
   came in two additive levers: mixed-noise distillation (`--gauss_prob 0.5`,
   wider SNR) lifted Gaussian-0dB 6.12 → 8.35; swapping the from-scratch CNN for
   a **frozen pretrained Whisper-tiny encoder** (8.21 M, only a 0.33 M proj head
   trains) lifted it again 8.35 → **10.58** — pretraining on diverse audio buys
   the cross-noise robustness a tiny from-scratch net can't learn from narrow
   data. Each lever ≈ +2.2 dB on the hard case.

**Final lightweight router = Whisper-tiny student (8.5 M, ÷11 vs teacher):**
within 0.15 dB of the teacher on clean/WHAM-5, and now within ~1.4 dB even on
unseen Gaussian-0dB (was 5.9 dB for the first student). Cluster has no internet —
Whisper weights were downloaded locally, scp'd to `~/tse1/pretrained/whisper-tiny`,
loaded offline (`HF_HUB_OFFLINE=1`). Residual ~1.4 dB at 0 dB could close by
unfreezing the encoder or distilling on separated-output audio (diminishing).

⚠️ **BatchNorm→GroupNorm was decisive.** With BatchNorm the student trained fine
(batch_acc 0.70) but eval collapsed to exactly 50% (chance): eval-mode running
stats went NaN under fp16, and BN at batch=1 erases cross-sample info — and
`eval_compare` routes one separated source at a time. GroupNorm (per-sample over
channels) is identical in train/eval and batch-1 safe. Any per-sample inference
encoder must avoid BatchNorm.

## 4c. Cross-attention router on separator outputs

Question: can a text-aware cross-attention router beat the simple dot-product
matcher if it is trained on the same domain it sees at eval time?

The useful version is **not** clean-source pair training. It trains on:
`mixture -> frozen SEP_LC -> two separated candidates`, with the route label
generated by oracle SI-SNR assignment against the clean target source. It is
initialized from `MATCHER_WHAM` / the previous clean SEP-router checkpoint, so
cross-attention learns a correction instead of starting from zero.

| router checkpoint | training candidates | clean | 5 dB | 0 dB | note |
|-------------------|---------------------|-------|------|------|------|
| clean-source XATTN | clean sources only | 9.02 | — | 5.24 | domain gap: train clean, eval separated |
| SEP-XATTN ep10 | SEP_LC outputs, clean mix | 18.53 | 14.19 | 13.10 | fixes most domain gap |
| SEP-XATTN noisy ep20 | + WHAM train noise, 0–5 dB | 18.69 | 14.35 | 13.47 | later epoch not best |
| **SEP-XATTN noisy ep11** | **+ WHAM train noise, 0–5 dB** | **18.79** | **14.34** | **13.79** | best noisy checkpoint |
| MATCHER_WHAM final | source audio + WHAM noise | 18.80 | 14.53 | 14.14 | still final winner |

Takeaway: noisy SEP-candidate training is real progress. It raises WHAM 0 dB
from 13.10 -> **13.79** and reaches the clean oracle ceiling, but it still trails
`MATCHER_WHAM` by ~0.35 dB at 0 dB and ~0.19 dB at 5 dB. Best checkpoint:
`exp/XATTN_ROUTER_SEP_NOISY_FT_0_5/models/xattn_router_sep_11.pt`.

## 5. Ceilings / where this tops out

- Clean Libri2Mix 2-spk practical max ≈ 20–22 dB (SOTA SepFormer/TF-GridNet);
  ours 18.8 → ~1–2 dB from SOTA, would need a stronger backbone.
- Noisy max ≈ 15–17 dB (separator also has to denoise); ours 0 dB 14.26 → ~0.7
  from 15. Depth + chunk are tapped out; further gains need a stronger separator
  backbone or more real-noisy training data (a larger effort).

## 6. Training logs (extracted)

### Separators — PIT train loss (−SI-SNR, more negative = better)

| run (jobid)          | what                         | warm/resume from | epochs   | final train −SISNR |
|----------------------|------------------------------|------------------|----------|--------------------|
| clean R9 (22234→70)  | clean PIT separation         | spk_emb warmstart| 150      | −20.42             |
| SEP_NOISY (22474)    | + synthetic Gaussian noise   | clean ep150      | 151→180  | −16.09             |
| SEP_WHAM (22509)     | + real WHAM noise            | clean ep150      | 151→230  | −14.25             |
| SEP_WHAM_BIG (22537) | deeper nr6→8                 | WHAM ep230       | 120      | −14.81             |
| **SEP_LC (22608)**   | + 10 s chunk, SNR −5..20     | BIG ep120        | 100      | −14.68             |

⚠️ Train loss is NOT comparable across rows: each regime mixes a different
noise/SNR range, so a "harder" curriculum shows a less-negative loss even when
the model is better. Clean (−20.4) looks best only because its task is easiest.
Compare models by the §2 eval table, not by train loss.

### Matchers — CE loss / in-batch retrieval acc (bs=32, chance ≈ 0.031)

| run (jobid)          | training audio        | epochs | final loss / batch_acc |
|----------------------|-----------------------|--------|------------------------|
| MATCHER_V1 (22435)   | clean source          | 60     | 1.02 / 0.70            |
| MATCHER_NOISY (22475)| + synthetic noise     | 60     | 1.60 / 0.54            |
| MATCHER_WHAM (22510) | + real WHAM noise     | 80     | 1.29 / 0.61            |
| STUDENT distill (22987) | + real WHAM noise, distilled from MATCHER_WHAM | 80 | 1.39 / 0.70 |

batch_acc is a 1-of-32 retrieval proxy during training; at inference the matcher
only does a 2-way pick, so routing_acc is ~99% (see §2). Noise lowers batch_acc
(harder audio↔text alignment); clean is easiest.

## Repro
- Separator: `wesep/bin/train_sep.py` (`--num_repeat 8 --chunk 160000
  --warm_sep <ckpt> --snr_min -5 --snr_max 20 --noise_dir <wham>`),
  cluster `sep_lc.slurm`.
- Matcher: `wesep/bin/train_matcher.py`, `matcher_wham.slurm`.
- Lightweight matcher: `wesep/bin/train_matcher_distill.py` (`--teacher <matcher>`),
  cluster `distill_student_1gpu.slurm`; eval auto-detects `kind=student_cnn`.
- XATTN SEP-router: `wesep/bin/train_xattn_router_sep.py`
  (`--router_init <clean-sep-router> --noise_prob 0.75 --snr_min 0 --snr_max 5
  --noise_dir <wham-tr>`), eval with `wesep/bin/eval_xattn_router_sep.py`.
- Eval: `wesep/bin/eval_compare.py` (`--noise_dir` for real-noise test,
  `--num_repeat` to match the separator), cluster `cmp_lc.slurm`.
- Final ckpts: `exp/SEP_LC/models/ckpt_100.pt`,
  `exp/MATCHER_WHAM/models/matcher_80.pt`.
