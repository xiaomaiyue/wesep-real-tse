# Text-guided TSE on Libri2Mix (Week 2–4 pipeline)

Text-cue extension of the wesep speaker-cue TSE recipe. The transcript of
the target utterance (ground truth from LibriSpeech, **no ASR**) is encoded
by a frozen text encoder and injected into BSRNN through the same
conditioning interface as the speaker embedding.

## Artifacts layout

```
libri_data/                              # unified data folder
├── LibriSpeech/                         # raw corpus (train-clean-100 / dev-clean / test-clean)
├── Libri2Mix/wav16k/min/{split}/        # generated audio: mix_clean/ s1/ s2/ + samples.jsonl
└── text_cues/
    ├── transcripts.json                 # utt_id -> oracle transcript (lookup, no ASR)
    ├── {split}.text_meta.jsonl          # per-mixture target/interferer texts (both orientations)
    ├── {split}.text_cues.csv            # flat human-readable table
    ├── {split}.text_{oracle|interferer|random}.json   # cue-resource ablation files
    └── emb/{encoder}/embeddings.npz     # cached embeddings + meta.json
```

## Pipeline scripts (local/)

| script | role |
|---|---|
| `make_mix_clean.py` | generate Libri2Mix mix_clean+s1+s2 (faithful to official; no WHAM; `--limit` for subsets) |
| `build_text_metadata.py` | Libri2Mix CSV + LibriSpeech trans.txt -> text metadata (pure lookup) |
| `build_text_cue_resources.py` | metadata -> oracle/interferer/random resource JSONs |
| `extract_text_embeddings.py` | frozen encoder (minilm/bert/distilbert/mpnet) -> embeddings.npz |
| `prepare_text_data.sh` | one-shot: all three steps for all splits |
| `sanity_check_text.py` | join audio+text+waveforms, verify alignment |
| `smoke_train_text.py` | tiny end-to-end training run (pipeline proof) |

Reproduce (laptop or cluster):

```bash
# 1) audio (subset via --limit; full = omit)
python local/make_mix_clean.py --librispeech_dir $LS --libri2mix_csv $META/libri2mix_train-clean-100.csv \
  --split_name train-100 --out_root $DATA --fs 16000 --mode min --limit 2000
python local/scan_librimix.py $DATA/Libri2Mix/wav16k/min/train-100/mix_clean \
  --outfile $DATA/Libri2Mix/wav16k/min/train-100/samples.jsonl   # NOTE: absolute path!

# 2) text cues (all splits, one command)
./local/prepare_text_data.sh --librispeech_dir $LS --libri2mix_meta $META \
  --out_dir $DATA/text_cues --splits "train-100 dev test" --encoders "minilm"

# 3) smoke (local)
python local/smoke_train_text.py --samples .../samples.jsonl --cues .../cues_text_oracle.yaml --device cpu
```

## wesep integration (5 files)

- `wesep/dataset/processor_text.py` (new): attach `text_spk{i}` embedding per speaker slot
- `wesep/dataset/cues.py`: registered `text` cue (policy fixed, key `mix_spk_id`, fields `resource` + `embeddings`)
- `wesep/dataset/collate.py`: `text_aux` batch key; `AUX_KEY_MAP["text"]`
- `wesep/modules/speaker/spk_frontend.py`: `TextEmbFeature` = projection MLP -> `SpeakerFuseLayer` (reuses multiply/additive/concat/FiLM)
- `wesep/models/tse_bsrnn_spk.py`: cue routing by ndim (3D waveform / 2D text vector), `textemb` fusion step C5

cues.yaml example:

```yaml
cues:
  text:
    type: embedding
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: .../train-100.text_oracle.json     # swap file = cue ablation
      embeddings: .../emb/minilm/embeddings.npz    # swap dir  = encoder ablation
```

## Experiment matrix (all switches are config-only)

| axis | how to switch |
|---|---|
| cue modality: spk / text / spk+text | config: `tse_bsrnn_spk.yaml` / `tse_bsrnn_text.yaml` / `tse_bsrnn_spk_text.yaml` |
| cue correctness: oracle / interferer / random | cues.yaml `resource:` path |
| text encoder: minilm / bert / ... | re-run extractor; cues.yaml `embeddings:` + config `text_dim` |
| fusion: multiply / additive / concat / FiLM | config `textemb.fusion` |

Primary diagnostic (the headline result): if the model truly uses the text
cue, interferer-text should flip the extracted speaker and random-text
should degrade gracefully; if it ignores text, all three tie.

## Status / verification log

- 2026-06-11 dev/test text metadata: 3000+3000 mixtures, **0 missing lookups**
- 2026-06-11 audio gen check: `mix == s1+s2` within PCM16 quantization, 20/20
- 2026-06-11 unit test: text-only model fwd/bwd, guards OK (3.96M params)
- 2026-06-11 smoke train (20 dev mixtures, 16 steps, CPU 22s):
  SISDR-loss 20.18 -> 0.09, **PASS**
- 2026-06-11 train-100 full metadata: 13,900 mixtures / 27,800 utts, 0 missing;
  embeddings 32,123 utts in 25s (MPS); audio subsets train 2000 / dev 1000 /
  test 1000 (3.0G), sanity 2000/2000 + 1000/1000 all green
- 2026-06-11 smoke train on train-100 subset (12 steps): 26.97 -> 0.75, **PASS**

## Local training experiment (2026-06-12/13) — findings & cluster guidance

Ran a full-size (21.6M) text-oracle training locally to get a first real
number. **Result: do NOT trust local full runs for headline numbers**, but
the pipeline and the text mechanism are both validated. Story:

1. **Full run** (2000 mix x 40 ep, MPS, ~19h) collapsed to **passthrough**:
   test SI-SDRi = 0.00 dB, SI-SDR(est vs mixture) = 42 dB (output == input).
   Looked like "text is ignored."
2. **Root cause = scale + a scheduler trap**, not the model:
   - local budget is ~1/28 of the cluster recipe (13900 mix x 150 ep);
   - validation kept hitting MPS OOM -> NaN -> `ReduceLROnPlateau` never saw
     improvement and decayed LR toward zero early, freezing the model at the
     passthrough attractor. (`best.pt` was even stuck at epoch 2 because that
     was the only epoch with a non-NaN val.)
3. **Constant-LR overfit probe** (48 mix, `--no_sched`, 60 ep, ~45 min):
   train/val SI-SDR climbed 0 -> +3.5 dB, still rising. Text path *can* drive
   separation.
4. **Swap test (decisive)**: same mixture/target, only the text cue changed:
   oracle text -> +3.13 dB, interferer text -> -10.79 dB. **13.9 dB swing =
   the model genuinely uses the text cue to pick the target.**

### What this means for the 4090 cluster
- The approach works mechanically; full-scale convergence just needs the real
  data/epoch budget and a sane LR schedule.
- **Use the recipe's epoch-based scheduler (`ExponentialDecrease`), not
  plateau-on-val.** On CUDA, val won't OOM, but epoch-based decay removes the
  dependency on validation entirely (safer). Best-model selection still needs
  working validation, which CUDA gives.
- **Pre-flight check, minutes per config:** before launching the full matrix,
  run `train_local_text.py --no_sched` on ~50 mixtures for each
  encoder/fusion variant and confirm train SI-SDR escapes 0 dB. Catches a dead
  config before you spend a GPU-day on it.
- Local MPS ceiling: full model fits only at batch<=2 x 2s (16GB unified mem);
  fine for the pipeline/probe, useless for headline numbers.

### Bugs fixed during this run (already in the code)
- validation length mismatch: istft trims to a hop multiple, so align est/tgt
  with `L = min(...)` before SI-SDR (was crashing every epoch).
- checkpoint now saved BEFORE validation, so a val crash can't lose an epoch.
- `--no_sched` flag added for clean overfit probes.
- Pitfall: `samples.jsonl` must use **absolute paths** (wesep warns+skips on
  read failure; with repeat_dataset this spins forever instead of erroring)
- BSRNN LSTMs do not run on MPS (fallback ping-pong is slower than CPU);
  use CPU locally, CUDA on cluster
