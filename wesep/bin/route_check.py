"""Route-B viability check: given the two CLEAN sources (perfect-separation
upper bound), can a frozen wav2vec2 + CER-to-cue-text pick the target speaker?

Reports routing accuracy. If high, route B = (any decent separator) + this
routing. Run on CPU:
  python wesep/bin/route_check.py --samples <test/samples.jsonl> \
    --meta <test.text_meta.jsonl> --n 60
"""
import json
import fire
import torch
import torchaudio


def _greedy_decode(emission, labels):
    # emission: (T, C) logits
    idx = emission.argmax(-1).tolist()
    out = []
    prev = None
    for i in idx:
        if i != prev and i != 0:        # 0 == blank '-'
            out.append(labels[i])
        prev = i
    return "".join(out).replace("|", " ").strip()


def _cer(hyp, ref):
    # char-level edit distance / len(ref)
    h, r = hyp, ref
    if len(r) == 0:
        return 1.0 if h else 0.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / len(r)


def main(samples, meta, n=60):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    labels = bundle.get_labels()
    for p in model.parameters():
        p.requires_grad_(False)

    # utt_id -> transcript
    text = {}
    with open(meta) as f:
        for line in f:
            d = json.loads(line)
            for utt, t in d.get("src_text", {}).items():
                text[utt] = t.upper()

    @torch.no_grad()
    def transcribe(path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0, keepdim=True)
        wav = (wav - wav.mean()) / (wav.std() + 1e-5)
        emis, _ = model(wav)
        return _greedy_decode(emis[0], labels)

    correct = tot = 0
    cer_t = cer_o = 0.0
    with open(samples) as f:
        lines = [json.loads(x) for x in f]
    for d in lines[:n]:
        spks = d["spk"]
        src = d["src"]
        if len(spks) < 2:
            continue
        # find each spk's utt_id (key piece starting with spk id)
        utts = {}
        for s in spks:
            for u in text:
                if u.split("-")[0] == s and (u in d["key"]):
                    utts[s] = u
                    break
        if len(utts) < 2:
            continue
        hyp = {s: transcribe(src[s][0]) for s in spks}
        for tgt in spks:
            ref = text[utts[tgt]]
            other = [s for s in spks if s != tgt][0]
            c_tgt = _cer(hyp[tgt], ref)
            c_oth = _cer(hyp[other], ref)
            routed = tgt if c_tgt <= c_oth else other
            correct += (routed == tgt)
            tot += 1
            cer_t += c_tgt
            cer_o += c_oth
    print("routing accuracy = %d/%d = %.1f%%" % (correct, tot,
                                                 100.0 * correct / max(tot, 1)))
    print("mean CER  target-source-vs-cue = %.3f | other-source-vs-cue = %.3f"
          % (cer_t / max(tot, 1), cer_o / max(tot, 1)))


if __name__ == "__main__":
    fire.Fire(main)
