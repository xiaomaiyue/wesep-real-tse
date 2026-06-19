"""Route-B end-to-end eval: separate (2 outputs) -> route by frozen-ASR CER to
the cue text -> measure SI-SNRi of the routed output vs the target.

Reports: routing accuracy, routed SI-SNRi (the real route-B score) and the
oracle-assignment SI-SNRi (separator ceiling) so we can see routing reaches it.

  python wesep/bin/eval_routeB.py --sep_ckpt <ckpt> \
    --samples <test samples.jsonl> --meta <test.text_meta.jsonl> --n 200
"""
import json
import fire
import torch
import torchaudio

from wesep.modules.separator.bsrnn import BSRNN
from wesep.utils.score import cal_SISNRi


def _greedy(emission, labels):
    idx = emission.argmax(-1).tolist()
    out, prev = [], None
    for i in idx:
        if i != prev and i != 0:
            out.append(labels[i])
        prev = i
    return "".join(out).replace("|", " ").strip()


def _cer(hyp, ref):
    if len(ref) == 0:
        return 1.0 if hyp else 0.0
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[len(hyp)] / len(ref)


def main(sep_ckpt, samples, meta, n=200, device="cpu"):
    dev = torch.device(device if device == "cpu" else "cuda:" + str(device))
    sep = BSRNN(sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
                causal=False, nspk=2, spec_dim=2).to(dev).eval()
    sd = torch.load(sep_ckpt, map_location="cpu")["model"]
    sep.load_state_dict(sd)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    asr = bundle.get_model().to(dev).eval()
    labels = bundle.get_labels()

    text = {}
    with open(meta) as f:
        for line in f:
            d = json.loads(line)
            for utt, t in d.get("src_text", {}).items():
                text[utt] = t.upper()

    @torch.no_grad()
    def load(p):
        w, sr = torchaudio.load(p)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        return w.mean(0)

    @torch.no_grad()
    def transcribe(wav):
        x = wav.unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-5)
        emis, _ = asr(x.to(dev))
        return _greedy(emis[0].cpu(), labels)

    lines = [json.loads(x) for x in open(samples)]
    route_ok = 0
    n_pair = 0
    sisnri_routed = 0.0
    sisnri_oracle = 0.0
    sisnri_interf = 0.0
    acc = 0
    with torch.no_grad():
        for d in lines[:n]:
            spks = d["spk"]
            if len(spks) < 2:
                continue
            utts = {}
            for s in spks:
                for u in text:
                    if u.split("-")[0] == s and u in d["key"]:
                        utts[s] = u
                        break
            if len(utts) < 2:
                continue
            mix = load(d["mix"]["default"][0]).to(dev)
            est = sep(mix.unsqueeze(0))[0]            # (2, T)
            L = min(est.shape[-1], mix.shape[-1])
            mix_np = mix[:L].cpu().numpy()
            hyp = [transcribe(est[k, :L]) for k in range(2)]
            for tgt in spks:
                ref = load(d["src"][tgt][0]).to(dev)[:L].cpu().numpy()
                cue = text[utts[tgt]]
                cers = [_cer(hyp[0], cue), _cer(hyp[1], cue)]
                pick = 0 if cers[0] <= cers[1] else 1
                # ABLATION: route with the WRONG (interferer) speaker's text.
                other = [s for s in spks if s != tgt][0]
                cue_i = text[utts[other]]
                ceri = [_cer(hyp[0], cue_i), _cer(hyp[1], cue_i)]
                pick_i = 0 if ceri[0] <= ceri[1] else 1
                # oracle assignment: the est that best matches this target
                e0 = est[0, :L].cpu().numpy()
                e1 = est[1, :L].cpu().numpy()
                _, d0 = cal_SISNRi(e0, ref, mix_np)
                _, d1 = cal_SISNRi(e1, ref, mix_np)
                opick = 0 if d0 >= d1 else 1
                route_ok += (pick == opick)
                n_pair += 1
                dr = [d0, d1][pick]
                do = [d0, d1][opick]
                sisnri_routed += dr
                sisnri_oracle += do
                sisnri_interf += [d0, d1][pick_i]
                acc += (dr > 1)
    m = max(n_pair, 1)
    print("pairs=%d" % n_pair)
    print("routing-vs-oracle agreement = %.1f%%" % (100.0 * route_ok / m))
    print("ROUTED   SI-SNRi = %.3f dB | accept>1dB = %.1f%%" %
          (sisnri_routed / m, 100.0 * acc / m))
    print("ORACLE   SI-SNRi = %.3f dB  (separator ceiling)" %
          (sisnri_oracle / m))
    print("ABLATION INTERFERER-text routing SI-SNRi = %.3f dB "
          "(should collapse if text drives routing)" % (sisnri_interf / m))


if __name__ == "__main__":
    fire.Fire(main)
