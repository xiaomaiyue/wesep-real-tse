"""Does the frozen-wav2vec2 feature-MSE actually discriminate passthrough?
Prints, per mixture: self (target vs target, ~0), mix (mixture vs target,
passthrough case), wrong (other speaker vs target). If mix/wrong >> self, the
content loss has discriminative power and only the weight needs raising.
"""
import fire
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (BASE_COLLECT_KEYS, build_collect_keys,
                                   tse_collate_fn)
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import parse_config_or_kwargs, set_seed


def _prep(x):
    if x.dim() == 3:
        x = x.squeeze(1)
    x = x.float()
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)


def main(config, test_data, test_cues, test_samples, n=5, data_type="raw"):
    configs = parse_config_or_kwargs(config, test_data=test_data,
                                     test_cues=test_cues,
                                     test_samples=test_samples)
    set_seed(42)
    w2v = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().eval()
    for p in w2v.parameters():
        p.requires_grad_(False)

    def feat(x):
        return w2v.extract_features(_prep(x))[0][-1]

    def cmse(a, b):
        return F.mse_loss(feat(a), feat(b)).item()

    configs["dataset_args"]["whole_utt"] = True
    ds = Dataset(data_type, test_data, configs["dataset_args"], state="test",
                 repeat_dataset=False, cues_yaml=test_cues)
    keys = build_collect_keys(load_yaml(test_cues), configs["dataset_args"],
                              BASE_COLLECT_KEYS)
    dl = DataLoader(ds, batch_size=1,
                    collate_fn=lambda b: tse_collate_fn(b, keys))

    print("# self=target-vs-target(~0) mix=passthrough wrong=other-speaker")
    done = 0
    with torch.no_grad():
        for batch in dl:
            mix = batch["wav_mix"].float()
            tgt = batch["wav_target"].float()
            if tgt.shape[0] < 2:
                continue
            wrong = tgt[[1, 0]]
            L = min(mix.shape[-1], tgt.shape[-1])
            mix, tgt, wrong = mix[..., :L], tgt[..., :L], wrong[..., :L]
            print("self=%.4f  mix=%.4f  wrong=%.4f" %
                  (cmse(tgt, tgt), cmse(mix, tgt), cmse(wrong, tgt)))
            done += 1
            if done >= n:
                break


if __name__ == "__main__":
    fire.Fire(main)
