"""Diagnose WHERE the text cue dies: cross-attn output vs FiLM gamma/beta vs
final output. For the same mixture, feed oracle cue (cueA) vs the other
speaker's cue (cueB) and measure relative differences at each stage.

  python wesep/bin/diag_cue.py --config <cfg> --checkpoint <ckpt> \
    --test_data <raw.list> --test_cues <cues.yaml> --test_samples <jsonl> \
    --n 5 --device cpu
"""
from __future__ import print_function

import fire
import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (BASE_COLLECT_KEYS, build_collect_keys,
                                   tse_collate_fn, AUX_KEY_MAP)
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import parse_config_or_kwargs, set_seed


def _reldiff(a, b):
    return (a - b).norm().item() / (a.norm().item() + 1e-9)


def main(config, checkpoint, test_data, test_cues, test_samples, n=5,
         device="cpu", data_type="raw"):
    configs = parse_config_or_kwargs(config, checkpoint=checkpoint,
                                     test_data=test_data, test_cues=test_cues,
                                     test_samples=test_samples)
    set_seed(configs.get("seed", 42))
    dev = torch.device(device if device == "cpu" else "cuda:" + str(device))

    if "spk_model_init" in configs["model_args"]["tse_model"]:
        configs["model_args"]["tse_model"]["spk_model_init"] = False
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"])
    load_pretrained_model(model, checkpoint)
    model = model.to(dev).eval()

    # locate conditioning modules by class name (robust to attr naming)
    attn = [m for _, m in model.named_modules()
            if type(m).__name__ == "CrossFuse"]
    film = [m for _, m in model.named_modules()
            if type(m).__name__ == "TimeWiseFiLM"]
    print("found CrossFuse:", len(attn), "| TimeWiseFiLM:", len(film))
    caps = {}

    def cap(name):
        def hook(m, i, o):
            caps[name] = (o[0] if isinstance(o, (list, tuple)) else o).detach()
        return hook

    if attn:
        attn[0].register_forward_hook(cap("attn"))
    if film:
        film[0].gamma_proj.register_forward_hook(cap("gamma"))
        film[0].beta_proj.register_forward_hook(cap("beta"))

    configs["dataset_args"]["whole_utt"] = True
    ds = Dataset(data_type, test_data, configs["dataset_args"],
                 state="test", repeat_dataset=False, cues_yaml=test_cues)
    keys = build_collect_keys(load_yaml(test_cues), configs["dataset_args"],
                              BASE_COLLECT_KEYS)
    dl = DataLoader(ds, batch_size=1,
                    collate_fn=lambda b: tse_collate_fn(b, keys))

    print("\n# same mixture, cueA vs cueB. reldiff = ||x_A - x_B|| / ||x_A||")
    print("# out~0 => output ignores cue ; attn/gamma~0 => mechanism collapsed")
    done = 0
    with torch.no_grad():
        for batch in dl:
            mix = batch["wav_mix"].float().to(dev)
            cue = None
            for k in AUX_KEY_MAP.values():
                if k in batch and batch[k] is not None:
                    cue = batch[k].float().to(dev)
            if cue is None or cue.shape[0] < 2:
                continue
            idx = list(range(cue.shape[0]))
            idx = [idx[1], idx[0]] + idx[2:]      # swap first two (the pair)

            out_o = model(mix, [cue])
            out_o = out_o[0] if isinstance(out_o, (list, tuple)) else out_o
            a_o = caps.get("attn"); g_o = caps.get("gamma"); b_o = caps.get("beta")
            a_o = a_o.clone() if a_o is not None else None
            g_o = g_o.clone() if g_o is not None else None
            b_o = b_o.clone() if b_o is not None else None

            out_s = model(mix, [cue[idx]])
            out_s = out_s[0] if isinstance(out_s, (list, tuple)) else out_s
            a_s = caps.get("attn"); g_s = caps.get("gamma"); b_s = caps.get("beta")

            msg = "sample {}: out_reldiff={:.4f}".format(
                done, _reldiff(out_o[0], out_s[0]))
            if a_o is not None:
                msg += " | attn_reldiff={:.4f}".format(_reldiff(a_o[0], a_s[0]))
            if g_o is not None:
                msg += " | gamma_reldiff={:.4f} | beta_reldiff={:.4f}".format(
                    _reldiff(g_o[0], g_s[0]), _reldiff(b_o[0], b_s[0]))
            print(msg)
            done += 1
            if done >= n:
                break


if __name__ == "__main__":
    fire.Fire(main)
