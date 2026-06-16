# Copyright (c) 2026
#
# SPDX-License-Identifier: Apache-2.0
#
# Text (semantic) cue processor: attaches precomputed text embeddings to
# samples. Mirrors processor_speaker.py, but the cue is a fixed-size vector
# loaded from an .npz cache instead of an enrollment waveform.

import logging

import numpy as np

from wesep.utils.file_utils import load_json

# module-level caches (per worker process)
_TEXT_RESOURCE_CACHE = {}
_TEXT_EMB_CACHE = {}


def _get_text_resource(resource_path):
    if resource_path not in _TEXT_RESOURCE_CACHE:
        _TEXT_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    return _TEXT_RESOURCE_CACHE[resource_path]


def _get_text_embeddings(emb_path):
    if emb_path not in _TEXT_EMB_CACHE:
        # np.load on npz is lazy; keep the handle and read members on demand
        _TEXT_EMB_CACHE[emb_path] = np.load(emb_path)
    return _TEXT_EMB_CACHE[emb_path]


def _build_lookup_key(sample, spk_slot, key_field):
    """Same semantics as processor_speaker._build_lookup_key."""
    if key_field == "spk_id":
        return sample[spk_slot]
    elif key_field == "mix_spk_id":
        mix_key = sample.get("key", None)
        if mix_key is None:
            raise KeyError("sample missing 'key' for mix_spk_id cue")
        return f"{mix_key}::{sample[spk_slot]}"
    elif key_field == "utt_id":
        # online-mix path: each speaker slot carries its utterance id in
        # sample["utt_<slot>"] (e.g. utt_spk1); the embedding npz is keyed by
        # that utterance id (emb_key == utt_id).
        utt = sample.get(f"utt_{spk_slot}", None)
        if utt is None:
            raise KeyError(f"sample missing 'utt_{spk_slot}' for utt_id cue")
        return utt
    else:
        raise ValueError(f"Unsupported key_field for text cue: {key_field}")


def attach_fixed_text_cue(
    data,
    resource_path,
    emb_path,
    key_field="mix_spk_id",
    scope="speaker",
    required=True,
):
    """Attach a fixed text embedding per (mixture, speaker) slot.

    resource JSON: lookup_key -> {"utt_id": str, "emb_key": str}
        ``emb_key`` indexes the npz; oracle/interferer/random ablations are
        just different resource files sharing one npz.
    Attaches: sample[f"text_spk{N}"] = float32 (D,) vector
    """
    if scope not in ("speaker", "utterance"):
        raise ValueError(f"Unsupported scope: {scope}")

    resource = _get_text_resource(resource_path)
    embs = _get_text_embeddings(emb_path)

    for sample in data:
        spk_slots = [k for k in sample.keys() if k.startswith("spk")]

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        for slot in spk_slots:
            lookup_key = _build_lookup_key(sample, slot, key_field)

            if lookup_key not in resource:
                if required:
                    raise KeyError(f"text cue not found: {lookup_key}")
                continue

            emb_key = resource[lookup_key]["emb_key"]
            if emb_key not in embs:
                msg = f"text embedding missing in npz: {emb_key}"
                if required:
                    raise KeyError(msg)
                logging.warning(msg)
                continue

            vec = np.asarray(embs[emb_key], dtype=np.float32).reshape(-1)
            sample[f"text_{slot}"] = vec

        yield sample
