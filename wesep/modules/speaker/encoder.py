import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi

_GET_SPEAKER_MODEL = None


def _find_wespeaker_dir():
    for root in sys.path:
        if not root:
            root = "."
        pkg_dir = Path(root) / "wespeaker"
        if (pkg_dir / "models" / "speaker_model.py").exists():
            return pkg_dir
    return None


def _install_lightweight_wespeaker(pkg_dir):
    for name in list(sys.modules):
        if name == "wespeaker" or name.startswith("wespeaker."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType("wespeaker")
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg.__path__ = [str(pkg_dir)]
    pkg.__package__ = "wespeaker"
    pkg.__spec__ = importlib.machinery.ModuleSpec(
        "wespeaker", loader=None, is_package=True)
    sys.modules["wespeaker"] = pkg


def get_wespeaker_model_factory():
    global _GET_SPEAKER_MODEL
    if _GET_SPEAKER_MODEL is not None:
        return _GET_SPEAKER_MODEL

    try:
        module = importlib.import_module("wespeaker.models.speaker_model")
    except Exception as first_error:
        pkg_dir = _find_wespeaker_dir()
        if pkg_dir is None:
            raise first_error
        _install_lightweight_wespeaker(pkg_dir)
        try:
            module = importlib.import_module("wespeaker.models.speaker_model")
        except Exception:
            raise first_error

    _GET_SPEAKER_MODEL = module.get_speaker_model
    return _GET_SPEAKER_MODEL


class Fbank_kaldi(nn.Module):
    """
    A wrapper module that performs:
    1) compute_fbank()
    2) apply_cmvn()

    Keep same with kaldi, not sure about the calculation efficiency.
    Strictly preserves the original arguments of both functions.
    """

    def __init__(
        self,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=1.0,
        sample_rate=16000,
        norm_mean=True,
        norm_var=False,
    ):
        super().__init__()
        # store parameters exactly as original function uses
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.sample_rate = sample_rate
        self.norm_mean = norm_mean
        self.norm_var = norm_var

    def compute_fbank(self, data):
        """Exact wrapper of the old compute_fbank()"""
        fbank_list = []
        for index_ in range(data.shape[0]):
            waveform = data[index_, :].unsqueeze(0)
            waveform = waveform * (1 << 15)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.num_mel_bins,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                sample_frequency=self.sample_rate,
                window_type="hamming",
                use_energy=False,
            )
            fbank_list.append(mat.unsqueeze(0))
        np_fbank = torch.cat(fbank_list, 0)
        return np_fbank

    def apply_cmvn(self, data):
        """Exact wrapper of the old apply_cmvn()"""
        mat_list = []
        for index_ in range(data.shape[0]):
            mat = data[index_, :, :]
            if self.norm_mean:
                mat = mat - torch.mean(mat, dim=0)
            if self.norm_var:
                mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
            mat_list.append(mat.unsqueeze(0))
        np_mat = torch.cat(mat_list, 0)
        return np_mat

    def forward(self, data):
        """Compute fbank → CMVN"""
        fb = self.compute_fbank(data)
        fb = self.apply_cmvn(fb)
        return fb


class SpeakerEncoder(nn.Module):
    """
    Wraps get_speaker_model + loading pretrained + freezing.

    Args:
        conf:
            model_name (str): name used in get_speaker_model(...)
            spk_args (dict): arguments passed to the speaker model constructor
            pretrained (str or None): path to pretrained model
            freeze (bool): whether to freeze all parameters
    """

    def __init__(self, conf):
        super().__init__()

        model_name = conf["model"]
        spk_args = conf.get("spk_args", {})
        pretrained = conf.get("pretrained", None)
        if pretrained:
            freeze = conf.get("freeze", True)
        else:
            freeze = conf.get("freeze", False)

        # 1. build model
        get_speaker_model = get_wespeaker_model_factory()
        self.spk_model = get_speaker_model(model_name)(**spk_args)

        # 2. load pretrained if provided
        if pretrained is not None:
            pretrained_model = torch.load(pretrained)
            state = self.spk_model.state_dict()

            for key in state.keys():
                if key in pretrained_model:
                    state[key] = pretrained_model[key]
                else:
                    print(f"not {key} loaded")

            self.spk_model.load_state_dict(state)

        # 3. freeze parameters if needed
        if freeze:
            for p in self.spk_model.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Forward simply calls the speaker encoder forward.
        """
        return self.spk_model(x)
