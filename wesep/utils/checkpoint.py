from typing import List, Optional

import torch

from wesep.utils.schedulers import BaseClass


def load_pretrained_model(model: torch.nn.Module,
                          path: str,
                          type: str = "generator",
                          strict: bool = False):
    """Warm-start ``model`` from a pretrained checkpoint.

    With ``strict=False`` (default) this performs a *partial* load: only the
    checkpoint tensors whose name and shape match the target model are copied,
    everything else (e.g. a different speaker/cue conditioning branch) is left
    at its initialized value. This lets a text-cued model reuse the BSRNN
    ``sep_model.*`` backbone from a speaker-embedding checkpoint while keeping
    its own ``spk_ft.textemb.*`` path random. A summary is logged so you can
    confirm the backbone actually transferred.
    """
    assert type in ["generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if type == "generator":
        state = states["models"][0]
    else:
        assert len(states["models"]) == 2
        state = states["models"][1]

    if isinstance(
            model,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        target = model.module
    else:
        target = model

    if strict:
        target.load_state_dict(state)
        return

    # strip a possible DDP "module." prefix from the checkpoint
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model_sd = target.state_dict()
    matched, shape_mismatch = {}, []
    for k, v in state.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                matched[k] = v
            else:
                shape_mismatch.append(k)
    missing = [k for k in model_sd if k not in matched]
    unused = [k for k in state if k not in model_sd]

    target.load_state_dict(matched, strict=False)
    print("[load_pretrained_model] partial warm-start from {}\n"
          "  loaded={} | shape_mismatch={} | "
          "model_keys_left_random={} | ckpt_keys_unused={}".format(
              path, len(matched), len(shape_mismatch), len(missing),
              len(unused)))
    if shape_mismatch:
        print("  [warn] shape mismatch (skipped): " +
              ", ".join(shape_mismatch[:8]) +
              (" ..." if len(shape_mismatch) > 8 else ""))


def load_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
    only_model: bool = False,
    mode: str = "all",
):
    assert mode in ["all", "generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if mode == "generator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][0]],
            [states["optimizers"][0]],
            [states["schedulers"][0]],
        )
    elif mode == "discriminator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][1]],
            [states["optimizers"][1]],
            [states["schedulers"][1]],
        )
    else:
        model_state, optimizer_state, scheduler_state = (
            states["models"],
            states["optimizers"],
            states["schedulers"],
        )

    for model, state in zip(models, model_state):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state, strict=False)
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
    if not only_model:
        for optimizer, state in zip(optimizers, optimizer_state):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, scheduler_state):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is not None:
                scaler.load_state_dict(states["scaler"])


def save_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
):
    if isinstance(models[0], torch.nn.DataParallel):
        state_dict = [model.module.state_dict() for model in models]
    elif isinstance(models[0], torch.nn.parallel.DistributedDataParallel):
        state_dict = [model.module.state_dict() for model in models]
    else:
        state_dict = [model.state_dict() for model in models]
    torch.save(
        {
            "models":
            state_dict,
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers":
            [s.state_dict() if s is not None else None for s in schedulers],
            "scaler":
            scaler.state_dict() if scaler is not None else None,
        },
        path,
    )
