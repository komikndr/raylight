from __future__ import annotations

import logging

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from raylight.diffusion_models.fsdp_utils import (
    freeze_and_detect_qt,
    fully_shard_bottom_up,
    load_from_full_model_state_dict,
)


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    has_qt = freeze_and_detect_qt(model)

    diffusion_model = model.diffusion_model

    fsdp_kwargs = {
        "reshard_after_forward": True,
    }
    fully_shard_bottom_up(
        diffusion_model,
        fsdp_kwargs=fsdp_kwargs,
        native_ignore_scale=not has_qt,
    )
    model.diffusion_model = diffusion_model

    if has_qt:
        load_from_full_model_state_dict(
            model=model,
            full_sd=model_state_dict,
            device=torch.device("cuda"),
            strict=False,
            cpu_offload=enable_cpu_offload,
            release_sd=False,
        )
    else:
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=False,
            cpu_offload=enable_cpu_offload,
        )
        set_model_state_dict(model=model, model_state_dict=model_state_dict, options=options)

    for name, param in model.named_parameters():
        if param.device.type == "meta":
            print("META PARAM:", name)
    return model
