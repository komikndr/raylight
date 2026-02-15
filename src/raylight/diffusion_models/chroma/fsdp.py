from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import detect_dtype_mismatch
import torch
import logging
from typing import cast
from torch.distributed.tensor import DTensor
from comfy.quant_ops import (
    QuantizedTensor,
    QUANT_ALGOS,
    TensorCoreFP8Layout,
    get_layout_class,
)
from raylight.diffusion_models.quant_fsdp_loader import load_from_full_model_state_dict

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout as _CKNvfp4Layout,
        register_layout_op,
        register_layout_class,
        get_layout_class,
    )
    _CK_AVAILABLE = True
    if torch.version.cuda is None:
        ck.registry.disable("cuda")
    else:
        cuda_version = tuple(map(int, str(torch.version.cuda).split('.')))
        if cuda_version < (13,):
            ck.registry.disable("cuda")
            logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")

    ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")

except ImportError as e:
    logging.error(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False

    class QuantizedTensor:
        pass

    class _CKFp8Layout:
        pass

    class _CKNvfp4Layout:
        pass

    def register_layout_class(name, cls):
        pass

    def get_layout_class(name):
        return None


def build_ignored_params(module, ref_dtype):
    ignored = set()

    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored.add(param)
            continue
        if not (name.endswith("weight") or name.endswith("bias")):
            ignored.add(param)
    return ignored


def _replace_param_by_name(model, name, new_param):
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)

    setattr(obj, parts[-1], new_param)


def sync_fp8_layout_with_state_dict(model, state_dict):
    for name, param in model.named_parameters():
        if not isinstance(param, QuantizedTensor):
            continue

        if name not in state_dict:
            continue

        sd_param = state_dict[name]

        if not isinstance(sd_param, QuantizedTensor):
            continue

        model_layout = param._layout_cls
        sd_layout = sd_param._layout_cls

        if model_layout == sd_layout:
            continue

        # recreate shell QuantizedTensor with SD layout
        new_q = QuantizedTensor(
            param._qdata,      # still meta
            sd_layout,        # <-- follow state_dict string
            param._params,
        )

        _replace_param_by_name(model, name, new_q)


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    sync_fp8_layout_with_state_dict(model, model_state_dict)
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        # noqa: W503
        if (
            not name.startswith("single_blocks.")
            and not name.startswith("double_blocks.")
            and not name.startswith("distilled_guidance_layer.")
        ):
            ignored_params.add(param)
    # Shard distilled_guidance_layer
    ref_dtype = diffusion_model.distilled_guidance_layer.layers[0].in_layer.weight.dtype
    distil_ignored_params = build_ignored_params(
        diffusion_model.distilled_guidance_layer,
        ref_dtype
    )

    diffusion_model.distilled_guidance_layer = fully_shard(
        module=diffusion_model.distilled_guidance_layer,
        mp_policy=MixedPrecisionPolicy(),
        reshard_after_forward=True,
        ignored_params=distil_ignored_params,
    )

    # Check dtype missmatch from scaled model
    ref_dtype = diffusion_model.double_blocks[0].img_attn.qkv.weight.dtype

    # Shard single_blocks
    for i, block in enumerate(diffusion_model.single_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
        )

    # Shard double_blocks
    for i, block in enumerate(diffusion_model.double_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
        )

    # Root wrap with ignored params
    fully_shard(diffusion_model,
                ignored_params=ignored_params,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True)

    model.diffusion_model = diffusion_model
    # print("MODEL PARAMS")
    # for p in model.parameters():
    #     print(p)
    # print("SD PARAMS")
    # for k, v in model_state_dict.items():
    #     print(k,":",v)
    load_from_full_model_state_dict(model,
                                    model_state_dict,
                                    torch.device("cuda"),
                                    strict=False,
                                    cpu_offload=False,
                                    use_distributed_state_dict=False,
                                    release_sd=False)

    # set_model_state_dict(
    #     model=model,
    #     model_state_dict=model_state_dict,
    #     options=StateDictOptions(
    #         full_state_dict=True,
    #         broadcast_from_rank0=True,
    #         cpu_offload=enable_cpu_offload
    #     ),
    # )

    return model
