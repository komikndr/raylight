# Modified from torchtune, for dealing with FSDP2 on NFP4, but it is modified to apply Comfy QuantizedTensor
# https://github.com/meta-pytorch/torchtune/blob/b3c964f360299da49f3a519aed18752ceedeffcf/torchtune/training/_distributed.py#L336
# Theoritically speaking, it is possible to FSDP GGUF format.
from typing import cast
import json
import torch
from torch.distributed.tensor import DTensor

from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class


def _decode_comfy_quant(conf):
    if conf is None:
        return None
    if isinstance(conf, dict):
        return conf
    if isinstance(conf, (bytes, bytearray)):
        return json.loads(conf.decode("utf-8"))
    if isinstance(conf, torch.Tensor):
        if conf.dtype == torch.uint8:
            return json.loads(conf.detach().cpu().numpy().tobytes())
        return json.loads(conf.detach().cpu().numpy().tobytes().decode("utf-8"))
    if isinstance(conf, str):
        return json.loads(conf)
    raise TypeError(f"Unsupported comfy_quant type: {type(conf)}")


def _shard_tensor(full_tensor, sharded_meta_param, device):
    if not hasattr(sharded_meta_param, "device_mesh"):
        return full_tensor.to(device=device)
    mesh = sharded_meta_param.device_mesh
    if mesh.ndim > 1:
        raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim}")
    shard_mesh_dim = 0
    shard_world_size = mesh.size(shard_mesh_dim)
    shard_rank = cast(
        torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)
    ).rank()
    chunk = list(torch.chunk(full_tensor, shard_world_size, dim=0))[shard_rank]
    sharded_param = full_tensor.new_zeros(chunk.size()).to(device=device)
    sharded_param[: chunk.size(0)].copy_(chunk)
    return sharded_param


def _build_quantized_tensor(param_name, full_sd, sharded_meta_param, device):
    if not param_name.endswith("weight"):
        return None
    prefix = param_name[: -len("weight")]
    conf = _decode_comfy_quant(full_sd.get(f"{prefix}comfy_quant"))
    if conf is None:
        if isinstance(sharded_meta_param, QuantizedTensor) or (
            hasattr(sharded_meta_param, "_local_tensor")
            and isinstance(sharded_meta_param._local_tensor, QuantizedTensor)
        ):
            raise ValueError(f"Missing comfy_quant for {param_name}")
        return None

    quant_format = conf.get("format", None)
    if quant_format is None or quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unknown quantization format for {param_name}: {quant_format}")

    qconfig = QUANT_ALGOS[quant_format]
    layout_name = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_name)
    if layout_cls is None:
        raise ValueError(f"Missing layout class for {layout_name}")

    full_qdata = full_sd.get(param_name)
    if full_qdata is None:
        raise ValueError(f"Missing quantized weight for {param_name}")

    if isinstance(full_qdata, QuantizedTensor):
        qdata = full_qdata._qdata
    else:
        qdata = full_qdata.to(device=device, dtype=qconfig["storage_t"])

    qdata = qdata.to(device=device)
    qdata = _shard_tensor(qdata, sharded_meta_param, device)

    local_meta = None
    if isinstance(sharded_meta_param, QuantizedTensor):
        local_meta = sharded_meta_param
    elif hasattr(sharded_meta_param, "_local_tensor") and isinstance(
        sharded_meta_param._local_tensor, QuantizedTensor
    ):
        local_meta = sharded_meta_param._local_tensor

    if local_meta is not None and hasattr(local_meta, "_params"):
        orig_dtype = getattr(local_meta._params, "orig_dtype", None)
        orig_shape = getattr(local_meta._params, "orig_shape", None)
    else:
        orig_dtype = None
        orig_shape = None

    params_kwargs = {}
    if orig_dtype is not None:
        params_kwargs["orig_dtype"] = orig_dtype
    if orig_shape is not None:
        params_kwargs["orig_shape"] = orig_shape
    else:
        params_kwargs["orig_shape"] = tuple(qdata.shape)

    if quant_format in ["float8_e4m3fn", "float8_e5m2"]:
        scale = full_sd.get(f"{prefix}weight_scale")
        if scale is not None:
            scale = scale.to(device=device)
        params_kwargs["scale"] = scale
    elif quant_format == "nvfp4":
        tensor_scale = full_sd.get(f"{prefix}weight_scale_2")
        block_scale = full_sd.get(f"{prefix}weight_scale")
        if tensor_scale is None or block_scale is None:
            raise ValueError(f"Missing NVFP4 scales for {param_name}")
        tensor_scale = tensor_scale.to(device=device)
        block_scale = block_scale.view(dtype=torch.float8_e4m3fn).to(device=device)
        block_scale = _shard_tensor(block_scale, sharded_meta_param, device)
        params_kwargs["scale"] = tensor_scale
        params_kwargs["block_scale"] = block_scale
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    params = layout_cls.Params(**params_kwargs)
    return QuantizedTensor(qdata, layout_name, params)


def load_from_full_model_state_dict(
    model,
    full_sd,
    device,
    strict,
    cpu_offload,
    use_distributed_state_dict,
    release_sd,
):
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model
    Args:
        model (FSDPModule): Model to generate fully qualified names for cpu_state_dict
        full_sd (dict[str, Any]): a full state dict to load into the model
        device (torch.device): device used to move full state dict tensors
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if offload to CPU is enabled
        use_distributed_state_dict (bool): Whether to use set_model_state_dict for loading
            state dict. Default: False. (TODO: this should be True once 3.2 Vision is fixed)
        release_sd (bool): whether to release memory of full_sd to save ram usage
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Raises:
        NotImplementedError: If got FSDP with more than 1D.
    """

    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        quant_tensor = _build_quantized_tensor(
            param_name, full_sd, sharded_meta_param, device
        )
        if quant_tensor is not None:
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = DTensor.from_local(
                    quant_tensor,
                    device_mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                )
            else:
                sharded_tensor = quant_tensor
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
            if release_sd:
                prefix = param_name[: -len("weight")]
                for key in (
                    param_name,
                    f"{prefix}weight_scale",
                    f"{prefix}weight_scale_2",
                    f"{prefix}comfy_quant",
                ):
                    if key in full_sd:
                        full_sd[key] = None
            continue

        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            raise ValueError(f"Missing parameter {param_name} in state_dict")
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if not hasattr(sharded_meta_param, "device_mesh"):
            sharded_tensor = full_tensor
        else:
            sharded_param = _shard_tensor(full_tensor, sharded_meta_param, device)
            sharded_tensor = DTensor.from_local(
                sharded_param,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        if release_sd:
            full_sd[param_name] = None

    return model.load_state_dict(sharded_sd, strict=strict, assign=True)
