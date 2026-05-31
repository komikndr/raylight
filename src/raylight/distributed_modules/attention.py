import os
import torch

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)
from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_forward

from yunchang.kernels import AttnType
from yunchang.globals import PROCESS_GROUP
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

_ATTN_TYPE = None
_SYNC_ULYSSES = None
_FORCE_RING_ONLY = False
_FORCE_RING_ATTENTION_CALL_COUNT = 0


def _force_ring_mem_log_interval():
    try:
        return int(os.environ.get("RAYLIGHT_FORCE_RING_MEM_LOG_INTERVAL", "1"))
    except ValueError:
        return 1


def _log_force_ring_memory(label, call_idx):
    if not torch.cuda.is_available():
        return
    interval = _force_ring_mem_log_interval()
    if interval <= 0 or call_idx % interval != 0:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"[Raylight][force-ring-attn][rank={rank}][call={call_idx}][{label}] "
        f"allocated={allocated:.2f}MiB reserved={reserved:.2f}MiB max_allocated={max_allocated:.2f}MiB"
    )


def set_attn_type(attn):
    global _ATTN_TYPE
    _ATTN_TYPE = attn


def get_attn_type():
    if _ATTN_TYPE is None:
        raise RuntimeError("_ATTN_TYPE is not initialized")
    else:
        return _ATTN_TYPE


def set_sync_ulysses(is_sync):
    global _SYNC_ULYSSES
    _SYNC_ULYSSES = is_sync


def get_sync_ulysses():
    if _SYNC_ULYSSES is None:
        raise RuntimeError("_SYNC_ULYSSES variable is not initialized")
    else:
        return _SYNC_ULYSSES


def set_force_ring_only(is_force):
    global _FORCE_RING_ONLY
    _FORCE_RING_ONLY = bool(is_force)


def get_force_ring_only():
    return bool(_FORCE_RING_ONLY)


def make_xfuser_attention(attn_type, sync_ulysses, force_ring_only=None):
    if force_ring_only is None:
        force_ring_only = get_force_ring_only()
    print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}, Force Ring Only: {force_ring_only}")
    attn = AttnType[attn_type]
    if attn_type == "SAGE_FP8_CUDA":
        ensure_hf_fp8_cuda_kernel()
    elif attn_type == "SAGE_FP8_SM90":
        ensure_hf_sm90_kernel

    xfuser_attn = None
    if not force_ring_only:
        xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn)

    def _attention_xfuser_unmask(
            q,
            k,
            v,
            heads,
            join_q=None,
            join_k=None,
            join_v=None,
            mask=None,
            attn_precision=None,
            skip_reshape=False,
            skip_output_reshape=False,
            *args,
            **kwargs):

        if skip_reshape:
            b, _, _, dim_head = q.shape
            if join_q is not None:
                j_b, _, _, j_dim_head = join_q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            if join_q is not None:
                j_b, _, j_dim_head = join_q.shape
                j_dim_head //= heads
                join_q, join_k, join_v = map(
                    lambda t: t.view(j_b, -1, heads, j_dim_head).transpose(1, 2),
                    (join_q, join_k, join_v),
                )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        # I am testing an issue with where uly=1 and ring>1 could cause vram leakage
        if force_ring_only:
            global _FORCE_RING_ATTENTION_CALL_COUNT
            _FORCE_RING_ATTENTION_CALL_COUNT += 1
            call_idx = _FORCE_RING_ATTENTION_CALL_COUNT
            ring_group = PROCESS_GROUP.RING_PG
            if ring_group is None:
                raise RuntimeError("Ring process group is not initialized")
            _log_force_ring_memory("before", call_idx)
            if join_q is not None:
                query = torch.cat([query, join_q.transpose(1, 2)], dim=1)
                out = xdit_ring_flash_attn_forward(
                    ring_group,
                    query,
                    key,
                    value,
                    None,
                    causal=False,
                    attn_type=attn,
                    joint_strategy="rear",
                    joint_tensor_key=join_k.transpose(1, 2),
                    joint_tensor_value=join_v.transpose(1, 2),
                )
            else:
                out = xdit_ring_flash_attn_forward(
                    ring_group,
                    query,
                    key,
                    value,
                    None,
                    causal=False,
                    attn_type=attn,
                )
            _log_force_ring_memory("after", call_idx)
            if type(out) == tuple:
                out = out[0]
            out = out.transpose(1, 2)
        # Check if using join attention, for MMDiT model
        elif join_q is not None:
            out = xfuser_attn(
                None,
                query,
                key,
                value,
                joint_strategy="rear",
                joint_tensor_query=join_q.transpose(1, 2),
                joint_tensor_key=join_k.transpose(1, 2),
                joint_tensor_value=join_v.transpose(1, 2),
            ).transpose(1, 2)
        else:
            out = xfuser_attn(
                None,
                query,
                key,
                value,
            ).transpose(1, 2)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out

    return _attention_xfuser_unmask
