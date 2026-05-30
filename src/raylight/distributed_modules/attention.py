import torch

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)
from xfuser.core.long_ctx_attention.ring import xdit_ring_flash_attn_func

from yunchang.kernels import AttnType
from yunchang.globals import PROCESS_GROUP
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

_ATTN_TYPE = None
_SYNC_ULYSSES = None
_FORCE_RING_ONLY = False


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
            ring_group = PROCESS_GROUP.RING_PG
            if ring_group is None:
                raise RuntimeError("Ring process group is not initialized")
            if join_q is not None:
                query = torch.cat([query, join_q.transpose(1, 2)], dim=1)
                out = xdit_ring_flash_attn_func(
                    query,
                    key,
                    value,
                    group=ring_group,
                    attn_type=attn,
                    joint_strategy="rear",
                    joint_tensor_key=join_k.transpose(1, 2),
                    joint_tensor_value=join_v.transpose(1, 2),
                )
            else:
                out = xdit_ring_flash_attn_func(
                    query,
                    key,
                    value,
                    group=ring_group,
                    attn_type=attn,
                )
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
