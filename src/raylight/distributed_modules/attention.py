import os
import torch

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType, select_flash_attn_impl
from yunchang.globals import PROCESS_GROUP
from yunchang.ring.utils import RingComm, update_out_and_lse
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


def _raylight_ring_flash_attn_forward(
    process_group,
    q,
    k,
    v,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    attn_type=AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    is_joint = False
    if joint_tensor_key is not None and joint_tensor_value is not None:
        supported_joint_strategy = ["front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supported. supported joint strategy: {supported_joint_strategy}"
            )
        is_joint = True
    elif joint_tensor_key is None and joint_tensor_value is None:
        pass
    else:
        raise ValueError("joint_tensor_key and joint_tensor_value should be None or not None simultaneously.")

    comm = RingComm(process_group)
    out = None
    lse = None
    next_k, next_v = None, None
    if comm.world_size > 1:
        # Ping-pong receive buffers avoid RingComm allocating empty_like() on every hop.
        k_recv_buffers = [torch.empty_like(k), torch.empty_like(k)]
        v_recv_buffers = [torch.empty_like(v), torch.empty_like(v)]
    else:
        k_recv_buffers = None
        v_recv_buffers = None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            recv_idx = step % 2
            next_k = comm.send_recv(k, recv_tensor=k_recv_buffers[recv_idx])
            next_v = comm.send_recv(v, recv_tensor=v_recv_buffers[recv_idx])
            comm.commit()

        if is_joint and joint_strategy == "rear":
            if step + 1 == comm.world_size:
                key = torch.cat([k, joint_tensor_key], dim=1)
                value = torch.cat([v, joint_tensor_value], dim=1)
            else:
                key, value = k, v
        elif is_joint and joint_strategy == "front":
            if step == 0:
                key = torch.cat([joint_tensor_key, k], dim=1)
                value = torch.cat([joint_tensor_value, v], dim=1)
            else:
                key, value = k, v
        else:
            key, value = k, v

        if not causal or step <= comm.rank:
            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            if attn_type == AttnType.FA3:
                block_out, block_lse = fn(
                    q,
                    key,
                    value,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal and step == 0,
                    window_size=window_size,
                    softcap=0.0,
                    alibi_slopes=alibi_slopes,
                    return_softmax=True and dropout_p > 0,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                block_out, block_lse = fn(
                    q,
                    key,
                    value,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal and step == 0,
                    window_size=window_size,
                    softcap=0.0,
                    alibi_slopes=alibi_slopes,
                    return_softmax=True and dropout_p > 0,
                )
            if attn_type == AttnType.SPARSE_SAGE:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    if attn_type != AttnType.SPARSE_SAGE:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


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
                out = _raylight_ring_flash_attn_forward(
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
                out = _raylight_ring_flash_attn_forward(
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
