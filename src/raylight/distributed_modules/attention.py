import functools

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel
from .inner_attention import INNER_ATTENTION_KEY, InnerAttentionDispatcher

_ATTN_TYPE = None
_SYNC_ULYSSES = None


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


def _patch_ring_lse_padding():
    """Trim the log-sum-exp the torch attention backends return.

    `_scaled_dot_product_efficient_attention` pads its log-sum-exp out to a 32
    element boundary, and yunchang hands it back unsliced, so ring attention
    combines an S row output with a ceil32(S) row lse and raises

        The size of tensor a (14880) must match the size of tensor b (14857)

    for any sequence that is not a multiple of 32. Ulysses never reaches this
    path, which is why it only appears once ring_degree > 1.
    """
    try:
        from yunchang import kernels
    except ImportError:
        return
    if getattr(kernels, "_raylight_lse_trimmed", False):
        return
    original = kernels.pytorch_attn_forward

    @functools.wraps(original)
    def trimmed(*args, **kwargs):
        out, lse = original(*args, **kwargs)[:2]
        seq = out.shape[1]
        if lse.shape[-1] > seq:
            lse = lse[..., :seq]
        return out, lse

    kernels.pytorch_attn_forward = trimmed
    kernels._raylight_lse_trimmed = True


def make_xfuser_attention(attn_type, sync_ulysses):
    print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
    _patch_ring_lse_padding()
    attn = AttnType[attn_type]
    if attn_type == "SAGE_FP8_CUDA":
        ensure_hf_fp8_cuda_kernel()
    elif attn_type == "SAGE_FP8_SM90":
        ensure_hf_sm90_kernel

    xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn)
    inner_dispatcher = InnerAttentionDispatcher(xfuser_attn)

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

        # For custom inner attentnion, such as SLA, maybe SOL
        transformer_options = kwargs.get("transformer_options", {})
        processor = transformer_options.get(INNER_ATTENTION_KEY)
        with inner_dispatcher.scope(processor, transformer_options):
            if join_q is not None:
                out = xfuser_attn(
                    None, query, key, value,
                    joint_strategy="rear",
                    joint_tensor_query=join_q.transpose(1, 2),
                    joint_tensor_key=join_k.transpose(1, 2),
                    joint_tensor_value=join_v.transpose(1, 2),
                    softmax_scale=kwargs.get("scale", None),
                ).transpose(1, 2)
            else:
                out = xfuser_attn(
                    None,
                    query,
                    key,
                    value,
                    softmax_scale=kwargs.get("scale", None),
                ).transpose(1, 2)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out

    return _attention_xfuser_unmask
