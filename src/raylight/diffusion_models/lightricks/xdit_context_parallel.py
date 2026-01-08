import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
import comfy
from comfy.ldm.lightricks.model import apply_rotary_emb
from ..utils import pad_to_world_size
attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_dit_forward(
    self,
    x,
    timestep,
    context,
    attention_mask,
    frame_rate=25,
    transformer_options={},
    keyframe_idxs=None,
    denoise_mask=None,
    **kwargs
):
    """
    Internal forward pass for LTX models.

    Args:
        x: Input tensor
        timestep: Timestep tensor
        context: Context tensor (e.g., text embeddings)
        attention_mask: Attention mask tensor
        frame_rate: Frame rate for temporal processing
        transformer_options: Additional options for transformer blocks
        keyframe_idxs: Keyframe indices for temporal processing
        **kwargs: Additional keyword arguments

    Returns:
        Processed output tensor
    """
    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if isinstance(x, list):
        input_dtype = x[0].dtype
        batch_size = x[0].shape[0]
    else:
        input_dtype = x.dtype
        batch_size = x.shape[0]
    # Process input
    merged_args = {**transformer_options, **kwargs}
    x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs, denoise_mask, **merged_args)
    merged_args.update(additional_args)

    # Prepare timestep and context
    timestep, embedded_timestep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
    context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)

    # Prepare attention mask and positional embeddings
    attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
    pe = self._prepare_positional_embeddings(pixel_coords, frame_rate, input_dtype)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x, x_orig_size = pad_to_world_size(x, dim=1)
    context, _ = pad_to_world_size(context, dim=1)
    pe, _ = pad_to_world_size(pe, dim=1)

    x = torch.chunk(x, sp_world_size, dim=1)[sp_rank]
    context = torch.chunk(context, sp_world_size, dim=1)[sp_rank]
    pe = torch.chunk(pe, sp_world_size, dim=2)[sp_rank]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # Process transformer blocks
    x = self._process_transformer_blocks(
        x, context, attention_mask, timestep, pe, transformer_options=transformer_options, **merged_args
    )

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x.contiguous(), dim=1)
    x = x[:, :x_orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # Process output
    x = self._process_output(x, embedded_timestep, keyframe_idxs, **merged_args)
    return x


def usp_cross_attn_forward(
        self,
        x,
        context=None,
        mask=None,
        pe=None,
        k_pe=None,
        transformer_options={}
):
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q = self.q_norm(q)
    k = self.k_norm(k)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    out = xfuser_optimized_attention(q, k, v, self.heads)
    out = xfuser_optimized_attention(q, k, v, self.heads)
    return self.to_out(out)
