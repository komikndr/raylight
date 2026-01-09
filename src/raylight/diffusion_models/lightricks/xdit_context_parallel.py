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


# A better solution is just to type.MethodType the x, context, and pe construction,
def pad_group_to_world_size(group, dim):
    """
    group: Tensor | list[Tensor] | tuple[Tensor]
    Returns: padded_group, orig_sizes
    """

    if torch.is_tensor(group):
        orig = group.size(dim)
        group, _ = pad_to_world_size(group, dim=dim)
        return group, orig

    elif isinstance(group, (list, tuple)):
        padded = []
        origs = []
        for g in group:
            o = g.size(dim)
            g, _ = pad_to_world_size(g, dim=dim)
            padded.append(g)
            origs.append(o)
        return type(group)(padded), origs

    else:
        return group, None


def pad_and_split_pe(pe, dim, sp_world_size, sp_rank):
    out = []

    for group in pe:              # pe[i]
        new_group = []

        for cos, sin, flag in group:  # pe[i][j]
            # Pad
            cos, _ = pad_to_world_size(cos, dim=dim)
            sin, _ = pad_to_world_size(sin, dim=dim)

            # Split (sequence parallel)
            cos = torch.chunk(cos, sp_world_size, dim=dim)[sp_rank]
            sin = torch.chunk(sin, sp_world_size, dim=dim)[sp_rank]

            new_group.append((cos, sin, flag))

        out.append(tuple(new_group))

    return out


def sp_chunk_group(group, sp_world_size, sp_rank, dim):
    if torch.is_tensor(group):
        return torch.chunk(group, sp_world_size, dim=dim)[sp_rank]

    elif isinstance(group, (list, tuple)):
        return type(group)(
            torch.chunk(g, sp_world_size, dim=dim)[sp_rank]
            for g in group
        )
    else:
        return group


def sp_gather_group(group, orig_sizes, dim):
    if torch.is_tensor(group):
        g = get_sp_group().all_gather(group.contiguous(), dim=dim)
        return g.narrow(dim, 0, orig_sizes)

    elif isinstance(group, (list, tuple)):
        out = []
        for g, o in zip(group, orig_sizes):
            g = get_sp_group().all_gather(g.contiguous(), dim=dim)
            g = g.narrow(dim, 0, o)
            out.append(g)
        return type(group)(out)

    else:
        return group


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
    x, x_orig = pad_group_to_world_size(x, dim=1)
    context, _ = pad_group_to_world_size(context, dim=1)
    pe = pad_and_split_pe(pe, dim=2, sp_world_size=sp_world_size, sp_rank=sp_rank)

    x = sp_chunk_group(x, sp_world_size, sp_rank, dim=1)
    context = sp_chunk_group(context, sp_world_size, sp_rank, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # Process transformer blocks
    x = self._process_transformer_blocks(
        x, context, attention_mask, timestep, pe, transformer_options=transformer_options, **merged_args
    )

    x = sp_gather_group(x, x_orig, dim=1)

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
