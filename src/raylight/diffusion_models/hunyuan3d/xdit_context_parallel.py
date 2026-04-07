import math
import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
from ..utils import pad_to_world_size

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)

# HunyuanVid use the same DiT single/double block as Flux


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def usp_dit_forward(self, x, timestep, context, guidance=None, transformer_options={}, *args, **kwargs):
    x = x.movedim(-1, -2)
    timestep = 1.0 - timestep
    txt = context
    img = self.latent_in(x)

    vec = self.time_in(timestep_embedding(timestep, 256, self.max_period).to(dtype=img.dtype))
    if self.guidance_in is not None:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256, self.max_period).to(img.dtype))

    txt = self.cond_in(txt)
    pe = None
    attn_mask = None

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img, img_orig_size = pad_to_world_size(img, dim=1)
    txt, txt_orig_size = pad_to_world_size(txt, dim=1)
    sp_world = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    img = torch.chunk(img, sp_world, dim=1)[sp_rank]
    txt = torch.chunk(txt, sp_world, dim=1)[sp_rank]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img, dim=1)
    txt = get_sp_group().all_gather(txt, dim=1)
    img = img[:, :img_orig_size, ...]
    txt = txt[:, :txt_orig_size, ...]

    img = torch.cat((txt, img), 1)

    img, joint_orig_size = pad_to_world_size(img, dim=1)
    img = torch.chunk(img, sp_world, dim=1)[sp_rank]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("single_block", i)](
                {"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img, dim=1)
    img = img[:, :joint_orig_size, ...]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = img[:, txt_orig_size:, ...]
    img = self.final_layer(img, vec)
    return img.movedim(-2, -1) * (-1.0)
