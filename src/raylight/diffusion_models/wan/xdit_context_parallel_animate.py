import torch
from typing import Optional
from einops import rearrange
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
from .xdit_context_parallel import sinusoidal_embedding_1d
from ..utils import pad_to_world_size

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_face_block_forward(
    self,
    x: torch.Tensor,
    motion_vec: torch.Tensor,
    motion_mask: Optional[torch.Tensor] = None,
    # use_context_parallel=False,
) -> torch.Tensor:

    B, T, N, C = motion_vec.shape
    T_comp = T

    x_motion = self.pre_norm_motion(motion_vec)
    x_feat = self.pre_norm_feat(x)

    kv = self.linear1_kv(x_motion)
    q = self.linear1_q(x_feat)

    k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
    q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)

    # Apply QK-Norm if needed.
    q = self.q_norm(q).to(v)
    k = self.k_norm(k).to(v)

    k = rearrange(k, "B L N H D -> (B L) N H D")
    v = rearrange(v, "B L N H D -> (B L) N H D")

    q = rearrange(q, "B (L S) H D -> (B L) S (H D)", L=T_comp)

    attn = xfuser_optimized_attention(q, k, v, heads=self.heads_num)

    attn = rearrange(attn, "(B L) S C -> B (L S) C", L=T_comp)

    output = self.linear2(attn)

    if motion_mask is not None:
        output = output * rearrange(motion_mask, "B T H W -> B (T H W)").unsqueeze(-1)

    return output


def usp_animate_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    pose_latents=None,
    face_pixel_values=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    full_ref = None
    if self.ref_conv is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x, orig_size = pad_to_world_size(x, dim=1)
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                return out
            out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

        if i % 5 == 0 and motion_vec is not None:
            x = x + self.face_adapter.fuser_blocks[i // 5](x, motion_vec)

    # head
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = self.head(x, e)

    # Context Parallel

    if full_ref is not None:
        x = x[:, full_ref.shape[1]:]

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x
