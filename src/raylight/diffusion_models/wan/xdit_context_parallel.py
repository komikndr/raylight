import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
import comfy
from ..utils import pad_to_world_size
attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculationk
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def pad_freqs(original_tensor, target_len):
    """
    original_tensor: [B, L_global, 1, D/2, 2, 2] — full freq tensor
    """
    b, seq_len, n, dim, f1, f2 = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    padding_tensor = torch.ones(
        b,
        pad_size,
        n,
        dim,
        f1,
        f2,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=1)
    return padded_tensor


# No need to this anymore in other model, just toch chunk the pe
def apply_rope_sp(xq, xk, freqs_cis):
    """
    xq, xk:       [B, L_local, 1, D]
    freqs_cis:    [B, L_global, 1, D/2, 2, 2] — full freq tensor
    Returns:      RoPE-applied local xq, xk
    """

    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()

    B, L_local, _, D = xq.shape
    L_global = L_local * sp_size

    # Ensure freqs_cis has length L_global
    freqs_cis = pad_freqs(freqs_cis, L_global)

    # Slice the correct frequency chunk for this rank
    start = sp_rank * L_local
    end = start + L_local
    freqs_local = freqs_cis[:, start:end]  # [B, L_local, 1, D/2, 2, 2]

    # Prepare xq/xk for RoPE (split real/imag)
    xq_ = xq.to(dtype=freqs_local.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_local.dtype).reshape(*xk.shape[:-1], -1, 1, 2)

    # Apply RoPE using local frequencies
    xq_out = freqs_local[..., 0] * xq_[..., 0] + freqs_local[..., 1] * xq_[..., 1]
    xk_out = freqs_local[..., 0] * xk_[..., 0] + freqs_local[..., 1] * xk_[..., 1]

    return xq_out.reshape_as(xq).type_as(xq), xk_out.reshape_as(xk).type_as(xk)


@torch.compiler.disable
def usp_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (Tensor):
            List of input video tensors with shape [B, C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [B, L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    # x = self.patch_embedding(x.to(next(self.patch_embedding.parameters()).dtype, copy=False)).to(x.dtype, copy=False)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype)
    )
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # head
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
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len
            )

    torch._dynamo.graph_break()
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    torch._dynamo.graph_break()

    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    return x


def usp_vace_dit_forward(
    self,
    x,
    t,
    context,
    vace_context,
    vace_strength,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    orig_shape = list(vace_context.shape)
    vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
    c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
    c = c.flatten(2).transpose(1, 2)
    c = list(c.split(orig_shape[0], dim=0))

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x, orig_size = pad_to_world_size(x, dim=1)
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x_orig = x

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

        ii = self.vace_layers_mapping.get(i, None)
        if ii is not None:
            for iii in range(len(c)):
                c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)
                x += c_skip * vace_strength[iii]
            del c_skip

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


def usp_camera_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    camera_conditions = None,
    transformer_options={},
    **kwargs,
):
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    if self.control_adapter is not None and camera_conditions is not None:
        x = x + self.control_adapter(camera_conditions).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

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

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


def usp_humo_dit_forward(
    self,
    x,
    t,
    context,
    freqs=None,
    audio_embed=None,
    reference_latent=None,
    transformer_options={},
    **kwargs,
):
    bs, _, time, height, width = x.shape

    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    if reference_latent is not None:
        ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
        ref = ref.flatten(2).transpose(1, 2)
        freqs_ref = self.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=time, device=x.device, dtype=x.dtype)
        x = torch.cat([x, ref], dim=1)
        freqs = torch.cat([freqs, freqs_ref], dim=1)
        del ref, freqs_ref

    # context
    context = self.text_embedding(context)
    context_img_len = None

    if audio_embed is not None:
        if reference_latent is not None:
            zero_audio_pad = torch.zeros(audio_embed.shape[0], reference_latent.shape[-3], *audio_embed.shape[2:], device=audio_embed.device, dtype=audio_embed.dtype)
            audio_embed = torch.cat([audio_embed, zero_audio_pad], dim=1)
        audio = self.audio_proj(audio_embed).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    else:
        audio = None

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
                out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, audio=audio, transformer_options=args["transformer_options"])
                return out
            out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, audio=audio, transformer_options=transformer_options)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


def usp_s2v_dit_forward(
    self,
    x,
    t,
    context,
    audio_embed=None,
    reference_latent=None,
    control_video=None,
    reference_motion=None,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    if audio_embed is not None:
        num_embeds = x.shape[-3] * 4
        audio_emb_global, audio_emb = self.casual_audio_encoder(audio_embed[:, :, :, :num_embeds])
    else:
        audio_emb = None

    # embeddings
    bs, _, time, height, width = x.shape
    x = self.patch_embedding(x.float()).to(x.dtype)
    if control_video is not None:
        x = x + self.cond_encoder(control_video)

    if t.ndim == 1:
        t = t.unsqueeze(1).repeat(1, x.shape[2])

    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)
    seq_len = x.size(1)

    cond_mask_weight = comfy.model_management.cast_to(self.trainable_cond_mask.weight, dtype=x.dtype, device=x.device).unsqueeze(1).unsqueeze(1)
    x = x + cond_mask_weight[0]

    if reference_latent is not None:
        ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
        ref = ref.flatten(2).transpose(1, 2)
        freqs_ref = self.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=max(30, time + 9), device=x.device, dtype=x.dtype)
        ref = ref + cond_mask_weight[1]
        x = torch.cat([x, ref], dim=1)
        freqs = torch.cat([freqs, freqs_ref], dim=1)
        t = torch.cat([t, torch.zeros((t.shape[0], reference_latent.shape[-3]), device=t.device, dtype=t.dtype)], dim=1)
        del ref, freqs_ref

    if reference_motion is not None:
        motion_encoded, freqs_motion = self.frame_packer(reference_motion, self)
        motion_encoded = motion_encoded + cond_mask_weight[2]
        x = torch.cat([x, motion_encoded], dim=1)
        freqs = torch.cat([freqs, freqs_motion], dim=1)

        t = torch.repeat_interleave(t, 2, dim=1)
        t = torch.cat([t, torch.zeros((t.shape[0], 3), device=t.device, dtype=t.dtype)], dim=1)
        del motion_encoded, freqs_motion

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    # context
    context = self.text_embedding(context)

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
                out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"])
                return out
            out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context)
        if audio_emb is not None:
            x = self.audio_injector(x, i, audio_emb, audio_emb_global, seq_len)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


def usp_self_attn_forward(self, x, freqs, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = apply_rope_sp(q, k, freqs)

    x = xfuser_optimized_attention(
        q.view(b, s, n * d),
        k.view(b, s, n * d),
        v,
        heads=self.num_heads,
    )
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_t2v_cross_attn_forward(self, x, context, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    # compute attention
    x = xfuser_optimized_attention(q, k, v, heads=self.num_heads)
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_i2v_cross_attn_forward(self, x, context, context_img_len, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    context_img = context[:, :context_img_len]
    context = context[:, context_img_len:]

    # compute query, key, value
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)
    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = xfuser_optimized_attention(q, k_img, v_img, heads=self.num_heads)
    x = xfuser_optimized_attention(q, k, v, heads=self.num_heads)
    x = x + img_x
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_t2v_cross_attn_gather_forward(self, x, context, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C] - video tokens
        context(Tensor): Shape [B, L2, C] - audio tokens with shape [B, frames*16, 1536]
    """
    b, n, d = x.size(0), self.num_heads, self.head_dim

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    # Handle audio temporal structure (16 tokens per frame)
    k = k.reshape(-1, 16, n, d).transpose(1, 2)
    v = v.reshape(-1, 16, n, d).transpose(1, 2)

    # Handle video spatial structure
    q = q.reshape(k.shape[0], -1, n, d).transpose(1, 2)

    x = xfuser_optimized_attention(q, k, v, heads=self.num_heads, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)

    x = x.transpose(1, 2).reshape(b, -1, n * d)
    x = self.o(x)
    return x
