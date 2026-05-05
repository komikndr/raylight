import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
import comfy
from comfy.ldm.flux.math import apply_rope1
from ..utils import pad_to_world_size

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _pad_and_split_for_sp(tensor, dim=1):
    if tensor is None:
        return None, None
    tensor, orig_size = pad_to_world_size(tensor, dim=dim)
    tensor = torch.chunk(tensor, get_sequence_parallel_world_size(), dim=dim)[get_sequence_parallel_rank()]
    return tensor, orig_size


def _sp_chunk_len(size):
    return (size + get_sequence_parallel_world_size() - 1) // get_sequence_parallel_world_size()


def _sync_ar_kv_cache_for_sampler(cache):
    if cache is None or "sp_full_end" not in cache:
        return

    full_end = cache["end"]
    sp_full_end = cache["sp_full_end"]
    if full_end >= sp_full_end:
        return

    full_delta = sp_full_end - full_end
    cache["sp_local_end"] = max(0, cache["sp_local_end"] - _sp_chunk_len(full_delta))
    cache["sp_full_end"] = full_end


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def usp_causal_ar_forward(
    self,
    x,
    timestep,
    context,
    clip_fea=None,
    time_dim_concat=None,
    transformer_options={},
    **kwargs,
):
    ar_state = transformer_options.get("ar_state")
    if ar_state is None:
        from comfy.ldm.wan.model import WanModel

        return WanModel.forward(
            self,
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=transformer_options,
            **kwargs,
        )

    bs = x.shape[0]
    block_frames = x.shape[2]
    t_per_frame = timestep.unsqueeze(1).expand(bs, block_frames)

    return self.forward_block(
        x=x,
        timestep=t_per_frame,
        context=context,
        start_frame=ar_state["start_frame"],
        kv_caches=ar_state["kv_caches"],
        crossattn_caches=ar_state["crossattn_caches"],
        clip_fea=clip_fea,
        transformer_options=transformer_options,
    )


def usp_causal_ar_forward_block(
    self,
    x,
    timestep,
    context,
    start_frame,
    kv_caches,
    crossattn_caches,
    clip_fea=None,
    transformer_options={},
):
    x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
    bs, c, t, h, w = x.shape

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)
    orig_size = x.shape[1]
    transformer_options["usp_ar_block_seq_len"] = orig_size

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(dtype=x.dtype))
    e = e.reshape(timestep.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None and self.img_emb is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    freqs = self.rope_encode(t, h, w, t_start=start_frame, device=x.device, dtype=x.dtype)

    x, _ = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    kv_cache=kv_caches[i],
                    crossattn_cache=crossattn_caches[i],
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x,
                e=e0,
                freqs=freqs,
                context=context,
                context_img_len=context_img_len,
                kv_cache=kv_caches[i],
                crossattn_cache=crossattn_caches[i],
                transformer_options=transformer_options,
            )

    x = get_sp_group().all_gather(x.contiguous(), dim=1)
    x = x[:, :orig_size, :]

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return x[:, :, :t, :h, :w]


def usp_causal_ar_self_attn_forward(self, x, freqs, kv_cache=None, transformer_options={}):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
    k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
    v = self.v(x).view(b, s, n, d)

    if kv_cache is None:
        x = xfuser_optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            v.view(b, s, n * d),
            heads=self.num_heads,
        )
        return self.o(x.flatten(2))

    _sync_ar_kv_cache_for_sampler(kv_cache)

    full_end = kv_cache["end"]
    local_end = kv_cache.get("sp_local_end", _sp_chunk_len(full_end))
    full_block_seq_len = transformer_options["usp_ar_block_seq_len"]
    local_new_end = local_end + s
    full_new_end = full_end + full_block_seq_len

    kv_cache["k"][:, local_end:local_new_end] = k
    kv_cache["v"][:, local_end:local_new_end] = v
    kv_cache["sp_local_end"] = local_new_end
    kv_cache["sp_full_end"] = full_new_end
    kv_cache["end"] = full_new_end

    x = xfuser_optimized_attention(
        q.view(b, s, n * d),
        kv_cache["k"][:, :local_new_end].view(b, local_new_end, n * d),
        kv_cache["v"][:, :local_new_end].view(b, local_new_end, n * d),
        heads=self.num_heads,
    )
    return self.o(x.flatten(2))


def usp_causal_ar_block_forward(
    self,
    x,
    e,
    freqs,
    context,
    context_img_len=257,
    kv_cache=None,
    crossattn_cache=None,
    transformer_options={},
):
    from comfy.ldm.wan.model import repeat_e

    if e.ndim < 4:
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
    else:
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

    y = self.self_attn(
        torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
        freqs,
        kv_cache=kv_cache,
        transformer_options=transformer_options,
    )
    x = torch.addcmul(x, y, repeat_e(e[2], x))
    del y

    if crossattn_cache is not None and crossattn_cache.get("is_init"):
        q = self.cross_attn.norm_q(self.cross_attn.q(self.norm3(x)))
        x_ca = xfuser_optimized_attention(q, crossattn_cache["k"], crossattn_cache["v"], heads=self.num_heads)
        x = x + self.cross_attn.o(x_ca.flatten(2))
    else:
        x = x + self.cross_attn(
            self.norm3(x),
            context,
            context_img_len=context_img_len,
            transformer_options=transformer_options,
        )
        if crossattn_cache is not None:
            crossattn_cache["k"] = self.cross_attn.norm_k(self.cross_attn.k(context))
            crossattn_cache["v"] = self.cross_attn.v(context)
            crossattn_cache["is_init"] = True

    y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
    x = torch.addcmul(x, y, repeat_e(e[5], x))
    return x


@torch.compiler.disable
def usp_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    *args,
    **kwargs,
):
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    sp_world = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
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

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

    torch._dynamo.graph_break()
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x.contiguous(), dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    torch._dynamo.graph_break()

    x = self.head(x, e)
    if full_ref is not None:
        x = x[:, full_ref.shape[1] :]

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
    *args,
    **kwargs,
):
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
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
    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    c = [_pad_and_split_for_sp(item, dim=1)[0] for item in c]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x_orig = x

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

        ii = self.vace_layers_mapping.get(i, None)
        if ii is not None:
            for iii in range(len(c)):
                c_skip, c[iii] = self.vace_blocks[ii](
                    c[iii],
                    x=x_orig,
                    e=e0,
                    freqs=freqs,
                    context=context,
                    context_img_len=context_img_len,
                    transformer_options=transformer_options,
                )
                x += c_skip * vace_strength[iii]
            del c_skip

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = self.head(x, e)

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
    camera_conditions=None,
    transformer_options={},
    *args,
    **kwargs,
):
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    if self.control_adapter is not None and camera_conditions is not None:
        x = x + self.control_adapter(camera_conditions).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
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
    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # head
    x = self.head(x, e)

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
    *args,
    **kwargs,
):
    bs, _, time, height, width = x.shape

    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    if reference_latent is not None:
        ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
        ref = ref.flatten(2).transpose(1, 2)
        freqs_ref = self.rope_encode(
            reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=time, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, ref], dim=1)
        freqs = torch.cat([freqs, freqs_ref], dim=1)
        del ref, freqs_ref

    # context
    context = self.text_embedding(context)
    context_img_len = None

    if audio_embed is not None:
        if reference_latent is not None:
            zero_audio_pad = torch.zeros(
                audio_embed.shape[0], reference_latent.shape[-3], *audio_embed.shape[2:], device=audio_embed.device, dtype=audio_embed.dtype
            )
            audio_embed = torch.cat([audio_embed, zero_audio_pad], dim=1)
        audio = self.audio_proj(audio_embed).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    else:
        audio = None

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    audio=audio,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, audio=audio, transformer_options=transformer_options
            )

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
    *args,
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

    cond_mask_weight = (
        comfy.model_management.cast_to(self.trainable_cond_mask.weight, dtype=x.dtype, device=x.device).unsqueeze(1).unsqueeze(1)
    )
    x = x + cond_mask_weight[0]

    if reference_latent is not None:
        ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
        ref = ref.flatten(2).transpose(1, 2)
        freqs_ref = self.rope_encode(
            reference_latent.shape[-3],
            reference_latent.shape[-2],
            reference_latent.shape[-1],
            t_start=max(30, time + 9),
            device=x.device,
            dtype=x.dtype,
        )
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
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    # context
    context = self.text_embedding(context)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], transformer_options=args["transformer_options"]
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, transformer_options=transformer_options)
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


def usp_self_attn_forward(self, x, freqs, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    patches = transformer_options.get("patches", {})

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    def qkv_fn_q(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        return apply_rope1(q, freqs)

    def qkv_fn_k(x):
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        return apply_rope1(k, freqs)

    q = qkv_fn_q(x)
    k = qkv_fn_k(x)

    x = xfuser_optimized_attention(
        q.view(b, s, n * d),
        k.view(b, s, n * d),
        self.v(x).view(b, s, n * d),
        heads=self.num_heads,
    )
    x = x.flatten(2)

    if "attn1_patch" in patches:
        for p in patches["attn1_patch"]:
            x = p({"x": x, "q": q, "k": k, "transformer_options": transformer_options})

    x = self.o(x)
    return x


def usp_t2v_cross_attn_forward(self, x, context, transformer_options={}, **kwargs):
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


def usp_i2v_cross_attn_forward(self, x, context, context_img_len, transformer_options={}, **kwargs):
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

    x = xfuser_optimized_attention(q, k, v, heads=self.num_heads, skip_reshape=True, skip_output_reshape=True)

    x = x.transpose(1, 2).reshape(b, -1, n * d)
    x = x.flatten(2)
    x = self.o(x)
    return x


@torch.compiler.disable
def usp_scail_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    pose_latents=None,
    reference_latent=None,
    transformer_options={},
    *args,
    **kwargs,
):
    if reference_latent is not None:
        x = torch.cat((reference_latent, x), dim=2)

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    x = x.flatten(2).transpose(1, 2)

    scail_pose_seq_len = 0
    if pose_latents is not None:
        scail_x = self.patch_embedding_pose(pose_latents.float()).to(x.dtype)
        scail_x = scail_x.flatten(2).transpose(1, 2)
        scail_pose_seq_len = scail_x.shape[1]
        x = torch.cat([x, scail_x], dim=1)
        del scail_x

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.cat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x,
                e=e0,
                freqs=freqs,
                context=context,
                context_img_len=context_img_len,
                transformer_options=transformer_options,
            )

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x.contiguous(), dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    x = self.head(x, e)

    if scail_pose_seq_len > 0:
        x = x[:, :-scail_pose_seq_len]

    x = self.unpatchify(x, grid_sizes)

    if reference_latent is not None:
        x = x[:, :, reference_latent.shape[2] :]

    return x


@torch.compiler.disable
def usp_multitalk_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    *args,
    **kwargs,
):
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    sp_world = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    x, orig_size = _pad_and_split_for_sp(x, dim=1)
    freqs, _ = _pad_and_split_for_sp(freqs, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.cat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x,
                e=e0,
                freqs=freqs,
                context=context,
                context_img_len=context_img_len,
                transformer_options=transformer_options,
            )

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x.contiguous(), dim=1)
    x = x[:, :orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return x
