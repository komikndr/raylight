import math

import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from yunchang.comm.all_to_all import SeqAllToAll4D
from ..utils import joint_rope_shards, pad_to_world_size, shard_ranges
from comfy.ldm.flux import layers as flux_layers
from comfy.ldm.flux.math import apply_rope, attention as native_attention
from comfy.ldm.flux.layers import apply_mod, modulated_norm
from comfy.ldm.flux.model import invert_slices
import raylight.distributed_modules.attention as xfuser_attn
attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


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


def _checkpoint_attention(q, k, v, mask, options, preferred_attention):
    lengths = options.get("raylight_key_lengths")
    if lengths is None:
        raise ValueError("Checkpoint-selected attention requires FLUX sequence shard lengths")
    group = get_sp_group()
    world = group.world_size
    if world == 1:
        return native_attention(q, k, v, None, mask=mask, transformer_options=options, preferred_attention=preferred_attention)

    # Gather tokens only for the heads owned by this rank. SOL routes token
    # blocks, so joining rank-local [text, image] pairs in rank order is wrong.
    widths = [(length + world - 1) // world for length in lengths]
    heads = q.shape[1]
    head_pad = -heads % world
    local_heads = (heads + head_pad) // world

    def gather_tokens(t):
        t = t.transpose(1, 2)
        if head_pad:
            t = torch.nn.functional.pad(t, (0, 0, 0, head_pad))
        t = SeqAllToAll4D.apply(group.device_group, t, 2, 1, sync_ulysses)
        if len(lengths) == 1:
            return t[:, :lengths[0]].transpose(1, 2)
        streams = t.unflatten(1, (world, sum(widths))).split(widths, dim=2)
        return torch.cat([s.flatten(1, 2)[:, :n] for s, n in zip(streams, lengths)], dim=1).transpose(1, 2)

    if mask is not None and mask.ndim == 4 and mask.shape[1] != 1:
        if mask.shape[1] != heads:
            raise ValueError("FLUX checkpoint attention mask must cover one or all heads")
        if head_pad:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, head_pad))
        mask = mask.narrow(1, group.rank_in_group * local_heads, local_heads)

    out = native_attention(gather_tokens(q), gather_tokens(k), gather_tokens(v), None,
                           mask=mask, transformer_options=options, preferred_attention=preferred_attention)
    parts = []
    for stream, length, width in zip(out.split(lengths, dim=1), lengths, widths):
        if width * world != length:
            stream = torch.nn.functional.pad(stream, (0, 0, 0, width * world - length))
        parts.append(stream.unflatten(1, (world, width)))
    out = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
    out = out.flatten(1, 2).unflatten(-1, (local_heads, v.shape[-1]))
    out = SeqAllToAll4D.apply(group.device_group, out, 1, 2, sync_ulysses)
    return out[:, :, :heads].flatten(2)


def attention(q, k, v, pe, mask=None, transformer_options={}, preferred_attention=None) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)

    if preferred_attention is not None and preferred_attention.function is not None:
        return _checkpoint_attention(q, k, v, mask, transformer_options, preferred_attention)
    heads = q.shape[1]
    x = xfuser_optimized_attention(q, k, v, heads, skip_reshape=True, key_lengths=transformer_options.get("raylight_key_lengths"))
    return x


def usp_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    timestep_zero_index=None,
    transformer_options={},
    attn_mask: Tensor = None,
    *args,
    **kwargs,
) -> Tensor:
    transformer_options = transformer_options.copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.txt_norm is not None:
        txt = self.txt_norm(txt)
    txt = self.txt_in(txt)

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    vec_orig = vec
    img, img_orig_size = pad_to_world_size(img, dim=1)
    txt, txt_orig_size = pad_to_world_size(txt, dim=1)
    txt_vec = vec
    modulation_dims = None
    extra_kwargs = {}
    if timestep_zero_index is not None:
        modulation_dims = []
        batch = vec.shape[0] // 2
        vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
        invert = invert_slices(timestep_zero_index, img_orig_size)
        for start, end in invert:
            modulation_dims.append((start, end, 0))
        for start, end in timestep_zero_index:
            modulation_dims.append((start, end, 1))
        txt_vec = vec[:batch]
    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

    # ===================== SP SPLIT ====================== #
    if img_ids is not None:
        img_ids, _ = pad_to_world_size(img_ids, dim=1)
        txt_ids, _ = pad_to_world_size(txt_ids, dim=1)
        pe_image, pe_combine = joint_rope_shards(self.pe_embedder, txt_ids, img_ids, txt_orig_size, img_orig_size)
    else:
        pe_combine = None
        pe_image = None

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    txt = torch.chunk(txt, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if modulation_dims is not None:
        extra_kwargs["modulation_dims_img"] = shard_ranges(modulation_dims, img.shape[1])

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    transformer_options["raylight_key_lengths"] = (txt_orig_size, img_orig_size)
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               transformer_options=args.get("transformer_options"),
                                               **extra_kwargs)
                return out

            out = blocks_replace[("double_block", i)]({
                "img": img,
                "txt": txt, "vec": vec,
                "pe": pe_image,
                "attn_mask": attn_mask,
                "transformer_options": transformer_options}, {
                "original_block": block_wrap}
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe_image,
                             attn_mask=attn_mask,
                             transformer_options=transformer_options,
                             **extra_kwargs)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    start = img.shape[1] * get_sequence_parallel_rank()
                    add = add[:, start:start + img.shape[1]]
                    img[:, :add.shape[1]] += add
    # ===================== SP GATHER ===================== #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    txt = get_sp_group().all_gather(txt.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    txt = txt[:, :txt_orig_size, :]

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)
    # ===================== SP SPLIT ====================== #
    img, img_orig_size = pad_to_world_size(img, dim=1)

    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)
    extra_kwargs = {}
    if timestep_zero_index is not None:
        assert modulation_dims is not None
        extra_kwargs["modulation_dims"] = list(
            map(lambda x: (0 if x[0] == 0 else x[0] + txt.shape[1], x[1] + txt.shape[1], x[2]), modulation_dims)
        )
    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    if timestep_zero_index is not None:
        extra_kwargs["modulation_dims"] = shard_ranges(extra_kwargs["modulation_dims"], img.shape[1])
    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["raylight_key_lengths"] = (img_orig_size,)
    transformer_options["img_slice"] = [max(0, min(img.shape[1], txt.shape[1] - get_sequence_parallel_rank() * img.shape[1])), img.shape[1]]
    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args.get("transformer_options"),
                                   **extra_kwargs)
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe_combine,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe_combine, attn_mask=attn_mask, transformer_options=transformer_options, **extra_kwargs)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    # Embed control signal in full concatenated [txt, img] space, then pad+chunk
                    sp_ws = get_sequence_parallel_world_size()
                    sp_r = get_sequence_parallel_rank()
                    txt_len = txt.shape[1]
                    full_add = torch.zeros(img.shape[0], img_orig_size, add.shape[-1],
                                           device=add.device, dtype=add.dtype)
                    full_add[:, txt_len:txt_len + add.shape[1]] = add
                    full_add, _ = pad_to_world_size(full_add, dim=1)
                    img += torch.chunk(full_add, sp_ws, dim=1)[sp_r]

    # ===================== SP GATHER ===================== #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    img = img[:, txt.shape[1]:, ...]

    extra_kwargs = {}
    if timestep_zero_index is not None:
        extra_kwargs["modulation_dims"] = modulation_dims

    img = self.final_layer(img, vec_orig, **extra_kwargs)  # (N, T, patch_size ** 2 * out_channels)
    return img


def usp_single_stream_forward(
    self,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims=None,
    transformer_options={},
    **kwargs
) -> Tensor:
    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec

    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()

    chunked = transformer_options.get("chunk_tokens") and flux_layers.chunk_tokens(x, transformer_options, modulation_dims)
    if chunked:
        chunk = flux_layers.CHUNK_TOKENS
        qkv = x.new_empty((*x.shape[:2], 3 * self.hidden_size))
        mlp = x.new_empty((*x.shape[:2], self.mlp_hidden_dim))
        for i, part in enumerate(self.qkv_chunks(x, mod, mlp, chunk)):
            qkv[:, i * chunk:(i + 1) * chunk] = part
            del part
    else:
        qkv, mlp = torch.split(self.linear1(modulated_norm(x, self.pre_norm, mod.scale, mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del qkv
    if chunked and pe is not None and "attn1_patch" not in transformer_patches:
        self.norm.norm_rope_(q, k, pe)
        pe = None
    else:
        q, k = self.norm(q, k, v)

    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        for p in patch:
            out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
            q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

    # compute attention
    attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
    del q, k, v

    if "attn1_output_patch" in transformer_patches:
        patch = transformer_patches["attn1_output_patch"]
        for p in patch:
            attn = p(attn, extra_options)

    # compute activation in mlp stream, cat again and run second linear layer
    if chunked:
        for s in range(0, x.shape[1], chunk):
            rows = slice(s, s + chunk)
            projected = self.linear2(torch.cat((attn[:, rows], mlp[:, rows]), dim=2))
            x[:, rows] += apply_mod(projected, mod.gate)
    else:
        if self.yak_mlp:
            mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
        else:
            mlp = self.mlp_act(mlp)
        output = self.linear2(torch.cat((attn, mlp), 2))
        x += apply_mod(output, mod.gate, None, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def usp_double_stream_forward(
    self,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    transformer_options={},
    **kwargs
):
    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()

    chunked = transformer_options.get("chunk_tokens") and flux_layers.chunk_tokens(img, transformer_options, modulation_dims_img)
    chunk_qkv = chunked and pe is not None and "attn1_patch" not in transformer_patches
    if chunked:
        chunk = flux_layers.CHUNK_TOKENS
    if not chunk_qkv:
        img_modulated = modulated_norm(img, self.img_norm1, img_mod1.scale, img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = modulated_norm(txt, self.txt_norm1, txt_mod1.scale, txt_mod1.shift, modulation_dims_txt)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    del txt_modulated
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del txt_qkv

    if chunk_qkv:
        n_txt = txt.shape[1]
        qkv = img.new_empty((img.shape[0], n_txt + img.shape[1], 3 * self.hidden_size))
        q, k, v = qkv.view(img.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q[:, :, :n_txt], k[:, :, :n_txt], v[:, :, :n_txt] = txt_q, txt_k, txt_v
        del txt_q, txt_k, txt_v
        for s in range(0, img.shape[1], chunk):
            qkv[:, n_txt + s:n_txt + s + chunk] = self.img_attn.qkv(modulated_norm(img[:, s:s + chunk], self.img_norm1, img_mod1.scale, img_mod1.shift))
        self.txt_attn.norm.norm_rope_(q[:, :, :n_txt], k[:, :, :n_txt], pe[:, :, :n_txt])
        self.img_attn.norm.norm_rope_(q[:, :, n_txt:], k[:, :, n_txt:], pe[:, :, n_txt:])
        pe = None
        del qkv
    else:
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v
    extra_options["img_slice"] = [txt.shape[1], q.shape[2]]
    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        for p in patch:
            out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
            q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

    # run actual attention
    attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
    del q, k, v

    if "attn1_output_patch" in transformer_patches:
        extra_options["img_slice"] = [txt.shape[1], attn.shape[1]]
        patch = transformer_patches["attn1_output_patch"]
        for p in patch:
            attn = p(attn, extra_options)

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img += apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    del img_attn
    if chunked:
        for s in range(0, img.shape[1], chunk):
            rows = slice(s, s + chunk)
            img[:, rows] += apply_mod(self.img_mlp(modulated_norm(img[:, rows], self.img_norm2, img_mod2.scale, img_mod2.shift)), img_mod2.gate)
    else:
        img += apply_mod(self.img_mlp(modulated_norm(img, self.img_norm2, img_mod2.scale, img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

    # calculate the txt bloks
    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    del txt_attn
    txt += apply_mod(self.txt_mlp(modulated_norm(txt, self.txt_norm2, txt_mod2.scale, txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt
