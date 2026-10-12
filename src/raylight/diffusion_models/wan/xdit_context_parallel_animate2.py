import torch

from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

import comfy
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention

from .xdit_context_parallel import sinusoidal_embedding_1d
from ..utils import pad_to_world_size


def _pad_and_split_for_sp(tensor, dim=1):
    if tensor is None:
        return None, None
    tensor, original_size = pad_to_world_size(tensor, dim=dim)
    return torch.chunk(tensor, get_sequence_parallel_world_size(), dim=dim)[get_sequence_parallel_rank()], original_size


def _gather(tensor):
    return get_sp_group().all_gather(tensor.contiguous(), dim=1)


def _local_frame_slice(frame, hw, local_start, local_end):
    start = max(frame * hw, local_start) - local_start
    end = min((frame + 1) * hw, local_end) - local_start
    if end <= start:
        return None
    return start, end


def usp_animate2_self_attn_forward_pose(self, x, freqs, transformer_options={}, **kwargs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
    k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
    v = self.v(x).view(b, s, n, d)
    out = optimized_attention(q.reshape(b, s, n * d), _gather(k).reshape(b, -1, n * d), _gather(v).reshape(b, -1, n * d), heads=n, transformer_options=transformer_options)
    for p in transformer_options.get("patches", {}).get("attn1_patch", []):
        out = p({"x": out, "q": q, "k": k, "transformer_options": transformer_options})
    return self.o(out), k, v


def usp_animate2_kv_from_input(self, x, freqs, transformer_options={}, **kwargs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
    return k, self.v(x).view(b, s, n, d)


def usp_animate2_self_attn_forward_gen(
    self, x, freqs, k_pose, v_pose, f_gen, hw, buffers, ref_strength=1.0, transformer_options={},
    **kwargs,
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
    k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
    v = self.v(x).view(b, s, n, d)

    world = get_sequence_parallel_world_size()
    local_start = get_sequence_parallel_rank() * ((f_gen * hw + world - 1) // world)
    local_end = local_start + s
    if ref_strength != 1.0:
        ref_slice = _local_frame_slice(0, hw, local_start, local_end)
        if ref_slice is not None:
            v[:, ref_slice[0]:ref_slice[1]] *= ref_strength

    full_k = _gather(k)
    full_v = _gather(v)
    out = x.new_empty(b, s, n, d)
    full_k_pose = None if k_pose is None else _gather(k_pose)
    full_v_pose = None if v_pose is None else _gather(v_pose)

    for frame in range(f_gen):
        local_slice = _local_frame_slice(frame, hw, local_start, local_end)
        if local_slice is None:
            continue
        start, end = local_slice
        frame_k, frame_v = full_k, full_v
        if full_k_pose is not None and frame > 0:
            pose_start, pose_end = (frame - 1) * hw, frame * hw
            frame_k = torch.cat((frame_k, full_k_pose[:, pose_start:pose_end]), dim=1)
            frame_v = torch.cat((frame_v, full_v_pose[:, pose_start:pose_end]), dim=1)
        out[:, start:end] = optimized_attention(
            q[:, start:end].reshape(b, end - start, n * d),
            frame_k.reshape(b, frame_k.shape[1], n * d),
            frame_v.reshape(b, frame_v.shape[1], n * d),
            heads=n,
            transformer_options=transformer_options,
        )

    for p in transformer_options.get("patches", {}).get("attn1_patch", []):
        out = p({"x": out, "q": q, "k": k, "transformer_options": transformer_options})
    return self.o(out)


def usp_animate2_dit_forward(
    self, x, t, context, clip_fea=None, freqs=None, freqs_pose=None, pose_latents=None, clip_fea_pose=None,
    context_pose=None, pose_strength=1.0, reference_strength=1.0, transformer_options={}, **kwargs
):
    x_input = x[:, :, 1:]
    x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    transformer_options["grid_sizes"] = grid_sizes
    f_gen, gh, gw = grid_sizes
    hw = gh * gw
    x = x.flatten(2).transpose(1, 2)
    apply_pose = pose_latents is not None
    if apply_pose and pose_latents.shape[2] != f_gen - 1:
        raise ValueError("pose branch has {} latent frames, expected {}".format(pose_latents.shape[2], f_gen - 1))
    cache = transformer_options.get("animate2_cache", None) if apply_pose else None
    if apply_pose:
        pose_latents = comfy.ldm.common_dit.pad_to_patch_size(pose_latents.to(x.dtype), self.patch_size)
        cache.select(pose_latents) if cache is not None else None
    cached = cache is not None and cache.filled(len(self.blocks))

    x, orig_size = _pad_and_split_for_sp(x)
    freqs, _ = _pad_and_split_for_sp(freqs)
    freqs_pose, _ = _pad_and_split_for_sp(freqs_pose)
    x_pose = None
    if apply_pose and not cached:
        x_pose = self.patch_embedding(torch.cat([pose_latents, torch.ones_like(pose_latents[:, :4]), pose_latents], dim=1).float()).to(x.dtype)
        x_pose, _ = _pad_and_split_for_sp(x_pose.flatten(2).transpose(1, 2))

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x.dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))
    e0_pose = None
    if apply_pose:
        e_pose = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, torch.ones_like(t.flatten())).to(dtype=x.dtype))
        e_pose = e_pose.reshape(t.shape[0], -1, e_pose.shape[-1])
        e0_pose = self.time_projection(e_pose).unflatten(2, (6, self.dim))

    context_gen = self.text_embedding(context)
    context_img_len = clip_fea.shape[-2] if clip_fea is not None else None
    if clip_fea is not None and self.img_emb is not None:
        context_gen = torch.cat([self.img_emb(clip_fea), context_gen], dim=1)
    context_img_len_pose = None
    if apply_pose and not cached:
        context_pose = self.text_embedding(context if context_pose is None else context_pose)
        clip_fea_pose = clip_fea if clip_fea_pose is None else clip_fea_pose
        if clip_fea_pose is not None:
            if self.img_emb is not None:
                context_pose = torch.cat([self.img_emb(clip_fea_pose), context_pose], dim=1)
            context_img_len_pose = clip_fea_pose.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    buffers = None

    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        if not apply_pose:
            k_pose = v_pose = None
        elif cached:
            x_pose_in = cache.take(i, x.device, x.dtype, x.shape[0])
            cache.prefetch(i + 1, x.device, x.dtype)
            k_pose, v_pose = block.kv_from_input(x_pose_in, e0_pose, freqs_pose, transformer_options=transformer_options)
        else:
            if cache is not None:
                cache.put(i, x_pose)
            x_pose, k_pose, v_pose = block.forward_pose(x_pose, e0_pose, freqs_pose, context_pose, context_img_len=context_img_len_pose, transformer_options=transformer_options)
        if v_pose is not None and pose_strength != 1.0:
            v_pose = v_pose * pose_strength
        if ("double_block", i) in blocks_replace:
            def block_wrap(args, block=block, k_pose=k_pose, v_pose=v_pose):
                return {"img": block.forward_gen(args["img"], args["vec"], args["pe"], args["txt"], k_pose, v_pose, f_gen, hw, buffers, ref_strength=reference_strength, context_img_len=context_img_len, transformer_options=args["transformer_options"])}
            x = blocks_replace[("double_block", i)]({"img": x, "txt": context_gen, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
        else:
            x = block.forward_gen(x, e0, freqs, context_gen, k_pose, v_pose, f_gen, hw, buffers, ref_strength=reference_strength, context_img_len=context_img_len, transformer_options=transformer_options)
        if "double_block" in transformer_options.get("patches", {}):
            for p in transformer_options["patches"]["double_block"]:
                x = p({"img": x, "x": x_input, "vec": e, "block_index": i, "img_offset": hw, "transformer_options": transformer_options})["img"]

    x = _gather(x)[:, :orig_size]
    return self.unpatchify(self.head(x, e), grid_sizes)
