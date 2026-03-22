from __future__ import annotations

import types

import torch

from comfy import model_base
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.wan.model import WanSelfAttention, sinusoidal_embedding_1d

from xfuser.core.distributed import get_pp_group, get_world_group

from raylight.distributed_worker.pipefusion_state import PIPEFUSION_CONTEXT_KEY


PIPEFUSION_PATCH_MODE_KEY = "pipefusion_patch_mode"
PIPEFUSION_PATCH_TOKEN_INDICES_KEY = "pipefusion_patch_token_indices"
PIPEFUSION_TOTAL_TOKENS_KEY = "pipefusion_total_tokens"


def _fallback_forward(self, x, t, context, clip_fea=None, freqs=None, transformer_options=None, **kwargs):
    if transformer_options is None:
        transformer_options = {}
    return self._raylight_pipefusion_original_forward_orig(
        x,
        t,
        context,
        clip_fea=clip_fea,
        freqs=freqs,
        transformer_options=transformer_options,
        **kwargs,
    )


def _ensure_base_wan_only(base_model):
    unsupported = [
        getattr(model_base, "WAN21_Vace", None),
        getattr(model_base, "WAN21_Camera", None),
        getattr(model_base, "WAN21_HuMo", None),
        getattr(model_base, "WAN21_FlowRVS", None),
        getattr(model_base, "WAN21_SCAIL", None),
    ]
    for unsupported_cls in unsupported:
        if unsupported_cls is not None and isinstance(base_model, unsupported_cls):
            raise ValueError(f"PipeFusion v1 currently supports base WAN21 only; got {type(base_model).__name__}")


def inject_wan21_pipefusion(model_patcher, base_model, *args):
    _ensure_base_wan_only(base_model)

    model = base_model.diffusion_model
    if not hasattr(base_model, "_raylight_pipefusion_original_concat_cond"):
        base_model._raylight_pipefusion_original_concat_cond = base_model.concat_cond
        base_model.concat_cond = types.MethodType(pipefusion_concat_cond, base_model)
    if not hasattr(model, "_raylight_pipefusion_original_forward_orig"):
        model._raylight_pipefusion_original_forward_orig = model.forward_orig
    if not hasattr(model, "reset_activation_cache"):
        model.reset_activation_cache = types.MethodType(reset_wan_pipefusion_cache, model)
    _patch_wan_self_attention_modules(model)
    if getattr(model, "_raylight_pipefusion_patched", False):
        return
    model.forward_orig = types.MethodType(pipefusion_dit_forward, model)
    model._raylight_pipefusion_patched = True


def _patch_wan_self_attention_modules(model):
    for module in model.modules():
        if type(module) is not WanSelfAttention:
            continue
        if getattr(module, "_raylight_pipefusion_patched", False):
            continue
        module._raylight_pipefusion_original_forward = module.forward
        module.forward = types.MethodType(pipefusion_wan_self_attention_forward, module)
        module._raylight_pipefusion_patched = True
        module._raylight_pipefusion_k_cache = None
        module._raylight_pipefusion_v_cache = None


def reset_wan_pipefusion_cache(self):
    for module in self.modules():
        if not getattr(module, "_raylight_pipefusion_patched", False):
            continue
        module._raylight_pipefusion_k_cache = None
        module._raylight_pipefusion_v_cache = None


def _wan_patch_in_channels(base_model):
    patch_embedding = getattr(base_model.diffusion_model, "patch_embedding", None)
    if patch_embedding is not None:
        return patch_embedding.weight.shape[1]

    patch_in_channels = getattr(base_model.diffusion_model, "_raylight_pipefusion_patch_in_channels", None)
    if patch_in_channels is None:
        raise RuntimeError("PipeFusion WAN patch input channels are unavailable on this stage")
    return patch_in_channels


def pipefusion_concat_cond(self, **kwargs):
    noise = kwargs.get("noise", None)
    extra_channels = _wan_patch_in_channels(self) - noise.shape[1]
    if extra_channels == 0:
        return None

    image = kwargs.get("concat_latent_image", None)
    device = kwargs["device"]

    if image is None:
        shape_image = list(noise.shape)
        shape_image[1] = extra_channels
        image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
    else:
        latent_dim = self.latent_format.latent_channels
        image = model_base.utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        for i in range(0, image.shape[1], latent_dim):
            image[:, i : i + latent_dim] = self.process_latent_in(image[:, i : i + latent_dim])
        image = model_base.utils.resize_to_batch_size(image, noise.shape[0])

    if extra_channels != image.shape[1] + 4:
        if not self.image_to_video or extra_channels == image.shape[1]:
            return image

    if image.shape[1] > (extra_channels - 4):
        image = image[:, : (extra_channels - 4)]

    mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
    if mask is None:
        mask = torch.zeros_like(noise)[:, :4]
    else:
        if mask.shape[1] != 4:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = 1.0 - mask
        mask = model_base.utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        if mask.shape[-3] < noise.shape[-3]:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode="constant", value=0)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 4, 1, 1, 1)
        mask = model_base.utils.resize_to_batch_size(mask, noise.shape[0])

    concat_mask_index = kwargs.get("concat_mask_index", 0)
    if concat_mask_index != 0:
        return torch.cat((image[:, :concat_mask_index], mask, image[:, concat_mask_index:]), dim=1)
    return torch.cat((mask, image), dim=1)


def partition_wan_for_pipefusion(base_model, stage):
    _ensure_base_wan_only(base_model)

    model = base_model.diffusion_model
    total_blocks = len(model.blocks)
    model._raylight_pipefusion_patch_in_channels = model.patch_embedding.weight.shape[1]
    model.blocks = torch.nn.ModuleList(
        [block if stage.stage_start <= block_idx < stage.stage_end else None for block_idx, block in enumerate(model.blocks)]
    )
    if not stage.is_first:
        model.patch_embedding = None
    if not stage.is_last:
        model.head = None

    model._raylight_pipefusion_partitioned = True
    model._raylight_pipefusion_total_blocks = total_blocks
    model._raylight_pipefusion_stage_start = stage.stage_start
    model._raylight_pipefusion_stage_end = stage.stage_end
    return model


def filter_wan_state_dict_for_stage(state_dict, stage):
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("blocks."):
            parts = key.split(".", 2)
            if len(parts) < 2 or not parts[1].isdigit():
                filtered[key] = value
                continue
            block_idx = int(parts[1])
            if stage.stage_start <= block_idx < stage.stage_end:
                filtered[key] = value
            continue
        if key.startswith("patch_embedding."):
            if stage.is_first:
                filtered[key] = value
            continue
        if key.startswith("head."):
            if stage.is_last:
                filtered[key] = value
            continue
        filtered[key] = value
    return filtered


def _grid_sizes_from_latent(self, x):
    return tuple(int((x.shape[idx + 2] + (self.patch_size[idx] // 2)) // self.patch_size[idx]) for idx in range(3))


def _split_tensor(tensor, num_chunks, dim=1):
    if tensor is None:
        return [None] * num_chunks
    return [chunk.contiguous() for chunk in torch.tensor_split(tensor, num_chunks, dim=dim)]


def _split_wan_tensor_by_grid(tensor, grid_sizes, num_chunks):
    if tensor is None:
        return [None] * num_chunks, [None] * num_chunks, None

    t_grid, h_grid, w_grid = grid_sizes
    orig_size = tensor.shape[1]
    expected_tokens = t_grid * h_grid * w_grid
    if orig_size != expected_tokens:
        raise RuntimeError(f"PipeFusion Wan tensor/grid mismatch: tokens={orig_size} grid={t_grid}x{h_grid}x{w_grid}")

    tensor = tensor.contiguous().reshape(tensor.shape[0], t_grid, h_grid, w_grid, *tensor.shape[2:])

    token_indices = torch.arange(orig_size, device=tensor.device, dtype=torch.long).reshape(1, t_grid, h_grid, w_grid)
    tail_shape = tensor.shape[4:]
    chunks = [chunk.contiguous().reshape(chunk.shape[0], -1, *tail_shape) for chunk in torch.tensor_split(tensor, num_chunks, dim=3)]
    index_chunks = [chunk.contiguous().reshape(-1) for chunk in torch.tensor_split(token_indices, num_chunks, dim=3)]
    return chunks, index_chunks, orig_size


def _merge_wan_tensor_chunks_by_grid(chunks, chunk_indices, orig_size):
    if not chunks:
        raise RuntimeError("PipeFusion Wan merge requires at least one chunk")

    tail_shape = chunks[0].shape[2:]
    tensor = chunks[0].new_zeros((chunks[0].shape[0], orig_size, *tail_shape))
    for chunk, token_indices in zip(chunks, chunk_indices):
        tensor.index_copy_(1, token_indices, chunk)
    return tensor


def _update_wan_attention_cache(attn, k, v, token_indices, total_tokens):
    k_cache = getattr(attn, "_raylight_pipefusion_k_cache", None)
    v_cache = getattr(attn, "_raylight_pipefusion_v_cache", None)
    cache_shape = (k.shape[0], total_tokens, k.shape[2], k.shape[3])

    if k_cache is None or tuple(k_cache.shape) != cache_shape or k_cache.dtype != k.dtype or k_cache.device != k.device:
        k_cache = k.new_zeros(cache_shape)
        v_cache = v.new_zeros(cache_shape)

    k_cache.index_copy_(1, token_indices, k)
    v_cache.index_copy_(1, token_indices, v)
    attn._raylight_pipefusion_k_cache = k_cache
    attn._raylight_pipefusion_v_cache = v_cache
    return k_cache, v_cache


def pipefusion_wan_self_attention_forward(self, x, freqs, transformer_options={}):
    patches = transformer_options.get("patches", {})
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
    k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
    v = self.v(x).view(b, s, n, d)

    if transformer_options.get(PIPEFUSION_PATCH_MODE_KEY, False):
        token_indices = transformer_options.get(PIPEFUSION_PATCH_TOKEN_INDICES_KEY)
        total_tokens = transformer_options.get(PIPEFUSION_TOTAL_TOKENS_KEY)
        if token_indices is None or total_tokens is None:
            raise RuntimeError("PipeFusion patch-mode Wan attention is missing token index metadata")
        k_attn, v_attn = _update_wan_attention_cache(self, k, v, token_indices, total_tokens)
    else:
        self._raylight_pipefusion_k_cache = k.contiguous()
        self._raylight_pipefusion_v_cache = v.contiguous()
        k_attn, v_attn = k, v

    x = optimized_attention(
        q.view(b, s, n * d),
        k_attn.view(b, k_attn.shape[1], n * d),
        v_attn.view(b, v_attn.shape[1], n * d),
        heads=self.num_heads,
        transformer_options=transformer_options,
    )

    if "attn1_patch" in patches:
        for p in patches["attn1_patch"]:
            x = p({"x": x, "q": q, "k": k_attn, "transformer_options": transformer_options})

    return self.o(x)


def _run_wan_block(block, block_idx, x, context, e0, freqs, context_img_len, transformer_options, blocks_replace):
    transformer_options["block_index"] = block_idx
    if ("double_block", block_idx) in blocks_replace:

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

        out = blocks_replace[("double_block", block_idx)](
            {
                "img": x,
                "txt": context,
                "vec": e0,
                "pe": freqs,
                "transformer_options": transformer_options,
            },
            {"original_block": block_wrap},
        )
        return out["img"]

    return block(
        x,
        e=e0,
        freqs=freqs,
        context=context,
        context_img_len=context_img_len,
        transformer_options=transformer_options,
    )


def _run_local_stage(
    self,
    chunk,
    chunk_freqs,
    context,
    e0,
    context_img_len,
    transformer_options,
    stage,
    blocks_replace,
    patch_mode=False,
    patch_token_indices=None,
    total_tokens=None,
):
    transformer_options[PIPEFUSION_PATCH_MODE_KEY] = patch_mode
    transformer_options[PIPEFUSION_PATCH_TOKEN_INDICES_KEY] = patch_token_indices
    transformer_options[PIPEFUSION_TOTAL_TOKENS_KEY] = total_tokens
    try:
        for block_idx in range(stage.stage_start, stage.stage_end):
            block = self.blocks[block_idx]
            if block is None:
                raise RuntimeError(f"PipeFusion stage {stage.rank} is missing owned Wan block {block_idx}")
            chunk = _run_wan_block(
                block,
                block_idx,
                chunk,
                context,
                e0,
                chunk_freqs,
                context_img_len,
                transformer_options,
                blocks_replace,
            )
        return chunk
    finally:
        transformer_options.pop(PIPEFUSION_PATCH_MODE_KEY, None)
        transformer_options.pop(PIPEFUSION_PATCH_TOKEN_INDICES_KEY, None)
        transformer_options.pop(PIPEFUSION_TOTAL_TOKENS_KEY, None)


def _prepare_pp_group(dtype):
    pp_group = get_pp_group()
    pp_group.reset_buffer()
    pp_group.set_config(dtype=dtype)
    return pp_group


def _run_sync_warmup(
    self, pp_group, x_full, freqs_full, context, e0, context_img_len, transformer_options, blocks_replace, pf_context, total_tokens
):
    stage = pf_context.stage
    if stage.is_first:
        chunk = x_full
    else:
        pf_context.trace(f"recv_start full from={pp_group.prev_rank}")
        chunk = pp_group.pipeline_recv(idx=0)
        pf_context.trace(f"recv_done full from={pp_group.prev_rank}")

    chunk = _run_local_stage(
        self,
        chunk,
        freqs_full,
        context,
        e0,
        context_img_len,
        transformer_options,
        stage,
        blocks_replace,
        patch_mode=False,
        patch_token_indices=None,
        total_tokens=total_tokens,
    )

    if stage.is_last:
        return chunk

    pf_context.trace(f"send_start full to={pp_group.next_rank} shape={tuple(chunk.shape)}")
    pp_group.pipeline_send(chunk, segment_idx=0)
    pf_context.trace(f"send_done full to={pp_group.next_rank}")
    return None


def _run_pipeline(
    self,
    pp_group,
    x_chunks,
    freqs_chunks,
    chunk_indices,
    context,
    e0,
    context_img_len,
    transformer_options,
    blocks_replace,
    pf_context,
    total_tokens,
):
    stage = pf_context.stage
    final_chunks = [None] * stage.num_pipeline_patch

    if not stage.is_first:
        for micro_idx in range(stage.num_pipeline_patch):
            pp_group.add_pipeline_recv_task(micro_idx)
        pf_context.trace(f"recv_start micro=0 from={pp_group.prev_rank}")
        pp_group.recv_next()

    for micro_idx in range(stage.num_pipeline_patch):
        if stage.is_first:
            chunk = x_chunks[micro_idx]
        else:
            chunk = pp_group.get_pipeline_recv_data(idx=micro_idx)
            pf_context.trace(f"recv_done micro={micro_idx} from={pp_group.prev_rank}")

        if not stage.is_first and micro_idx + 1 < stage.num_pipeline_patch:
            pf_context.trace(f"recv_start micro={micro_idx + 1} from={pp_group.prev_rank}")
            pp_group.recv_next()

        chunk = _run_local_stage(
            self,
            chunk,
            freqs_chunks[micro_idx],
            context,
            e0,
            context_img_len,
            transformer_options,
            stage,
            blocks_replace,
            patch_mode=True,
            patch_token_indices=chunk_indices[micro_idx],
            total_tokens=total_tokens,
        )

        if stage.is_last:
            final_chunks[micro_idx] = chunk
        else:
            pf_context.trace(f"send_start micro={micro_idx} to={pp_group.next_rank} shape={tuple(chunk.shape)}")
            pp_group.pipeline_isend(chunk, segment_idx=micro_idx)
            pf_context.trace(f"send_queued micro={micro_idx} to={pp_group.next_rank}")

    return final_chunks


@torch.compiler.disable
def pipefusion_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    pf_context = transformer_options.get(PIPEFUSION_CONTEXT_KEY)
    if pf_context is None:
        return _fallback_forward(
            self,
            x,
            t,
            context,
            clip_fea=clip_fea,
            freqs=freqs,
            transformer_options=transformer_options,
            **kwargs,
        )

    if kwargs.get("reference_latent") is not None:
        raise ValueError("PipeFusion v1 does not support Wan reference_latent inputs yet")
    if pf_context.runtime.parallel.sequence_world_size != 1:
        raise NotImplementedError("Wan PipeFusion does not yet combine pipeline execution with USP sequence sharding")
    if pf_context.runtime.parallel.config.cfg_degree != 1:
        raise NotImplementedError("Wan PipeFusion currently ignores CFG parallel execution")

    stage = pf_context.stage
    latent_x = x
    grid_sizes = _grid_sizes_from_latent(self, latent_x)
    transformer_options["grid_sizes"] = grid_sizes
    transformer_options["pipefusion_mode"] = pf_context.mode
    transformer_options["total_blocks"] = stage.total_blocks
    transformer_options["block_type"] = "double"

    if freqs is None:
        raise ValueError("PipeFusion requires Wan RoPE frequencies")
    if stage.num_pipeline_patch > freqs.shape[1]:
        raise ValueError(f"PipeFusion num_pipeline_patch cannot exceed Wan token count: {stage.num_pipeline_patch} > {freqs.shape[1]}")

    pp_group = _prepare_pp_group(latent_x.dtype)

    time_embedding = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=latent_x.dtype))
    e = time_embedding.reshape(t.shape[0], -1, time_embedding.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    context = self.text_embedding(context)
    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    freqs_chunks, chunk_indices, orig_token_count = _split_wan_tensor_by_grid(freqs, grid_sizes, stage.num_pipeline_patch)

    full_freqs = freqs[:, :orig_token_count, ...]
    full_tokens = None

    if stage.is_first:
        if self.patch_embedding is None:
            raise RuntimeError("PipeFusion first stage requires Wan patch_embedding to be present")
        full_tokens = self.patch_embedding(latent_x.float()).to(latent_x.dtype)
        full_tokens = full_tokens.flatten(2).transpose(1, 2)
        x_chunks, x_chunk_indices, x_orig_token_count = _split_wan_tensor_by_grid(full_tokens, grid_sizes, stage.num_pipeline_patch)
        if x_orig_token_count != orig_token_count:
            raise RuntimeError(f"PipeFusion token/freq count mismatch: tokens={x_orig_token_count} freqs={orig_token_count}")
        for expected, actual in zip(chunk_indices, x_chunk_indices):
            if not torch.equal(expected, actual):
                raise RuntimeError("PipeFusion token/freq chunk index mismatch")
    else:
        x_chunks = None

    if pf_context.is_warmup():
        full_tokens = _run_sync_warmup(
            self,
            pp_group,
            full_tokens,
            full_freqs,
            context,
            e0,
            context_img_len,
            transformer_options,
            blocks_replace,
            pf_context,
            orig_token_count,
        )
    else:
        final_chunks = _run_pipeline(
            self,
            pp_group,
            x_chunks,
            freqs_chunks,
            chunk_indices,
            context,
            e0,
            context_img_len,
            transformer_options,
            blocks_replace,
            pf_context,
            orig_token_count,
        )

    if stage.is_last:
        if self.head is None:
            raise RuntimeError("PipeFusion last stage requires Wan head to be present")
        if pf_context.is_warmup():
            tokens = full_tokens
        else:
            tokens = _merge_wan_tensor_chunks_by_grid(final_chunks, chunk_indices, orig_token_count)
        tokens = self.head(tokens, e)
        output = self.unpatchify(tokens, grid_sizes)
    else:
        output = torch.empty(
            (latent_x.shape[0], self.out_dim, latent_x.shape[2], latent_x.shape[3], latent_x.shape[4]),
            dtype=latent_x.dtype,
            device=latent_x.device,
        )

    output = get_world_group().broadcast(output.contiguous(), src=pf_context.runtime.parallel.global_world_size - 1)
    pf_context.trace(f"broadcast_done output_shape={tuple(output.shape)}")
    return output
