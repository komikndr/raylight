from __future__ import annotations

import types

import torch

from comfy import model_base
from comfy.ldm.wan.model import sinusoidal_embedding_1d

from xfuser.core.distributed import get_pp_group, get_world_group

from raylight.distributed_worker.pipefusion_state import PIPEFUSION_CONTEXT_KEY


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
    if not hasattr(model, "_raylight_pipefusion_original_forward_orig"):
        model._raylight_pipefusion_original_forward_orig = model.forward_orig
    if getattr(model, "_raylight_pipefusion_patched", False):
        return
    model.forward_orig = types.MethodType(pipefusion_dit_forward, model)
    model._raylight_pipefusion_patched = True


def partition_wan_for_pipefusion(base_model, stage):
    _ensure_base_wan_only(base_model)

    model = base_model.diffusion_model
    total_blocks = len(model.blocks)
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
        return [None] * num_chunks, None

    t_grid, h_grid, w_grid = grid_sizes
    orig_size = tensor.shape[1]
    expected_tokens = t_grid * h_grid * w_grid
    if orig_size != expected_tokens:
        raise RuntimeError(f"PipeFusion Wan tensor/grid mismatch: tokens={orig_size} grid={t_grid}x{h_grid}x{w_grid}")

    padded_w_grid = ((w_grid + num_chunks - 1) // num_chunks) * num_chunks
    if padded_w_grid != w_grid:
        pad_tokens = t_grid * h_grid * (padded_w_grid - w_grid)
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_tokens
        tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)

    if tensor.ndim == 3:
        tensor = tensor.view(tensor.shape[0], t_grid, h_grid, padded_w_grid, tensor.shape[-1])
        chunks = [chunk.contiguous().view(chunk.shape[0], -1, chunk.shape[-1]) for chunk in torch.chunk(tensor, num_chunks, dim=3)]
    else:
        tensor = tensor.view(tensor.shape[0], t_grid, h_grid, padded_w_grid, *tensor.shape[2:])
        chunks = [chunk.contiguous().view(chunk.shape[0], -1, *chunk.shape[4:]) for chunk in torch.chunk(tensor, num_chunks, dim=3)]
    return chunks, orig_size


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


def _run_local_stage(self, chunk, chunk_freqs, context, e0, context_img_len, transformer_options, stage, blocks_replace):
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


def _prepare_pp_group(dtype):
    pp_group = get_pp_group()
    pp_group.reset_buffer()
    pp_group.set_config(dtype=dtype)
    return pp_group


def _run_sync_warmup(self, pp_group, x_chunks, freqs_chunks, context, e0, context_img_len, transformer_options, blocks_replace, pf_context):
    stage = pf_context.stage
    final_chunks = []

    for micro_idx in range(stage.num_pipeline_patch):
        if stage.is_first:
            chunk = x_chunks[micro_idx]
        else:
            pf_context.trace(f"recv_start micro={micro_idx} from={pp_group.prev_rank}")
            chunk = pp_group.pipeline_recv(idx=micro_idx)
            pf_context.trace(f"recv_done micro={micro_idx} from={pp_group.prev_rank}")

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
        )

        if stage.is_last:
            final_chunks.append(chunk)
            continue

        pf_context.trace(f"send_start micro={micro_idx} to={pp_group.next_rank} shape={tuple(chunk.shape)}")
        pp_group.pipeline_send(chunk, segment_idx=micro_idx)
        pf_context.trace(f"send_done micro={micro_idx} to={pp_group.next_rank}")

    return final_chunks


def _run_pipeline(self, pp_group, x_chunks, freqs_chunks, context, e0, context_img_len, transformer_options, blocks_replace, pf_context):
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
    freqs_chunks, orig_token_count = _split_wan_tensor_by_grid(freqs, grid_sizes, stage.num_pipeline_patch)

    x_chunks = None
    if stage.is_first:
        if self.patch_embedding is None:
            raise RuntimeError("PipeFusion first stage requires Wan patch_embedding to be present")
        tokens = self.patch_embedding(latent_x.float()).to(latent_x.dtype)
        tokens = tokens.flatten(2).transpose(1, 2)
        x_chunks, x_orig_token_count = _split_wan_tensor_by_grid(tokens, grid_sizes, stage.num_pipeline_patch)
        if x_orig_token_count != orig_token_count:
            raise RuntimeError(f"PipeFusion token/freq count mismatch: tokens={x_orig_token_count} freqs={orig_token_count}")

    if pf_context.is_warmup():
        final_chunks = _run_sync_warmup(
            self,
            pp_group,
            x_chunks,
            freqs_chunks,
            context,
            e0,
            context_img_len,
            transformer_options,
            blocks_replace,
            pf_context,
        )
    else:
        final_chunks = _run_pipeline(
            self,
            pp_group,
            x_chunks,
            freqs_chunks,
            context,
            e0,
            context_img_len,
            transformer_options,
            blocks_replace,
            pf_context,
        )

    if stage.is_last:
        if self.head is None:
            raise RuntimeError("PipeFusion last stage requires Wan head to be present")
        tokens = torch.cat(final_chunks, dim=1)[:, :orig_token_count, :]
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
