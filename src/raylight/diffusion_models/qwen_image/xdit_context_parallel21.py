import torch

import comfy.model_prefetch
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.qwen_image21.model import _split_rows, block_causal_attention
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

from ..utils import pad_to_world_size


def _target_attention(q, k, v, heads, prefix_kv, target_len, transformer_options, preferred_attention=None):
    group = get_sp_group()
    k = group.all_gather(k.contiguous(), dim=1)[:, :target_len]
    v = group.all_gather(v.contiguous(), dim=1)[:, :target_len]
    if prefix_kv:
        k = torch.cat((prefix_kv[0], k), dim=1)
        v = torch.cat((prefix_kv[1], v), dim=1)
    return optimized_attention(q.flatten(2), k.flatten(2), v.flatten(2), heads,
                               transformer_options=transformer_options, preferred_attention=preferred_attention)


def usp_dit_forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
    b, _, h, w = x.shape
    dtype = x.dtype
    ref_latents = list(ref_latents or [])
    image_slots = list(image_slots or [])

    hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
    prefix_len = hidden_states.shape[1] - h * w
    patches = transformer_options.get("patches", {})
    for p in patches.get("post_input", []):
        out = p({"img": hidden_states, "pe": pe, "transformer_options": transformer_options})
        hidden_states, pe = out["img"], out.get("pe", pe)

    t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
    temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
    scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
    mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))

    target_len = hidden_states.shape[1] - prefix_len
    prefix = hidden_states[:, :prefix_len]
    prefix_pe = pe[:, :prefix_len]
    target, _ = pad_to_world_size(hidden_states[:, prefix_len:], dim=1)
    target_pe, _ = pad_to_world_size(pe[:, prefix_len:], dim=1)
    rank = get_sequence_parallel_rank()
    world_size = get_sequence_parallel_world_size()
    target = torch.chunk(target, world_size, dim=1)[rank]
    target_pe = torch.chunk(target_pe, world_size, dim=1)[rank]

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(self.transformer_blocks)
    transformer_options["block_type"] = "single"
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), x.device, transformer_options)
    for i, block in enumerate(self.transformer_blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, dtype)
        transformer_options["block_index"] = i
        if (("single_block", i) in blocks_replace or patches.get("single_block") or patches.get("attn1_patch") or
                transformer_options.get("optimized_attention_override") is not None or block.attn.comfy_attention.function is not None):
            full_target = get_sp_group().all_gather(target.contiguous(), dim=1)[:, :target_len]
            hidden_states = torch.cat((prefix, full_target), dim=1)
            attn_fn = block_causal_attention(segments, transformer_options, block_index=i, prefix_len=prefix_len)
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], mod, args["pe"], attn_fn, prefix_len, args["transformer_options"])}
                hidden_states = blocks_replace[("single_block", i)](
                    {"img": hidden_states, "vec": temb, "pe": pe, "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                hidden_states = block(hidden_states, mod, pe, attn_fn, prefix_len, transformer_options)
            for p in patches.get("single_block", []):
                hidden_states = p({"img": hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})["img"]
            prefix = hidden_states[:, :prefix_len].clone()
            full_target = hidden_states[:, prefix_len:].clone()
            full_target, _ = pad_to_world_size(full_target, dim=1)
            target = torch.chunk(full_target, world_size, dim=1)[rank]
            continue

        prefix_kv = []
        if prefix_len:
            causal = block_causal_attention(segments[:-1], transformer_options, block_index=i, prefix_len=prefix_len)

            def prefix_attention(q, k, v, heads, preferred_attention=None):
                prefix_kv.extend((k, v))
                return causal(q, k, v, heads, preferred_attention=preferred_attention)

            prefix = block(prefix, mod, prefix_pe, prefix_attention, prefix_len, transformer_options)

        def target_attention(q, k, v, heads, preferred_attention=None):
            return _target_attention(q, k, v, heads, prefix_kv, target_len, transformer_options, preferred_attention)

        target = block(target, mod, target_pe, target_attention, 0, transformer_options)

    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
    target = get_sp_group().all_gather(target.contiguous(), dim=1)[:, :target_len]
    target = self.norm_out(target, temb[:-1])
    target = self.proj_out(target)
    return target.transpose(1, 2).reshape(b, self.out_channels, h, w)
