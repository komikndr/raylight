import torch

import comfy.model_prefetch
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux3.model import Block, Flux3
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn


xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(xfuser_attn.get_attn_type(), xfuser_attn.get_sync_ulysses())
_SEQUENCE_LENGTH = "raylight_flux3_sequence_length"


def _split_sequence(tensor):
    tensor, _ = pad_to_world_size(tensor, dim=1)
    return tensor.chunk(get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()].clone()


def usp_attention(original_attention, q, k, v, heads, transformer_options={}, **kwargs):
    length = transformer_options[_SEQUENCE_LENGTH]
    if length % get_sequence_parallel_world_size():
        # Padding must not enter the softmax. Keep queries local and gather only
        # valid keys/values when the sequence cannot be evenly distributed.
        k = get_sp_group().all_gather(k.contiguous(), dim=1)[:, :length]
        v = get_sp_group().all_gather(v.contiguous(), dim=1)[:, :length]
        return original_attention(q, k, v, heads, transformer_options=transformer_options, **kwargs)
    return xfuser_optimized_attention(q, k, v, heads, transformer_options=transformer_options, **kwargs)


def usp_block_forward(self, x, spans, pe, transformer_options={}):
    if _SEQUENCE_LENGTH not in transformer_options:
        return Block.forward(self, x, spans, pe, transformer_options)
    transformer_options = transformer_options.copy()
    transformer_options["optimized_attention_override"] = usp_attention
    return Block.forward(self, x, spans, pe, transformer_options)


def usp_flux3_forward(self, x, timestep, context=None, ref_latents=None, transformer_options={}, txt_states=None, ref_states=None, ref_shapes=None, **kwargs):
    if self.raylight_native_attention or "optimized_attention_override" in transformer_options:
        return Flux3._forward(
            self, x, timestep, context, ref_latents, transformer_options,
            txt_states=txt_states, ref_states=ref_states, ref_shapes=ref_shapes, **kwargs,
        )

    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    vec_target = None
    if self.flow_map:
        sample_sigmas = transformer_options["sample_sigmas"]
        i = int((sample_sigmas - timestep[0]).abs().argmin())
        target = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        vec_target = self.ref_time_in(timestep_embedding(target.reshape(1).to(device=device, dtype=dtype), 256))

    if txt_states is None:
        txt_states, ref_states = self.preprocess_conds(context, ref_latents, vec_target, transformer_options)
        ref_shapes = [ref.shape[-2:] for ref in ref_latents] if ref_latents else None
    transformer_options = transformer_options.copy()

    vec = self.time_in(timestep_embedding(torch.cat([timestep, timestep.new_zeros(1)]).to(dtype), 256))
    if vec_target is not None:
        vec = vec + vec_target
    vec_img, vec_zero = vec[:-1], vec[-1:]
    early = self.early_stream_modulations["image"](vec_img)[0]

    img = self.emb_in["image"](x.flatten(2).transpose(1, 2))
    image_length = H * W
    parts = [(None, "image", self.image_ids(H, W, 0, device))]
    lengths = [image_length]
    if ref_states is not None:
        parts.append((ref_states, "image_cond", self.ref_ids(ref_shapes, device)))
        lengths.append(ref_states.shape[1])
    parts.append((txt_states, "txt", self.text_ids(txt_states.shape[1], device)))
    lengths.append(txt_states.shape[1])

    pe = self.rope(torch.cat([ids for _, _, ids in parts]))
    image_pe = _split_sequence(pe[:, :image_length])
    pe = _split_sequence(pe)
    spans, length = [], 0
    for (_, mode, _), count in zip(parts, lengths):
        mod = self.single_stream_modulations[mode](vec_img if mode == "image" else vec_zero)[0]
        spans.append((length, length + count, mod))
        length += count

    world_size = get_sequence_parallel_world_size()
    local_length = (length + world_size - 1) // world_size
    local_start = get_sequence_parallel_rank() * local_length
    local_end = local_start + local_length
    joint = torch.empty(B, local_length, img.shape[-1], device=device, dtype=img.dtype)
    if local_end > length:
        joint[:, max(0, length - local_start):].zero_()
    img = _split_sequence(img)

    queue = list(self.content_mode_blocks["image"]) + list(self.single_blocks)
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(queue, device, transformer_options)
    comfy.model_prefetch.malloc_graph_begin(device)
    transformer_options[_SEQUENCE_LENGTH] = image_length
    for block in self.content_mode_blocks["image"]:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        img = block(img, [(0, img.shape[1], early)], image_pe, transformer_options)

    img = get_sp_group().all_gather(img.contiguous(), dim=1)[:, :image_length]
    parts[0] = (img, "image", parts[0][2])
    local_spans = []
    for (tokens, _, _), (start, end, mod) in zip(parts, spans):
        begin, stop = max(start, local_start), min(end, local_end)
        if begin < stop:
            joint[:, begin - local_start:stop - local_start].copy_(tokens[:, begin - start:stop - start])
            local_spans.append((begin - local_start, stop - local_start, mod))
    if not local_spans:
        local_spans = [(0, local_length, spans[-1][2])]
    del img, parts

    transformer_options[_SEQUENCE_LENGTH] = length
    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    for i, block in enumerate(self.single_blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        transformer_options["block_index"] = i
        joint = block(joint, local_spans, pe, transformer_options)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
    comfy.model_prefetch.malloc_graph_end()

    img = get_sp_group().all_gather(joint.contiguous(), dim=1)[:, :image_length]
    out = self.final_layer["image"](img, vec_img)
    return out.transpose(1, 2).reshape(B, self.in_channels, H, W)
