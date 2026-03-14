import torch
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


def attention(q, k, v, heads, transformer_options={}):
    return xfuser_optimized_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        heads=heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )


def usp_dit_forward(self, x, timestep, context, y, freqs, freqs_text, transformer_options={}, *args, **kwargs):
    patches_replace = transformer_options.get("patches_replace", {})
    context = self.text_embeddings(context)
    time_embed = self.time_embeddings(timestep, x.dtype) + self.pooled_text_embeddings(y)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    context, context_orig_size = pad_to_world_size(context, dim=1)
    freqs_text, _ = pad_to_world_size(freqs_text, dim=1)
    context = torch.chunk(context, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    freqs_text = torch.chunk(freqs_text, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    for block in self.text_transformer_blocks:
        context = block(context, time_embed, freqs_text, transformer_options=transformer_options)

    context = get_sp_group().all_gather(context.contiguous(), dim=1)
    context = torch.chunk(context, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    visual_embed = self.visual_embeddings(x)
    visual_shape = visual_embed.shape[:-1]
    visual_embed = visual_embed.flatten(1, -2)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    visual_embed, visual_embed_orig_size = pad_to_world_size(visual_embed, dim=1)
    freqs, _ = pad_to_world_size(freqs, dim=1)

    visual_embed = torch.chunk(visual_embed, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.visual_transformer_blocks)
    transformer_options["block_type"] = "double"

    for i, block in enumerate(self.visual_transformer_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return block(x=args["x"],
                             context=args["context"],
                             time_embed=args["time_embed"],
                             freqs=args["freqs"],
                             transformer_options=args.get("transformer_options"))

            visual_embed = blocks_replace[("double_block", i)]({"x": visual_embed,
                                                                "context": context,
                                                                "time_embed": time_embed,
                                                                "freqs": freqs,
                                                                "transformer_options": transformer_options},
                                                               {"original_block": block_wrap})["x"]
        else:
            visual_embed = block(visual_embed,
                                 context,
                                 time_embed,
                                 freqs=freqs,
                                 transformer_options=transformer_options)

    visual_embed = get_sp_group().all_gather(visual_embed.contiguous(), dim=1)
    visual_embed = visual_embed[:, :visual_embed_orig_size, :]
    visual_embed = visual_embed.reshape(*visual_shape, -1)
    return self.out_layer(visual_embed, time_embed)


# Both for visual block and text block
def usp_self_attn_foward(self, x, freqs, transformer_options={}, **kwargs):
    q = self._compute_qk(x, freqs, self.to_query, self.query_norm)
    k = self._compute_qk(x, freqs, self.to_key, self.key_norm)
    v = self.to_value(x).view(*x.shape[:-1], self.num_heads, -1)
    out = attention(q, k, v, self.num_heads, transformer_options=transformer_options)
    return self.out_layer(out)


# idk if this is necessary split chunk since x seq size would be halved and this would not get called
def usp_self_attn_forward_chunked(self, x, freqs, transformer_options={}, **kwargs):
    def process_chunks(proj_fn, norm_fn):
        x_chunks = torch.chunk(x, self.num_chunks, dim=1)
        freqs_chunks = torch.chunk(freqs, self.num_chunks, dim=1)
        chunks = []
        for x_chunk, freqs_chunk in zip(x_chunks, freqs_chunks):
            chunks.append(self._compute_qk(x_chunk, freqs_chunk, proj_fn, norm_fn))
        return torch.cat(chunks, dim=1)

    q = process_chunks(self.to_query, self.query_norm)
    k = process_chunks(self.to_key, self.key_norm)
    v = self.to_value(x).view(*x.shape[:-1], self.num_heads, -1)
    out = attention(q, k, v, self.num_heads, transformer_options=transformer_options)
    return self.out_layer(out)


def usp_cross_attn_forward(self, x, context, transformer_options={}, *args, **kwargs):
    q, k, v = self.get_qkv(x, context)
    out = attention(self.query_norm(q), self.key_norm(k), v, self.num_heads, transformer_options=transformer_options)
    return self.out_layer(out)
