from .registry import active_inner_attention, inner_attention_scope
import torch


def _ring_world_size(attention):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    process_group = getattr(attention, "ring_pg", None)
    return torch.distributed.get_world_size(process_group)


# Keeps XFuser ring callback stable while selecting a scoped processor
class InnerAttentionDispatcher:
    def __init__(self, xfuser_attention):
        self.ring_world_size = _ring_world_size(xfuser_attention)
        self._dense_ring_attn_fn = xfuser_attention.ring_attn_fn

        def dispatch(q, k, v, *args, **kwargs):
            active = active_inner_attention()
            if active is None:
                return self._dense_ring_attn_fn(q, k, v, *args, **kwargs)
            processor, transformer_options = active
            dense = lambda query, key, value, **options: self._dense_ring_attn_fn(
                query, key, value, *args, **options)
            return processor(
                dense,
                q,
                k,
                v,
                transformer_options=transformer_options,
                **kwargs
            )

        xfuser_attention.ring_attn_fn = dispatch

    def scope(self, processor, transformer_options):
        return inner_attention_scope(processor, transformer_options, self.ring_world_size)
