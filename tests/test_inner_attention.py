import pytest
from pathlib import Path

from raylight.distributed_modules.inner_attention import (
    INNER_ATTENTION_KEY,
    InnerAttentionDispatcher,
    InnerAttentionRegistry,
    inner_attention_scope,
    clear_inner_attention,
    set_inner_attention,
)
from raylight.distributed_modules.inner_attention.h3_sla import MiniMaxH3SLA


class Processor:
    supports_ring = True

    def __call__(self, dense, q, k, v, *, transformer_options, **kwargs):
        return ("custom", transformer_options, q, k, v, kwargs)


def test_registry_and_set_inner_attention_copy_options():
    name = "test:processor"
    InnerAttentionRegistry.register(name)(Processor)
    assert name in InnerAttentionRegistry.available()
    assert isinstance(InnerAttentionRegistry.create(name), Processor)
    with pytest.raises(ValueError):
        InnerAttentionRegistry.register(name)(Processor)

    class Model:
        model_options = {"keep": {"value": 1}, "transformer_options": {"old": 2}}

        def clone(self):
            result = Model()
            result.model_options = self.model_options.copy()
            return result

    model = Model()
    clone = set_inner_attention(model, Processor())
    assert clone is not model
    assert clone.model_options["transformer_options"][INNER_ATTENTION_KEY].__class__ is Processor
    assert "raylight_inner_attention" not in model.model_options["transformer_options"]
    assert clone.model_options["keep"] is model.model_options["keep"]
    cleared = clear_inner_attention(clone)
    assert INNER_ATTENTION_KEY not in cleared.model_options["transformer_options"]
    assert INNER_ATTENTION_KEY in clone.model_options["transformer_options"]
    preserved = clear_inner_attention(clone, type("Other", (), {}))
    assert INNER_ATTENTION_KEY in preserved.model_options["transformer_options"]


def test_dispatcher_scopes_custom_and_resets():
    class Attention:
        def __init__(self):
            self.ring_attn_fn = lambda *args, **kwargs: ("dense", args, kwargs)

    attention = Attention()
    dispatcher = InnerAttentionDispatcher(attention)
    dense = attention.ring_attn_fn(1, 2, 3)
    with dispatcher.scope(Processor(), {"step": 4}):
        custom = attention.ring_attn_fn(1, 2, 3, flag=True)
    assert dense[0] == "dense"
    assert custom[0] == "custom"
    assert custom[1] == {"step": 4}
    assert attention.ring_attn_fn(1, 2, 3)[0] == "dense"


def test_dispatcher_none_scope_stays_dense():
    class Attention:
        ring_attn_fn = lambda *args, **kwargs: "dense"

    attention = Attention()
    dispatcher = InnerAttentionDispatcher(attention)
    with dispatcher.scope(None, {}):
        assert attention.ring_attn_fn(1, 2, 3) == "dense"


def test_non_ring_processor_is_rejected():
    with pytest.raises(RuntimeError):
        with inner_attention_scope(MiniMaxH3SLA(), {}, ring_world_size=2):
            pass


def test_h3_preparation_ranges_and_tail():
    class Layout:
        segments = [(0, 5, "text"), (5, 7, "ref_audio"), (7, 10, "audio"), (10, 14, "video")]

    processor = MiniMaxH3SLA(dense_last_steps=1, protect_audio=True)
    payload = {"layout": Layout(), "text_token_tags": [0, 1, 1, 0, 1]}
    prepared = processor.prepare({"sample_sigmas": [1.0, 0.5, 0.2, 0.0], "sigmas": [0.2]}, payload)
    assert prepared["dense_tail"]
    assert prepared["protected_ranges"] == [(1, 3), (4, 5), (5, 7), (7, 10)]
    assert MiniMaxH3SLA().prepare({}, {})["protected_ranges"] is None


def test_h3_malformed_tags_protect_text():
    class Layout:
        segments = [(0, 3, "text")]

    assert MiniMaxH3SLA().prepare({}, {"layout": Layout()})["protected_ranges"] == [(0, 3)]


def test_block_map_is_bounded_by_block_count():
    import torch
    from raylight.distributed_modules.inner_attention._h3_sla_block_map import get_block_map

    q = torch.randn(1, 65, 2, 4, dtype=torch.float16)
    lut, ordinary = get_block_map(q, q, blkq=32, blkk=32, sparsity_ratio=0.5, protected_ranges=[(0, 1), (64, 65)])
    assert lut.dtype == torch.int32 and lut.is_contiguous()
    assert lut.shape[-1] >= ordinary
    for batch in range(lut.shape[0]):
        for head in range(lut.shape[1]):
            for query in range(lut.shape[2]):
                assert bool((lut[batch, head, query] == 0).any())
                assert bool((lut[batch, head, query] == 2).any())


def test_xfuser_normal_and_joint_calls_share_dispatch_scope():
    source = (Path(__file__).parents[1] / "src/raylight/distributed_modules/attention.py").read_text()
    assert source.count("with inner_dispatcher.scope") == 1
    assert source.count("xfuser_attn(") == 2


def test_h3_kernel_uses_fixed_head_dim_and_launch_ladder():
    source = (Path(__file__).parents[1] / "src/raylight/distributed_modules/inner_attention/_h3_sla_kernel.py").read_text()
    assert "_H3_HEAD_DIM = 128" in source
    assert "(8, 3), (4, 3), (8, 2), (4, 1)" in source
    assert "BLOCK_N=block_k" in source
    assert "torch.einsum" not in source


def test_h3_node_is_registered():
    from raylight.comfy_extra_dist.nodes_minimax_h3 import NODE_CLASS_MAPPINGS

    assert "RayMiniMaxH3SLA" in NODE_CLASS_MAPPINGS
