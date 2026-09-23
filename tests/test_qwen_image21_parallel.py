from types import SimpleNamespace

import torch

import raylight.diffusion_models.qwen_image.xdit_context_parallel21 as usp
from comfy.ldm.qwen_image21.model import block_causal_attention


def test_qwen21_target_attention_matches_full_sequence_with_padding():
    torch.manual_seed(0)
    prefix_k = torch.randn(1, 3, 2, 4)
    prefix_v = torch.randn(1, 3, 2, 4)
    target_k = torch.randn(1, 5, 2, 4)
    target_v = torch.randn(1, 5, 2, 4)
    padded_k = torch.cat((target_k, torch.zeros(1, 1, 2, 4)), dim=1)
    padded_v = torch.cat((target_v, torch.zeros(1, 1, 2, 4)), dim=1)
    group = SimpleNamespace(all_gather=lambda tensor, dim: padded_k if tensor.data_ptr() == local_k.data_ptr() else padded_v)
    original_group = usp.get_sp_group
    usp.get_sp_group = lambda: group
    try:
        for rank in range(2):
            local_k = padded_k[:, rank * 3:(rank + 1) * 3].contiguous()
            local_v = padded_v[:, rank * 3:(rank + 1) * 3].contiguous()
            q = torch.randn(1, 3, 2, 4)
            actual = usp._target_attention(q, local_k, local_v, 2, (prefix_k, prefix_v), 5, {})
            expected = usp.optimized_attention(q.flatten(2), torch.cat((prefix_k, target_k), dim=1).flatten(2),
                                                torch.cat((prefix_v, target_v), dim=1).flatten(2), 2)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    finally:
        usp.get_sp_group = original_group


def test_qwen21_sharded_forward_matches_block_causal_attention():
    torch.manual_seed(1)
    prefix = torch.randn(1, 4, 4)
    image = torch.randn(1, 4, 1, 5)
    text_mask = torch.ones(2, 2, dtype=torch.bool).tril()
    segments = [(0, 2, text_mask), (2, 4, None), (4, 9, None)]

    class Block:
        def __init__(self):
            self.attn = SimpleNamespace(comfy_attention=SimpleNamespace(function=None))

        def __call__(self, states, mod, pe, attn_fn, prefix_len, options):
            qkv = states.unsqueeze(2)
            return states + attn_fn(qkv, qkv, qkv, 1)

    class Model:
        out_channels = 4

        def __init__(self):
            self.transformer_blocks = [Block(), Block()]
            self.modulation = lambda temb: torch.zeros(temb.shape[0], 16)
            self.time_text_embed = lambda timesteps, dtype: torch.zeros(timesteps.shape[0], 4)
            self.norm_out = lambda states, temb: states
            self.proj_out = lambda states: states

        def build_sequence(self, x, context, refs, slots):
            return torch.cat((prefix, x.flatten(2).transpose(1, 2)), dim=1), torch.empty(1, 9, 1, 2, 2, 2), segments

    model = Model()
    full = torch.cat((prefix, image.flatten(2).transpose(1, 2)), dim=1)
    block_inputs = []
    for block in model.transformer_blocks:
        block_inputs.append(full.clone())
        full = block(full, None, None, block_causal_attention(segments), 4, {})
    reference_target = full[:, 4:]

    original_group = usp.get_sp_group
    original_rank = usp.get_sequence_parallel_rank
    original_world = usp.get_sequence_parallel_world_size
    original_pad = usp.pad_to_world_size
    original_make = usp.comfy.model_prefetch.make_prefetch_queue
    original_pop = usp.comfy.model_prefetch.prefetch_queue_pop
    usp.get_sequence_parallel_world_size = lambda: 2
    usp.pad_to_world_size = lambda tensor, dim: (torch.cat((tensor, torch.zeros_like(tensor[:, :1])), dim=dim)
                                                 if tensor.shape[dim] % 2 else tensor, tensor.shape[dim])
    usp.comfy.model_prefetch.make_prefetch_queue = lambda *args: None
    usp.comfy.model_prefetch.prefetch_queue_pop = lambda *args: None
    try:
        for rank in range(2):
            calls = []

            def all_gather(tensor, dim):
                calls.append(tensor)
                if len(calls) <= 2 * len(block_inputs):
                    block_input = block_inputs[(len(calls) - 1) // 2][:, 4:]
                    return torch.cat((block_input, torch.zeros(1, 1, 4)), dim=1).unsqueeze(2)
                expected_shard = torch.cat((reference_target, torch.zeros(1, 1, 4)), dim=1)[:, rank * 3:(rank + 1) * 3]
                valid = min(3, 5 - rank * 3)
                torch.testing.assert_close(tensor[:, :valid], expected_shard[:, :valid], rtol=1e-4, atol=1e-4)
                return torch.cat((reference_target, torch.zeros(1, 1, 4)), dim=1)

            usp.get_sp_group = lambda: SimpleNamespace(all_gather=all_gather)
            usp.get_sequence_parallel_rank = lambda: rank
            result = usp.usp_dit_forward(model, image, torch.ones(1), prefix, transformer_options={})
            torch.testing.assert_close(result, reference_target.transpose(1, 2).reshape(1, 4, 1, 5), rtol=1e-4, atol=1e-4)
            assert len(calls) == 5
    finally:
        usp.get_sp_group = original_group
        usp.get_sequence_parallel_rank = original_rank
        usp.get_sequence_parallel_world_size = original_world
        usp.pad_to_world_size = original_pad
        usp.comfy.model_prefetch.make_prefetch_queue = original_make
        usp.comfy.model_prefetch.prefetch_queue_pop = original_pop


def test_qwen21_block_replacement_receives_full_sequence():
    torch.manual_seed(2)
    prefix = torch.randn(1, 2, 4)
    image = torch.randn(1, 4, 1, 3)
    segments = [(0, 2, torch.ones(2, 2, dtype=torch.bool).tril()), (2, 5, None)]
    calls = []

    def replace(args, extra):
        calls.append(args["img"].shape[1])
        out = extra["original_block"](args)
        out["img"] = out["img"] + 0.5
        return out

    class Block:
        def __call__(self, states, mod, pe, attn_fn, prefix_len, options):
            qkv = states.unsqueeze(2)
            return states + attn_fn(qkv, qkv, qkv, 1)

    class Model:
        out_channels = 4
        transformer_blocks = [Block()]
        modulation = staticmethod(lambda temb: torch.zeros(temb.shape[0], 16))
        time_text_embed = staticmethod(lambda timesteps, dtype: torch.zeros(timesteps.shape[0], 4))
        norm_out = staticmethod(lambda states, temb: states)
        proj_out = staticmethod(lambda states: states)

        def build_sequence(self, x, context, refs, slots):
            return torch.cat((prefix, x.flatten(2).transpose(1, 2)), dim=1), torch.empty(1, 5, 1, 2, 2, 2), segments

    model = Model()
    full = torch.cat((prefix, image.flatten(2).transpose(1, 2)), dim=1)
    reference = replace({"img": full, "pe": None, "transformer_options": {}},
                        {"original_block": lambda args: {"img": model.transformer_blocks[0](args["img"], None, None,
                         block_causal_attention(segments), 2, {})}})["img"][:, 2:]
    calls.clear()

    original_group = usp.get_sp_group
    original_rank = usp.get_sequence_parallel_rank
    original_world = usp.get_sequence_parallel_world_size
    original_pad = usp.pad_to_world_size
    original_make = usp.comfy.model_prefetch.make_prefetch_queue
    original_pop = usp.comfy.model_prefetch.prefetch_queue_pop
    usp.get_sequence_parallel_world_size = lambda: 2
    usp.pad_to_world_size = lambda tensor, dim: (torch.cat((tensor, torch.zeros_like(tensor[:, :1])), dim=dim)
                                                 if tensor.shape[dim] % 2 else tensor, tensor.shape[dim])
    usp.comfy.model_prefetch.make_prefetch_queue = lambda *args: None
    usp.comfy.model_prefetch.prefetch_queue_pop = lambda *args: None
    try:
        for rank in range(2):
            gathered = []

            def all_gather(tensor, dim):
                gathered.append(tensor)
                if len(gathered) == 1:
                    return torch.cat((full[:, 2:], torch.zeros(1, 1, 4)), dim=1)
                valid = min(2, 3 - rank * 2)
                torch.testing.assert_close(tensor[:, :valid], reference[:, rank * 2:rank * 2 + valid], rtol=1e-4, atol=1e-4)
                return torch.cat((reference, torch.zeros(1, 1, 4)), dim=1)

            usp.get_sp_group = lambda: SimpleNamespace(all_gather=all_gather)
            usp.get_sequence_parallel_rank = lambda: rank
            options = {"patches_replace": {"dit": {("single_block", 0): replace}}}
            output = usp.usp_dit_forward(model, image, torch.ones(1), prefix, transformer_options=options)
            torch.testing.assert_close(output, reference.transpose(1, 2).reshape(1, 4, 1, 3), rtol=1e-4, atol=1e-4)
            assert len(gathered) == 2
        assert calls == [5, 5]
    finally:
        usp.get_sp_group = original_group
        usp.get_sequence_parallel_rank = original_rank
        usp.get_sequence_parallel_world_size = original_world
        usp.pad_to_world_size = original_pad
        usp.comfy.model_prefetch.make_prefetch_queue = original_make
        usp.comfy.model_prefetch.prefetch_queue_pop = original_pop
