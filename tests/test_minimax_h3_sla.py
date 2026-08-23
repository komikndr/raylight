import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
SLA_DIR = ROOT / "src/raylight/diffusion_models/minimax/sla"
FORWARD_PATH = ROOT / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"
WORKER_PATH = ROOT / "src/raylight/distributed_worker/ray_worker.py"
SAMPLING_CONFIG_PATH = ROOT / "src/raylight/distributed_worker/sampling_config.py"


def _load_sla():
    """Load the sla package standalone (its imports are torch-only)."""
    spec = importlib.util.spec_from_file_location(
        "h3_sla_under_test", SLA_DIR / "__init__.py", submodule_search_locations=[str(SLA_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["h3_sla_under_test"] = module
    spec.loader.exec_module(module)
    return module


sla = _load_sla()


def _config(**overrides):
    values = dict(
        enabled=True,
        sparsity_ratio=0.90,
        block_size=64,
        min_seq_len=8192,
        dense_last_steps=0,
        protect_audio=True,
    )
    values.update(overrides)
    return sla.MiniMaxH3SLAConfig(**values)


def _runtime(**overrides):
    # 5-step schedule: sigmas 1.0 -> 0.0
    return _config(**overrides).create_runtime([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])


def _begin(runtime, sigma=0.8, prefix=128, global_len=36188, local_len=18094, rank=0, world_size=2):
    runtime.begin_step(sigma, prefix=prefix, global_len=global_len, local_len=local_len, rank=rank, world_size=world_size)


@pytest.fixture(autouse=True)
def _clear_active_runtime():
    yield
    sla.set_active_runtime(None)


def _hook(original):
    return sla.make_sla_ring_hook(original)


# -- disabled -------------------------------------------------------------------


def test_hook_without_runtime_is_pass_through():
    calls = []

    def original(q, k, v, *args, **kwargs):
        calls.append((q, k, v, args, kwargs))
        return "dense-out"

    assert sla.get_active_runtime() is None
    q = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16)  # far below min_seq_len
    out = _hook(original)(q, q, q, softmax_scale=0.088)
    assert out == "dense-out"
    assert len(calls) == 1


def test_disabled_config_is_pass_through_without_stats():
    calls = []

    def original(q, k, v, *args, **kwargs):
        calls.append(1)
        return "dense-out"

    runtime = _runtime(enabled=False)
    _begin(runtime)
    sla.set_active_runtime(runtime)

    q = torch.randn(1, 2048, 8, 128, dtype=torch.bfloat16)
    assert _hook(original)(q, q, q) == "dense-out"
    assert len(calls) == 1
    assert runtime.calls == 0
    assert runtime.dense_fallthroughs == 0


# -- threshold ------------------------------------------------------------------


def test_short_sequence_falls_back_to_dense():
    calls = []

    def original(q, k, v, *args, **kwargs):
        calls.append(1)
        return "dense-out"

    runtime = _runtime(min_seq_len=8192)
    _begin(runtime, global_len=2048, local_len=1024)
    sla.set_active_runtime(runtime)

    q = torch.randn(1, 2048, 8, 128, dtype=torch.bfloat16)  # 2048 < 8192
    assert _hook(original)(q, q, q) == "dense-out"
    assert len(calls) == 1
    assert runtime.calls == 0
    assert runtime.dense_fallthroughs == 1
    assert runtime.fallthrough_reasons == {"min_seq_len": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_long_sequence_with_min_seq_len_zero_is_sparse(monkeypatch):
    runtime = _runtime(min_seq_len=0)
    _begin(runtime, global_len=2048, local_len=1024)
    sla.set_active_runtime(runtime)

    captured = {}

    def fake_kernel(q, k, v, lut, topk, block_m, block_n, qk_scale=None):
        captured["topk"] = topk
        return torch.zeros_like(q)

    monkeypatch.setattr(sla, "block_sparse_attention", fake_kernel)
    q = torch.randn(1, 2048, 8, 128, device="cuda", dtype=torch.bfloat16)

    def dense_original(*args, **kwargs):
        pytest.fail("dense path must not run")

    out = sla.make_sla_ring_hook(dense_original)(q, q, q)
    assert isinstance(out, torch.Tensor)
    assert runtime.calls == 1
    # base top-k budget plus the pinned prefix blocks (protect_audio=True, prefix=128)
    base_topk = max(1, int((1.0 - 0.90) * (2048 // 64)))
    n_pinned = (128 + 64 - 1) // 64
    assert captured["topk"] == base_topk + n_pinned


# -- sparsity ---------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_block_map_keeps_requested_subset():
    torch.manual_seed(0)
    q = torch.randn(1, 1024, 8, 16, device="cuda")
    k = torch.randn(1, 1024, 8, 16, device="cuda")

    lut, topk = sla.get_block_map(q, k, topk_ratio=0.1, BLKQ=64, BLKK=64, protect_upto=0)

    nk = 1024 // 64
    assert topk == max(1, int(0.1 * nk))
    assert lut.dtype == torch.int32
    assert lut.is_contiguous()
    assert lut.shape == (1, 8, 1024 // 64, topk)
    for h in range(8):
        for m in range(lut.shape[2]):
            row = lut[0, h, m].tolist()
            assert len(set(row)) == topk  # unique key blocks
            assert all(0 <= b < nk for b in row)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_hook_records_sparse_stats(monkeypatch):
    runtime = _runtime(min_seq_len=0)
    _begin(runtime, prefix=128, global_len=2048, local_len=1024, world_size=2)
    sla.set_active_runtime(runtime)

    def fake_kernel(q, k, v, lut, topk, block_m, block_n, qk_scale=None):
        assert lut.shape == (1, 8, 2048 // 64, topk)
        return torch.zeros_like(q)

    monkeypatch.setattr(sla, "block_sparse_attention", fake_kernel)
    q = torch.randn(1, 2048, 8, 128, device="cuda", dtype=torch.bfloat16)
    out = _hook(lambda *a, **kw: pytest.fail("dense path must not run"))(q, q, q, softmax_scale=0.088)
    assert out.shape == q.shape

    nk = 2048 // 64
    base_topk = max(1, int(0.1 * nk))
    n_pinned = (128 + 64 - 1) // 64  # prefix pinned on top of the budget
    assert runtime.calls == 1
    assert runtime.seq == 2048
    assert runtime.kept_blocks == base_topk + n_pinned
    assert runtime.total_blocks == nk
    assert runtime.pinned_blocks == n_pinned
    assert runtime.heads == 8
    assert runtime.dense_fallthroughs == 0


# -- audio protection ---------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_protect_audio_pins_prefix_blocks():
    torch.manual_seed(1)
    q = torch.randn(1, 1024, 8, 16, device="cuda")
    k = torch.randn(1, 1024, 8, 16, device="cuda")

    base_topk = max(1, int(0.1 * (1024 // 64)))
    lut, topk = sla.get_block_map(q, k, topk_ratio=0.1, BLKQ=64, BLKK=64, protect_upto=150)

    n_pinned = (150 + 64 - 1) // 64  # 3
    assert topk == min(1024 // 64, base_topk + n_pinned)
    for h in range(8):
        for m in range(lut.shape[2]):
            row = set(lut[0, h, m].tolist())
            assert set(range(n_pinned)) <= row, "prefix blocks must be pinned into every query block"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_protect_audio_off_keeps_plain_budget():
    torch.manual_seed(1)
    q = torch.randn(1, 1024, 8, 16, device="cuda")
    k = torch.randn(1, 1024, 8, 16, device="cuda")

    lut_on, topk_on = sla.get_block_map(q, k, topk_ratio=0.1, BLKQ=64, BLKK=64, protect_upto=150)
    lut_off, topk_off = sla.get_block_map(q, k, topk_ratio=0.1, BLKQ=64, BLKK=64, protect_upto=0)

    assert topk_off == max(1, int(0.1 * (1024 // 64)))
    assert topk_on > topk_off


def test_hook_uses_prefix_only_when_protect_audio(monkeypatch):
    runtime = _runtime(min_seq_len=0, protect_audio=False)
    _begin(runtime, prefix=128, global_len=2048, local_len=1024)
    sla.set_active_runtime(runtime)

    seen = {}

    def fake_map(q, k, topk_ratio, blkq, blkk, protect_upto):
        seen["protect_upto"] = protect_upto
        return torch.zeros(1, 8, 32, 1, dtype=torch.int32), 1

    monkeypatch.setattr(sla, "get_block_map", fake_map)
    monkeypatch.setattr(sla, "block_sparse_attention", lambda *a, **kw: torch.zeros(1, 2048, 8, 128))
    q = torch.randn(1, 2048, 8, 128, dtype=torch.bfloat16)
    _hook(lambda *a, **kw: pytest.fail("dense path must not run"))(q, q, q)
    assert seen["protect_upto"] == 0


# -- multi-rank (Ulysses) consistency -------------------------------------------------


def _well_separated_qk(b=1, seq=1024, heads=8, dim=16, seed=42):
    """Key block j has a distinct mean vector, so per-head block scores are
    strictly ordered (no near-ties) and the top-k choice is deterministic."""
    g = torch.Generator().manual_seed(seed)
    nk = seq // 64
    u = torch.randn(1, 1, heads, dim, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    k = torch.empty(b, seq, heads, dim)
    for j in range(nk):
        block_mean = (j + 1.0) * u
        noise = 0.01 * torch.randn(b, 64, heads, dim, generator=g)
        k[:, j * 64:(j + 1) * 64] = block_mean.expand(b, 64, heads, dim) + noise
    q = torch.randn(b, seq, heads, dim, generator=g)
    return q.cuda().contiguous(), k.cuda().contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_multirank_selection_matches_single_gpu_per_head():
    """Ulysses W=2: after the all-to-all, rank r owns heads [r*H/2, (r+1)*H/2)
    over the FULL global sequence. The per-head SLA selection must be identical
    to single-GPU SLA for that head -- i.e. independent of which rank computes
    it and of the local shard."""
    q, k = _well_separated_qk()
    topk_ratio = 0.25
    lut_ref, topk_ref = sla.get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64, protect_upto=100)

    world_size = 2
    per_rank = q.shape[2] // world_size
    for rank in range(world_size):
        h0 = rank * per_rank
        q_r = q[:, :, h0:h0 + per_rank].contiguous()
        k_r = k[:, :, h0:h0 + per_rank].contiguous()
        lut_r, topk_r = sla.get_block_map(q_r, k_r, topk_ratio, BLKQ=64, BLKK=64, protect_upto=100)

        assert topk_r == topk_ref
        for local_h in range(per_rank):
            global_h = h0 + local_h
            ref_row = sorted(lut_ref[0, global_h].flatten().tolist())
            rank_row = sorted(lut_r[0, local_h].flatten().tolist())
            assert rank_row == ref_row, (
                f"rank {rank} head {local_h} (global {global_h}) selection diverges from single-GPU"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton block map")
def test_multirank_prefix_pinning_is_global_on_every_rank():
    q, k = _well_separated_qk()
    world_size = 2
    per_rank = q.shape[2] // world_size
    n_pinned = (100 + 64 - 1) // 64  # 2

    for rank in range(world_size):
        h0 = rank * per_rank
        lut_r, topk_r = sla.get_block_map(
            q[:, :, h0:h0 + per_rank].contiguous(), k[:, :, h0:h0 + per_rank].contiguous(),
            0.25, BLKQ=64, BLKK=64, protect_upto=100,
        )
        for local_h in range(per_rank):
            for m in range(lut_r.shape[2]):
                assert set(range(n_pinned)) <= set(lut_r[0, local_h, m].tolist())


# -- dense last steps ------------------------------------------------------------------


def test_dense_last_steps_window():
    runtime = _runtime(dense_last_steps=1)  # schedule has n_steps=5
    _begin(runtime, sigma=0.2)  # step index 4 == n_steps - 1 -> last step
    assert runtime.dense_this_step is True

    _begin(runtime, sigma=0.4)  # step index 3
    assert runtime.dense_this_step is False


def test_dense_last_steps_zero_never_dense():
    runtime = _runtime(dense_last_steps=0)
    _begin(runtime, sigma=0.0)
    assert runtime.dense_this_step is False


def test_hook_honours_dense_last_steps(monkeypatch):
    calls = []

    def original(q, k, v, *args, **kwargs):
        calls.append(1)
        return "dense-out"

    runtime = _runtime(min_seq_len=0, dense_last_steps=1)
    _begin(runtime, sigma=0.2)  # last step
    sla.set_active_runtime(runtime)

    q = torch.randn(1, 2048, 8, 128, dtype=torch.bfloat16)
    assert _hook(original)(q, q, q) == "dense-out"
    assert runtime.dense_fallthroughs == 1
    assert runtime.fallthrough_reasons == {"dense_last_steps": 1}


# -- worker lifecycle --------------------------------------------------------------------


def _load_worker_method(name, **extra_globals):
    tree = ast.parse(WORKER_PATH.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = {
        "H3_SLA_CONFIG_KEY": "minimax_h3_sla_config",
        "H3_SLA_RUNTIME_KEY": "minimax_h3_sla",
    }
    namespace.update(extra_globals)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(WORKER_PATH), "exec"),
        namespace,
    )
    return namespace


def test_worker_configure_sets_config_and_clears_state():
    cleared = []
    ns = _load_worker_method("configure_minimax_h3_sla", h3_sla_set_active_runtime=cleared.append)
    configure = ns["configure_minimax_h3_sla"]

    class _Model:
        def __init__(self):
            self.model_options = {}

    class _Worker:
        def __init__(self):
            self.model = _Model()

    worker = _Worker()
    stale = _runtime()
    worker.model.model_options["transformer_options"] = {"minimax_h3_sla": stale}

    config = {
        "enabled": True,
        "sparsity_ratio": 0.9,
        "block_size": 64,
        "min_seq_len": 8192,
        "dense_last_steps": 0,
        "protect_audio": True,
        "debug": False,
    }
    configure(worker, config)

    options = worker.model.model_options["transformer_options"]
    assert options["minimax_h3_sla_config"] == config
    assert "minimax_h3_sla" not in options
    assert cleared == [None]


def test_sampling_dispatch_pushes_sla_config():
    source = SAMPLING_CONFIG_PATH.read_text()
    assert "SLA_ACTORS_CONFIG_KEY" in source
    assert "DISABLED_H3_SLA_CONFIG" in source
    assert "actor.configure_minimax_h3_block_cache.remote(config)" in source
    assert "actor.configure_minimax_h3_sla.remote(sla_config)" in source


# -- forward wiring ------------------------------------------------------------------------


def test_forward_publishes_sla_state_to_the_hook():
    source = FORWARD_PATH.read_text()
    assert 'transformer_options.get(h3_sla.RUNTIME_KEY)' in source
    assert "h3_sla.set_active_runtime(None)" in source
    assert "sla_runtime.begin_step(" in source
    assert 'seg_kind == "video"' in source
    assert "global_len=layout.seq_len" in source
    assert "local_len=h.shape[0]" in source


def test_hook_installed_on_the_xfuser_instance():
    source = FORWARD_PATH.read_text()
    assert "h3_sla.install_on(xfuser_attn.get_last_xfuser_attention())" in source


# -- node registration -------------------------------------------------------------------------


def _load_nodes_module():
    src = str(ROOT / "src")
    comfy_root = str(ROOT.parents[1])
    for path in (src, comfy_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    import raylight.comfy_extra_dist.nodes_minimax_h3 as module

    return module


def test_node_registered_with_expected_signature_and_defaults():
    module = _load_nodes_module()

    assert "RayMiniMaxH3SLAAttention" in module.NODE_CLASS_MAPPINGS
    assert module.NODE_DISPLAY_NAME_MAPPINGS["RayMiniMaxH3SLAAttention"] == "MiniMax H3 SLA Attention (Ray USP)"

    node_cls = module.RayMiniMaxH3SLAAttention
    assert node_cls.RETURN_TYPES == ("RAY_ACTORS",)
    assert node_cls.FUNCTION == "patch"

    inputs = node_cls.INPUT_TYPES()
    required = inputs["required"]
    assert list(required.keys()) == [
        "ray_actors", "enabled", "sparsity_ratio", "block_size",
        "min_seq_len", "dense_last_steps", "protect_audio",
    ]
    assert required["enabled"][1]["default"] is False
    assert required["sparsity_ratio"][1]["default"] == 0.90
    assert required["block_size"] == (["64", "128"], {"default": "64"})
    assert required["min_seq_len"][1]["default"] == 8192
    assert required["dense_last_steps"][1]["default"] == 0
    assert required["protect_audio"][1]["default"] is True
    assert inputs["optional"]["debug"][1]["default"] is False


# -- stats / logging ------------------------------------------------------------------------------


def test_stats_accumulate_across_calls():
    runtime = _runtime(min_seq_len=0)
    _begin(runtime, prefix=128, global_len=1024, local_len=512, world_size=2)
    runtime.record_sparse(1024, 8, 16, 2, 28)
    runtime.record_sparse(1024, 8, 16, 2, 28)
    assert runtime.calls == 2
    assert runtime.kept_blocks == 16
    assert runtime.total_blocks == 32
    assert runtime.pinned_blocks == 4


def test_summary_log_reports_required_fields(capsys):
    runtime = _runtime(min_seq_len=0)
    _begin(runtime, prefix=128, global_len=36188, local_len=18094, rank=0, world_size=2)
    runtime.record_sparse(36188, 122, 566, 2, 28)
    runtime.log_summary()

    out = capsys.readouterr().out
    assert "S_global=36188" in out
    assert "S_local=18094" in out
    assert "blocks 122/566 kept" in out
    assert "78.4% sparse" in out
    assert "asked 90%" in out
    assert "2 pinned" in out
    assert "BLK=64x64" in out
    assert "0 dense fall-throughs" in out
    assert "rank0/world2" in out


def test_summary_warns_when_never_invoked(capsys):
    runtime = _runtime(min_seq_len=1 << 30)
    _begin(runtime, prefix=0, global_len=64, local_len=32, rank=0, world_size=2)
    runtime.record_dense_fallthrough("min_seq_len")
    runtime.log_summary()

    out = capsys.readouterr().out
    assert "never invoked" in out


# -- Triton kernel (CUDA only) --------------------------------------------------------------------


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton SLA kernel")


@cuda_required
def test_mean_pool_divides_each_block_by_its_valid_token_count():
    block_map = importlib.import_module(f"{sla.__name__}.block_map")
    x = torch.empty(1, 129, 1, 16, device="cuda", dtype=torch.bfloat16)
    x[:, :64] = 1.0
    x[:, 64:128] = 2.0
    x[:, 128:] = 3.0

    pooled = block_map.mean_pool(x.contiguous(), 64)

    assert torch.equal(pooled[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0], device="cuda"))


@cuda_required
def test_sparse_kernel_matches_dense_when_all_blocks_kept():
    if sla.block_sparse_attention is None:
        pytest.skip(f"triton kernel unavailable: {sla._KERNEL_IMPORT_ERROR}")

    torch.manual_seed(0)
    b, seq, heads, dim = 1, 512, 4, 64
    q = torch.randn(b, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, seq, heads, dim, device="cuda", dtype=torch.bfloat16)

    lut, topk = sla.get_block_map(q, k, 1.0, BLKQ=64, BLKK=64, protect_upto=0)
    assert topk == seq // 64  # all blocks kept -> sparse kernel must equal dense attention

    out = sla.block_sparse_attention(q, k, v, lut, topk, 64, 64)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@cuda_required
def test_sparse_kernel_with_real_sparsity_runs_and_shapes():
    if sla.block_sparse_attention is None:
        pytest.skip(f"triton kernel unavailable: {sla._KERNEL_IMPORT_ERROR}")

    torch.manual_seed(1)
    b, seq, heads, dim = 1, 1024, 8, 128
    q = torch.randn(b, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, seq, heads, dim, device="cuda", dtype=torch.bfloat16)

    lut, topk = sla.get_block_map(q, k, 0.1, BLKQ=64, BLKK=64, protect_upto=128)
    out = sla.block_sparse_attention(q, k, v, lut, topk, 64, 64)

    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()
