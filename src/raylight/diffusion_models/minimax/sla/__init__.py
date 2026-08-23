"""MiniMax H3 SLA (Sparse Linear Attention) for the RayLight USP path.

RayLight port of PlagueKind's ``H3 SLA Attention`` node
(``ComfyUI-PlagueKind-Nodes/ComfyUI-H3-SLA-Attention``, Apache-2.0), which
itself vendors the block-sparse kernel from LightX2V (Apache-2.0). The
attention kernel and block selection (``sla/kernel.py`` and
``sla/block_map.py``) are vendored verbatim.

Why the hook lives where it does
--------------------------------
The original node hooks ComfyUI's ``optimized_attention_override`` and sees
the FULL packed sequence on a single GPU. Under RayLight USP (Ulysses=2,
Ring=1) each rank only holds a shard of the sequence until xFuser's Ulysses
all-to-all inside ``xFuserLongContextAttention`` gathers it: after that
collective every rank owns the FULL global sequence for its own head slice
(H/W heads).

The SLA block selection is per-head and depends only on (pooled q of this
head, pooled k of this head) over the full sequence, so computing it after
the Ulysses all-to-all yields exactly what single-GPU SLA computes for those
heads: no extra collective is needed, and the ``[text | cond | audio]``
prefix pinning operates on global token indices (identical on every rank).

If the ring degree were > 1 the sequence would still be sharded across the
ring group and per-shard selection would be wrong; in that case (as with any
unsupported shape: quant descales, joint attention, causal, bad dtype) the
hook falls back to the original dense attention and counts a dense
fall-through instead of pretending to be sparse.

Sage interaction: the sparse path runs the Triton SLA kernel (displacing the
dense backend for that call, exactly like the original node displaces
ComfyUI's optimized attention); every dense path calls the original
``ring_attn_fn`` (SAGE_AUTO) completely unchanged. Sage is never disabled or
replaced.

State flow (mirrors the MiniMax H3 Block Cache pattern)
-------------------------------------------------------
The xFuser call chain does not carry ``transformer_options``, so the
attention hook cannot read the model options directly. Per-sample state
therefore travels through a worker-local singleton (``WORKER_STATE``):

1. The node installs the config on the worker model's ``transformer_options``
   and an OUTER_SAMPLE wrapper that creates a per-sample runtime (from the
   sampler sigmas).
2. ``configure_sampling_features`` re-pushes the config to the workers at
   sampler dispatch (same backstop as the block cache).
3. ``usp_dit_forward`` publishes the current runtime to ``WORKER_STATE``
   once per step, after updating the prefix (start of the video segment in
   the packed layout) and the dense-last-steps flag.
4. The wrapped ``ring_attn_fn`` reads ``WORKER_STATE.runtime`` on every
   attention call.

The singleton is per-worker-process state: each RayWorker runs one model and
one xFuser instance, so there is no cross-model ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

CONFIG_KEY = "minimax_h3_sla_config"
RUNTIME_KEY = "minimax_h3_sla"
ACTORS_CONFIG_KEY = "_raylight_minimax_h3_sla_config"

# H3 is the only model this hook supports; head_dim 128 matches its config.
_H3_HEAD_DIM = 128
_OK_DTYPES = (torch.bfloat16, torch.float16)

try:  # Triton is only needed for the sparse kernel, not for selection.
    from .kernel import block_sparse_attention
    _KERNEL_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 -- a missing kernel must not kill the run
    block_sparse_attention = None
    _KERNEL_IMPORT_ERROR = _exc

from .block_map import get_block_map


@dataclass(frozen=True)
class MiniMaxH3SLAConfig:
    enabled: bool
    sparsity_ratio: float
    block_size: int
    min_seq_len: int
    dense_last_steps: int
    protect_audio: bool
    debug: bool = False

    @property
    def blkk(self):
        # BLKK=64 is not a typo (upstream): on sm_120 the 128x128 tile needs
        # 160 KB of shared memory against a ~99 KB limit and cannot launch at
        # all; 128x64 both fits and measured fastest.
        return 64 if self.block_size == 128 else self.block_size

    def to_dict(self):
        return {
            "enabled": bool(self.enabled),
            "sparsity_ratio": float(self.sparsity_ratio),
            "block_size": int(self.block_size),
            "min_seq_len": int(self.min_seq_len),
            "dense_last_steps": int(self.dense_last_steps),
            "protect_audio": bool(self.protect_audio),
            "debug": bool(self.debug),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            enabled=bool(data.get("enabled", False)),
            sparsity_ratio=float(data.get("sparsity_ratio", 0.90)),
            block_size=int(data.get("block_size", 64)),
            min_seq_len=int(data.get("min_seq_len", 8192)),
            dense_last_steps=int(data.get("dense_last_steps", 0)),
            protect_audio=bool(data.get("protect_audio", True)),
            debug=bool(data.get("debug", False)),
        )

    def create_runtime(self, sigmas):
        if isinstance(sigmas, torch.Tensor):
            schedule = sigmas.detach().flatten().float().cpu().tolist()
        else:
            schedule = [float(sigma) for sigma in sigmas]
        return MiniMaxH3SLARuntime(self, schedule)


class MiniMaxH3SLARuntime:
    """Per-sample SLA state, created by the OUTER_SAMPLE wrapper on each worker."""

    def __init__(self, config, schedule):
        self.config = config
        self.schedule = list(schedule)
        # per-step values, published to the attention hook by usp_dit_forward
        self.step = 0
        self.prefix = 0
        self.dense_this_step = False
        self.global_len = 0
        self.local_len = 0
        self.rank = None
        self.world_size = None
        # stats (per worker rank; the summary is logged from rank 0)
        self.calls = 0
        self.seq = 0
        self.kept_blocks = 0
        self.total_blocks = 0
        self.pinned_blocks = 0
        self.heads = 0
        self.dense_fallthroughs = 0
        self.fallthrough_reasons = {}
        self.kernel_failure = None
        self.started = False

    @property
    def enabled(self):
        return self.config.enabled

    # -- step tracking ------------------------------------------------------

    def _schedule_step(self, sigma):
        if len(self.schedule) <= 1:
            return 0
        total_steps = len(self.schedule) - 1
        return min(range(total_steps), key=lambda index: abs(self.schedule[index] - sigma))

    def begin_step(self, sigma, prefix, global_len, local_len, rank, world_size):
        """Called once per model forward (per step) by usp_dit_forward."""
        self.log_start(rank)
        self.rank = rank
        self.world_size = world_size
        self.global_len = int(global_len)
        self.local_len = int(local_len)
        self.prefix = int(prefix)
        self.step = self._schedule_step(sigma)
        n_steps = max(1, len(self.schedule) - 1)
        self.dense_this_step = (
            self.config.dense_last_steps > 0 and self.step >= n_steps - self.config.dense_last_steps
        )
        if self.config.debug:
            print(
                f"[H3 SLA][rank{rank}] step={self.step} sigma={sigma:.6f} "
                f"dense={self.dense_this_step} prefix={self.prefix} "
                f"S_global={self.global_len} S_local={self.local_len} W={world_size}",
                flush=True,
            )

    # -- stats ---------------------------------------------------------------

    def record_sparse(self, seq_len, topk, total_blocks, pinned_blocks, heads):
        self.calls += 1
        self.seq = int(seq_len)
        self.kept_blocks += int(topk)
        self.total_blocks += int(total_blocks)
        self.pinned_blocks += int(pinned_blocks)
        self.heads = int(heads)

    def record_dense_fallthrough(self, reason):
        self.dense_fallthroughs += 1
        bucket = reason.split(":", 1)[0]
        self.fallthrough_reasons[bucket] = self.fallthrough_reasons.get(bucket, 0) + 1

    # -- logging ---------------------------------------------------------------

    def log_start(self, rank=None):
        if self.started:
            return
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        self.rank = rank
        self.started = True
        cfg = self.config
        print(
            f"[H3 SLA][rank{rank}] installed | sparsity={cfg.sparsity_ratio:.2f} "
            f"BLK={cfg.block_size}x{cfg.blkk} | min_seq_len={cfg.min_seq_len} "
            f"dense_last_steps={cfg.dense_last_steps} | protect_audio={cfg.protect_audio}",
            flush=True,
        )

    def log_summary(self):
        if self.rank != 0 or (self.calls + self.dense_fallthroughs) == 0:
            return
        real = 1.0 - (self.kept_blocks / self.total_blocks) if self.total_blocks else 0.0
        cfg = self.config
        print(
            f"[H3 SLA] {self.calls} calls x {self.heads} heads | "
            f"S_global={self.global_len} S_local={self.local_len} (padded S={self.seq}) | "
            f"blocks {self.kept_blocks}/{self.total_blocks} kept "
            f"({real * 100.0:.1f}% sparse, asked {cfg.sparsity_ratio * 100.0:.0f}%) | "
            f"{self.pinned_blocks} pinned | BLK={cfg.block_size}x{cfg.blkk} | "
            f"{self.dense_fallthroughs} dense fall-throughs | "
            f"rank0/world{self.world_size}",
            flush=True,
        )
        if self.calls == 0:
            print(
                f"[H3 SLA] WARNING: patch active but never invoked -- attention was "
                f"NOT sparsified ({self.dense_fallthroughs} dense fall-throughs "
                f"{self.fallthrough_reasons}).",
                flush=True,
            )
        if self.kernel_failure is not None:
            print(f"[H3 SLA] WARNING: kernel fell back to dense at least once: {self.kernel_failure}", flush=True)

    def clear(self):
        self.calls = 0
        self.seq = 0
        self.kept_blocks = 0
        self.total_blocks = 0
        self.pinned_blocks = 0
        self.heads = 0
        self.dense_fallthroughs = 0
        self.fallthrough_reasons = {}
        self.kernel_failure = None
        self.started = False


class _WorkerSLAState:
    """Per-worker-process state bridging usp_dit_forward and the attention hook."""

    def __init__(self):
        self.runtime: Optional[MiniMaxH3SLARuntime] = None


WORKER_STATE = _WorkerSLAState()


def get_active_runtime():
    return WORKER_STATE.runtime


def set_active_runtime(runtime):
    WORKER_STATE.runtime = runtime


# -- the attention hook --------------------------------------------------------


def make_sla_ring_hook(original):
    """Wrap xFuser's ``ring_attn_fn`` with the SLA sparse path.

    ``original`` is called with ``(q, k, v, **kwargs)`` where q/k/v are
    ``(B, S_full, H_local, D)`` -- the full global sequence for this rank's
    head slice, after the Ulysses all-to-all. Every guard that cannot prove
    the sparse path is semantically valid falls back to ``original`` (dense)
    and counts a fall-through; the hook never fabricates sparsity.
    """

    def sla_ring_attn(q, k, v, *args, **kwargs):
        runtime = WORKER_STATE.runtime
        if runtime is None or not runtime.config.enabled:
            return original(q, k, v, *args, **kwargs)

        config = runtime.config

        def dense(reason=None):
            if reason is not None:
                runtime.record_dense_fallthrough(reason)
            return original(q, k, v, *args, **kwargs)

        # Anything that is not the plain full-sequence bidirectional MHA path
        # goes straight through dense.
        if kwargs.get("attn_processor") is not None or kwargs.get("causal"):
            return dense("unsupported attention flags")
        if kwargs.get("joint_tensor_key") is not None or kwargs.get("joint_tensor_value") is not None:
            return dense("joint attention")
        if (
            kwargs.get("q_descale") is not None
            or kwargs.get("k_descale") is not None
            or kwargs.get("v_descale") is not None
        ):
            return dense("quantized descale")

        group = kwargs.get("group")
        if group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            # ring degree > 1: the sequence is still sharded across the ring
            # group, so a per-rank selection would be wrong -> stay dense.
            if torch.distributed.get_world_size(group) > 1:
                return dense("ring world size > 1")

        if q.ndim != 4 or q.shape[-1] != _H3_HEAD_DIM or q.dtype not in _OK_DTYPES:
            return dense("unexpected tensor layout/dtype")

        seq_len = q.shape[1]
        if seq_len < config.min_seq_len:
            return dense("min_seq_len")
        if runtime.dense_this_step:
            return dense("dense_last_steps")
        if block_sparse_attention is None:
            return dense(f"kernel unavailable: {_KERNEL_IMPORT_ERROR}")

        try:
            if not q.is_contiguous():
                q = q.contiguous()
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()

            # Pin the [text | cond | audio] prefix into every query's
            # selection. 0 when the layout is unavailable (usp_dit_forward
            # never sets one), which simply disables the protection rather
            # than guessing.
            prefix = int(runtime.prefix) if config.protect_audio else 0
            if prefix >= seq_len:
                prefix = 0

            topk_ratio = 1.0 - config.sparsity_ratio
            lut, topk = get_block_map(q, k, topk_ratio, config.block_size, config.blkk, protect_upto=prefix)
            out = block_sparse_attention(
                q, k, v, lut, topk, config.block_size, config.blkk,
                qk_scale=kwargs.get("softmax_scale"),
            )

            nk = (seq_len + config.blkk - 1) // config.blkk
            n_pinned = (prefix + config.blkk - 1) // config.blkk if prefix > 0 else 0
            runtime.record_sparse(seq_len, topk, nk, n_pinned, k.shape[2])
            return out

        except Exception as exc:  # noqa: BLE001 -- a bad kernel must not kill the run
            if runtime.kernel_failure is None:
                runtime.kernel_failure = "%s: %s" % (exc.__class__.__name__, exc)
            return dense(f"kernel: {exc.__class__.__name__}")

    return sla_ring_attn


def install_on(xfuser_instance):
    """Wrap the ``ring_attn_fn`` of a ``xFuserLongContextAttention`` instance.

    Idempotent per instance: the hook is installed once at import time of
    ``minimax/xdit_context_parallel.py`` and stays a no-op (dense pass-through)
    until a sample publishes an enabled runtime to ``WORKER_STATE``.
    """
    if getattr(xfuser_instance, "_h3_sla_hooked", False):
        return xfuser_instance
    xfuser_instance.ring_attn_fn = make_sla_ring_hook(xfuser_instance.ring_attn_fn)
    xfuser_instance._h3_sla_hooked = True
    return xfuser_instance
