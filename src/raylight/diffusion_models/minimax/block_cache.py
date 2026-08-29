import math
from dataclasses import dataclass
from typing import Optional

import torch


CONFIG_KEY = "minimax_h3_block_cache_config"
RUNTIME_KEY = "minimax_h3_block_cache"
ACTORS_CONFIG_KEY = "_raylight_minimax_h3_block_cache_config"


@dataclass(frozen=True)
class MiniMaxH3BlockCacheConfig:
    sigma_threshold: float
    start_percent: float
    end_percent: float
    max_cached_steps: int
    cache_depth: float
    debug: bool = False

    @property
    def enabled(self):
        return self.sigma_threshold > 0.0 and self.max_cached_steps > 0 and self.cache_depth > 0.0

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "sigma_threshold": self.sigma_threshold,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "max_cached_steps": self.max_cached_steps,
            "cache_depth": self.cache_depth,
            "debug": self.debug,
        }

    def create_runtime(self, sigmas):
        if isinstance(sigmas, torch.Tensor):
            schedule = sigmas.detach().flatten().float().cpu().tolist()
        else:
            schedule = [float(sigma) for sigma in sigmas]
        return MiniMaxH3BlockCacheRuntime(self, schedule)


@dataclass
class MiniMaxH3BlockCachePlan:
    mode: str
    prefix: int
    residual: Optional[torch.Tensor]


class _StreamState:
    def __init__(self):
        self.last_sigma = None
        self.last_mode = "FULL"
        self.consecutive_cached = 0
        self.signature = None
        self.residual = None


class MiniMaxH3BlockCacheRuntime:
    def __init__(self, config, schedule):
        self.config = config
        self.schedule = schedule
        self.streams = {}
        self.full_steps = 0
        self.cache_steps = 0
        self.executed_blocks = 0
        self.avoided_blocks = 0
        self.rank = None
        self.started = False

    @property
    def enabled(self):
        return self.config.enabled

    @staticmethod
    def conditioning_key(transformer_options):
        uuids = transformer_options.get("uuids")
        if not uuids:
            return None
        return tuple(str(uuid) for uuid in uuids)

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
        if self.config.debug:
            print(
                f"[H3 Block Cache][rank{rank}] enabled={self.enabled} "
                f"threshold={self.config.sigma_threshold:.2f} start={self.config.start_percent:.2f} "
                f"end={self.config.end_percent:.2f} mcs={self.config.max_cached_steps} depth={self.config.cache_depth:.2f}",
                flush=True,
            )

    def prefix_blocks(self, block_count):
        return max(1, min(block_count - 1, math.ceil(block_count * (1.0 - self.config.cache_depth))))

    def _schedule_step(self, sigma):
        if len(self.schedule) <= 1:
            return 0
        total_steps = len(self.schedule) - 1
        return min(range(total_steps), key=lambda index: abs(self.schedule[index] - sigma))

    def _schedule_position(self, sigma):
        if len(self.schedule) <= 1:
            return 0.0
        return self._schedule_step(sigma) / (len(self.schedule) - 1)

    def plan(self, key, sigma, signature, block_count, rank):
        self.log_start(rank)
        self.rank = rank
        prefix = self.prefix_blocks(block_count)
        step = self._schedule_step(sigma)
        if key is None:
            if self.config.debug:
                print(
                    f"[H3 Block Cache][rank{rank}] step={step} sigma={sigma:.6f} mode=FULL "
                    f"prefix_blocks={block_count}/{block_count} cached_steps_count=0 residual_valid=False "
                    "reason=missing_conditioning_uuid",
                    flush=True,
                )
            return MiniMaxH3BlockCachePlan("FULL", prefix, None)

        state = self.streams.setdefault(key, _StreamState())
        same_step = state.last_sigma is not None and math.isclose(sigma, state.last_sigma, rel_tol=0.0, abs_tol=1e-7)
        signature_matches = state.signature == signature
        residual_valid = state.residual is not None and signature_matches
        if not signature_matches:
            state.residual = None
            state.signature = None
            state.consecutive_cached = 0

        if same_step and signature_matches:
            mode = state.last_mode
        else:
            delta = math.inf if state.last_sigma is None else abs(state.last_sigma - sigma)
            position = self._schedule_position(sigma)
            can_cache = (
                residual_valid
                and self.config.start_percent <= position <= self.config.end_percent
                and delta < self.config.sigma_threshold
                and state.consecutive_cached < self.config.max_cached_steps
            )
            mode = "CACHE" if can_cache else "FULL"
            state.last_sigma = sigma
            state.last_mode = mode
            if mode == "CACHE":
                state.consecutive_cached += 1
                self.cache_steps += 1
                self.executed_blocks += prefix
                self.avoided_blocks += block_count - prefix
            else:
                state.consecutive_cached = 0
                self.full_steps += 1
                self.executed_blocks += block_count

        residual = state.residual if mode == "CACHE" else None
        if self.config.debug:
            executed_prefix = prefix if mode == "CACHE" else block_count
            print(
                f"[H3 Block Cache][rank{rank}] step={step} sigma={sigma:.6f} mode={mode} "
                f"prefix_blocks={executed_prefix}/{block_count} cached_steps_count={state.consecutive_cached} "
                f"residual_valid={residual_valid}",
                flush=True,
            )
        return MiniMaxH3BlockCachePlan(mode, prefix, residual)

    def store_residual(self, key, signature, residual):
        if key is None:
            return
        state = self.streams[key]
        state.signature = signature
        state.residual = residual.detach()

    def log_summary(self):
        if self.rank != 0 or self.full_steps + self.cache_steps == 0:
            return
        print(
            f"[H3 Block Cache] FULL={self.full_steps} CACHE={self.cache_steps} "
            f"effective_blocks={self.executed_blocks} skipped_blocks={self.avoided_blocks}",
            flush=True,
        )

    def clear(self):
        self.streams.clear()
        self.full_steps = 0
        self.cache_steps = 0
        self.executed_blocks = 0
        self.avoided_blocks = 0
        self.rank = None
        self.started = False
