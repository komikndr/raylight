# From https://github.com/komikndr/raylight/pull/116
# Thanks to https://github.com/Karmabu, Karmabu
# Doing this since the original PR implementation quite disruptive when it comes to code base,
import logging
import torch

from .registry import register_inner_attention

H3_SLA_PREPARED_KEY = "raylight_minimax_h3_sla_prepared"
_H3_HEAD_DIM = 128


def _values(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().reshape(-1).tolist()
    return list(value)


def _dense_tail(options, count):
    if count <= 0:
        return False
    schedule = _values(options.get("sample_sigmas"))[:-1]
    current = _values(options.get("sigmas"))
    n_steps = len(schedule)
    if not schedule or not current:
        return False
    try:
        index = min(range(len(schedule)), key=lambda i: abs(float(schedule[i]) - float(current[0])))
    except (TypeError, ValueError):
        return False
    return index >= max(0, n_steps - count)


def _ranges(payload, protect_audio):
    if not isinstance(payload, dict):
        return None
    layout = payload.get("layout")
    if layout is None or not hasattr(layout, "segments"):
        return None
    result = []
    tags = payload.get("text_token_tags")
    tags = _values(tags)
    for segment in layout.segments:
        if len(segment) == 3:
            start, stop, kind = segment
        else:
            return None
        if not isinstance(start, int) or not isinstance(stop, int) or stop < start:
            return None
        if kind == "text" and (len(tags) != stop - start or any(tag not in (0, 1) for tag in tags)):
            result.append((start, stop))
        elif kind == "text":
            run = None
            for index, tag in enumerate(tags[:stop - start] + [None]):
                if tag == 1 and run is None:
                    run = start + index
                elif tag != 1 and run is not None:
                    result.append((run, start + index))
                    run = None
        elif protect_audio and kind in ("audio", "ref_audio", "cond_audio"):
            result.append((start, stop))
    return result


@register_inner_attention("raylight:minimax_h3_sla")
class MiniMaxH3SLA:
    supports_ring = False

    def __init__(self, enabled=True, sparsity_ratio=0.90, block_size=64, min_seq_len=8192,
                 dense_last_steps=1, protect_audio=True):
        self.enabled = enabled
        self.sparsity_ratio = float(sparsity_ratio)
        self.block_size = int(block_size)
        self.min_seq_len = int(min_seq_len)
        self.dense_last_steps = int(dense_last_steps)
        self.protect_audio = protect_audio
        self._kernel = None
        self._kernel_checked = False
        self._logged_failure = False

    def prepare(self, transformer_options=None, minimax_payload=None):
        options = transformer_options or {}
        payload = minimax_payload or options.get("minimax_payload") or {}
        return {
            "dense_tail": _dense_tail(options, self.dense_last_steps),
            "protected_ranges": _ranges(payload, self.protect_audio),
        }

    def _load_kernel(self):
        if not self._kernel_checked:
            self._kernel_checked = True
            try:
                from ._h3_sla_kernel import forward
            except ImportError:
                pass
            else:
                self._kernel = forward
        return self._kernel

    def __call__(self, dense_attention, q, k, v, *, transformer_options, **kwargs):
        prepared = transformer_options.get(H3_SLA_PREPARED_KEY)
        if (not self.enabled or not isinstance(prepared, dict) or prepared.get("dense_tail", True) or
                prepared.get("protected_ranges") is None or q.device.type != "cuda" or
                q.dtype not in (torch.float16, torch.bfloat16) or q.ndim != 4 or q.shape[-1] != _H3_HEAD_DIM or
                q.shape[1] < self.min_seq_len or
                q.shape != k.shape or q.dtype != k.dtype or q.dtype != v.dtype):
            return dense_attention(q, k, v, **kwargs)
        kernel = self._load_kernel()
        if kernel is None:
            return dense_attention(q, k, v, **kwargs)
        try:
            return kernel(q, k, v, sparsity_ratio=self.sparsity_ratio, block_size=self.block_size,
                          protected_ranges=prepared["protected_ranges"], **kwargs)
        except Exception as error:
            if not self._logged_failure:
                logging.warning("[Raylight] H3 SLA kernel failed; using dense attention (%s)", error)
                self._logged_failure = True
            return dense_attention(q, k, v, **kwargs)
