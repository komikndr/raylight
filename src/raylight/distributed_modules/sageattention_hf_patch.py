import logging
import inspect
import sys
import types
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PATCHED: set[str] = set()


def _supports_argument(fn, name: str) -> bool:
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
        return name in params
    except (TypeError, ValueError):
        code = getattr(fn, "__code__", None)
        return bool(code and name in code.co_varnames)


def _ensure_sageattention_core():
    module = sys.modules.get("sageattention")
    if module is None:
        try:
            import sageattention as module  # type: ignore
        except ImportError:
            module = types.ModuleType("sageattention")
            sys.modules["sageattention"] = module

    core = getattr(module, "core", None)
    if core is None:
        try:
            from sageattention import core as core_module  # type: ignore
        except ImportError:
            core_module = types.SimpleNamespace()
        module.core = core = core_module

    return module, core


def _load_hf_kernel(attr_name: str, log_label: str) -> Optional[Callable[..., Any]]:
    hf_kernel = None

    try:
        from kernels import get_kernel

        try:
            hf_module = get_kernel("kernels-community/sage_attention")
            hf_kernel = getattr(hf_module, "sageattn", None)
            if hf_kernel is None:
                logger.warning("HuggingFace kernels SageAttention missing 'sageattn'; skipping patch.")
        except Exception as exc:
            logger.warning("Failed to load kernels-community/sage_attention: %s", exc)
    except ImportError:
        hf_kernel = None

    if hf_kernel is None:
        try:
            import sage_attention  # type: ignore
            hf_kernel = getattr(sage_attention, attr_name)
        except (ImportError, AttributeError):
            logger.warning(
                "HuggingFace SageAttention %s kernel not available; continuing with original implementation.",
                log_label,
            )
            return None

    return hf_kernel


def _make_hf_wrapper(hf_kernel: Callable[..., Any]):
    def _hf_wrapper(
        q,
        k,
        v,
        *,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        sm_scale=None,
        smooth_k: bool = True,
        return_lse: bool = False,
        **kwargs,
    ):
        call_kwargs = {
            "tensor_layout": tensor_layout,
            "is_causal": is_causal,
            "sm_scale": sm_scale,
            "return_lse": return_lse,
        }

        if _supports_argument(hf_kernel, "smooth_k"):
            call_kwargs["smooth_k"] = smooth_k

        call_kwargs.update(kwargs)

        result = hf_kernel(q, k, v, **call_kwargs)

        if return_lse:
            out, lse = result
            return out, lse

        return result

    return _hf_wrapper


def _ensure_kernel(attr_name: str, log_label: str) -> bool:
    if attr_name in _PATCHED:
        return True

    hf_kernel = _load_hf_kernel(attr_name, log_label)
    if hf_kernel is None:
        return False

    sageattention, official_core = _ensure_sageattention_core()
    _hf_wrapper = _make_hf_wrapper(hf_kernel)

    setattr(official_core, attr_name, _hf_wrapper)
    setattr(sageattention, attr_name, _hf_wrapper)
    _PATCHED.add(attr_name)

    logger.info("Patched SageAttention %s kernel to use HuggingFace implementation.", log_label)
    return True


def ensure_hf_sm90_kernel() -> bool:
    # try:
    #     import sageattention
    #     from sageattention import core as official_core
    # except ImportError:
    #     logger.warning("Unable to patch SageAttention SM90 kernel: package 'sageattention' not found.")
    #     return False
    return _ensure_kernel("sageattn_qk_int8_pv_fp8_cuda_sm90", "SM90")


def ensure_hf_fp8_cuda_kernel() -> bool:
    return _ensure_kernel("sageattn_qk_int8_pv_fp8_cuda", "FP8 CUDA")
