from contextlib import contextmanager
from contextvars import ContextVar


INNER_ATTENTION_KEY = "raylight_inner_attention"
_PROCESSORS = {}
_ACTIVE = ContextVar("raylight_inner_attention", default=None)


class InnerAttentionRegistry:
    @classmethod
    def register(cls, name):
        if not isinstance(name, str) or ":" not in name:
            raise ValueError("inner-attention names must be namespaced strings")

        def decorator(processor):
            if name in _PROCESSORS:
                raise ValueError("inner-attention processor already registered: {}".format(name))
            _PROCESSORS[name] = processor
            return processor

        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        try:
            processor = _PROCESSORS[name]
        except KeyError:
            raise KeyError("unknown inner-attention processor: {}".format(name))
        return processor(**kwargs)

    @classmethod
    def available(cls):
        return tuple(sorted(_PROCESSORS))


def register_inner_attention(name):
    return InnerAttentionRegistry.register(name)


def create_inner_attention(name, **kwargs):
    return InnerAttentionRegistry.create(name, **kwargs)


def set_inner_attention(model, processor):
    cloned = model.clone()
    model_options = cloned.model_options.copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    transformer_options[INNER_ATTENTION_KEY] = processor
    model_options["transformer_options"] = transformer_options
    cloned.model_options = model_options
    return cloned


def clear_inner_attention(model, processor_type=None):
    cloned = model.clone()
    model_options = cloned.model_options.copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    if processor_type is None or isinstance(transformer_options.get(INNER_ATTENTION_KEY), processor_type):
        transformer_options.pop(INNER_ATTENTION_KEY, None)
    model_options["transformer_options"] = transformer_options
    cloned.model_options = model_options
    return cloned


@contextmanager
def inner_attention_scope(processor, transformer_options=None, ring_world_size=1):
    if processor is not None and ring_world_size != 1 and not getattr(processor, "supports_ring", False):
        raise RuntimeError("inner-attention processor does not support ring attention")
    token = _ACTIVE.set(None if processor is None else (processor, transformer_options if transformer_options is not None else {}))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_inner_attention():
    return _ACTIVE.get()
