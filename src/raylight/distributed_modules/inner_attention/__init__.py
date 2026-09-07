from .dispatcher import InnerAttentionDispatcher
from .h3_sla import H3_SLA_PREPARED_KEY, MiniMaxH3SLA
from .registry import (
    INNER_ATTENTION_KEY,
    InnerAttentionRegistry,
    active_inner_attention,
    clear_inner_attention,
    create_inner_attention,
    inner_attention_scope,
    register_inner_attention,
    set_inner_attention,
)

__all__ = [
    "H3_SLA_PREPARED_KEY", "INNER_ATTENTION_KEY", "InnerAttentionDispatcher", "InnerAttentionRegistry", "MiniMaxH3SLA",
    "active_inner_attention", "clear_inner_attention", "create_inner_attention", "inner_attention_scope", "register_inner_attention",
    "set_inner_attention",
]


# Patch registration flow:

# RayMiniMaxH3SLA.patch
#   -> @ray_patch
#   -> model_function_runner
#   -> create_inner_attention("raylight:minimax_h3_sla")
#   -> InnerAttentionRegistry.create
#   -> MiniMaxH3SLA
#   -> set_inner_attention
#   -> clone ModelPatcher
#   -> transformer_options
#   -> _h3_sla_diffusion_wrapper
#   -> patched RAY_ACTORS

# Third-party registration flow:

# @register_inner_attention("plugin:attention")
#   -> InnerAttentionRegistry
#   -> Plugin processor class

# Plugin patch
#   -> create_inner_attention("plugin:attention")
#   -> set_inner_attention
#   -> shared dispatcher
