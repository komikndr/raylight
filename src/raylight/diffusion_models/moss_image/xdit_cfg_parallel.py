from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    # ref_latents/ref_kv/context_attention_bias are explicit _forward parameters,
    # not **kwargs, so they are chunked by name; batch-1 reference values stay shared
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=("x", "timestep", "context", "ref_latents", "ref_kv", "context_attention_bias"),
        **kwargs,
    )
