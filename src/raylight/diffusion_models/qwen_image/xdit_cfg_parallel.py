from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=("x", "timesteps", "context", "attention_mask", "ref_latents", "additional_t_cond"),
        **kwargs,
    )
