from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=("x", "timestep", "context", "clip_fea", "time_dim_concat"),
        auto_chunk_extra_kwargs=True,
        **kwargs,
    )
