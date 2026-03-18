from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper_ltx(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=("x", "timestep", "context", "attention_mask", "keyframe_idxs", "denoise_mask"),
        auto_chunk_extra_kwargs=True,
        **kwargs,
    )


def cfg_parallel_forward_wrapper_ltxav(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=("x", "timestep", "context", "attention_mask", "keyframe_idxs", "denoise_mask"),
        auto_chunk_extra_kwargs=True,
        **kwargs,
    )


cfg_parallel_forward_wrapper = cfg_parallel_forward_wrapper_ltx
