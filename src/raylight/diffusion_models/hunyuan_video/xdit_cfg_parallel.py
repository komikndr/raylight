from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=(
            "x",
            "timestep",
            "context",
            "y",
            "txt_byt5",
            "clip_fea",
            "guidance",
            "attention_mask",
            "guiding_frame_index",
            "ref_latent",
        ),
        **kwargs,
    )
