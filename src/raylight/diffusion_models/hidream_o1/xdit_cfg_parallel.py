from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=(
            "x",
            "timesteps",
            "context",
            "input_ids",
            "attention_mask",
            "position_ids",
            "vinput_mask",
            "ref_patches",
        ),
        auto_chunk_extra_kwargs=True,
        skip_extra_kwargs=("ar_len", "ref_pixel_values", "ref_image_grid_thw"),
        **kwargs,
    )
