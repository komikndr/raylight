import torch
import itertools
import ray
import comfy.model_management


@torch.inference_mode()
def tiled_scale_multidim(
    samples,
    function,
    tile=(64, 64),
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    downscale=False,
    index_formulas=None,
    pbar=None
):
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b + 1]

        # handle entire input fitting in a single tile
        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b:b + 1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

            if pbar is not None:
                pbar.update(1)

        output[b:b + 1] = out / out_div.clamp(min=1e-8)
    return output


def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None
):
    return tiled_scale_multidim(
        samples,
        function,
        (tile_y, tile_x),
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        pbar=pbar
    )


def cancellable_get(refs, timeout=0.1):
    """
    A Ray.get() replacement that checks for ComfyUI cancellation.
    Returns a list of results if refs is a list, otherwise a single result.
    """
    # Handle single ref
    is_list = isinstance(refs, (list, tuple))
    remaining = list(refs) if is_list else [refs]
    
    # Store results in original order
    results = [None] * len(remaining)
    ref_to_idx = {ref: i for i, ref in enumerate(remaining)}
    
    # print(f"[Raylight] Awaiting {len(remaining)} Ray tasks with cancellation support...")
    
    try:
        while remaining:
            # Check ComfyUI cancel status
            if comfy.model_management.processing_interrupted():
                print("[Raylight] Cancellation detected! Force-canceling Ray tasks...")
                for ref in remaining:
                    try:
                        # Use force=True (SIGKILL) and recursive=True for immediate cleanup
                        ray.cancel(ref, force=True, recursive=True)
                    except Exception as e:
                        print(f"[Raylight] Error canceling task: {e}")
                # Raise exception to stop ComfyUI execution
                raise Exception("Raylight: Job canceled by user.")

            # Wait for some tasks to complete
            ready, remaining = ray.wait(remaining, timeout=timeout)
            for ref in ready:
                idx = ref_to_idx[ref]
                results[idx] = ray.get(ref)
    except Exception as e:
        # Rethrow if it's our cancellation exception, otherwise log and rethrow
        if "Job canceled by user" not in str(e):
            print(f"[Raylight] Error during cancellable_get: {e}")
        raise e
            
    return results if is_list else results[0]
