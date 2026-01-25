import torch
import torch.nn as nn
import types


def patch_ffn_forward(module, num_chunks, verbose=True):
    """
    Monkey-patches the forward method of a FeedForward module to run in chunks.
    This is safer than module replacement as it keeps the original weights and 
    module tree intact for ComfyUI's ModelPatcher.
    """
    if hasattr(module, "_raylight_ffn_patched"):
        # Just update the chunk count
        module._raylight_ffn_num_chunks = num_chunks
        return True

    # Store original forward
    module._original_forward = module.forward
    module._raylight_ffn_num_chunks = num_chunks
    module._raylight_ffn_patched = True

    def chunked_forward(self, x):
        num_chunks = self._raylight_ffn_num_chunks
        if num_chunks <= 1:
            return self._original_forward(x)

        batch, seq_len, dim = x.shape
        # Debug print once per inference start for large sequences (video)
        if seq_len > 1000 and not hasattr(self, "_ffn_logged"):
            print(f"[RayWorker] Video FFN Chunking Active: seq_len={seq_len}, chunks={num_chunks}")
            self._ffn_logged = True

        if seq_len < num_chunks:
            return self._original_forward(x)

        # Pre-allocate output tensor to avoid memory peaks from concatenation
        out = torch.empty_like(x)

        chunk_size = (seq_len + num_chunks - 1) // num_chunks

        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            chunk = x[:, i:end, :]

            # Use original forward logic (usually self.net(chunk))
            chunk_out = self._original_forward(chunk)

            out[:, i:end, :] = chunk_out

            # Explicitly clear chunk references
            del chunk, chunk_out

        return out

    # Bind the new forward method to this instance
    module.forward = types.MethodType(chunked_forward, module)
    return True


def wrap_ffn_layers(model, num_chunks=8, verbose=True):
    """
    Finds and patches FFN layers in LTX-2 / LTXAV models.
    """
    info = {"ffn_wrapped": 0, "already_wrapped": 0}

    # Target LTX-2 / LTXAV structure
    target = model
    # If it's a ComfyUI ModelPatcher, get the inner model
    if hasattr(target, 'model'):
        target = target.model
    elif hasattr(target, 'get_model_object'):
        # Fallback for older or different patcher versions
        try:
            target = target.get_model_object("diffusion_model")
        except:
            pass

    # Handle nested diffusion_model (some versions wrap it again)
    if hasattr(target, 'diffusion_model'):
        target = target.diffusion_model

    patched_count = 0
    already_patched = 0

    for name, module in target.named_modules():
        # LTX-2 uses 'FeedForward' class
        # Target both video 'ff' and audio 'audio_ff' (for LTXAV)
        if name.endswith('.ff') or name.endswith('.audio_ff'):
            if hasattr(module, "_raylight_ffn_patched"):
                module._raylight_ffn_num_chunks = num_chunks
                already_patched += 1
            else:
                patch_ffn_forward(module, num_chunks, verbose)
                patched_count += 1
                if verbose:
                    print(f"[RayLTXFFNChunker] Patched FFN: {name}")

    info["ffn_wrapped"] = patched_count
    info["already_wrapped"] = already_patched

    if verbose:
        print(f"[RayLTXFFNChunker] Patched {patched_count} FFN layers, updated {already_patched}.")

    return info


class RayLTXFFNChunker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "ffn_chunks": ("INT", {"default": 8, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    FUNCTION = "apply_chunking"
    CATEGORY = "Raylight/extra"

    def apply_chunking(self, ray_actors, ffn_chunks):
        import ray
        gpu_actors = ray_actors["workers"]

        # Apply to all workers
        ray.get([actor.apply_ffn_chunking.remote(ffn_chunks) for actor in gpu_actors])

        return (ray_actors,)

NODE_CLASS_MAPPINGS = {
    "RayLTXFFNChunker": RayLTXFFNChunker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayLTXFFNChunker": "Ray LTX FFN Chunker (VRAM Saver)",
}
