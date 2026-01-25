import torch
import types
import comfy.utils
import comfy.sd
import logging
from raylight.utils.memory import monitor_memory
from raylight.utils.common import cleanup_memory, force_malloc_trim

class VaeManager:
    def __init__(self, worker):
        self.worker = worker
        self.vae_model = None

    def load_vae(self, vae_path):
        from raylight.comfy_dist.sd import decode_tiled_1d, decode_tiled_, decode_tiled_3d
        state_dict = {}
        metadata = None
        if "pixel_space" in vae_path:
            state_dict["pixel_space_vae"] = torch.tensor(1.0)
        else:
            # Use return_metadata=True to ensure correct versioning (LTX-1 vs LTX-2)
            res = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            if isinstance(res, tuple):
                state_dict, metadata = res
            else:
                state_dict = res
                metadata = None

        # Diagnostics for VAE identification
        key_check = "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"
        if key_check in state_dict:
            channels = state_dict[key_check].shape[0]
            print(f"[RayWorker {self.worker.local_rank}] VAE Identification: {key_check} has {channels} channels.")
            # Replicate ComfyUI logic for clarity in logs
            version = 0
            if channels == 512: version = 0
            elif channels == 1024:
                version = 1
                if "encoder.down_blocks.1.conv.conv.bias" in state_dict: version = 2
            print(f"[RayWorker {self.worker.local_rank}] VAE Identification: Guessed version {version}")
        
        if metadata:
            print(f"[RayWorker {self.worker.local_rank}] VAE Identification: Metadata found, contains 'config': {'config' in metadata}")

        original_level = logging.getLogger().getEffectiveLevel()
        # logging.getLogger().setLevel(logging.ERROR) # Let's see the warnings for now
        try:
            # Pass metadata to VAE constructor to correctly handle configs
            vae_model = comfy.sd.VAE(sd=state_dict, metadata=metadata)
            vae_model.throw_exception_if_invalid()
        finally:
            logging.getLogger().setLevel(original_level)

        vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
        vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
        vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)

        if self.worker.local_rank == 0:
            print(f"VAE loaded in {self.worker.global_world_size} GPUs")
        self.vae_model = vae_model

    def release_vae(self):
        """Explicitly release VAE model from memory to free RAM for other operations."""
        if self.vae_model is not None:
            print(f"[RayWorker {self.worker.local_rank}] Releasing VAE model from RAM...")
            del self.vae_model
            self.vae_model = None
            
            # Aggressive cleanup
            cleanup_memory()
            
            # Force OS to reclaim freed memory
            force_malloc_trim()
            
            print(f"[RayWorker {self.worker.local_rank}] VAE released.")
        return True

    def get_temporal_compression(self):
        if self.vae_model is None: return None
        return self.vae_model.temporal_compression_decode()

    def get_spatial_compression(self):
        if self.vae_model is None: return None
        return self.vae_model.spacial_compression_decode()

    def check_health(self, samples, shard_index):
        # Diagnostic: Check input latent statistics
        l_min, l_max, l_mean = samples["samples"].min().item(), samples["samples"].max().item(), samples["samples"].mean().item()
        print(f"[RayWorker {self.worker.local_rank}] Input Latent Stats (shard {shard_index}): min={l_min:.4f}, max={l_max:.4f}, mean={l_mean:.4f}")
        if torch.isnan(samples["samples"]).any():
            print(f"[RayWorker {self.worker.local_rank}] CRITICAL: Input latents for shard {shard_index} contain NaNs!")
        
        # Check if overwrite_cast_dtype is set (from patch_temp_fix_ck_ops)
        if getattr(self.worker, "overwrite_cast_dtype", None) is not None:
            print(f"[RayWorker {self.worker.local_rank}] Debug: overwrite_cast_dtype is set to {self.worker.overwrite_cast_dtype}")
        
        # Check VAE weights for NaNs BEFORE decoding
        if self.vae_model is not None:
            nan_params = []
            for name, p in self.vae_model.first_stage_model.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(name)
            if nan_params:
                print(f"[RayWorker {self.worker.local_rank}] CRITICAL: VAE parameters contain NaNs: {nan_params[:5]}...")
            else:
                print(f"[RayWorker {self.worker.local_rank}] VAE parameters verified healthy (no NaNs).")

    def decode(
        self,
        shard_index,
        samples,
        tile_size,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        discard_latent_frames=0,
        vae_dtype="auto",
        mmap_path=None,
        mmap_shape=None,
        output_offset=0,
    ):
        with monitor_memory(f"RayWorker {self.worker.local_rank} - ray_vae_decode", device=self.worker.device):
            print(f"[RayWorker {self.worker.local_rank}] Entering ray_vae_decode (direct method call) for shard {shard_index}...")
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        
        temporal_compression = self.vae_model.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(
                1, min(temporal_size // 2, temporal_overlap // temporal_compression)
            )
        else:
            temporal_size = None
            temporal_overlap = None

        compression = self.vae_model.spacial_compression_decode()

        # Determine VAE dtype based on user selection
        if vae_dtype == "auto":
            # Auto: Use bfloat16 if supported (RTX 3000+), else fall back to float32
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                print(f"[RayWorker {self.worker.local_rank}] Using bfloat16 VAE (auto: bf16 supported)")
            else:
                dtype = torch.float32
                print(f"[RayWorker {self.worker.local_rank}] Using float32 VAE (auto: bf16 not supported)")
        elif vae_dtype == "bf16":
            dtype = torch.bfloat16
            print(f"[RayWorker {self.worker.local_rank}] Using bfloat16 VAE (user selected)")
        elif vae_dtype == "fp16":
            dtype = torch.float16
            print(f"[RayWorker {self.worker.local_rank}] Using float16 VAE (user selected - may cause NaN on some models)")
        else:  # fp32
            dtype = torch.float32
            print(f"[RayWorker {self.worker.local_rank}] Using float32 VAE (user selected - stable but 2x memory)")
        
        self.vae_model.first_stage_model.to(dtype)
        self.vae_model.vae_dtype = dtype
        
        latents_to_decode = samples["samples"].to(dtype)
        print(f"[RayWorker {self.worker.local_rank}] latents_to_decode shape: {latents_to_decode.shape}")
        print(f"[RayWorker {self.worker.local_rank}] tiling: tile_t={temporal_size}, overlap_t={temporal_overlap}")

        images = self.vae_model.decode_tiled(
            latents_to_decode,
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
        print(f"[RayWorker {self.worker.local_rank}] decode_tiled complete. Shape: {images.shape}")
        
        if torch.isnan(images).any():
            print(f"[RayWorker {self.worker.local_rank}] CRITICAL: VAE output STILL contains NaNs even in float32!")
        
        if len(images.shape) == 5:
            # VAE.decode_tiled returns [B, T, H, W, C]
            # Reshape to [B*T, H, W, C] and squeeze leading dim if B=1
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        elif len(images.shape) == 4:
            # Already [T, H, W, C]
            pass
        else:
            # Fallback for unexpected shapes
            while len(images.shape) > 4 and images.shape[0] == 1:
                images = images.squeeze(0)
            if len(images.shape) == 3: # H, W, C -> 1, H, W, C
                images = images.unsqueeze(0)
            
        # If we have overlap, discard the warmup frames
        if discard_latent_frames > 0:
            # 1 latent frame = 8 video frames
            temporal_compression = self.vae_model.temporal_compression_decode() or 8
            discard_video_frames = discard_latent_frames * temporal_compression
            print(f"[RayWorker {self.worker.local_rank}] Discarding {discard_video_frames} redundant warmup frames")
            images = images[discard_video_frames:]
            
        # Move to CPU explicitly before transport to avoid Ray GPU serialization issues
        # and print stats to troubleshoot "Black Video" issues.
        stats_min, stats_max, stats_mean = images.min().item(), images.max().item(), images.mean().item()
        print(f"[RayWorker {self.worker.local_rank}] Shard {shard_index} statistics: min={stats_min:.4f}, max={stats_max:.4f}, mean={stats_mean:.4f}")
        
        if mmap_path and mmap_shape:
            # Direct write to shared memory optimization
            print(f"[RayWorker {self.worker.local_rank}] Writing directly to shared mmap: {mmap_path} (offset={output_offset})...")
            
            # Open shared mmap
            num_elements = 1
            for dim in mmap_shape: num_elements *= dim
            
            # We assume float32 output
            out_buffer = torch.from_file(mmap_path, shared=True, size=num_elements, dtype=torch.float32).reshape(mmap_shape)
            
            # Write slice
            write_len = images.shape[0]
            if output_offset + write_len > out_buffer.shape[0]:
                 # Safety clip
                 write_len = out_buffer.shape[0] - output_offset
                 images = images[:write_len]

            # Write (auto-cast to float32 if needed)
            out_buffer[output_offset : output_offset + write_len] = images.to(torch.float32).cpu()
            
            # Return stats only
            result_payload = {
                "mmap": True,
                "shape": images.shape,
                "stats": (stats_min, stats_max, stats_mean)
            }
            del out_buffer
            
        else:
            # Fallback: serializing large tensor over Ray Object Store (high RAM usage)
            print(f"[RayWorker {self.worker.local_rank}] Moving to CPU and converting to float16 for transport...")
            images = images.cpu().to(torch.float16)
            result_payload = images

        # Proactive memory cleanup - release decoded images tensor
        del images
        del latents_to_decode
        
        # Release VAE from GPU VRAM (keep in CPU RAM for potential reuse within same prompt)
        # This is critical to avoid VRAM accumulation across shards
        if self.vae_model is not None and hasattr(self.vae_model, 'first_stage_model'):
            self.vae_model.first_stage_model.to('cpu')
        
        print(f"[RayWorker {self.worker.local_rank}] Shard {shard_index} complete. Cleanup and return.")
        cleanup_memory()

        # Force OS to reclaim freed memory (fixes RSS creep on Worker 0)
        force_malloc_trim()
            
        return (shard_index, result_payload)
