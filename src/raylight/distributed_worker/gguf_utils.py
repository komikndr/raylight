import os
import sys
import hashlib
import pickle
import ctypes
import gc
import torch
import ray

def evict_page_cache(path):
    """Tells the kernel to evict pages of the given file from the Page Cache."""
    if not path or not os.path.exists(path):
        return
    
    try:
        if sys.platform.startswith("linux"):
            fd = os.open(path, os.O_RDONLY)
            file_size = os.path.getsize(path)
            # POSIX_FADV_DONTNEED = 4 - tells kernel to evict pages from cache
            os.posix_fadvise(fd, 0, file_size, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            print(f"[Raylight] SUCCESS: Evicted {os.path.basename(path)} from page cache ({file_size / 1024**3:.2f} GB)")
    except (OSError, AttributeError) as e:
        # posix_fadvise not available on all platforms
        print(f"[Raylight] Note: Could not evict {os.path.basename(path)} from page cache: {e}")

def check_mmap_leak(path):
    """Checks /proc/self/maps to see if the file is still mapped."""
    if not sys.platform.startswith("linux"):
        return False
    
    try:
        with open("/proc/self/maps", "r") as f:
            maps = f.read()
            
        if path in maps:
            # Count occurrences
            count = maps.count(path)
            print(f"[Raylight] WARNING: GGUF file is still mapped {count} times in memory: {path}")
            return True
        else:
            print(f"[Raylight] SUCCESS: GGUF file is NOT mapped in memory (Clean Release): {path}")
            return False
    except Exception as e:
        print(f"[Raylight] Failed to check maps: {e}")
        return False

class GGUFStreamer:
    """Handles streaming serialization of GGUF models to shared memory."""
    
    @staticmethod
    def stream_to_shm(model, current_unet_path, local_rank):
        """
        Stream CUDA weights to /dev/shm cache before nuking VRAM.
        Atomic-ish: Writes tensors one at a time to avoid RAM spikes.
        """
        # Unique ID per Node + Model (prevents cross-node /dev/shm collisions)
        node_id = ray.get_runtime_context().get_node_id()[:8]
        model_hash = hashlib.md5(current_unet_path.encode()).hexdigest()[:8] if current_unet_path else "unknown"
        cache_path = f"/dev/shm/raylight_{node_id}_{model_hash}_v2.pt"

        if local_rank == 0 and not os.path.exists(cache_path):
            print(f"[RayWorker 0] Streaming VRAM state to SHM (true streaming mode)...")
            
            # TRUE STREAMING: Write tensors one at a time using pickle protocol
            param_count = 0
            buf_count = 0
            bytes_written = 0
            
            # We'll build a streaming dict by writing key-value pairs one at a time
            tensor_index = {}  # name -> (offset, dtype, shape) for verification
            
            # Use buffered I/O and protocol 5 for speed
            with open(cache_path, 'wb', buffering=8*1024*1024) as f:  # 8MB buffer
                pickler = pickle.Pickler(f, protocol=5)
                
                # Write header placeholder
                data_start = f.tell()
                pickler.dump({"version": 2, "streaming": True, "count": 0})
                
                # Stream parameters one at a time (NO gc.collect during loop - major speedup)
                for name, param in model.model.named_parameters():
                    if param.device.type == "cuda":
                        # Copy single tensor to CPU
                        cpu_tensor = param.detach().cpu()
                        
                        # Fix for UnpicklingError & Metadata Loss
                        metadata = None
                        if hasattr(cpu_tensor, "tensor_type") or hasattr(param, "tensor_type"):
                            t_type = getattr(cpu_tensor, "tensor_type", getattr(param, "tensor_type", None))
                            t_shape = getattr(cpu_tensor, "tensor_shape", getattr(param, "tensor_shape", None))
                            metadata = {
                                "tensor_type": t_type,
                                "tensor_shape": t_shape,
                                "patches": [] # Explicitly clear patches for clean base weights
                            }
                        
                        plain_tensor = cpu_tensor.as_subclass(torch.Tensor)
                        
                        tensor_index[name] = (f.tell() - data_start, str(plain_tensor.dtype), tuple(plain_tensor.shape))
                        pickler.dump((name, plain_tensor, metadata))
                        pickler.clear_memo()
                        param_count += 1
                        del cpu_tensor
                        del plain_tensor
                
                # Stream buffers
                for name, buf in model.model.named_buffers():
                    if buf.device.type == "cuda":
                        cpu_tensor = buf.detach().cpu()
                        key = f"buf_{name}"
                        tensor_index[key] = (f.tell() - data_start, str(cpu_tensor.dtype), tuple(cpu_tensor.shape))
                        pickler.dump((key, cpu_tensor, None)) # None metadata for buffers
                        pickler.clear_memo()
                        buf_count += 1
                        del cpu_tensor
                
                # Write end marker
                pickler.dump(None)
                pickler.clear_memo()
                bytes_written = f.tell()
            
            cache_size_mb = bytes_written / 1024**2
            print(f"[RayWorker 0] Cache created: {cache_path} ({cache_size_mb:.1f} MB)")
            print(f"[RayWorker 0] Cache contains: {param_count} params + {buf_count} buffers (streamed)")
            
            # Cleanup only at end
            gc.collect()
            torch.cuda.empty_cache()
            
            # Force libc to release freed memory back to OS
            try:
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except Exception:
                pass
            
            # Evict original GGUF file from page cache
            if current_unet_path:
                evict_page_cache(current_unet_path)
                
        elif local_rank != 0:
            print(f"[RayWorker {local_rank}] Waiting for cache from Worker 0...")
        else:
            print(f"[RayWorker 0] Cache already exists: {cache_path}")
        
        return cache_path
