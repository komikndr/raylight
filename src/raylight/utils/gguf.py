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


