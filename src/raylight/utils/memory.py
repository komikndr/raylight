import os
import sys
import glob
import torch
import psutil
import gc


def get_system_ram_gb():
    """Returns (used_gb, total_gb, available_gb, cached_gb)."""
    try:
        vm = psutil.virtual_memory()
        cached = getattr(vm, "cached", 0)
        return (vm.used / 1024**3, vm.total / 1024**3, vm.available / 1024**3, cached / 1024**3)
    except Exception:
        return (0, 0, 0, 0)


def get_process_rss_gb():
    """Returns RSS of current process in GB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    except Exception:
        return 0


def get_vram_gb(device=None):
    """Returns (allocated_gb, reserved_gb) for the specified or current device."""
    if not torch.cuda.is_available():
        return (0, 0)

    try:
        if device is not None:
            with torch.cuda.device(device):
                alloc = torch.cuda.memory_allocated()
                res = torch.cuda.memory_reserved()
        else:
            alloc = torch.cuda.memory_allocated()
            res = torch.cuda.memory_reserved()
        return (alloc / 1024**3, res / 1024**3)
    except Exception:
        return (0, 0)


def get_shm_usage_gb():
    """Returns total size of Raylight cache files in /dev/shm in GB."""
    total_bytes = 0
    try:
        files = glob.glob("/dev/shm/raylight_*.pt")
        for f in files:
            try:
                total_bytes += os.path.getsize(f)
            except OSError:
                pass
    except Exception:
        pass
    return total_bytes / 1024**3


def get_gguf_mmap_gb():
    """Returns total size of .gguf files mapped in /proc/self/maps in GB."""
    total_bytes = 0
    try:
        if os.path.exists("/proc/self/maps"):
            with open("/proc/self/maps", "r") as f:
                for line in f:
                    if ".gguf" in line:
                        parts = line.split()
                        if len(parts) > 0:
                            # format: 7f7f...-7f7f... r--p ...
                            addr_range = parts[0]
                            start_hex, end_hex = addr_range.split("-")
                            start = int(start_hex, 16)
                            end = int(end_hex, 16)
                            total_bytes += (end - start)
    except Exception:
        pass
    return total_bytes / 1024**3


def log_memory_stats(tag="Memory", device=None):
    """Logs comprehensive memory statistics to stdout."""
    # Data collection
    sys_used, sys_total, sys_avail, sys_cached = get_system_ram_gb()
    rss = get_process_rss_gb()
    vram_alloc, vram_res = get_vram_gb(device)
    shm = get_shm_usage_gb()
    mmap = get_gguf_mmap_gb()

    # Formatting
    header = f"[{tag}] Memory Stats"
    print(f"{header:-^60}")

    # System Level
    print(f"System RAM : {sys_used:5.2f} GB Used / {sys_total:5.2f} GB Total")
    print(f"           : {sys_avail:5.2f} GB Available | {sys_cached:5.2f} GB Cached (Benign)")

    # Process Level
    print(f"Process RSS: {rss:5.2f} GB (This Process)")

    # VRAM
    if vram_res > 0:
        frag = vram_res - vram_alloc
        print(f"VRAM       : {vram_alloc:5.2f} GB Alloc / {vram_res:5.2f} GB Rsrvd | Frag: {frag:5.2f} GB")

    # Shared Resources
    if shm > 0:
        print(f"Shared Mem : {shm:5.2f} GB (Raylight /dev/shm cache)")

    if mmap > 0:
        print(f"GGUF Mmap  : {mmap:5.2f} GB (Mapped into this process)")

    print("-" * 60)


class monitor_memory:
    """Context manager to log memory stats before and after a block."""
    def __init__(self, tag="Block", device=None):
        self.tag = tag
        self.device = device
        self.peak_start = 0

    def __enter__(self):
        log_memory_stats(f"[START] {self.tag}", self.device)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        except:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "SUCCESS"

        peak_gb = 0.0
        try:
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated(self.device)
                peak_gb = peak_bytes / 1024**3
        except:
            pass

        print(f"[{'END:' + status}] {self.tag} | Peak VRAM Delta: {peak_gb:.2f} GB")
        log_memory_stats(f"[END:{status}] {self.tag}", self.device)
