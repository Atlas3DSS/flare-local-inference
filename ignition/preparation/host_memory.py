"""Host-memory limits for model preparation workers."""

import ctypes
import gc
from pathlib import Path

_WORKER_BYTES = 2 << 30
_HEADROOM_BYTES = 8 << 30


def release_unused():
    """Return freed preparation objects and glibc arenas to the OS."""
    gc.collect()
    trim = ctypes.CDLL(None).malloc_trim
    trim.argtypes = (ctypes.c_size_t,)
    trim.restype = ctypes.c_int
    trim(0)


def available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is required to size preparation workers")


def device_memory_info(device):
    """Report allocatable physical memory, including reclaimable RAM on integrated GPUs."""
    import torch

    free, total = torch.cuda.mem_get_info(device)
    if torch.cuda.get_device_properties(device).is_integrated:
        free = min(total, available_bytes())
    return free, total


def worker_limit(limit, *, pools=1, available=None, device_bytes=0):
    """Bound worker RAM, including device storage on integrated GPUs, after an 8 GiB reserve."""
    if limit <= 0 or pools <= 0 or device_bytes < 0:
        raise ValueError("compiler and pool limits must be positive and device bytes nonnegative")
    available = available_bytes() if available is None else available
    worker_bytes = _WORKER_BYTES + device_bytes
    capacity = (available - _HEADROOM_BYTES) // (pools * worker_bytes)
    if capacity < 1:
        raise MemoryError(f"preparation requires at least {(_HEADROOM_BYTES + pools * worker_bytes) >> 20} MiB "
                          f"available host memory; found {available >> 20} MiB")
    return min(limit, capacity)
