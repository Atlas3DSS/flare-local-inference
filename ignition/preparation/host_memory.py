"""Host-memory limits for preparation workers beside resident model tables."""

import ctypes
import gc
from pathlib import Path

_WORKER_BYTES = 2 << 30
_HEADROOM_BYTES = 8 << 30


def release_unused():
    """Return freed preparation objects and glibc arenas before pinning host tables."""
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


def worker_limit(limit, *, pools=1, available=None):
    """Keep aggregate compiler capacity within free RAM after an 8 GiB reserve."""
    if limit <= 0 or pools <= 0:
        raise ValueError("compiler and pool limits must be positive")
    available = available_bytes() if available is None else available
    capacity = (available - _HEADROOM_BYTES) // (pools * _WORKER_BYTES)
    if capacity < 1:
        raise MemoryError(f"preparation requires at least {(_HEADROOM_BYTES + pools * _WORKER_BYTES) >> 20} MiB "
                          f"available host memory; found {available >> 20} MiB")
    return min(limit, capacity)
