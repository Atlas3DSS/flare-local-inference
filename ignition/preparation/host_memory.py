"""Physical-memory budgets for preparation workers and kernel tuning."""

import ctypes
import gc
import os
from pathlib import Path

_WORKER_BYTES = 2 << 30
_HEADROOM_BYTES = 8 << 30
# Kernel modules and temporary compiler objects retained by each tuning rank.
_TUNING_GROWTH_BYTES = 1 << 30


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


def worker_limit(limit, *, pools=1, available=None, device_bytes=0, headroom_bytes=0):
    """Bound worker RAM after the greater of system headroom and the requested reserve."""
    if limit <= 0 or pools <= 0 or device_bytes < 0 or headroom_bytes < 0:
        raise ValueError("compiler and pool limits must be positive; device bytes and headroom must be nonnegative")
    available = available_bytes() if available is None else available
    headroom = max(_HEADROOM_BYTES, headroom_bytes)
    worker_bytes = _WORKER_BYTES + device_bytes
    capacity = (available - headroom) // (pools * worker_bytes)
    if capacity < 1:
        raise MemoryError(f"preparation requires at least {(headroom + pools * worker_bytes) >> 20} MiB "
                          f"available host memory; found {available >> 20} MiB")
    return min(limit, capacity)


def tuning_limits(device, workers, *, pools=1, headroom_bytes=0, race_budget=None):
    """Size compiler pools and candidate races against their physical memory supply."""
    import torch

    if not torch.cuda.get_device_properties(device).is_integrated:
        return worker_limit(workers, pools=pools) if workers else 0, race_budget
    available = device_memory_info(device)[0]
    headroom = max(_HEADROOM_BYTES, headroom_bytes)
    usable = (available - headroom) // pools - _TUNING_GROWTH_BYTES
    if usable <= 0:
        raise MemoryError(f"kernel tuning needs more than {(headroom + pools * _TUNING_GROWTH_BYTES) >> 20} MiB "
                          f"available shared memory; found {available >> 20} MiB")
    race = max(1, usable // 4)
    if race_budget is not None:
        race = min(race, race_budget)
    return min(workers, (usable - race) // _WORKER_BYTES), race


def release_page_cache(paths) -> None:
    """Drop files' clean pages from the page cache once their contents have been loaded.

    An integrated GPU's free memory excludes reclaimable page cache, so checkpoint pages
    read while loading weights would otherwise count against tuning, graphs and KV pages.
    """
    for path in paths:
        with open(path, "rb") as file:
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
