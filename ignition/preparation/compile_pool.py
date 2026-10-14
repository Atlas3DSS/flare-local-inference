"""Trace and compile prepared forwards in workers without model storage."""

import os
import tempfile
from pathlib import Path

import torch

from .compilation import _OPTIONS, _defer_full_collection
from .host_memory import compile_threads, device_memory_info, worker_limit

# Device memory one compiler worker can hold: its CUDA context plus Inductor's constant folding.
_WORKER_DEVICE_BYTES = 2 << 30


@_defer_full_collection()
def _compile_forward(payload, device, options, path, num_threads, grad_enabled):
    from .aot import aot_forward
    from .forward_metadata import loads

    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(grad_enabled)
    torch.cuda.set_device(device)
    _OPTIONS.update(options)
    forward = loads(payload)
    artifact = aot_forward(forward)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        artifact.save_compiled_function(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CompilePool:
    def __init__(self, device, *, headroom_bytes=0, pools=1):
        """Size workers for `device`, beside `pools - 1` other ranks' pools compiling on this host."""
        self.device = torch.device(device)
        self.pool = None
        if type(headroom_bytes) is not int or headroom_bytes < 0:
            raise ValueError("headroom_bytes must be a nonnegative byte count")
        if type(pools) is not int or pools < 1:
            raise ValueError("pools must be a positive count")
        # Ranks sharing this host share its compile threads.
        budget = max(1, _OPTIONS['compile_threads'] // pools)
        # Workers open contexts on the owning device; a nearly full device compiles with fewer of them.
        free = device_memory_info(self.device)[0]
        shared = torch.cuda.get_device_properties(self.device).is_integrated
        capacity = (free - headroom_bytes) // _WORKER_DEVICE_BYTES
        if capacity < 1:
            raise MemoryError("forward compilation requires memory for a CUDA worker beyond preparation headroom")
        self.workers = worker_limit(min(8, budget, capacity), pools=pools, available=free if shared else None,
                                   device_bytes=_WORKER_DEVICE_BYTES if shared else 0,
                                   headroom_bytes=headroom_bytes if shared else 0)
        threads = max(1, budget // self.workers)
        if shared:
            # Each compile thread is an Inductor subprocess in the worker's shared memory.
            threads = min(threads, compile_threads(threads, workers=pools * self.workers, available=free,
                                                   device_bytes=_WORKER_DEVICE_BYTES, headroom_bytes=headroom_bytes))
        self.options = dict(_OPTIONS, compile_threads=threads)

    def submit(self, forward, path):
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool
        from .forward_metadata import dumps

        payload = dumps(forward)
        if self.pool is None:
            self.pool = SubprocPool(self.workers, kind=SubprocKind.SPAWN)
        return self.pool.submit(_compile_forward, payload, self.device, self.options, str(path),
                                torch.get_num_threads(), torch.is_grad_enabled())

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.pool is not None:
            self.pool.shutdown()
