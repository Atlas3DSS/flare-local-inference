"""Trace and compile prepared forwards in workers without model storage."""

import os
import tempfile
from pathlib import Path

import torch

from .compilation import _OPTIONS, _defer_full_collection
from .host_memory import device_memory_info, worker_limit

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
    def __init__(self, device):
        self.device = torch.device(device)
        budget = max(1, _OPTIONS['compile_threads'])
        # Workers open contexts on the owning device; a nearly full device compiles with fewer of them.
        free = device_memory_info(self.device)[0]
        shared = torch.cuda.get_device_properties(self.device).is_integrated
        self.workers = worker_limit(max(1, min(8, budget, free // _WORKER_DEVICE_BYTES)),
                                   device_bytes=_WORKER_DEVICE_BYTES if shared else 0)
        self.options = dict(_OPTIONS, compile_threads=max(1, budget // self.workers))
        self.pool = None

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
