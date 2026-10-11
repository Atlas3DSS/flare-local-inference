"""Trace and compile prepared forwards in workers without model storage."""

import os
import tempfile
from pathlib import Path

import torch

from .compilation import _OPTIONS, _defer_full_collection


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
        self.workers = min(8, budget)
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
