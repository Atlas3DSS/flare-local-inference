"""Trace and compile prepared forwards in workers without model storage."""

import os
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from .compilation import _OPTIONS, _defer_full_collection
from .host_memory import compile_threads, device_memory_info, worker_limit

# Device memory one compiler worker can hold: its CUDA context plus Inductor's constant folding.
_WORKER_DEVICE_BYTES = 2 << 30
_WORKER_HOST_BYTES = 8 << 30


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


def _free_device_bytes(uuids):
    """Inspect explicitly selected GPUs without opening CUDA contexts on them."""
    from cuda.bindings import nvml

    nvml.init_v2()
    try:
        result = {}
        for uuid in uuids:
            handle = nvml.device_get_handle_by_uuid(uuid)
            name = nvml.device_get_name(handle)
            if isinstance(name, bytes):
                name = name.decode()
            result[uuid] = (nvml.device_get_memory_info_v2(handle).free,
                            tuple(nvml.device_get_cuda_compute_capability(handle)), name)
        return result
    finally:
        nvml.shutdown()


@contextmanager
def _visible_devices(mask):
    """Give a compiler subprocess one physical GPU as logical device zero."""
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = mask
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


class _Slot:
    def __init__(self, mask, capacity):
        self.mask, self.capacity = mask, capacity
        self.workers, self.pool, self.running = 0, None, []

    def idle(self):
        self.running = [future for future in self.running if not future.done()]
        return self.workers - len(self.running)


class CompilePool:
    """Compile on the owner and explicitly selected GPUs within their memory budgets.

    External GPUs must match the owner's model and architecture. They are exposed
    to their subprocesses as device zero, matching the owning rank's serialized
    tensor metadata. This requires an owner on logical device zero. Integrated
    GPUs retain their shared-memory worker and Inductor-thread limits.
    """

    def __init__(self, device, *, headroom_bytes=0, devices=(), jobs=None):
        self.device = torch.device(device)
        self._owners = ExitStack()
        self.num_threads = torch.get_num_threads()
        if type(headroom_bytes) is not int or headroom_bytes < 0:
            raise ValueError("headroom_bytes must be a nonnegative byte count")
        if jobs is not None and (type(jobs) is not int or jobs < 1):
            raise ValueError("compiler job count must be a positive integer")
        budget = max(1, _OPTIONS['compile_threads'])
        # Workers open contexts on the owning device; a nearly full device compiles with fewer of them.
        free = device_memory_info(self.device)[0]
        properties = torch.cuda.get_device_properties(self.device)
        shared = properties.is_integrated
        capacity = max(0, (free - headroom_bytes) // _WORKER_DEVICE_BYTES)
        self.slots = [_Slot(None, capacity)]
        if devices:
            if shared:
                raise ValueError("external compiler GPUs require a discrete owning GPU")
            if self.device.index != 0:
                raise ValueError("external compiler GPUs require owning logical device cuda:0")
            owner = str(properties.uuid)
            owner = owner if owner.startswith("GPU-") else "GPU-" + owner
            uuids = tuple(uuid for uuid in dict.fromkeys(devices) if uuid != owner)
            for uuid, (available, capability, name) in _free_device_bytes(uuids).items():
                if capability != (properties.major, properties.minor) or name != properties.name:
                    raise ValueError(f"compiler GPU {uuid} must match {properties.name} "
                                     f"SM{properties.major}{properties.minor}; got {name} {capability}")
                self.slots.append(_Slot(uuid, max(0, (available - headroom_bytes) // _WORKER_DEVICE_BYTES)))
        total_capacity = sum(slot.capacity for slot in self.slots)
        if total_capacity < 1:
            raise MemoryError("forward compilation requires memory for a CUDA worker beyond preparation headroom")
        if len(self.slots) > 1:
            cpus = len(os.sched_getaffinity(0))
            budget = min(budget, cpus)
            # Prefer the explicitly selected devices for small compilation catalogs.
            self.slots = self.slots[1:] + self.slots[:1]
            # Workers trace tensor metadata. The inherited intra-op setting is a
            # Dynamo guard, not a reservation of that many active compiler CPUs.
            limit = min(cpus, total_capacity)
            self.workers = worker_limit(limit if jobs is None else min(limit, jobs),
                                        worker_bytes=_WORKER_HOST_BYTES)
        else:
            limit = min(8, budget, capacity)
            self.workers = worker_limit(limit if jobs is None else min(limit, jobs),
                                       available=free if shared else None,
                                       device_bytes=_WORKER_DEVICE_BYTES if shared else 0,
                                       headroom_bytes=headroom_bytes if shared else 0)
        remaining = self.workers
        while remaining:
            for slot in self.slots:
                if remaining and slot.workers < slot.capacity:
                    slot.workers += 1
                    remaining -= 1
        self.slots = [slot for slot in self.slots if slot.workers]
        threads = max(1, budget // self.workers)
        if shared:
            # Each compile thread is an Inductor subprocess in the worker's shared memory.
            threads = min(threads, compile_threads(threads, workers=self.workers, available=free,
                                                   device_bytes=_WORKER_DEVICE_BYTES, headroom_bytes=headroom_bytes))
        self.options = dict(_OPTIONS, compile_threads=threads)

    def submit(self, forward, path):
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool
        from .forward_metadata import dumps

        payload = dumps(forward)
        slot = max(self.slots, key=_Slot.idle)
        if slot.pool is None:
            if slot.mask is None:
                slot.pool = SubprocPool(slot.workers, kind=SubprocKind.SPAWN)
            else:
                with _visible_devices(slot.mask):
                    slot.pool = SubprocPool(slot.workers, kind=SubprocKind.SPAWN)
            self._owners.callback(slot.pool.shutdown)
        future = slot.pool.submit(_compile_forward, payload, self.device, self.options, str(path),
                                  self.num_threads, torch.is_grad_enabled())
        slot.running.append(future)
        return future

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        return self._owners.__exit__(kind, value, traceback)
