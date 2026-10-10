"""Caller-owned byte storage for serialized planned operators."""

from __future__ import annotations

from math import prod

import torch


class ScratchArena:
    """Grow during preparation, then retain one address for graph capture."""

    def __init__(self, device):
        self.tensor = torch.empty(0, dtype=torch.uint8, device=device)
        self.frozen = False
        self._finalizers = []
        self._parallel_plans = []
        self._auxiliary_stream = None

    @property
    def device(self):
        return self.tensor.device

    @property
    def nbytes(self):
        return self.tensor.nbytes

    @property
    def auxiliary_stream(self):
        if self._auxiliary_stream is None:
            self._auxiliary_stream = torch.cuda.Stream(device=self.device)
        return self._auxiliary_stream

    def register_parallel(self, *plans):
        """Declare plans whose workspaces must coexist within a captured fork."""
        if self.frozen:
            raise RuntimeError("scratch declarations are frozen")
        if len(plans) < 2:
            raise ValueError("parallel scratch requires at least two plans")
        self._parallel_plans.append(plans)

    def parallel_nbytes(self, requirement):
        sizes = {}
        def aligned(plan):
            key = id(plan)
            if key not in sizes:
                sizes[key] = (requirement(plan) + 255) // 256 * 256
            return sizes[key]
        return max((sum(aligned(plan) for plan in group)
                    for group in self._parallel_plans), default=0)

    def reserve(self, nbytes):
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("scratch size must be a nonnegative integer")
        if nbytes > self.nbytes:
            if self.frozen:
                raise RuntimeError(f"frozen scratch arena has {self.nbytes} bytes, requires {nbytes}")
            self.tensor = torch.empty((nbytes + 255) // 256 * 256,
                                      dtype=torch.uint8, device=self.device)
        return self.tensor

    def freeze(self):
        self.frozen = True

    def trim(self, nbytes):
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("scratch size must be a nonnegative integer")
        capacity = (nbytes + 255) // 256 * 256
        if capacity > self.nbytes:
            raise ValueError("selected scratch exceeds the reserved candidate capacity")
        if capacity < self.nbytes:
            if self.frozen:
                raise RuntimeError("cannot trim a frozen scratch arena")
            self.tensor = torch.empty(capacity, dtype=torch.uint8, device=self.device)

    def views(self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]) -> tuple[torch.Tensor, ...]:
        """Return disjoint views for simultaneous use within one serialized call.

        Calls reuse the same offsets. Preparation must refresh retained views
        after growth and freeze the arena before recording CUDA graph pointers.
        """
        sizes = []
        for shape, dtype in shapes_and_dtypes:
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("scratch dimensions must be nonnegative integers")
            if not isinstance(dtype, torch.dtype):
                raise TypeError("scratch dtype must be a torch dtype")
            sizes.append(prod(shape) * dtype.itemsize)
        storage = self.reserve(sum((size + 255) // 256 * 256 for size in sizes))
        result, offset = [], 0
        for (shape, dtype), size in zip(shapes_and_dtypes, sizes, strict=True):
            result.append(storage.narrow(0, offset, size).view(dtype).reshape(shape))
            offset += (size + 255) // 256 * 256
        return tuple(result)

    def register_finalizer(self, callback):
        if self.frozen:
            raise RuntimeError("scratch binding declarations are frozen")
        if not callable(callback):
            raise TypeError("scratch binding finalizer must be callable")
        self._finalizers.append(callback)

    def finalize(self):
        for callback in self._finalizers:
            callback()
        self._finalizers.clear()


def scratch_storage(workspace, *, nbytes, device):
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("scratch size must be a nonnegative integer")
    device = torch.device(device)
    if isinstance(workspace, ScratchArena):
        if workspace.device != device:
            raise ValueError("scratch arena and operator must share a device")
        return workspace.reserve(nbytes)
    if workspace is None:
        return torch.empty(nbytes, dtype=torch.uint8, device=device)
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8:
        raise TypeError("planned scratch must be a uint8 tensor or ScratchArena")
    if workspace.device != device or workspace.ndim != 1 or not workspace.is_contiguous():
        raise ValueError("planned scratch requires contiguous one-dimensional storage on the operator device")
    if workspace.nbytes < nbytes:
        raise ValueError(f"planned scratch has {workspace.nbytes} bytes, requires {nbytes}")
    return workspace


def planned_scratch(specs, workspace=None):
    specs = tuple(specs)
    if isinstance(workspace, ScratchArena):
        if not specs or any(spec.device != workspace.device for spec in specs):
            raise ValueError("scratch specs must be nonempty and share the arena device")
        views = workspace.views(*((spec.shape, spec.dtype) for spec in specs))
        return views[0] if len(views) == 1 else views
    if len(specs) != 1 or specs[0].dtype != torch.uint8 or len(specs[0].shape) != 1:
        raise ValueError("operator scratch plan must expose one uint8 byte buffer")
    spec, = specs
    return scratch_storage(workspace, nbytes=spec.nbytes, device=spec.device)


def prime_owners(workspace, *owners):
    """Keep the arena in prepared plans without pinning pre-finalization views.

    b12x retains priming owners until a plan is released. Serving components
    own their finalized bindings; benchmark calls retain concrete trial views.
    """
    return (workspace,) if isinstance(workspace, ScratchArena) else owners
