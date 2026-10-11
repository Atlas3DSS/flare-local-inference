"""Aligned slot storage for immutable continuation states and live model state."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateRegion:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class StateLayout:
    regions: tuple[StateRegion, ...]
    row_bytes: int

    @classmethod
    def from_specs(cls, specifications):
        regions, names, offset = [], set(), 0
        for name, shape, dtype in specifications:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("state region names must be nonempty and unique")
            shape = tuple(shape)
            if not shape or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError("state region extents must be positive integers")
            if not isinstance(dtype, torch.dtype):
                raise TypeError("state regions require a tensor dtype")
            offset = (offset + 255) // 256 * 256
            region = StateRegion(name, offset, shape, dtype)
            regions.append(region)
            names.add(name)
            offset += region.nbytes
        if not regions:
            raise ValueError("state storage requires at least one region")
        return cls(tuple(regions), (offset + 255) // 256 * 256)

    def region(self, name):
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def snapshot_capacity(self, budget_bytes=4 << 30):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("snapshot budget must be a nonnegative byte count")
        return budget_bytes // self.row_bytes


@dataclass(frozen=True)
class StateStorage:
    backing: torch.Tensor
    layout: StateLayout

    @classmethod
    def allocate(cls, layout, slots, device):
        if type(slots) is not int or slots < 2:
            raise ValueError("state storage requires null and live slots")
        return cls(torch.zeros((slots, layout.row_bytes), dtype=torch.uint8, device=device), layout)

    def region(self, name):
        region = self.layout.region(name)
        return self.backing[:, region.offset:region.offset + region.nbytes].view(
            region.dtype).view(self.backing.shape[0], *region.shape)


def save_hidden(source, destination, slots, active):
    """Save packed final hidden rows to validated physical request slots.

    Destination contains only live slots. Admission must reject duplicate active
    slots before forward execution. Inactive destinations retain their bytes.
    """
    physical = torch.arange(destination.shape[0], device=destination.device)
    packed = torch.arange(source.shape[0], device=source.device)
    matches = (physical[:, None] == slots[None, :]) & active[None, :].bool()
    indices = torch.where(matches, packed[None, :], 0).amax(1)
    destination.copy_(torch.where(matches.any(1)[:, None], source[indices], destination))
