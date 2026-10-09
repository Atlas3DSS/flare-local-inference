"""Minimal tensor-parallel group for TP2 over NCCL.

Wraps torch.distributed with a thin interface: init a pair of processes
on two GPUs, expose allreduce_sum_. Nothing else.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


class TPGroup:
    """TP communication group backed by NCCL."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        process_group: dist.ProcessGroup,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.process_group = process_group

    def allreduce_sum_(self, tensor: torch.Tensor) -> None:
        """In-place sum allreduce across the TP group."""
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)

    @staticmethod
    def init_process(
        rank: int,
        world_size: int,
        device_id: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ) -> TPGroup:
        """Initialize this process as one member of a TP group.

        Call from each spawned process. Uses NCCL backend.
        """
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device)

        return TPGroup(
            rank=rank,
            world_size=world_size,
            device=device,
            process_group=dist.group.WORLD,
        )

    @staticmethod
    def destroy():
        """Clean up distributed state."""
        if dist.is_initialized():
            dist.destroy_process_group()


def _pad_to_multiple(tensor: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad a tensor with zeros along `dim` to the next multiple."""
    size = tensor.shape[dim]
    target = ((size + multiple - 1) // multiple) * multiple
    if target == size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target - size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def tp_shard_dim0(tensor: torch.Tensor, rank: int, world_size: int,
                  unit: int = 1) -> torch.Tensor:
    """Shard a tensor along dim 0 for TP. Pads if not evenly divisible.

    `unit`: pad to the next multiple of `world_size * unit` so each shard
    has a whole number of `unit`-sized blocks (e.g. unit=head_dim for head sharding).
    """
    tensor = _pad_to_multiple(tensor, 0, world_size * unit)
    shard_size = tensor.shape[0] // world_size
    return tensor.narrow(0, rank * shard_size, shard_size).contiguous()


def tp_shard_dim1(tensor: torch.Tensor, rank: int, world_size: int,
                  unit: int = 1) -> torch.Tensor:
    """Shard a tensor along dim 1 for TP. Pads if not evenly divisible."""
    tensor = _pad_to_multiple(tensor, 1, world_size * unit)
    shard_size = tensor.shape[1] // world_size
    return tensor.narrow(1, rank * shard_size, shard_size).contiguous()
