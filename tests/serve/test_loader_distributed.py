from __future__ import annotations

import os
import socket
from multiprocessing import get_context
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from serve.model.loader import ShardedLoader
from serve.tp.group import tp_shard_dim0


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor]) -> str:
    model_path = tmp_path / "model"
    model_path.mkdir()
    save_file(tensors, str(model_path / "model.safetensors"))
    return str(model_path)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _loader_worker(rank: int, world_size: int, port: int, model_path: str, mode: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        tp_group = SimpleNamespace(
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
            process_group=dist.group.WORLD,
        )
        loader = ShardedLoader(model_path, device="cpu", tp_group=tp_group, backend="distributed")
        if mode == "tensor":
            result = loader.tensor("w")
        elif mode == "dim0":
            result = loader.dim0_shard("w", unit=2)
        else:
            raise ValueError(mode)
        queue.put((rank, result))
        loader.evict_all()
    finally:
        dist.destroy_process_group()


def _run_distributed(mode: str, model_path: str):
    ctx = get_context("spawn")
    queue = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=_loader_worker, args=(rank, 2, port, model_path, mode, queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=10.0) for _ in procs]
    for proc in procs:
        proc.join(timeout=10.0)
        assert proc.exitcode == 0
    return [tensor for _rank, tensor in sorted(results)]


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_distributed_tensor_fanout_matches_source(tmp_path):
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    model_path = _write_checkpoint(tmp_path, {"w": weight})
    rank0, rank1 = _run_distributed("tensor", model_path)
    assert torch.equal(rank0, weight)
    assert torch.equal(rank1, weight)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_distributed_dim0_shard_matches_tp_helper(tmp_path):
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    model_path = _write_checkpoint(tmp_path, {"w": weight})
    rank0, rank1 = _run_distributed("dim0", model_path)
    assert torch.equal(rank0, tp_shard_dim0(weight, 0, 2, unit=2))
    assert torch.equal(rank1, tp_shard_dim0(weight, 1, 2, unit=2))
