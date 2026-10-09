from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from serve.model.loader import ShardedLoader
from serve.tp.group import tp_shard_dim0, tp_shard_dim1


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor]) -> str:
    model_path = tmp_path / "model"
    model_path.mkdir()
    save_file(tensors, str(model_path / "model.safetensors"))
    return str(model_path)


def _loader(model_path: str, *, rank: int, world_size: int) -> ShardedLoader:
    tp_group = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        process_group=None,
    )
    return ShardedLoader(model_path, device="cpu", tp_group=tp_group, backend="local")


def test_local_dim0_shard_matches_tp_helper(tmp_path):
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    model_path = _write_checkpoint(tmp_path, {"w": weight})

    for rank in range(4):
        loader = _loader(model_path, rank=rank, world_size=4)
        shard = loader.dim0_shard("w", unit=2)
        expected = tp_shard_dim0(weight, rank, 4, unit=2)
        assert torch.equal(shard, expected)
        loader.evict_all()


def test_local_dim1_shard_matches_tp_helper(tmp_path):
    weight = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    model_path = _write_checkpoint(tmp_path, {"w": weight})

    for rank in range(2):
        loader = _loader(model_path, rank=rank, world_size=2)
        shard = loader.dim1_shard("w", unit=2)
        expected = tp_shard_dim1(weight, rank, 2, unit=2)
        assert torch.equal(shard, expected)
        loader.evict_all()


def test_local_dim0_shard_over_slice(tmp_path):
    weight = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    model_path = _write_checkpoint(tmp_path, {"w": weight})
    sliced = weight.narrow(0, 2, 6)

    for rank in range(3):
        loader = _loader(model_path, rank=rank, world_size=3)
        shard = loader.dim0_shard("w", start=2, length=6, unit=2)
        expected = tp_shard_dim0(sliced, rank, 3, unit=2)
        assert torch.equal(shard, expected)
        loader.evict_all()


def test_local_dim0_shard_replica_groups(tmp_path):
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    model_path = _write_checkpoint(tmp_path, {"w": weight})

    for rank in range(4):
        loader = _loader(model_path, rank=rank, world_size=4)
        shard = loader.dim0_shard("w", unit=4, shard_world_size=2, replica_group_size=2, pad=False)
        expected = tp_shard_dim0(weight, rank // 2, 2, unit=4)
        assert torch.equal(shard, expected)
        loader.evict_all()


def test_load_into_dim1_shard(tmp_path):
    weight = torch.arange(64, dtype=torch.uint8).reshape(8, 8)
    model_path = _write_checkpoint(tmp_path, {"w": weight})

    loader = _loader(model_path, rank=1, world_size=2)
    out = torch.empty(8, 4, dtype=torch.uint8)
    loader.load_into_dim1_shard(out, "w", unit=4, pad=False)
    expected = tp_shard_dim1(weight, 1, 2, unit=4)
    assert torch.equal(out, expected)
    loader.evict_all()


def test_shard_shape_helpers_match_materialized_shapes(tmp_path):
    weight = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    model_path = _write_checkpoint(tmp_path, {"w": weight})
    loader = _loader(model_path, rank=1, world_size=3)

    assert loader.dim0_shard_shape("w", unit=2) == loader.dim0_shard("w", unit=2).shape
    assert loader.dim1_shard_shape("w", unit=2) == loader.dim1_shard("w", unit=2).shape
    loader.evict_all()


def test_optional_and_scalar(tmp_path):
    tensors = {
        "vec": torch.arange(4, dtype=torch.float32),
        "scalar": torch.tensor(3.5, dtype=torch.float32),
    }
    model_path = _write_checkpoint(tmp_path, tensors)
    loader = _loader(model_path, rank=0, world_size=1)

    assert loader.optional("missing") is None
    assert loader.scalar("missing", default=7.0) == 7.0
    assert torch.equal(loader.tensor("vec"), tensors["vec"])
    assert loader.scalar("scalar") == 3.5
    loader.evict_all()


def test_fp4_down_proj_must_dequantize_before_dim1_shard():
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )

    def dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        packed = weight.to(torch.int32)
        lo = lut[packed & 0xF]
        hi = lut[(packed >> 4) & 0xF]
        vals = torch.stack([lo, hi], dim=-1).reshape(weight.shape[0], weight.shape[1] * 2)
        grouped_scale = scale.float().unsqueeze(-1).expand(-1, -1, 16).reshape(weight.shape[0], -1)
        return vals * grouped_scale

    weight = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    scale = torch.ones(2, 1, dtype=torch.float32)

    old_rank0 = tp_shard_dim1(dequant(weight, scale), 0, 2, unit=8)
    old_rank1 = tp_shard_dim1(dequant(weight, scale), 1, 2, unit=8)
    new_rank0 = dequant(tp_shard_dim1(weight, 0, 2, unit=8), tp_shard_dim1(scale, 0, 2))
    new_rank1 = dequant(tp_shard_dim1(weight, 1, 2, unit=8), tp_shard_dim1(scale, 1, 2))

    assert not torch.equal(old_rank0, new_rank0)
    assert not torch.equal(old_rank1, new_rank1)
