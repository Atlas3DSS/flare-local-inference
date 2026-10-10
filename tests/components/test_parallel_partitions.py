"""Explicit recipe partitions preserve fused projections and grouped KV heads."""

import pytest

from flare.components.parallel import TensorParallelShard


@pytest.mark.parametrize("world", (1, 2, 4))
def test_fused_projections_partition_each_component(world):
    shards = [TensorParallelShard(rank, world) for rank in range(world)]
    extents = (2048, 2048, 6144)
    partitions = [shard.fused_partitions(extents) for shard in shards]
    offset = 0
    for index, extent in enumerate(extents):
        assert [partitions[rank][index] for rank in range(world)] == [
            (offset + rank * extent // world, offset + (rank + 1) * extent // world)
            for rank in range(world)
        ]
        offset += extent


@pytest.mark.parametrize("world,expected", [(1, [(0, 2)]), (2, [(0, 1), (1, 2)]),
                                           (4, [(0, 1), (0, 1), (1, 2), (1, 2)])])
def test_query_rank_groups_share_kv_heads(world, expected):
    assert [TensorParallelShard(rank, world).replicated_partition(2) for rank in range(world)] == expected


def test_unrepresentable_partitions_fail_before_loading():
    with pytest.raises(ValueError):
        TensorParallelShard(0, 4).partition(640, alignment=128)
    with pytest.raises(ValueError):
        TensorParallelShard(0, 4).replicated_partition(3)
    assert TensorParallelShard(3, 4).partition(640, alignment=16) == (480, 640)
