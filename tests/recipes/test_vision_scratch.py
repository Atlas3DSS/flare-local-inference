"""Vision graph storage shares caller-owned pools across serialized capacities."""

from dataclasses import fields, is_dataclass
from importlib import import_module

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.scratch import GraphBufferPool


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))


@pytest.mark.parametrize("recipe,large,small,sharded", [
    ("qwen38", 1024, 256, False),
    ("deepseek4", 576, 144, False),
    ("mimo_v2", 1024, 256, True),
    ("kimi3", 1024, 256, False),
    ("nemotron_h", 256, 64, False),
    ("step5", 256, 64, True),
])
def test_vision_capacities_reuse_pool_storage(recipe, large, small, sharded):
    scratch_type = import_module(f"ignition.recipes.{recipe}.vision").VisionScratch
    pool = GraphBufferPool("cpu", chunk_bytes=1 << 20)
    args = (TensorParallelShard(0, 4), "cpu") if sharded else ("cpu",)
    first = scratch_type.allocate(large, *args, buffers=pool.cursor())
    capacity = pool.nbytes
    storage = {chunk.untyped_storage()._cdata for chunks in pool.chunks.values() for chunk in chunks}
    assert all(t.untyped_storage()._cdata in storage for t in tensors(first))
    for size in (small, large, small):
        for chunks in pool.chunks.values():
            for chunk in chunks:
                chunk.fill_(float("nan"))
        scratch = scratch_type.allocate(size, *args, buffers=pool.cursor())
        assert pool.nbytes == capacity
        assert all(t.untyped_storage()._cdata in storage for t in tensors(scratch))
        assert all(torch.count_nonzero(t) == 0 for t in tensors(scratch))


def test_qwen_decoder_and_vision_share_pool_with_disjoint_live_operands():
    from ignition.recipes.qwen38.decoder import DecoderScratch
    from ignition.recipes.qwen38.vision import VisionMetadata, VisionScratch

    pool = GraphBufferPool("cpu", chunk_bytes=1 << 20)
    decoder = DecoderScratch.allocate(128, "cpu", sample_rows=4, world_size=4, buffers=pool.cursor())
    decoder_storage = {t.untyped_storage()._cdata for t in tensors(decoder)}
    cursor = pool.cursor()
    vision = VisionScratch.allocate(256, "cpu", buffers=cursor)
    metadata = VisionMetadata.allocate(256, 1, "cpu", buffers=cursor)
    assert decoder_storage & {t.untyped_storage()._cdata for t in tensors(vision)}
    intervals = sorted((t.data_ptr(), t.data_ptr() + t.nbytes) for obj in (vision, metadata) for t in tensors(obj))
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    for index, t in enumerate(tensors(vision), 1):
        t.fill_(index)
    for index, t in enumerate(tensors(vision), 1):
        assert torch.all(t == index)
