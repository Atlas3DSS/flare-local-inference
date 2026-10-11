"""Overlap streams remain distinct from pooled capture and tuning streams."""

import os

import pytest
import torch

from flare.preparation.scratch import ScratchArena


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


def test_auxiliary_stream_remains_independent_after_pool_wraparound():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    arenas = [ScratchArena("cuda:0") for _ in range(2)]
    streams = [arena.auxiliary_stream for arena in arenas]
    pooled = [torch.cuda.Stream(device="cuda:0") for _ in range(96)]
    assert len({stream.cuda_stream for stream in streams}) == 2
    assert not {stream.cuda_stream for stream in streams} & {stream.cuda_stream for stream in pooled}
    values = [torch.zeros(128, device="cuda:0") for _ in streams]
    for index, (stream, value) in enumerate(zip(streams, values, strict=True)):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            value.fill_(index + 1)
        torch.cuda.current_stream().wait_stream(stream)
    for index, value in enumerate(values):
        torch.testing.assert_close(value, torch.full_like(value, index + 1))
