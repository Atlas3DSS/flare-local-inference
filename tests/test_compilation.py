"""Ordering of writes through typed aliases in captured inference forwards."""

import os

import pytest
import torch

from flare.preparation.compilation import compile_forward


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_shared_storage_write_order():
    storage = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    fp32 = storage.view(torch.float32)
    bf16 = storage.view(torch.bfloat16)
    fp32_offset = fp32[32:96]
    before = torch.empty_like(bf16)
    middle = torch.empty_like(bf16)
    after = torch.empty_like(fp32)

    def forward():
        with torch.profiler.record_function("shared_storage_initialization"):
            fp32.fill_(1.25)
            before.copy_(bf16)
        fp32_offset.fill_(2.5)
        middle.copy_(bf16)
        bf16[4:20].fill_(0.5)
        after.copy_(fp32)

    forward()
    expected = [tensor.clone() for tensor in (storage, before, middle, after)]
    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for value in (0, 19, 128):
        storage.fill_(value)
        graph.replay()
        for tensor, reference in zip((storage, before, middle, after), expected, strict=True):
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("rows", [1, 512])
@torch.no_grad()
def test_compiled_bf16_gate_rounding(rows):
    logits = torch.randn(rows, 1, dtype=torch.bfloat16, device="cuda")
    projected = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
    probabilities = torch.empty_like(logits)
    output = torch.empty_like(projected)

    def forward():
        torch.sigmoid(logits, out=probabilities)
        torch.mul(projected, probabilities, out=output)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(3):
        logits.normal_()
        projected.normal_()
        forward()
        expected = output.clone()
        expected_probabilities = probabilities.clone()
        output.fill_(float("nan"))
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(probabilities, expected_probabilities, rtol=0, atol=0)
