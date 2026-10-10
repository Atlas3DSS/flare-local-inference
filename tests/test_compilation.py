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


@torch.library.custom_op("test_compilation::fill_output", mutates_args=("out",))
def _fill_output(out: torch.Tensor, value: float) -> None:
    out.fill_(value)


@_fill_output.register_fake
def _fake_fill_output(out, value):
    pass


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("separate_views", [False, True])
@torch.no_grad()
def test_captured_scratch_aliases_preserve_intermediate_values(separate_views):
    workspace = torch.zeros(5, 336, device="cuda", dtype=torch.bfloat16)
    first_writer = workspace[:] if separate_views else workspace
    projection = workspace[:, :320] if separate_views else workspace
    injection = workspace[:, 320:324]
    snapshot = torch.empty_like(projection)
    injection_snapshot = torch.empty_like(injection)
    final = torch.empty_like(projection)

    def forward(writer=first_writer):
        _fill_output(writer, 1.)
        snapshot.copy_(projection)
        injection_snapshot.copy_(injection)
        _fill_output(workspace, 2.)
        final.copy_(projection)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for value in (0., 3., -7.):
        workspace.fill_(value)
        graph.replay()
        torch.testing.assert_close(snapshot, torch.ones_like(snapshot), rtol=0, atol=0)
        torch.testing.assert_close(injection_snapshot, torch.ones_like(injection_snapshot), rtol=0, atol=0)
        torch.testing.assert_close(final, torch.full_like(final, 2.), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("rows", [1, 512])
@torch.no_grad()
def test_compiled_head_norm_preserves_fp32_reduction_order(rows):
    packed = torch.zeros(rows, 13312, device="cuda", dtype=torch.bfloat16)
    source = packed[:, 12288:12800].view(rows, 2, 256)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)
    variance = torch.empty(rows, 2, 1, device="cuda", dtype=torch.float32)

    def forward():
        value = source.float()
        mean = value.square().mean(-1, keepdim=True)
        variance.copy_(mean)
        normalized = value * torch.rsqrt(mean + 1e-6)
        output.copy_((normalized * (1.0 + weight.float())).bfloat16())

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    generator = torch.Generator(device="cuda").manual_seed(903)
    for _ in range(3):
        source.normal_(generator=generator)
        forward()
        expected, expected_variance = output.clone(), variance.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(variance, expected_variance, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_snapshots_of_reused_copy_destinations():
    source = torch.zeros(1, 6144, device="cuda", dtype=torch.bfloat16)
    staged = torch.empty_like(source)
    snapshots = [torch.empty_like(source), torch.empty_like(source)]

    def forward():
        for value, snapshot in enumerate(snapshots, 1):
            _fill_output(source, float(value))
            staged.copy_(source)
            snapshot.copy_(staged)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    graph.replay()
    for value, snapshot in enumerate(snapshots, 1):
        torch.testing.assert_close(snapshot, torch.full_like(snapshot, value), rtol=0, atol=0)
