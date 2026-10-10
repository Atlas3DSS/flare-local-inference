"""Ordering of writes through typed aliases in captured inference forwards."""

import os
import subprocess
import sys

import pytest
import torch
import triton
import triton.language as tl

from flare.preparation.compilation import compile_forward


@triton.jit
def _cached_fill_triton(out, value: tl.constexpr):
    offsets = tl.arange(0, 64)
    tl.store(out + offsets, value)


@triton.jit
def _cached_negate_triton(out, value: tl.constexpr):
    offsets = tl.arange(0, 64)
    tl.store(out + offsets, -value)


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


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_fp32_sigmoid_preserves_bf16_gate_rounding():
    gate = torch.tensor([[-0.0296630859375]], dtype=torch.bfloat16, device="cuda")
    normalized = torch.tensor([[-0.23046875]], dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(normalized)

    def forward():
        output.copy_((normalized * torch.sigmoid(gate.float())).bfloat16())

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for value in (-0.0296630859375, 0., 1.):
        gate.fill_(value)
        forward()
        expected = output.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("rows", [1, 4096])
@torch.no_grad()
def test_layer_norm_preserves_aten_rounding(rows):
    source = torch.zeros((rows, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros_like(weight)
    output = torch.empty_like(source)

    def forward():
        output.copy_(torch.nn.functional.layer_norm(source, (128,), weight, bias, 1e-6))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    generator = torch.Generator(device="cuda").manual_seed(532128)
    for _ in range(3):
        source.normal_(generator=generator)
        weight.normal_(generator=generator)
        bias.normal_(generator=generator)
        forward()
        expected = output.clone()
        output.fill_(float("nan"))
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    graph.reset()


@torch.library.custom_op("test_compilation::fill_output", mutates_args=("out",))
def _fill_output(out: torch.Tensor, value: float) -> None:
    out.fill_(value)


@_fill_output.register_fake
def _fake_fill_output(out, value):
    pass


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@torch.no_grad()
def test_compilation_cache_binds_each_invocation_and_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))

    def prepare(value):
        source = torch.empty(64, device="cuda")
        output = torch.empty_like(source)

        def forward():
            _fill_output(source, value)
            output.copy_(source + 1)

        return compile_forward(forward), output

    first, first_output = prepare(2.)
    second, second_output = prepare(8.)
    assert not first.cache_hit
    assert second.cache_hit
    for compiled, output, value in ((first, first_output, 3.), (second, second_output, 9.)):
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        output.zero_()
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("native_kind", ["custom", "triton"])
def test_compilation_cache_survives_process_restart(tmp_path, monkeypatch, native_kind):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    program = '''
import sys
from unittest.mock import patch
import torch
from flare.preparation.compilation import compile_forward

@torch.library.custom_op("flare_cache_test::fill", mutates_args=("out",))
def fill(out: torch.Tensor, value: float) -> None:
    out.fill_(value)
@fill.register_fake
def fake_fill(out, value):
    pass

warm = sys.argv[1] == "warm"
source = torch.empty(64, device="cuda")
output = torch.empty_like(source)
if sys.argv[2] == "triton":
    sys.path.insert(0, sys.argv[3])
    from test_compilation import _cached_fill_triton, _cached_negate_triton
    from flare.preparation.triton import prepare_triton
    kernel = _cached_negate_triton if warm else _cached_fill_triton
    launch = prepare_triton(kernel, (source, 7. if warm else 3.), (1,))
else:
    launch = lambda: fill(source, 7. if warm else 3.)
def forward():
    launch()
    output.copy_(source * 2)
with torch.no_grad():
    if warm:
        with patch("torch._inductor.standalone_compile", side_effect=AssertionError("cache miss")):
            compiled = compile_forward(forward)
    else:
        compiled = compile_forward(forward)
    assert compiled.cache_hit == warm
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    output.zero_()
    graph.replay()
    expected = 14. if warm else 6.
    if warm and sys.argv[2] == "triton":
        expected = -expected
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
'''
    for mode in ("cold", "warm"):
        environment = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(tmp_path / (mode + "-inductor")))
        subprocess.run([sys.executable, "-c", program, mode, native_kind, os.path.dirname(__file__)],
                       env=environment, check=True)


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
