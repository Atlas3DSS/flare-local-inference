"""Strided fused projection outputs through compiled RMS normalization and replay."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.rms_norm import PreparedRMSNorm

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


class NormalizationForward:
    def __init__(self, source, operation):
        self.source, self.operation = source, operation

    def __call__(self):
        return self.operation(self.source)


@torch.no_grad()
def test_rms_norm_compiles_in_metadata_worker(tmp_path):
    from ignition.preparation.compile_pool import CompilePool

    device = torch.device("cuda:0")
    source = torch.randn((4, 1792), dtype=torch.bfloat16, device=device)[:, :1280]
    weight = torch.ones(1280, device=device)
    output = torch.empty((4, 1280), dtype=source.dtype, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, session=session,
            name="worker_rms", staging=torch.empty_like(output))
        forward = NormalizationForward(source, operation)
        path = tmp_path / "rms.bin"
        with CompilePool(device) as pool:
            pool.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        compiled(forward)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled(forward)
        source.fill_(2)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize("tokens", [1, 4, 17, 4096])
@pytest.mark.parametrize("width", [128, 512, 1280])
@torch.no_grad()
def test_strided_rms_norm_compiled_replay(tokens, width):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4124 + tokens + width)
    parent = torch.randn((tokens, 1792), dtype=torch.bfloat16, device=device, generator=generator)
    source = parent[:, 1792 - width:]
    output = torch.empty((tokens, width), dtype=source.dtype, device=device)
    staging = torch.empty_like(output)
    weight = torch.linspace(.5, 1.5, width, device=device).bfloat16()
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, session=session, name="rms", staging=staging)
        forward = torch.compile(lambda: operation(source), fullgraph=True)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for gain in (1, -.5, 2):
            parent.mul_(gain)
            output.fill_(float("nan"))
            if operation.staging is not None:
                operation.staging.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            values = source.float()
            expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                        * weight.float()).bfloat16()
            lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
            upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
            assert ((output >= lower) & (output <= upper)).all()
            assert (output != expected).float().mean() < .001
            del values, expected, lower, upper
        graph.reset()
