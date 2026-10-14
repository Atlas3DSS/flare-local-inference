"""Compile on a second GPU and bind the resulting forward for native replay on its owner."""

import os

import pytest
import torch

from ignition.preparation.compile_pool import CompilePool
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires two explicitly assigned matching CUDA devices")


class AffineForward:
    def __init__(self, source, output):
        self.source, self.output = source, output

    def __call__(self):
        self.output.copy_(self.source * 2 + 3)


@pytest.fixture
def constrained_compiler_cpu_budget(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    monkeypatch.setattr("ignition.preparation.compile_pool.os.sched_getaffinity", lambda _: {0, 1})
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@torch.no_grad()
def test_external_compiler_artifact_replays_on_owning_gpu(tmp_path, constrained_compiler_cpu_budget):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two assigned GPUs")
    own, other = (torch.cuda.get_device_properties(index) for index in (0, 1))
    if (own.name, own.major, own.minor) != (other.name, other.major, other.minor) or own.is_integrated:
        pytest.skip("requires matching discrete GPUs")
    uuid = str(other.uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    source = torch.arange(8192, device=device, dtype=torch.float32).view(256, 32) / 128
    output = torch.empty_like(source)
    forward = AffineForward(source, output)
    path = tmp_path / "forward.bin"
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    with CompilePool(device, devices=(uuid,), jobs=1) as compiler:
        assert len(compiler.slots) == 1 and compiler.slots[0].mask == uuid
        compiler.submit(forward, path).result(timeout=120)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == mask
    with path.open("rb") as file:
        compiled = torch.compiler.load_compiled_function(file)
    assert compiled.guard_check(forward)
    compiled(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(forward)
    bundle = PreparedGraphBundle(device=0, graphs={"affine": graph},
        buffers={"source": source, "output": output}, resources=(forward, compiled))
    worker = bundle.start_worker()
    try:
        for gain in (1., -.5, 2.):
            source.mul_(gain).add_(.125)
            expected = source * 2 + 3
            output.fill_(float("nan"))
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)
            ticket = worker.submit(0, repetitions=3)
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        worker.close()
        graph.reset()
