"""Restored forwards bind live storage and native handles before graph capture."""

import os
from contextlib import nullcontext

import pytest
import torch
import triton
import triton.language as tl
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

from flare.preparation.aot import ForwardCache, _guard_code_cache, aot_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


@torch.library.custom_op("test_aot::fill", mutates_args=("output",))
def _fill(output: torch.Tensor, handle: int) -> None:
    output.fill_(handle + .25)


@_fill.register_fake
def _fake_fill(output, handle):
    pass


@triton.jit
def _copy(source, output):
    index = tl.arange(0, 32)
    tl.store(output + index, tl.load(source + index))


@triton.jit
def _unrelated(output):
    tl.store(output + tl.arange(0, 32), -19.)


class BufferForward:
    def __init__(self, handle):
        self.handle = handle
        self.storage = torch.zeros(1024, dtype=torch.uint8, device="cuda")
        self.part = self.storage.view(torch.float32)[16:48]
        self.alias = self.storage.view(torch.bfloat16)
        self.before = torch.empty_like(self.alias)
        self.source = torch.randn(32, device="cuda")
        self.output = torch.empty_like(self.source)
        self.negate = False

    def __call__(self):
        self.storage.zero_()
        _fill(self.part, self.handle)
        self.before.copy_(self.alias)
        torch.library.wrap_triton(_copy)[(1,)](self.source, self.output)
        self.output.add_(self.part)
        if self.negate:
            self.output.neg_()
        return self.output


@pytest.mark.parametrize("handle", [0, 1, 73])
@torch.no_grad()
def test_restored_handles_aliases_and_triton_registration(handle, tmp_path):
    original = BufferForward(handle)
    compiled = aot_forward(original)
    compiled(original)
    path = tmp_path / "forward.bin"
    compiled.save_compiled_function(str(path))
    unrelated = kernel_side_table.add_kernel(_unrelated)
    with path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    assert kernel_side_table.get_kernel(unrelated) is _unrelated
    for handle in (127, 1, 0):
        fresh = BufferForward(handle)
        returned = restored(fresh)
        assert returned.data_ptr() == fresh.output.data_ptr()
        assert returned.shape == fresh.output.shape and returned.stride() == fresh.output.stride()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            restored(fresh)
        for _ in range(2):
            fresh.source.normal_()
            graph.replay()
            torch.testing.assert_close(fresh.output, fresh.source + handle + .25)
            torch.testing.assert_close(fresh.before, fresh.alias, rtol=0, atol=0)
        fresh.negate = True
        assert not restored.guard_check(fresh)
        graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("cached_guards", [False, True])
def test_restored_forward_rejects_changed_storage_aliases(tmp_path, rebind, cached_guards):
    original = BufferForward(19)
    compiled = aot_forward(original)
    compiled(original)
    path = tmp_path / "forward.bin"
    compiled.save_compiled_function(str(path))
    restoration = _guard_code_cache(tmp_path / "guards") if cached_guards else nullcontext()
    with restoration, path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    fresh = BufferForward(23)
    different = torch.empty_like(fresh.storage).view(torch.float32)[16:48]
    if rebind:
        restored(fresh)
        fresh.part.set_(different)
    else:
        fresh.part = different
    with pytest.raises((ValueError, RuntimeError), match="storage aliases changed|GuardManager check failed"):
        restored(fresh)


@torch.no_grad()
def test_forward_cache_restores_without_tracing_and_rebuilds_changed_guards(tmp_path, monkeypatch):
    import flare.preparation.aot as aot
    from flare.preparation.execution import GraphSignature
    from torch._dynamo.guards import PyExprCSEPass

    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    cache = ForwardCache(namespace={'model': 'buffer'}, device='cuda:0')
    signature = GraphSignature('decode', 1, 1)
    from types import SimpleNamespace

    events = []
    progress = SimpleNamespace(update=events.append)
    first = cache.prepare(BufferForward(0), name='decode', signature=signature, progress=progress)
    assert any("tracing and compiling" in event for event in events)
    assert not first.cache_hit

    def forbid_trace(*args, **kwargs):
        raise AssertionError('warm forward cache must not trace')

    fresh = BufferForward(73)
    with monkeypatch.context() as patch:
        patch.setattr(aot, 'aot_forward', forbid_trace)
        patch.setattr(aot, 'make_fx', forbid_trace)
        patch.setattr(PyExprCSEPass, 'count', forbid_trace)
        events.clear()
        restored = cache.prepare(fresh, name='decode', signature=signature, progress=progress)
        assert any("restoring compiled" in event for event in events)
        assert any("warming restored" in event for event in events)
        assert not any("tracing and compiling" in event for event in events)
        assert restored.cache_hit
        torch.testing.assert_close(fresh.output, fresh.source + 73.25)

    fresh.negate = True
    rebuilt = cache.prepare(fresh, name='decode', signature=signature)
    assert not rebuilt.cache_hit
    torch.testing.assert_close(fresh.output, -(fresh.source + 73.25))
