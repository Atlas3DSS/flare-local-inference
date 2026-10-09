from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import torch

from sparkgap.recipes.qwen38 import linear
from sparkgap.preparation.scratch import ScratchArena


@dataclass
class Query:
    workspace_form: str = "provided"
    workspace_nbytes: int | None = None


def projection_fixture(monkeypatch):
    from b12x.gemm.blockscaled import _a16

    requests = []
    calls = []
    values = torch.arange(30, dtype=torch.float32).view(5, 6).to(torch.bfloat16) / 32
    weight = object()
    monkeypatch.setattr(_a16, "_weight_parts", lambda value: (values, None, None, False))
    monkeypatch.setattr(linear.blockscaled, "query_from_call", lambda *args, **kwargs: Query())
    plan = SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs),
                           memory_requirements=lambda: SimpleNamespace(scratch_nbytes=12))
    monkeypatch.setattr(linear.blockscaled, "plan", lambda query: plan)

    def run(source, values, scales, gain, *, out, workspace):
        calls.append((source, out, workspace))
        out.copy_(source @ values.T)
        return out

    state = SimpleNamespace(required_workspace=12, run=run)

    def prepare(batch):
        requests.extend(batch)
        for request in batch:
            request.prepare_call(state).invoke()

    session = SimpleNamespace(prepare=prepare)
    return session, weight, values, state, requests, calls


@pytest.mark.parametrize("storage", ["arena", "tensor", "none"])
def test_dense_benchmark_uses_finite_private_strided_inputs_and_planned_workspace(monkeypatch, storage):
    session, weight, values, state, requests, calls = projection_fixture(monkeypatch)
    source = torch.zeros((4, 12), dtype=torch.bfloat16)[:, ::2]
    output = torch.empty((4, 10), dtype=torch.bfloat16)[:, :5]
    bias = torch.arange(5, dtype=torch.bfloat16)
    workspace = (ScratchArena("cpu") if storage == "arena" else
                 torch.empty(64, dtype=torch.uint8) if storage == "tensor" else None)
    linear.QuantizedProjection(source, weight, output, session=session, name="projection",
                               workspace=workspace, bias=bias, activation_mode="a16")
    trial = requests[0].benchmark_call(state)
    trial.produce()
    actual = trial.invoke()
    trial_source, trial_output, trial_workspace = calls[-1]
    assert trial_source.stride() == source.stride()
    assert trial_output.stride() == output.stride()
    assert trial_source.data_ptr() != source.data_ptr()
    if isinstance(workspace, ScratchArena):
        assert trial_workspace is workspace.tensor and workspace.nbytes == 256
    else:
        assert trial_workspace is workspace
    if trial_workspace is not None:
        assert trial_workspace.nbytes >= state.required_workspace
    assert torch.isfinite(trial_source).all() and trial_source.count_nonzero() > 0
    torch.testing.assert_close(actual, trial_source @ values.T + bias, rtol=0, atol=0)
    assert source.count_nonzero() == 0 and output.count_nonzero() == 0
    saved = trial_source.clone()
    trial_source.zero_()
    trial.produce()
    torch.testing.assert_close(trial_source, saved, rtol=0, atol=0)


def test_selected_prime_and_finalizer_resolve_grown_arena(monkeypatch):
    session, weight, values, state, requests, calls = projection_fixture(monkeypatch)
    arena = ScratchArena("cpu")
    source = torch.zeros((4, 6), dtype=torch.bfloat16)
    output = torch.empty((4, 5), dtype=torch.bfloat16)
    projection = linear.QuantizedProjection(source, weight, output, session=session,
                                            name="projection", workspace=arena)
    original = arena.tensor
    assert calls[-1][2] is original and projection.workspace is original
    grown = arena.reserve(512)
    assert grown is not original
    requests[0].prepare_call(state).invoke()
    arena.finalize()
    arena.freeze()
    assert calls[-1][2] is grown and projection.workspace is grown
    too_large = SimpleNamespace(required_workspace=513, run=state.run)
    with pytest.raises(RuntimeError, match="frozen scratch arena"):
        requests[0].benchmark_call(too_large)
