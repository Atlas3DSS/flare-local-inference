"""Ordering contracts for complete-decoder numerical qualification."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).resolve().parents[2] / "scripts/qualify_decoder.py"
    spec = importlib.util.spec_from_file_location("decoder_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_capture_waits_for_final_shared_arena_bindings(harness):
    events, models = [], []

    def build(mode, capacity):
        events.append(("build", mode, capacity))
        model = SimpleNamespace(refreshed=False)
        models.append(model)
        return model

    def finalize():
        assert len(models) == 2
        events.append("finalize")
        for model in models:
            model.refreshed = True

    def capture(mode, capacity, model):
        assert model.refreshed and "freeze" in events
        events.append(("capture", mode, capacity))

    arena = SimpleNamespace(finalize=finalize, freeze=lambda: events.append("freeze"))
    assert harness._prepare_decoder_graphs(
        object(), arena, build, capture, prefill_capacity=512, tune=False,
        progress=lambda message: pytest.fail(message),
    ) is None
    assert events == [
        ("build", "prefill", 512), ("build", "decode", 1), "finalize", "freeze",
        ("capture", "prefill", 512), ("capture", "decode", 1),
    ]


def test_default_finalization_failure_cannot_capture_stale_bindings(harness):
    captured = []

    def fail():
        raise RuntimeError("binding refresh failed")

    with pytest.raises(RuntimeError, match="binding refresh failed"):
        harness._prepare_decoder_graphs(
            object(), SimpleNamespace(finalize=fail, freeze=lambda: pytest.fail("froze after failure")),
            lambda *args: object(), lambda *args: captured.append(args),
            prefill_capacity=512, tune=False, progress=lambda message: None)
    assert not captured


def test_tuned_capture_waits_for_all_declarations_and_final_binding_refresh(harness):
    events, models = [], {}
    report = {"selection_counts": {"benchmark": 2}}

    def build(mode, capacity):
        events.append(("build", mode, capacity))
        models[mode] = SimpleNamespace(selected=False)
        return models[mode]

    workspace = object()

    def register(arena):
        assert set(models) == {"prefill", "decode"}
        assert arena is workspace
        events.append("register_scratch")

    def tune(*, progress):
        assert events[-1] == "register_scratch"
        assert callable(progress)
        events.append("tune")
        for model in models.values():
            model.selected = True
        return report

    def capture(mode, capacity, model):
        assert model is models[mode] and model.selected
        events.append(("capture", mode, capacity))

    actual = harness._prepare_decoder_graphs(
        SimpleNamespace(tune=tune, register_scratch=register),
        workspace,
        build,
        capture,
        prefill_capacity=512,
        tune=True,
        progress=lambda message: None,
    )
    assert actual is report
    assert "expert_workspace_cleanup" not in actual
    assert events == [
        ("build", "prefill", 512),
        ("build", "decode", 1),
        "register_scratch",
        "tune",
        ("capture", "prefill", 512),
        ("capture", "decode", 1),
    ]


def test_tuning_failure_cannot_capture_default_bindings(harness):
    captured = []

    def fail(*, progress):
        raise RuntimeError("candidate measurement failed")

    with pytest.raises(RuntimeError, match="candidate measurement failed"):
        harness._prepare_decoder_graphs(
            SimpleNamespace(tune=fail, register_scratch=lambda arena: None),
            object(),
            lambda mode, capacity: object(),
            lambda *args: captured.append(args),
            prefill_capacity=512,
            tune=True,
            progress=lambda message: None,
        )
    assert captured == []
