"""Persistent graph-catalog output through the shared capture boundary."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ignition.preparation import capture
from ignition.preparation.execution import GraphSignature


@pytest.fixture
def capture_options(monkeypatch):
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    for name, value in {
        "current_stream": lambda device: stream,
        "stream": lambda stream: nullcontext(),
        "CUDAGraph": lambda **options: object(),
        "graph": lambda *args, **options: nullcontext(),
        "synchronize": lambda device: None,
        "mem_get_info": lambda device: (1024, 2048),
        "memory_allocated": lambda device: 0,
        "memory_reserved": lambda device: 0,
    }.items():
        monkeypatch.setattr(capture.torch.cuda, name, value)
    monkeypatch.setattr(capture.torch.distributed, "is_initialized", lambda: False)
    session = SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext)
    return dict(session=session, device="cpu", stream=stream, pool=None)


@pytest.mark.parametrize("compiled_cache_hit", (False, True))
def test_capture_reports_catalog_once_before_compilation_and_on_recapture(capture_options, monkeypatch,
                                                                         compiled_cache_hit):
    messages, calls = [], []
    def forward():
        calls.append("forward")
    class Cached:
        cache_hit = True
        def __call__(self):
            calls.append("cached")
    class Cache:
        def __init__(self, **options):
            assert compiled_cache_hit
            assert any("Graph catalog: 1 graphs" in line for line in messages)
        def prepare_many(self, forwards, signatures, **options):
            return {"decode": Cached()}
    monkeypatch.setattr(capture, "ForwardCache", Cache)
    options = dict(capture_options, compile_decoder=compiled_cache_hit, progress=messages.append)
    signatures = {"decode": GraphSignature("decode", 1, 1)}
    for _ in range(2):
        capture.capture_forwards({"decode": forward}, signatures, **options)
    assert sum("Graph catalog: 1 graphs" in line for line in messages) == 1
    assert calls == (["cached"] * 2 if compiled_cache_hit else ["forward"] * 4)
    assert any("compiled cache hit" in line for line in messages) == compiled_cache_hit
    options["session"] = SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext)
    capture.capture_forwards({"decode": forward}, signatures, **options)
    assert sum("Graph catalog: 1 graphs" in line for line in messages) == 2


def test_capture_catalog_is_quiet_for_workers_and_nonzero_host_ranks(capture_options, monkeypatch):
    capture.capture_forwards({}, {}, compile_decoder=False, progress=None, **capture_options)
    assert not hasattr(capture_options["session"], "_graph_catalog_reported")
    messages = []
    monkeypatch.setattr(capture.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(capture.torch.distributed, "get_rank", lambda: 1)
    capture.capture_forwards({}, {}, compile_decoder=False, progress=messages.append, **capture_options)
    assert not any("Graph catalog" in line for line in messages)
    assert not hasattr(capture_options["session"], "_graph_catalog_reported")


def test_catalog_display_failure_does_not_mark_the_session_reported(capture_options):
    def failed(message):
        raise RuntimeError("output failed")
    with pytest.raises(RuntimeError, match="output failed"):
        capture.capture_forwards({}, {}, compile_decoder=False, progress=failed, **capture_options)
    assert not hasattr(capture_options["session"], "_graph_catalog_reported")


@pytest.fixture
def deepseek_catalog(monkeypatch):
    from ignition.components.prompt_logprobs import prefill_modes
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig
    from ignition.recipes.deepseek41.contract import PATCH_CAPACITIES
    from tests.recipes.test_deepseek41_catalog import register_catalog

    config = PreparationConfig()
    registered = register_catalog.__wrapped__(monkeypatch)(config)[0]
    signatures = [signature for _, _, signature in registered]
    for enabled in (True, False):
        for kind, mode, capacities in prefill_modes(config.prefill_capacities,
                                                   () if enabled else config.batch_capacities, None):
            signatures.extend(GraphSignature(kind, 1 if mode == "prefill" else capacity,
                capacity if mode == "prefill" else 1, speculation_enabled=enabled) for capacity in capacities)
        for kind in ("resume", "init", "restore", "snapshot"):
            signatures.extend(GraphSignature(kind, batch, 1, speculation_enabled=enabled)
                              for batch in config.batch_capacities)
    signatures.extend(GraphSignature("vision", 1, capacity) for capacity in PATCH_CAPACITIES)
    assert len(signatures) == 113
    return signatures


def test_complete_deepseek_catalog_is_rendered_without_losing_shapes(deepseek_catalog):
    from ignition.progress import graph_catalog

    messages = []
    graph_catalog(messages.append, deepseek_catalog)
    output = "\n".join(messages)
    assert "Graph catalog: 113 graphs (before compilation/capture)" in output
    packed = [line.split() for line in messages if line.strip().startswith("packed_speculative")]
    assert packed == [
        ["packed_speculative", "on", "32", "6", "64,96,128", "5", "-", "3"],
        ["packed_speculative", "on", "16", "6", "32,48,64,80,96", "5", "-", "5"],
        ["packed_speculative", "on", "8", "6", "16,24,32,40,48", "5", "-", "5"],
        ["packed_speculative", "on", "4", "6", "8,12,16,20,24", "5", "-", "5"],
        ["packed_speculative", "on", "2", "6", "4,6,8,10,12", "5", "-", "5"],
        ["packed_speculative", "on", "1", "6", "2,3,4,5,6", "5", "-", "5"],
    ]
    kinds = {signature.kind for signature in deepseek_catalog}
    rows = [line.split() for line in messages if line.split() and line.split()[0] in kinds]
    assert sum(int(row[-1]) for row in rows) == 113
    assert {row[0] for row in rows} == kinds
