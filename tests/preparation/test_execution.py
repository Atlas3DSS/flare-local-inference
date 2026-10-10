from types import SimpleNamespace

import pytest

from flare.preparation.execution import GraphSignature, PreparedExecution


def _tensor(dtype, shape):
    strides, running = [], 1
    for extent in reversed(shape):
        strides.insert(0, running)
        running *= extent
    return SimpleNamespace(dtype=f"torch.{dtype}", shape=shape, stride=lambda: tuple(strides))


def _execution(**changes):
    arguments = dict(device=0, graphs={"decode": object(), "prefill": object()},
        signatures={"decode": GraphSignature("decode", 1, 1, ("tokens",)),
                    "prefill": GraphSignature("prefill", 1, 16, ("tokens", "fresh"))},
        buffers={"frame_index": _tensor("int32", (1,)),
                 "submission_sequence": _tensor("int64", (1,)),
                 "completion_records": _tensor("int64", (2, 4, 8)),
                 "tokens": _tensor("int64", (16,)), "fresh": _tensor("bool", (1,))})
    return PreparedExecution(**(arguments | changes))


def test_manifest_offsets_are_stable_across_graphs(monkeypatch):
    execution = _execution()
    spans = [(4096, 4), (8192, 8), (12288, 512), (16384, 128), (20480, 1)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = execution.manifest()
    assert manifest["context"] == 9
    assert manifest["abi_version"] == 2
    assert manifest["input_frame_bytes"] == 152
    decode, prefill = manifest["graphs"]
    assert decode["sampling_mode"] == prefill["sampling_mode"] == "general"
    assert decode["input_bindings"] == prefill["input_bindings"][:3]
    assert decode["input_bindings"][-1] == {"buffer_name": "tokens", "host_offset": 16, "nbytes": 128}
    assert prefill["input_bindings"][-1] == {"buffer_name": "fresh", "host_offset": 144, "nbytes": 1}
    assert all(binding["buffer_name"] != "completion_records" for graph in manifest["graphs"]
               for binding in graph["input_bindings"])
    assert manifest["buffers"][-1]["dtype"] == "bool"


def test_speculative_manifest_exports_vector_abi(monkeypatch):
    execution = _execution()
    buffers = dict(execution.buffers)
    buffers["completion_records"] = _tensor("int64", (2, 4, 24))
    execution = _execution(buffers=buffers,
        signatures={"decode": GraphSignature("speculative", 1, 8, ("tokens",)),
                    "prefill": GraphSignature("prefill", 1, 16, ("tokens", "fresh"))})
    spans = [(4096, 4), (8192, 8), (12288, 1536), (16384, 128), (20480, 1)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = execution.manifest()
    assert manifest["abi_version"] == 3
    assert manifest["graphs"][0]["kind"] == "speculative"
    assert manifest["graphs"][0]["token_capacity"] == 8
    assert manifest["input_frame_bytes"] == 152


@pytest.mark.parametrize("signature", [
    lambda: GraphSignature("callback", 1, 1),
    lambda: GraphSignature("decode", True, 1),
    lambda: GraphSignature("decode", 1, 0),
    lambda: GraphSignature("decode", 1, 1, ("tokens", "tokens")),
    lambda: GraphSignature("decode", 1, 1, ("completion_records",)),
    lambda: GraphSignature("decode", 1, 1, sampling_mode="automatic"),
    lambda: GraphSignature("speculative", 3, 4),
    lambda: GraphSignature("speculative", 1, 9),
    lambda: GraphSignature("speculative", 4, 3),
])
def test_invalid_execution_signature_fails_before_cuda(signature):
    with pytest.raises(ValueError):
        signature()


def test_signature_references_must_cover_exact_bundle():
    with pytest.raises(ValueError, match="exactly one signature"):
        _execution(signatures={"decode": GraphSignature("decode", 1, 1)})
    with pytest.raises(ValueError, match="unregistered"):
        _execution(signatures={"decode": GraphSignature("decode", 1, 1, ("missing",)),
                               "prefill": GraphSignature("prefill", 1, 16)})
