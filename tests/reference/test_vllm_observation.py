"""Worker metadata must expose selected arithmetic without executing a plan."""

from dataclasses import dataclass
import hashlib
import json
import mmap
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from b12x.gemm.blockscaled._preparation import _PackedExecutionState, _PackedRegimeState
from b12x.gemm.blockscaled._tuning import BlockscaledConfig, BlockscaledQuery
from b12x.preparation.types import FrozenMapping, Plan, Selection, _CompositePlan, _Prepared
from qualify import vllm_observation as observation


def forbidden(*args, **kwargs):
    raise AssertionError("observation must not resolve plans or read tensor values")


def leaf(rows, mode):
    query = BlockscaledQuery(recipe="mxfp8", num_tokens=rows, in_features=256,
                            padded_in_features=256, out_features=256,
                            activation_mode="auto", expected_m=rows)
    config = BlockscaledConfig(mode=mode)
    state = _PackedExecutionState(query=query, config=config, device=torch.device("cpu"),
                                  programs={}, dense=None, alpha_one=None, offsets=None,
                                  required_workspace=32)
    plan = Plan(contract=SimpleNamespace(component_id="gemm.blockscaled_precision"),
                query=query, _compile_jobs=forbidden, _memory_requirements=forbidden,
                _materialize=forbidden)
    selection = Selection(component_id=plan.component_id, query=FrozenMapping({"rows": rows}),
                          config=config, source="cached")
    payload = _Prepared(state=state, selection=selection, programs=frozenset(),
                        retained=None, owners=(), closers=(), device=None, scratch=())
    object.__setattr__(plan, "_prepared", payload)
    return plan


def test_worker_observation_preserves_actual_composite_selection_and_module_aliases(monkeypatch):
    small, large = leaf(1, "quantized"), leaf(512, "a16")
    composite = _CompositePlan(component_id="gemm.blockscaled_precision",
                               capacity_metadata=FrozenMapping({"max_rows": 512}),
                               variants={1: small, 512: large}, _assemble=forbidden)
    state = _PackedRegimeState(capacity=large.prepared.state,
                              exact=MappingProxyType({1: small.prepared.state}))
    object.__setattr__(composite, "_prepared", _Prepared(
        state=state, selection=None, programs=frozenset(), retained=None, owners=(),
        closers=(), device=None, variants=composite.variants, scratch=()))
    model = torch.nn.Module()
    model.attn_hc = torch.nn.Linear(2, 3, dtype=torch.bfloat16, device="meta")
    model.attn_hc.b12x_linear = SimpleNamespace(
        recipe="mxfp8", activation_mode="auto", a16_max_tokens=0,
        a4_prefill_enabled=False, _plan_key=(512, (1,)), plan=composite, a4_plan=None)
    model.gdn = torch.nn.Module()
    model.gdn.A_log = torch.nn.Parameter(torch.empty(2, dtype=torch.float32, device="meta"))
    model.gdn._b12x_prefill_plans = {512: large}
    session = SimpleNamespace(_plans=[small, large, composite], namespace=FrozenMapping({}),
                              state="FROZEN", autotune=True)
    worker = SimpleNamespace(get_model=lambda: model, model_runner=SimpleNamespace(),
                             vllm_config=SimpleNamespace(), _b12x_session=session)
    monkeypatch.setattr(observation, "_libraries", lambda: {})
    for method in ("cpu", "tolist", "item", "numpy"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    report = observation.observe_worker(worker)
    json.dumps(report, allow_nan=False)
    assert len(report["plans"]) == 3
    selected = report["plans"][str(composite.handle)]
    assert selected["variants"] == {"1": str(small.handle), "512": str(large.handle)}
    assert selected["prepared_variants"] == selected["variants"]
    assert selected["materialized_state"]["capacity"]["config"]["mode"] == "a16"
    assert selected["materialized_state"]["exact"]["1"]["config"]["mode"] == "quantized"
    assert report["plans"][str(small.handle)]["query"]["activation_mode"] == "auto"
    assert report["plans"][str(small.handle)]["selection"]["config"]["mode"] == "quantized"
    assert report["modules"]["attn_hc"]["b12x_linear"]["plans"]["plan"] == str(composite.handle)
    assert report["modules"]["gdn"]["plans"]["_b12x_prefill_plans"]["512"] == str(large.handle)
    assert report["modules"]["attn_hc"]["parameters"]["weight"]["dtype"] == "torch.bfloat16"
    assert report["modules"]["gdn"]["parameters"]["A_log"]["dtype"] == "torch.float32"


def test_metadata_records_tensor_extents_without_serializing_values():
    @dataclass
    class Config:
        dtype: object
        shared: object

    tensor = torch.empty_strided((3, 4), (8, 1), device="meta", dtype=torch.bfloat16)
    shared = {"tensor": tensor}
    config = Config(torch.bfloat16, shared)
    shared["config"] = config
    result = observation._metadata(config)
    assert result["dtype"] == "torch.bfloat16"
    assert result["shared"]["tensor"]["shape"] == [3, 4]
    assert result["shared"]["tensor"]["stride"] == [8, 1]
    assert result["shared"]["config"]["recursive_reference"]
    json.dumps(result, allow_nan=False)


def test_loaded_binary_hash_is_bound_to_actual_process_mapping(tmp_path):
    path = tmp_path / "sampling.so"
    content = bytes(range(256)) * 16
    path.write_bytes(content)
    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ):
        report = observation._libraries()
    entry, = [entry for entry in report["mapped_native_binaries"] if entry["path"] == str(path)]
    assert entry["file_matches_mapping"] is True
    assert entry["mapped_inode"] == path.stat().st_ino
    assert entry["file"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert str(path) in report["sampling_binary_paths"]


def test_worker_without_preparation_session_is_not_reported_as_qualified():
    worker = SimpleNamespace(get_model=lambda: torch.nn.Module(), model_runner=object())
    with pytest.raises(RuntimeError, match="no b12x preparation session"):
        observation.observe_worker(worker)
