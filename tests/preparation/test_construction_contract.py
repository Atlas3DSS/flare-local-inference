"""Check the complete metadata comparison used for constructor refactors."""

from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec
from b12x.norm import hyperconnection as hc
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory
from b12x.preparation import device
from b12x.preparation.types import DeviceIdentity, _CompositePlan

from ignition.preparation.execution import GraphSignature
from tests.preparation.test_bootstrap_construction import construction_contract
from tests.preparation.test_tuning import backend, declaration


@pytest.fixture
def declarations(monkeypatch):
    detected = device.DetectedDevice(
        ordinal=0,
        identity=DeviceIdentity("nvidia", (12, 0), 170, "RTX PRO 6000"),
    )
    monkeypatch.setattr(device, "detect_device", lambda *args: detected)

    def make(*, shared=True):
        owners = [object(), object()]

        def child(tokens, owner):
            plan = hc.plan(
                hc.Caps(device="cuda:0", max_tokens=tokens, hidden_size=2560),
                invocation=FrozenMapping({"operation": "scaled_silu"}),
            )
            requirement = MemoryRequirements(
                scratch=(ScratchBufferSpec(name="work", shape=(tokens, 320),
                                          dtype=torch.bfloat16, device=torch.device("cuda:0")),),
                persistent=(PersistentMemory(("fixture", owner), 64, 32),),
            )
            return replace(plan, _memory_requirements=lambda *args: requirement)

        plans = {1: child(1, owners[0]), 8: child(8, owners[0 if shared else 1])}
        composite = _CompositePlan(
            component_id="test.capacity", capacity_metadata=FrozenMapping({"max_tokens": 8}),
            variants=plans, _assemble=lambda *args: None,
        )
        callbacks = dict.fromkeys(plans, lambda state: None)
        request = composite.request(name="layer.op", prepare_calls=callbacks,
                                    benchmark_calls=callbacks)
        session = SimpleNamespace(requests={request.name: request}, arena=SimpleNamespace(nbytes=8192))
        signatures = [GraphSignature("prefill", 1, 8, inputs=("input_ids",))]
        return session, signatures

    return make


def test_contract_normalizes_owner_identity_but_preserves_shared_allocations(declarations):
    first = construction_contract(*declarations())
    assert first == construction_contract(*declarations())
    assert first != construction_contract(*declarations(shared=False))
    children = first["requests"][0]["plan"]["variants"]
    assert children["1"]["memory"]["scratch"][0] == {
        "name": "work", "shape": [1, 320], "dtype": "torch.bfloat16", "device": "cuda:0"
    }
    assert children["8"]["memory"]["scratch_nbytes"] == 5120
    assert children["8"]["memory"]["pending_persistent_nbytes"] == 32


@pytest.mark.parametrize("field", ["query", "invocation", "override", "device", "memory", "signature"])
def test_contract_detects_nested_changes_without_catalog_count_changes(declarations, field):
    session, signatures = declarations()
    original = construction_contract(session, signatures)
    request = session.requests["layer.op"]
    child = request.plan.variants[8]
    if field == "query":
        child = replace(child, query=replace(child.query, lowrank=321))
    elif field == "invocation":
        child = replace(child, invocation=FrozenMapping({"operation": "scaled_silu", "label": "changed"}))
    elif field == "override":
        child = replace(child, override=child.contract.default_config(child.query, None))
    elif field == "device":
        child = replace(child, _device=torch.device("cuda:1"))
    elif field == "memory":
        memory = child.memory_requirements()
        child = replace(child, _memory_requirements=lambda *args: replace(
            memory, persistent=(replace(memory.persistent[0], required_nbytes=96),)))
    else:
        signatures = [replace(signatures[0], inputs=("different_input",))]
    composite = replace(request.plan, variants={1: request.plan.variants[1], 8: child})
    session.requests[request.name] = replace(request, plan=composite)
    changed = construction_contract(session, signatures)
    assert len(original["requests"]) == len(changed["requests"])
    assert len(original["graph_signatures"]) == len(changed["graph_signatures"])
    assert original != changed


def test_contract_rejects_unencoded_device_objects(declarations):
    session, signatures = declarations()
    request = session.requests["layer.op"]
    child = replace(request.plan.variants[8], _device=object())
    composite = replace(request.plan, variants={1: request.plan.variants[1], 8: child})
    session.requests[request.name] = replace(request, plan=composite)
    with pytest.raises(TypeError, match="unsupported construction metadata"):
        construction_contract(session, signatures)


@pytest.mark.parametrize("fail_finalizer", [False, True])
def test_extracted_attention_fixture_lives_through_trials_and_releases_before_capture(
    backend, monkeypatch, fail_finalizer
):
    from b12x.preparation import PreparationSession
    from ignition.preparation.fixtures.varlen_attention import benchmark_call
    from ignition.preparation.tuning import RecordedPreparationSession

    session = RecordedPreparationSession(device=0, checkpoint="fixture-lifetime",
        workload_schema="attention-fixture-v1", numerical_policy="default")
    arguments = {name: torch.zeros(4, 2, 8) for name in ("q", "k", "v")}
    arguments["cu_seqlens_q"] = torch.zeros(3, dtype=torch.int32)
    calls, references, events = [], [], []
    spec = ScratchBufferSpec(name="work", shape=(256,), dtype=torch.uint8,
                             device=torch.device("cpu"))
    state = SimpleNamespace(
        scratch_plan=SimpleNamespace(scratch_specs=lambda: (spec,)),
        bind=lambda **kwargs: SimpleNamespace(**kwargs),
        run=lambda binding: binding.q + binding.k + binding.v,
    )

    def trial(_):
        call = benchmark_call(state, arguments=arguments, session=session, workspace=None)
        calls.append(call)
        inputs = call.owners[0]
        references.append(weakref.ref(inputs))
        call.produce()
        expected = inputs.q.clone()
        inputs.q.fill_(float("nan"))
        call.produce()
        assert torch.equal(inputs.q, expected)
        assert inputs.cu_seqlens_q.tolist() == [0, 4, 4]
        assert torch.isfinite(call.invoke()).all()
        assert all(not tensor.count_nonzero() for tensor in arguments.values())

    original_prepare = PreparationSession.prepare

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        if kwargs["autotune"]:
            close = result.close

            def close_trials():
                assert references and all(reference() is not None for reference in references)
                assert len(calls) == 2
                events.append("trial_close")
                close()
                calls.clear()

            result.close = close_trials
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    session.declare((declaration("first", benchmark=trial), declaration("second", benchmark=trial)))

    def finalize():
        assert references[0]() is references[1]()
        events.append("finalize")
        if fail_finalizer:
            raise RuntimeError("fixture binding finalization failed")

    session.register_finalizer(finalize)
    if fail_finalizer:
        with pytest.raises(RuntimeError, match="fixture binding finalization failed"):
            session.tune()
        with pytest.raises(RuntimeError, match="completed tuning"):
            with session.capture():
                pytest.fail("failed finalization must prevent capture")
    else:
        session.tune()
        with session.capture():
            assert all(reference() is None for reference in references)
    assert events == ["finalize", "trial_close"]
    assert not calls and all(reference() is None for reference in references)
    assert not session._benchmark_resources and not session._recorded_requests
