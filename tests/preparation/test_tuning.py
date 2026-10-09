from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import Event
from types import SimpleNamespace
import weakref

import pytest
import torch
from b12x.preparation import FrozenMapping, PreparationSession, MemoryRequirements, PreparedCall
from b12x._lib.scratch import ScratchBufferSpec

from sparkgap.preparation.scratch import ScratchArena, scratch_storage

from sparkgap.preparation.tuning import (
    RecordedPreparationSession,
    benchmark_resource,
    register_finalizer,
)


@pytest.fixture
def backend(monkeypatch):
    events = []
    controls = SimpleNamespace(cancel=False, skip_prime=set(), close_error=False)

    def initialize(self, *, device, namespace, **kwargs):
        assert kwargs["autotune"] is False and kwargs["compile_workers"] == 0
        self.device = SimpleNamespace(ordinal=0)
        self.namespace = FrozenMapping(namespace)
        self._stop = Event()
        self.compile_workers = 0
        self._published_owners = {}

    def prepare(self, requests, *, autotune, **kwargs):
        events.append(("prepare", autotune, tuple(item.name for item in requests)))
        retained_trials = []
        for request in requests:
            if autotune and request.benchmark_call is not None:
                trial = request.benchmark_call(request.plan)
                if getattr(controls, "retain_trials", False):
                    retained_trials.append(trial)
            request.plan.selection = SimpleNamespace(
                source=request.plan.source if autotune else "default", config={"tile": 32})
            if request.name not in controls.skip_prime:
                call = request.prepare_call(request.plan)
                if call is not None:
                    call.invoke()
                # b12x PreparationJob._publish retains this tuple until plan release.
                self._published_owners[request.name] = () if call is None else tuple(call.owners)
        if autotune and controls.cancel:
            self._stop.set()

        def close():
            events.append(("result_close", autotune))
            if controls.close_error:
                raise RuntimeError("result close failed")
            if autotune and getattr(controls, "retain_trials", False):
                controls.before_trial_close()
                retained_trials.clear()

        return SimpleNamespace(coverage={}, cache_hits=2 if autotune else 0,
            benchmarked_candidates=7 if autotune else 0, elapsed_seconds=1.25,
            parent_cute_compilations=0, parent_triton_compilations=0, close=close)

    def envelope(self, requests):
        events.append(("envelope", tuple(item.name for item in requests)))
        return SimpleNamespace(scratch_nbytes=max((item.plan.bytes for item in requests), default=0),
                               pending_persistent_nbytes=0)

    def workers(self, count):
        assert count is None
        events.append(("workers",))
        self.compile_workers = 8

    @contextmanager
    def capture(self):
        events.append(("capture",))
        yield

    monkeypatch.setattr(PreparationSession, "__init__", initialize)
    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    monkeypatch.setattr(PreparationSession, "candidate_memory_envelope", envelope)
    monkeypatch.setattr(PreparationSession, "configure_compile_workers", workers)
    monkeypatch.setattr(PreparationSession, "_check_thread", lambda self: None)
    monkeypatch.setattr(PreparationSession, "_synchronize", lambda self: events.append(("drain",)))
    monkeypatch.setattr(PreparationSession, "_allocated", lambda self: 4096 + sum(
        getattr(value, "allocated_nbytes", value.nbytes)
        for value in getattr(self, "_benchmark_resources", {}).values()))
    monkeypatch.setattr(PreparationSession, "capture", capture)
    monkeypatch.setattr(PreparationSession, "freeze", lambda self: events.append(("freeze",)))
    def close_session(self):
        events.append(("close",))
        self._published_owners.clear()

    monkeypatch.setattr(PreparationSession, "close", close_session)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache",)))
    return events, controls


def declaration(name, *, component="sequence.gdn_prefill", size=128, selected_size=None, source="tuned",
                prepare=lambda state: None, benchmark=None):
    requirement = MemoryRequirements(scratch=(ScratchBufferSpec(name="workspace",
        shape=(size if selected_size is None else selected_size,), dtype=torch.uint8,
        device=torch.device("cpu")),))
    plan = SimpleNamespace(component_id=component, bytes=size, source=source, selection=None,
        memory_requirements=lambda: requirement,
        contract=SimpleNamespace(config_payload=lambda config: FrozenMapping(config)))
    return SimpleNamespace(name=name, plan=plan, prepare_call=prepare,
                           benchmark_call=benchmark, retain_benchmark_call=False)




def test_benchmark_resource_shared_across_requests_and_released_before_capture(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    observed = []
    references = []

    class Arena:
        nbytes = 512

    def create():
        value = Arena()
        references.append(weakref.ref(value))
        return value

    def trial(state):
        observed.append(id(benchmark_resource(session, "strided-state", create)))

    session.prepare((declaration("a", benchmark=trial), declaration("b", benchmark=trial)))
    report = session.tune()
    assert len(references) == 1 and observed[0] == observed[1]
    assert references[0]() is None
    assert report["shared_benchmark_bytes"] == 512
    assert report["released_preparation_bytes"] == 512
    assert report["allocated_after_trial_release"] == report["allocated_before_tuning"] == 4096
    with pytest.raises(RuntimeError, match="only during"):
        benchmark_resource(session, "late", create)






@pytest.mark.parametrize("source", ["fixed", "override", "cached", "tuned"])
def test_completed_selection_sources_are_accepted_and_reported(backend, source):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    session.prepare((declaration("op", source=source),))
    assert session.tune()["selection_counts"] == {source: 1}


@pytest.mark.parametrize("failure", ["default", "cancel", "finalizer", "result_close"])
def test_incomplete_tuning_never_allows_capture_or_freeze(backend, failure):
    events, controls = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    session.prepare((declaration("op", source="default" if failure == "default" else "tuned"),))
    controls.cancel = failure == "cancel"
    controls.close_error = failure == "result_close"
    if failure == "finalizer":
        def fail():
            raise RuntimeError("binding refresh failed")
        session.register_finalizer(fail)
    with pytest.raises(RuntimeError):
        session.tune()
    assert session._tune_status == "failed"
    assert not session._recorded_requests and not session._benchmark_resources
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()


def test_disabled_tuning_does_not_silently_keep_defaults(backend):
    events, _ = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    session.prepare((declaration("op"),))
    session._stop.set()
    with pytest.raises(RuntimeError, match="enabled, uncancelled"):
        session.tune()
    assert not any(event[0] == "prepare" and event[1] is True for event in events)


def test_names_registration_and_one_shot_lifecycle_are_strict(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    item = declaration("op")
    with pytest.raises(ValueError, match="unique"):
        session.prepare((item, item))
    session.prepare((item,))
    with pytest.raises(ValueError, match="unique"):
        session.prepare((item,))
    with pytest.raises(ValueError, match="default local"):
        session.prepare((declaration("later"),), autotune=True)
    session.tune()
    for action in (lambda: session.tune(), lambda: session.prepare((declaration("late"),)),
                   lambda: session.register_finalizer(lambda: None)):
        with pytest.raises(RuntimeError, match="closed after tuning"):
            action()
    session.close()
    assert session._tune_status == "closed"


def test_namespace_is_exact_and_reserved_policy_cannot_be_overridden(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision", namespace={"headers": "digest"})
    assert session.namespace["checkpoint"] == "revision"
    assert session.namespace["headers"] == "digest"
    for key in ("checkpoint", "workload_schema", "numerical_policy"):
        with pytest.raises(ValueError, match="cannot override"):
            RecordedPreparationSession(device=0, checkpoint="revision", namespace={key: "different"})




def test_helpers_preserve_ordinary_session_behavior():
    session = object.__new__(PreparationSession)
    value = object()
    register_finalizer(session, lambda: pytest.fail("ordinary session finalizer invoked"))
    assert benchmark_resource(session, "arena", lambda: value) is value


@pytest.mark.parametrize("source", ["cached", "tuned"])
def test_borrowed_state_restores_after_result_close_and_owner_release(backend, monkeypatch, source):
    events, _ = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    storage = torch.zeros(32, dtype=torch.uint8)
    resource = SimpleNamespace(nbytes=32, allocated_nbytes=0, borrowed_nbytes=32)
    calls = []

    def restore():
        assert not session._benchmark_resources and not session._recorded_requests
        assert not session._finalizers
        assert events[-2:] == [("result_close", True), ("drain",)]
        assert session._tune_status == "releasing"
        assert storage[0].item() == 9
        storage.zero_()
        calls.append("restore")

    session.register_benchmark_resource("state", resource, restore=restore)
    original_prepare = PreparationSession.prepare

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        close = result.close

        def final_trial_restore():
            close()
            storage.fill_(9)

        result.close = final_trial_restore
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)

    def trial(state):
        assert benchmark_resource(session, "state", lambda: pytest.fail("borrow ignored")) is resource
        storage.fill_(7)

    session.prepare((declaration("op", source=source,
        benchmark=trial if source == "tuned" else None),))
    report = session.tune()
    assert calls == ["restore"] and not storage.count_nonzero()
    assert events.index(("result_close", True)) < len(events) - 1
    assert report["shared_benchmark_bytes"] == report["borrowed_benchmark_bytes"] == 32
    assert report["allocated_shared_benchmark_bytes"] == report["released_preparation_bytes"] == 0
    assert not session._benchmark_restorers
    with session.capture():
        assert events[-2][0] == "empty_cache"
        assert not storage.count_nonzero()


@pytest.mark.parametrize("failure", ["prepare", "finalizer", "result_close"])
def test_borrowed_state_restores_on_tuning_failures(backend, monkeypatch, failure):
    _, controls = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    storage = torch.full((8,), 7, dtype=torch.uint8)
    session.register_benchmark_resource("state", SimpleNamespace(nbytes=8), restore=storage.zero_)
    session.prepare((declaration("op"),))
    if failure == "prepare":
        def fail(*args, **kwargs):
            raise RuntimeError("candidate failed")
        monkeypatch.setattr(PreparationSession, "prepare", fail)
    elif failure == "finalizer":
        def fail():
            raise RuntimeError("finalizer failed")
        session.register_finalizer(fail)
    else:
        controls.close_error = True
    with pytest.raises(RuntimeError):
        session.tune()
    assert not storage.count_nonzero()
    assert not session._benchmark_restorers
    assert session._tune_status == "failed"
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()


@pytest.mark.parametrize("failure", ["reset", "pre_sync", "post_sync"])
def test_failed_restore_retains_owner_and_close_retries_before_freeze(backend, monkeypatch, failure):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    attempts = []

    class Owner:
        def restore(self):
            attempts.append("reset")
            if failure == "reset" and len(attempts) == 1:
                raise RuntimeError("reset failed")

    owner = Owner()
    reference = weakref.ref(owner)
    session.register_benchmark_resource("state", SimpleNamespace(nbytes=16), restore=owner.restore)
    del owner
    original_sync = session._synchronize
    sync_attempts = []
    if failure in ("pre_sync", "post_sync"):
        def sync():
            if session._tune_status == "releasing":
                sync_attempts.append("sync")
                if len(sync_attempts) == (1 if failure == "pre_sync" else 2):
                    raise RuntimeError("restoration fence failed")
            original_sync()
        monkeypatch.setattr(session, "_synchronize", sync)
    session.prepare((declaration("op"),))
    with pytest.raises(RuntimeError):
        session.tune()
    assert session._tune_status == "failed" and reference() is not None
    assert not session._benchmark_resources and session._benchmark_restorers
    assert len(attempts) == (0 if failure == "pre_sync" else 1)
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()
    monkeypatch.setattr(session, "_synchronize", original_sync)
    session.close()
    assert len(attempts) == (1 if failure == "pre_sync" else 2) and reference() is None
    assert session._tune_status == "closed" and not session._benchmark_restorers


def test_borrowed_registration_is_unique_collecting_only_and_restores_on_close(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    resource = SimpleNamespace(nbytes=4)
    restored = []
    with pytest.raises(TypeError, match="restore callback"):
        session.register_benchmark_resource("state", resource, restore=None)
    session.register_benchmark_resource("state", resource, restore=lambda: restored.append(True))
    with pytest.raises(ValueError, match="unique"):
        session.register_benchmark_resource("state", resource, restore=lambda: None)
    session.close()
    assert restored == [True]
    with pytest.raises(RuntimeError, match="closed after tuning"):
        session.register_benchmark_resource("late", resource, restore=lambda: None)


def test_gdn_registration_reuses_exact_strided_key_and_drops_arena_before_reset(backend):
    from sparkgap.preparation.fixtures.gdn import register_benchmark_state

    session = RecordedPreparationSession(device=0, checkpoint="revision")
    storage = torch.full((2, 32), 7, dtype=torch.uint8)
    template = torch.empty_strided((2, 4), (8, 1))
    references = []

    def restore():
        assert references[0]() is None
        storage.zero_()

    register_benchmark_state(session, template, storage=storage, restore=restore)
    key = ("qwen_gdn_state", "cpu", 2, 32)
    references.append(weakref.ref(session._benchmark_resources[key]))

    def trial(state):
        arena = benchmark_resource(session, key, lambda: pytest.fail("registered arena not reused"))
        assert arena.view(template).data_ptr() == storage.data_ptr()
        assert arena.view(template).stride() == template.stride()
        arena.view(template).fill_(23)

    session.prepare((declaration("op", benchmark=trial),))
    report = session.tune()
    assert not storage.count_nonzero()
    assert report["borrowed_benchmark_bytes"] == storage.nbytes
    assert report["allocated_shared_benchmark_bytes"] == 0


def test_close_retries_restoration_after_base_session_has_closed(backend, monkeypatch):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    session.state = "OPEN"
    attempts = []

    def base_close(self):
        self.state = "CLOSED"

    def check_thread(self):
        if self.state == "CLOSED":
            raise RuntimeError("preparation session is closed")

    def restore():
        attempts.append("restore")
        if len(attempts) == 1:
            raise RuntimeError("reset failed after base close")

    monkeypatch.setattr(PreparationSession, "close", base_close)
    monkeypatch.setattr(PreparationSession, "_check_thread", check_thread)
    session.register_benchmark_resource("state", SimpleNamespace(nbytes=4), restore=restore)
    with pytest.raises(RuntimeError, match="reset failed"):
        session.close()
    assert session.state == "CLOSED" and session._tune_status == "failed"
    assert session._benchmark_restorers
    session.close()
    session.close()
    assert attempts == ["restore", "restore"]
    assert session._tune_status == "closed" and not session._benchmark_restorers








def test_all_component_envelope_reserves_one_arena_before_tuning_and_refreshes_aliases(backend):
    events, controls = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(4)
    session.register_scratch(arena)
    owners = [SimpleNamespace(binding=None) for _ in range(3)]
    primes = []

    def declaration_for(index, component, size, selected_size):
        def prime(state):
            value = scratch_storage(arena, nbytes=4, device="cpu")
            primes.append((index, value.nbytes))
            owners[index].binding = value
            assert not arena.frozen

        request = declaration(f"layer{index}", component=component, size=size,
                              selected_size=selected_size, prepare=prime)
        session.prepare((request,))
        register_finalizer(session, lambda: setattr(owners[index], "binding", arena.tensor))
        return request

    requests = [declaration_for(0, "sequence.gdn_prefill", 300, 120),
                declaration_for(1, "attention.qsa", 513, 257),
                declaration_for(2, "moe.decode", 400, 129)]
    controls.skip_prime.add("layer1")
    report = session.tune()
    assert primes == [(0, 256), (1, 256), (2, 256), (0, 768), (2, 768)]
    assert arena.nbytes == 512 and arena.frozen
    assert all(owner.binding is arena.tensor for owner in owners)
    assert report["scratch_arena"] == {"initial_bytes": 256,
        "candidate_scratch_bytes": 513, "candidate_reserved_bytes": 768,
        "selected_scratch_bytes": 512, "reserved_bytes": 512, "pending_persistent_bytes": 0}
    envelope = ("envelope", tuple(request.name for request in requests))
    assert events.count(envelope) == 1
    assert events.index(envelope) < events.index(("prepare", True, envelope[1]))
    assert report["selection_counts"] == {"tuned": 3}
    assert report["cache_hits"] == 2 and report["benchmarked_candidates"] == 7
    with session.capture():
        assert all(owner.binding is arena.tensor for owner in owners)
    session.freeze()


def test_default_storage_released_after_alias_rebind_and_trial_cleanup_before_report(backend, monkeypatch):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    original = arena.reserve(4)
    reference = weakref.ref(original)
    owner = SimpleNamespace(binding=original)
    session.register_scratch(arena)
    session.prepare((declaration("op", size=513, selected_size=120, prepare=lambda state, saved=original: None),))
    session.register_finalizer(lambda saved=original: setattr(owner, "binding", arena.tensor))
    monkeypatch.setattr(session, "_allocated", lambda: 4096 + arena.nbytes +
        (256 if reference() is not None and reference() is not arena.tensor else 0))
    del original

    def empty_cache():
        assert reference() is None
        assert owner.binding is arena.tensor and arena.nbytes == 256

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    report = session.tune()
    assert reference() is None
    assert report["released_preparation_bytes"] == 256
    assert report["allocated_after_trial_release"] == 4096 + 256


@pytest.mark.parametrize("source", ["cached", "tuned"])
@pytest.mark.parametrize("borrowed_state", [False, True])
def test_selected_arena_retained_after_trials_restoration_and_capture(backend, monkeypatch, source, borrowed_state):
    events, _ = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    references, restored = {}, []
    original_prepare = PreparationSession.prepare

    class Trial:
        def __init__(self, scratch):
            self.scratch = scratch

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        if kwargs["autotune"]:
            trial = Trial(arena.tensor)
            references["trial"] = weakref.ref(trial)
            references["tensor"] = weakref.ref(trial.scratch)
            close = result.close

            def close_trial():
                assert references["tensor"]() is arena.tensor
                trial.scratch.fill_(17)
                close()

            result.close = close_trial
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    session.prepare((declaration("expert", component="moe.decode", source=source),))

    def restore():
        assert references["trial"]() is None
        assert references["tensor"]() is arena.tensor
        assert arena.tensor.eq(17).all()
        assert events[-1] == ("drain",)
        restored.append(True)

    if borrowed_state:
        session.register_benchmark_resource("state", SimpleNamespace(nbytes=0), restore=restore)
    monkeypatch.setattr(session, "_allocated", lambda: 4096 + arena.nbytes)

    def empty_cache():
        assert references["trial"]() is None
        assert references["tensor"]() is arena.tensor

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    report = session.tune()
    assert restored == ([True] if borrowed_state else [])
    assert report["scratch_arena"] == {"initial_bytes": 0, "candidate_scratch_bytes": 128,
        "candidate_reserved_bytes": 256, "selected_scratch_bytes": 256,
        "reserved_bytes": 256, "pending_persistent_bytes": 0}
    assert report["released_preparation_bytes"] == 0
    assert report["allocated_after_trial_release"] == 4096 + 256
    with session.capture():
        assert references["tensor"]() is arena.tensor
    session.freeze()
    session.close()
    assert references["tensor"]() is arena.tensor
    assert arena.tensor.eq(17).all()


def test_arena_registration_is_unique_collecting_only_and_optional(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    with pytest.raises(TypeError):
        session.register_scratch(torch.empty(4, dtype=torch.uint8))
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    with pytest.raises(ValueError):
        session.register_scratch(arena)
    session.prepare((declaration("op"),))
    session.tune()
    with pytest.raises(RuntimeError, match="closed after tuning"):
        session.register_scratch(ScratchArena("cpu"))
    ordinary = RecordedPreparationSession(device=0, checkpoint="revision")
    ordinary.prepare((declaration("op"),))
    assert ordinary.tune()["scratch_arena"] == {}


def test_failed_close_retains_arena_owner_until_successful_drain(backend, monkeypatch):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(17)
    reference = weakref.ref(arena)
    session.register_scratch(arena)
    del arena
    attempts = []

    def close(self):
        attempts.append("close")
        if len(attempts) == 1:
            raise RuntimeError("drain failed")

    sync_attempts = []

    def synchronize():
        sync_attempts.append("sync")
        if len(sync_attempts) == 1:
            raise RuntimeError("fallback drain failed")

    monkeypatch.setattr(PreparationSession, "close", close)
    monkeypatch.setattr(session, "_synchronize", synchronize)
    with pytest.raises(RuntimeError, match="drain failed"):
        session.close()
    assert reference() is not None and session._tune_status == "failed"
    session.close()
    assert reference() is None and session._tune_status == "closed"
    assert attempts == ["close", "close"] and sync_attempts == ["sync", "sync"]


def test_arena_binding_finalizers_precede_other_finalizers_and_close_trial(backend):
    events, controls = backend
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(1)
    session.register_scratch(arena)
    owners = [SimpleNamespace(binding=arena.tensor) for _ in range(2)]
    session.prepare((declaration("owner", size=512), declaration("alias", size=512)))
    controls.skip_prime.update(("owner", "alias"))
    for index, owner in enumerate(owners):
        def finalize(index=index, owner=owner):
            owner.binding = arena.tensor
            events.append(("arena_finalize", index))
        register_finalizer(session, finalize, workspace=arena)

    def inspect_bindings():
        assert all(owner.binding is arena.tensor for owner in owners)
        events.append(("inspect_bindings",))

    register_finalizer(session, inspect_bindings)
    session.tune()
    ordered = [("arena_finalize", 0), ("arena_finalize", 1),
               ("inspect_bindings",), ("result_close", True)]
    assert [events.index(event) for event in ordered] == sorted(events.index(event) for event in ordered)
    assert all(owner.binding.nbytes == 512 for owner in owners)


def test_arena_binding_failure_keeps_callback_owner_until_session_cleanup(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    session.prepare((declaration("op"),))

    class Owner:
        def finalize(self):
            raise RuntimeError("selected binding failed")

    owner = Owner()
    reference = weakref.ref(owner)
    register_finalizer(session, owner.finalize, workspace=arena)
    del owner
    with pytest.raises(RuntimeError, match="selected binding failed"):
        session.tune()
    assert reference() is not None and session._tune_status == "failed"
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()
    session.close()
    assert reference() is None
    assert arena.nbytes == 256


def test_candidate_storage_survives_trial_close_then_dies_before_selected_memory_report(backend, monkeypatch):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    references, order = {}, []
    owner = SimpleNamespace(binding=None)
    original_prepare = PreparationSession.prepare

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        if kwargs["autotune"]:
            candidate = arena.tensor
            references["candidate"] = weakref.ref(candidate)
            close = result.close

            def close_trial():
                assert candidate is not arena.tensor
                assert owner.binding is arena.tensor
                assert candidate.nbytes == 1024 and arena.nbytes == 512
                assert arena.tensor.eq(23).all()
                candidate.fill_(17)
                order.append("close_trial")
                close()

            result.close = close_trial
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    session.prepare((declaration("expert", component="moe.decode", size=1024, selected_size=257),))

    def finalize():
        assert references["candidate"]() is not None
        assert arena.nbytes == 512 and arena.frozen
        arena.tensor.fill_(23)
        owner.binding = arena.tensor
        order.append("finalize")

    register_finalizer(session, finalize, workspace=arena)
    monkeypatch.setattr(session, "_allocated", lambda: 4096 + arena.nbytes +
        (1024 if references.get("candidate", lambda: None)() is not None
         and references["candidate"]() is not arena.tensor else 0))

    def empty_cache():
        assert references["candidate"]() is None
        assert owner.binding is arena.tensor and arena.tensor.eq(23).all()
        order.append("empty_cache")

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    report = session.tune()
    assert order == ["finalize", "close_trial", "empty_cache"]
    assert report["released_preparation_bytes"] == 1024
    assert report["allocated_after_trial_release"] == 4096 + 512
    assert report["scratch_arena"]["candidate_reserved_bytes"] == 1024
    assert report["scratch_arena"]["selected_scratch_bytes"] == report["scratch_arena"]["reserved_bytes"] == 512


def test_undeclared_candidate_growth_fails_before_freeze_finalization_or_capture(backend):
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    finalized = []

    def benchmark(state):
        arena.reserve(513)

    session.prepare((declaration("op", size=128, benchmark=benchmark),))
    register_finalizer(session, lambda: finalized.append(True), workspace=arena)
    with pytest.raises(RuntimeError):
        session.tune()
    assert not finalized and not arena.frozen
    assert session._tune_status == "failed"
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    session.close()


@pytest.mark.parametrize("retain_arena_owner", [False, True])
def test_published_priming_owners_can_release_candidate_storage_only_when_arena_is_retained(
        backend, monkeypatch, retain_arena_owner):
    from sparkgap.preparation.scratch import prime_owners

    session = RecordedPreparationSession(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(16)
    session.register_scratch(arena)
    source, output = torch.ones(4), torch.zeros(4)
    owners = [SimpleNamespace(binding=None), SimpleNamespace(binding=None)]
    references = {}

    def prime(state):
        scratch = arena.tensor
        binding = SimpleNamespace(scratch=scratch, source=source, output=output)
        for owner in owners:
            owner.binding = binding
        references[state.selection.source] = weakref.ref(scratch)
        retained = prime_owners(arena, scratch, binding) if retain_arena_owner else (scratch, binding)
        return PreparedCall(run=lambda: binding.output.copy_(binding.source), owners=retained)

    session.prepare((declaration("operator", size=1024, selected_size=257, prepare=prime),))
    assert session._published_owners["operator"]
    for owner in owners:
        def finalize(owner=owner):
            owner.binding = SimpleNamespace(scratch=arena.tensor, source=source, output=output)
        register_finalizer(session, finalize, workspace=arena)
    del finalize
    monkeypatch.setattr(session, "_allocated", lambda: 4096 + arena.nbytes +
        (1024 if references.get("tuned", lambda: None)() is not None
         and references["tuned"]() is not arena.tensor else 0))
    report = session.tune()
    assert arena.nbytes == 512 and arena.frozen
    assert references["default"]() is None
    assert all(owner.binding.scratch is arena.tensor for owner in owners)
    assert output.eq(source).all()
    if retain_arena_owner:
        assert session._published_owners["operator"] == (arena,)
        assert references["tuned"]() is None
        assert report["allocated_after_trial_release"] == 4096 + 512
    else:
        assert references["tuned"]() is not None
        assert session._published_owners["operator"][0] is references["tuned"]()
        assert report["allocated_after_trial_release"] == 4096 + 512 + 1024
    session.close()
    assert not session._published_owners
    assert references["tuned"]() is None
    assert all(owner.binding.scratch is arena.tensor for owner in owners)


@pytest.mark.parametrize("operator", ["gdn", "fused_moe", "hyperconnection", "varlen_attention", "blockscaled_gemm", "mtp_feedback"])
def test_operator_benchmark_owners_release_after_trial_close(backend, operator):
    from sparkgap.preparation.fixtures import (
        blockscaled_gemm, fused_moe, gdn, hyperconnection, varlen_attention, mtp_feedback,
    )
    from sparkgap.recipes.qwen38.moe import RouterScratch

    events, controls = backend
    controls.retain_trials = True
    session = RecordedPreparationSession(device=0, checkpoint="revision")
    source = torch.zeros(4, 32, dtype=torch.bfloat16)
    output = torch.zeros_like(source)
    references = []
    trial_alive_at_close = []
    controls.before_trial_close = lambda: trial_alive_at_close.append(all(ref() is not None for ref in references))
    specs = lambda: (ScratchBufferSpec(name="workspace", shape=(0,),
        dtype=torch.uint8, device=torch.device("cpu")),)
    state = SimpleNamespace(scratch=SimpleNamespace(scratch_specs=specs),
        scratch_plan=SimpleNamespace(scratch_specs=specs),
        layout=SimpleNamespace(scratch_specs=specs), required_workspace=0,
        bind=lambda **kwargs: SimpleNamespace(**kwargs), run=lambda *args, **kwargs: None,
        launch=lambda *args, **kwargs: None)

    def benchmark(_):
        if operator == "gdn":
            arguments = dict(x=source, out=output, conv_state=torch.zeros(3, 32, 3),
                conv_weight=torch.ones(32, 4), query_start_loc=torch.zeros(2, dtype=torch.int32),
                state_slot_ids=torch.full((1,), -1, dtype=torch.int64),
                state_is_fresh=torch.zeros(1, dtype=torch.bool),
                num_accepted_tokens=torch.ones(1, dtype=torch.int32),
                num_seqs=torch.zeros(1, dtype=torch.int32), num_tokens=torch.zeros(1, dtype=torch.int32))
            call = gdn.benchmark_call(state, arguments=arguments, session=session, name="op",
                mode="prefill", scratch_specs=lambda _: specs(), workspace=None,
                run_options={}, binding_plan=object())
        elif operator == "fused_moe":
            call = fused_moe.benchmark_call(state, source=source,
                experts=SimpleNamespace(num_experts=512), top_k=10, session=session, name="op",
                workspace=None, allocate_router=RouterScratch.allocate, router_weight=None, router_plan=None)
        elif operator == "hyperconnection":
            call = hyperconnection.benchmark_call(state, "scaled_silu", (source, output), {})
        elif operator == "varlen_attention":
            arguments = dict(q=source.view(4, 2, 16), k=source.view(4, 2, 16),
                v=source.view(4, 2, 16), cu_seqlens_q=torch.zeros(3, dtype=torch.int32))
            call = varlen_attention.benchmark_call(state, arguments=arguments,
                session=session, workspace=None)
        elif operator == "mtp_feedback":
            call = mtp_feedback.benchmark_call(state,
                arguments=dict(token_embedding=source, multi_state=source, output=output),
                session=session, name="op", workspace=None, eps=1e-6)
        else:
            call = blockscaled_gemm.benchmark_call(state, source=source, output=output,
                values=torch.ones(32, 32), scales=None, gain=None, bias=None,
                weight=object(), selected_workspace=lambda _: None)
        if call.reset is not None:
            call.reset()
        call.produce()
        call.invoke()
        references.append(weakref.ref(call.owners[0]))
        return call

    session.prepare((declaration("op", benchmark=benchmark),))
    session.tune()
    assert trial_alive_at_close == [True]
    assert references and all(ref() is None for ref in references)
    assert not session._benchmark_resources
    assert not source.count_nonzero() and not output.count_nonzero()
    with session.capture():
        pass
