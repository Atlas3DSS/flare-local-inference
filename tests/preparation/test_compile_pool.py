"""Compiler workers receive tensor layouts and aliases without tensor contents."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses import FakeTensor

from ignition.preparation.forward_metadata import dumps, loads


@pytest.mark.parametrize("integrated,expected", [(False, 8), (True, 4)])
def test_compiler_concurrency_accounts_for_shared_device_ram(monkeypatch, integrated, expected):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: ((2 if integrated else 40) << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 << 30)
    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    pool = compile_pool.CompilePool("cuda:0")
    assert pool.workers == expected
    assert pool.options["compile_threads"] * pool.workers == 16


def test_metadata_preserves_tensor_aliases_and_offsets():
    storage = torch.empty(1024, dtype=torch.uint8)
    original = SimpleNamespace(storage=storage, part=storage.view(torch.float32)[16:48],
        alias=storage.view(torch.bfloat16), independent=torch.empty(1024, dtype=torch.uint8))
    original.again = original.part
    restored = loads(dumps(original))
    assert restored.again is restored.part
    assert isinstance(restored.storage, FakeTensor)
    for name in ('storage', 'part', 'alias', 'independent'):
        source, target = getattr(original, name), getattr(restored, name)
        assert (target.shape, target.stride(), target.storage_offset(), target.dtype, target.device) == (
            source.shape, source.stride(), source.storage_offset(), source.dtype, source.device)
    assert restored.storage.untyped_storage()._cdata == restored.part.untyped_storage()._cdata
    assert restored.storage.untyped_storage()._cdata == restored.alias.untyped_storage()._cdata
    assert restored.storage.untyped_storage()._cdata != restored.independent.untyped_storage()._cdata


def test_metadata_size_is_independent_of_storage_size():
    small = torch.empty(16, device='meta')
    large = torch.empty(1 << 35, device='meta')
    assert len(dumps(large)) < len(dumps(small)) + 32
    restored = loads(dumps(large))
    assert restored.numel() == 1 << 35
    assert restored.untyped_storage().device.type == 'meta'


def test_preparation_closures_cannot_execute_in_compiler():
    owner = SimpleNamespace(callback=lambda: pytest.fail('preparation must not run'))
    restored = loads(dumps(owner))
    with pytest.raises(RuntimeError, match='process-local resource'):
        restored.callback.anything


def test_native_lock_methods_are_not_rebound_in_worker():
    from threading import Lock

    lock = Lock()
    restored = loads(dumps(SimpleNamespace(lock=lock, acquire=lock.acquire)))
    assert not lock.locked()
    with pytest.raises(RuntimeError, match='process-local resource'):
        restored.lock.acquire


def test_selected_plan_metadata_survives_without_registering_a_worker_plan():
    from b12x.preparation.types import Plan, plan_from_handle, require_prepared

    plan = Plan(contract=SimpleNamespace(component_id='test.layout'), query=None,
                _compile_jobs=lambda *args: (), _memory_requirements=lambda *args: None,
                _materialize=lambda *args: None)
    selected = SimpleNamespace(closed=False, state=SimpleNamespace(tiles=17, scratch=torch.empty(32)),
                               selection=SimpleNamespace(config='selected'))
    object.__setattr__(plan, '_prepared', selected)
    restored = loads(dumps(plan))
    state = require_prepared(restored, 'test.layout')
    assert state.tiles == 17 and isinstance(state.scratch, FakeTensor)
    assert restored.selection.config == 'selected'
    assert restored.handle == plan.handle and plan_from_handle(plan.handle) is plan


def test_attention_layout_restores_before_forwarding_attribute_lookups():
    from b12x.attention._shared.contiguous.api import AttentionPlan, VarlenAttentionPlan

    for kind in (AttentionPlan, VarlenAttentionPlan):
        plan = kind(key=SimpleNamespace(tokens=128), compiled=object(), cutlass_dtype=float)
        restored = loads(dumps(plan))
        assert restored.tokens == 128
