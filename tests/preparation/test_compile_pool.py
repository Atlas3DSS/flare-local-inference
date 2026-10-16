"""Compiler workers receive tensor layouts and aliases without tensor contents."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses import FakeTensor

from ignition.preparation.forward_metadata import dumps, loads


@pytest.mark.parametrize("integrated,expected,threads", [(False, 8, 2), (True, 4, 1)])
def test_compiler_concurrency_accounts_for_shared_device_ram(monkeypatch, integrated, expected, threads):
    from ignition.preparation import compile_pool, host_memory

    free = (24 if host_memory.os.name == "nt" else 2) if integrated else 40
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 << 30)
    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    pool = compile_pool.CompilePool("cuda:0")
    assert pool.workers == expected
    # Four integrated workers leave no RAM for additional Inductor subprocesses.
    assert pool.options["compile_threads"] == threads


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


def test_checkpoint_model_class_loads_in_fresh_worker(tmp_path, monkeypatch):
    import subprocess
    import sys
    from transformers import dynamic_module_utils

    cache = tmp_path / 'modules'
    source = tmp_path / 'checkpoint'
    source.mkdir()
    (source / 'modeling_test.py').write_text(
        'import torch\n'
        'class CheckpointLayer(torch.nn.Linear):\n'
        '    pass\n'
    )
    monkeypatch.setattr(dynamic_module_utils, 'HF_MODULES_CACHE', str(cache))
    layer = dynamic_module_utils.get_class_from_dynamic_module(
        'modeling_test.CheckpointLayer', str(source), local_files_only=True)
    payload = tmp_path / 'forward.pkl'
    payload.write_bytes(dumps(layer(16, 32, device='meta')))
    code = '''
import sys
from pathlib import Path
from transformers import dynamic_module_utils
from torch._subclasses import FakeTensor
from ignition.preparation.forward_metadata import loads
dynamic_module_utils.HF_MODULES_CACHE = sys.argv[1]
assert sys.argv[1] not in sys.path
assert not any(name.startswith('transformers_modules.') for name in sys.modules)
layer = loads(Path(sys.argv[2]).read_bytes())
assert type(layer).__name__ == 'CheckpointLayer'
assert isinstance(layer.weight, FakeTensor)
assert tuple(layer.weight.shape) == (32, 16)
'''
    subprocess.run([sys.executable, '-c', code, str(cache), str(payload)],
                   check=True, capture_output=True, text=True)


def test_preparation_closures_cannot_execute_in_compiler():
    owner = SimpleNamespace(callback=lambda: pytest.fail('preparation must not run'))
    restored = loads(dumps(owner))
    with pytest.raises(RuntimeError, match='process-local resource'):
        restored.callback.anything


@pytest.mark.parametrize("kind", ["storage", "session", "recorded_session"])
def test_preparation_resources_do_not_serialize_active_package_storage(kind):
    from b12x.preparation import PreparationSession
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tensor_metadata import NumericStorage, _NUMERIC_STORAGE
    from ignition.preparation.tuning import RecordedPreparationSession

    with NumericStorage() as registry:
        storage = PackageStorage.__new__(PackageStorage)
        storage.numeric_storage = registry
        resource = storage
        if kind != "storage":
            cls = PreparationSession if kind == "session" else RecordedPreparationSession
            resource = cls.__new__(cls)
            resource.benchmark_storage = storage
        restored = loads(dumps(SimpleNamespace(resource=resource, alias=resource)))
        assert restored.resource is restored.alias
        with pytest.raises(RuntimeError, match="process-local resource"):
            restored.resource.numeric_storage
        assert storage.numeric_storage is registry and _NUMERIC_STORAGE.get() is registry


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



@pytest.mark.parametrize(("available_gib", "threads"), [(12, 1), (16, 8), (64, 16)])
def test_integrated_compile_threads_fit_the_ram_workers_leave(available_gib, threads):
    from ignition.preparation.host_memory import compile_threads

    assert compile_threads(16, workers=1, available=available_gib << 30, device_bytes=2 << 30) == threads


@pytest.mark.parametrize(("free_gib", "workers", "threads"), [(95, 8, 2), (3, 1, 16)])
def test_compile_pool_sizes_workers_from_free_device_memory(monkeypatch, free_gib, workers, threads):
    from ignition.preparation import compile_pool

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_gib << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: 128 << 30)
    pool = compile_pool.CompilePool("cuda:0")
    assert (pool.workers, pool.options["compile_threads"]) == (workers, threads)


@pytest.mark.parametrize("integrated,workers,threads", [(False, 8, 2), (True, 4, 1)])
def test_compiler_reuses_its_budgeted_worker_pool(monkeypatch, tmp_path, integrated, workers, threads):
    from concurrent.futures import Future
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 << 30)
    executors = []

    class Executor:
        def __init__(self, count, *, kind=None, mp_context=None):
            self.calls, self.closed = [], False
            assert count == workers
            if compile_pool.os.name == "nt":
                assert mp_context.get_start_method() == "spawn"
            else:
                assert kind == subproc_pool.SubprocKind.SPAWN
            executors.append(self)

        def submit(self, function, *arguments):
            self.calls.append(arguments)
            future = Future()
            future.set_result(None)
            return future

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(subproc_pool, "SubprocPool", Executor)
    monkeypatch.setattr(compile_pool, "ProcessPoolExecutor", Executor)
    with compile_pool.CompilePool("cuda:0") as pool:
        pool.submit(SimpleNamespace(), tmp_path / "first.bin")
        pool.submit(SimpleNamespace(), tmp_path / "second.bin")
    (executor,) = executors
    assert [call[3] for call in executor.calls] == [str(tmp_path / "first.bin"), str(tmp_path / "second.bin")]
    assert all(call[2]["compile_threads"] == threads for call in executor.calls)
    assert executor.closed and compile_pool._OPTIONS["compile_threads"] == 16


@pytest.mark.parametrize("integrated,workers", [(False, 8), (True, 2)])
def test_compilation_preserves_configured_headroom(monkeypatch, integrated, workers):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 << 30)
    assert compile_pool.CompilePool("cuda:0", headroom_bytes=16 << 30).workers == workers
    with pytest.raises(MemoryError, match="headroom"):
        compile_pool.CompilePool("cuda:0", headroom_bytes=39 << 30)


@pytest.mark.parametrize("pools,workers", [(1, 8), (2, 4), (4, 2)])
def test_ranks_sharing_a_host_split_its_compilers(monkeypatch, pools, workers):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 << 30)
    pool = compile_pool.CompilePool("cuda:0", pools=pools)
    # Every rank's pool together stays within the host's threads and RAM.
    assert pool.workers == workers and pools * pool.workers * pool.options["compile_threads"] == 16


def test_windows_compiler_workers_fit_host_ram_with_larger_device_budget(monkeypatch):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setattr(compile_pool, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(compile_pool, "device_memory_info", lambda device: (105 << 30, 110 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 << 30)
    pool = compile_pool.CompilePool("cuda:0", headroom_bytes=8 << 30)
    assert (pool.workers, pool.options["compile_threads"]) == (4, 1)
