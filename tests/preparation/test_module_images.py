"""Lazy module loads bind queried functions and discard bindings at module unload."""

import ctypes as C
import threading
import weakref

import pytest

from ignition.preparation import module_images
from ignition.preparation.module_images import ModuleImages


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(module_images, "cupti_library", lambda library=None: object())
    return ModuleImages()


def launch(recorder, name, handle, context, module_id):
    """Record a materializing driver call binding `handle` to a freshly loaded module."""
    buffer = C.create_string_buffer(b"cubin")
    module = module_images._Module(module_id, len(buffer.value), C.cast(buffer, C.c_void_p))
    resource = module_images._Resource(C.c_void_p(context), None, C.cast(C.pointer(module), C.c_void_p))
    recorder._record(None, 1, 0, C.pointer(module_images._Call(site=0, name=name.encode())))
    recorder._record(None, 3, 6, C.pointer(resource))
    values = (C.c_uint64 * 2)(*((handle, 0) if name == "cuLaunchKernel" else (0, handle)))
    result = C.c_int(0)
    recorder._record(None, 1, 0, C.pointer(module_images._Call(
        site=1, name=name.encode(), params=C.cast(values, C.c_void_p),
        result=C.pointer(result), context=C.c_void_p(context))))


def unload(recorder, context, module_id):
    module = module_images._Module(module_id, 0, None)
    resource = module_images._Resource(C.c_void_p(context), None, C.cast(C.pointer(module), C.c_void_p))
    recorder._record(None, 3, 7, C.pointer(resource))


@pytest.mark.parametrize("name,index,kernel", [
    ("cuFuncGetAttribute", 2, False),
    ("cuKernelGetAttribute", 2, True),
    ("cuOccupancyMaxActiveBlocksPerMultiprocessor", 1, False),
    ("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", 1, False),
    ("cuOccupancyMaxPotentialBlockSize", 2, False),
    ("cuOccupancyMaxPotentialBlockSizeWithFlags", 2, False),
    ("cuOccupancyAvailableDynamicSMemPerBlock", 1, False),
    ("cuOccupancyMaxPotentialClusterSize", 1, False),
    ("cuOccupancyMaxActiveClusters", 1, False),
])
@pytest.mark.parametrize("status", [0, 1])
def test_query_binds_module_even_when_launch_configuration_is_rejected(recorder, name, index, kernel, status):
    image = C.create_string_buffer(b"queried cubin")
    module = module_images._Module(7, len(image.value), C.cast(image, C.c_void_p))
    resource = module_images._Resource(C.c_void_p(0x1000), None, C.cast(C.pointer(module), C.c_void_p))
    recorder._record(None, 1, 0, C.pointer(module_images._Call(site=0, name=name.encode())))
    recorder._record(None, 3, 6, C.pointer(resource))
    values = (C.c_uint64 * (index + 1))()
    values[index] = 0x1234
    result = C.c_int(status)
    recorder._record(None, 1, 0, C.pointer(module_images._Call(
        site=1, name=name.encode(), params=C.cast(values, C.c_void_p), result=C.pointer(result))))
    assert recorder._error is None
    bindings = recorder._kernels if kernel else recorder._functions
    assert bindings[0x1234].cubin == b"queried cubin"
    unload(recorder, 0x1000, 7)
    assert 0x1234 not in bindings


def test_unload_drops_only_the_unloaded_module(recorder):
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    launch(recorder, "cuKernelSetAttribute", 0x22, 0x1000, 7)
    launch(recorder, "cuLaunchKernel", 0x33, 0x1000, 9)
    unload(recorder, 0x1000, 7)
    assert 0x11 not in recorder._functions
    assert 0x22 not in recorder._kernels
    assert recorder._functions[0x33].module_id == 9
    assert recorder._bindings == {(0x1000, 9): {0x33}}
    assert set(recorder.images) == {(0x1000, 9)}


def test_rebound_handle_survives_unload(recorder):
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 9)
    unload(recorder, 0x1000, 7)
    assert recorder._functions[0x11].module_id == 9
    assert recorder._bindings == {(0x1000, 9): {0x11}}


def test_unload_respects_context(recorder):
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    launch(recorder, "cuLaunchKernelEx", 0x22, 0x2000, 7)
    unload(recorder, 0x1000, 7)
    assert 0x11 not in recorder._functions
    assert recorder._functions[0x22].context == 0x2000
    assert set(recorder.images) == {(0x2000, 7)}


def test_unloading_an_unbound_module_changes_nothing(recorder):
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    unload(recorder, 0x1000, 42)
    assert recorder._functions[0x11].module_id == 7


def test_repeated_launches_do_not_grow_bookkeeping(recorder):
    for _ in range(50):
        launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    assert len(recorder._bindings[(0x1000, 7)]) == 1
    assert len(recorder._functions) == 1
    assert len(recorder.images) == 1
    unload(recorder, 0x1000, 7)
    assert recorder._bindings == {}
    assert recorder._functions == {}
    assert recorder.images == {}


def test_unloaded_images_are_released_while_live_images_remain(recorder):
    launch(recorder, "cuLaunchKernel", 0x11, 0x1000, 7)
    live = weakref.ref(recorder.images[(0x1000, 7)])
    for module_id in range(8, 58):
        launch(recorder, "cuKernelSetAttribute", 0x22, 0x1000, module_id)
        discarded = weakref.ref(recorder.images[(0x1000, module_id)])
        unload(recorder, 0x1000, module_id)
        assert discarded() is None
        assert live() is recorder._functions[0x11]
        assert len(recorder.images) == 1


def test_context_destruction_releases_its_images_and_bindings(recorder):
    launch(recorder, "cuLaunchKernel", 0x11, 0x1000, 7)
    launch(recorder, "cuKernelSetAttribute", 0x22, 0x1000, 8)
    launch(recorder, "cuLaunchKernel", 0x33, 0x2000, 7)
    discarded = [weakref.ref(recorder.images[(0x1000, module_id)]) for module_id in (7, 8)]
    resource = module_images._Resource(C.c_void_p(0x1000), None, None)
    recorder._record(None, 3, 2, C.pointer(resource))
    assert recorder._error is None
    assert all(ref() is None for ref in discarded)
    assert set(recorder.images) == {(0x2000, 7)}
    assert set(recorder._functions) == {0x33}
    assert not recorder._kernels
    assert recorder._bindings == {(0x2000, 7): {0x33}}


def test_rebind_keeps_the_other_table_indexed(recorder):
    launch(recorder, "cuKernelSetAttribute", 0x11, 0x1000, 7)
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 9)
    unload(recorder, 0x1000, 7)
    assert 0x11 not in recorder._kernels
    assert recorder._functions[0x11].module_id == 9
    assert recorder._bindings == {(0x1000, 9): {0x11}}


def test_unload_waits_for_a_concurrent_binding_to_be_indexed(recorder):
    """An unload racing a binding on another driver thread must observe the index update.

    The binding side continues only once the unload has engaged — either by
    attempting the binding lock or by running its index pop — so the index
    update provably completes after the unload started, every run.
    """
    engaged = threading.Event()
    unloader = []
    lock = threading.Lock()

    class HandshakeLock:
        def __enter__(self):
            if threading.current_thread() is next(iter(unloader), None):
                engaged.set()
            lock.acquire()
            return self

        def __exit__(self, *exception):
            lock.release()
            return False

    class UnloadingBindings(dict):
        def setdefault(self, key, default):
            value = super().setdefault(key, default)
            thread = threading.Thread(target=recorder._drop_module_bindings, args=key)
            unloader.append(thread)
            thread.start()
            engaged.wait(5)  # failure guard; the handshake itself is not timed
            return value

        def pop(self, key, default=None):
            value = super().pop(key, default)
            engaged.set()
            return value

    recorder._binding_lock = HandshakeLock()
    recorder._bindings = UnloadingBindings()
    launch(recorder, "cuLaunchKernelEx", 0x11, 0x1000, 7)
    for thread in unloader:
        thread.join()
    assert 0x11 not in recorder._functions
    assert recorder._bindings == {}
