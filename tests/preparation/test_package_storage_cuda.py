"""Preparation storage preserves GPU capabilities and mapped mailbox publication."""

import os
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_package_weights_support_gpudirect_registration():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    subprocess.run([sys.executable, "-c", """
import ctypes as C
import torch
from ignition.preparation.package_storage import PackageStorage

torch.empty(1, device="cuda:0")
with PackageStorage(device=0) as storage:
    weight = torch.empty(4 << 20, dtype=torch.uint8, device="cuda:0")
    supported, capable = C.c_int(), C.c_uint()
    storage.cuda.call("cuDeviceGetAttribute", C.byref(supported), C.c_int(116), C.c_int(0))
    storage.cuda.call("cuPointerGetAttribute", C.byref(capable), C.c_int(15), C.c_uint64(weight.data_ptr()))
    assert bool(capable.value) == bool(supported.value), (capable.value, supported.value)
"""], check=True, timeout=60)


@pytest.fixture(scope="module")
def mailbox_storage():
    import torch
    from ignition.preparation.package_storage import PackageStorage

    torch.cuda.set_device(0)
    with PackageStorage(device=0) as storage:
        yield storage


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
@pytest.mark.parametrize("count", [96, 65536])
def test_mapped_mailbox_publishes_rows_during_graph_replay(count, mailbox_storage):
    import torch
    from ignition.components.ngram_delivery import publish_requests

    with torch.cuda.device(0):
        storage = mailbox_storage
        region = storage.host_allocate("requests", (count + 1,), torch.int64)
        graph = torch.cuda.CUDAGraph()
        try:
            assert region.host_view.data_ptr() == region.device_view.data_ptr() == region.address
            ids = torch.arange(count, device="cuda", dtype=torch.int64)
            sequence = torch.ones(1, device="cuda", dtype=torch.int64)
            expected = torch.arange(count, dtype=torch.int64)
            publish_requests(ids, region.device_view, sequence)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                publish_requests(ids, region.device_view, sequence)
            for value in (17, 127, 2048):
                ids.add_(1)
                expected.add_(1)
                sequence.fill_(value)
                torch.cuda.synchronize()
                region.host_view.fill_(-1)
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                deadline = time.monotonic() + 10
                while region.host_view[-1].item() != value:
                    assert time.monotonic() < deadline, "mailbox publication timed out"
                    time.sleep(.001)
                torch.testing.assert_close(region.host_view[:-1], expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        finally:
            graph.reset()
            region.close()
