"""Preparation allocations expose the device's GPUDirect memory capability."""

import os
import subprocess
import sys

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
