"""Device discovery with discrete VRAM and Linux shared system memory."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ignition import planning
from ignition.errors import CapacityError


GIB = 1 << 30


@pytest.fixture
def devices(monkeypatch):
    class NotSupportedError(Exception):
        pass

    calls = []
    nvml = SimpleNamespace(
        NotSupportedError=NotSupportedError,
        init_v2=lambda: calls.append("init"),
        shutdown=lambda: calls.append("shutdown"),
        device_get_count_v2=lambda: 1,
        device_get_handle_by_index_v2=lambda index: index,
        device_get_name=lambda handle: "NVIDIA GB10",
        device_get_uuid=lambda handle: "GPU-spark",
        device_get_cuda_compute_capability=lambda handle: (12, 1),
        device_get_pci_info_v3=lambda handle: SimpleNamespace(bus_id="0000000f:01:00.0"),
        device_get_memory_info_v2=lambda handle: SimpleNamespace(total=96 * GIB, free=80 * GIB),
    )

    def by_bus(bus):
        assert bus == b"0000000f:01:00.0"
        return 0, 7

    def attribute(attribute, device):
        assert attribute == 18 and device == 7
        return 0, 1

    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUdevice_attribute=SimpleNamespace(CU_DEVICE_ATTRIBUTE_INTEGRATED=18),
        cuInit=lambda flags: (calls.append("driver_init") or 0,),
        cuDeviceGetByPCIBusId=by_bus,
        cuDeviceGetAttribute=attribute,
        cuDeviceTotalMem=lambda device: (0, 121 * GIB),
    )
    bindings = ModuleType("cuda.bindings")
    bindings.nvml, bindings.driver = nvml, driver
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    return nvml, driver, calls


def test_discrete_discovery_does_not_initialize_cuda(devices):
    _, _, calls = devices
    gpu, = planning.discover_gpus()
    assert (gpu.total_bytes, gpu.free_bytes) == (96 * GIB, 80 * GIB)
    assert not gpu.unified
    assert calls == ["init", "shutdown"]


def unsupported(nvml, monkeypatch):
    def memory(handle):
        raise nvml.NotSupportedError()
    monkeypatch.setattr(nvml, "device_get_memory_info_v2", memory)


@pytest.mark.parametrize("available,expected", [(100, 100), (128, 121)])
def test_integrated_discovery_uses_available_dram_without_swap(devices, monkeypatch, available, expected):
    nvml, _, calls = devices
    unsupported(nvml, monkeypatch)
    def meminfo(path):
        assert str(path) == "/proc/meminfo"
        return f"MemFree: 1024 kB\nMemAvailable: {available * (GIB // 1024)} kB\nSwapFree: 999999999 kB\n"
    monkeypatch.setattr(planning.Path, "read_text", meminfo)
    gpu, = planning.discover_gpus()
    assert (gpu.total_bytes, gpu.free_bytes, gpu.capability) == (121 * GIB, expected * GIB, (12, 1))
    assert gpu.unified
    assert calls == ["init", "driver_init", "shutdown"]


def test_missing_vram_report_requires_an_integrated_device(devices, monkeypatch):
    nvml, driver, calls = devices
    unsupported(nvml, monkeypatch)
    monkeypatch.setattr(driver, "cuDeviceGetAttribute", lambda attribute, device: (0, 0))
    with pytest.raises(CapacityError, match="not an integrated GPU"):
        planning.discover_gpus()
    assert calls[-1] == "shutdown"


def test_integrated_discovery_requires_available_memory(devices, monkeypatch):
    nvml, _, calls = devices
    unsupported(nvml, monkeypatch)
    monkeypatch.setattr(planning.Path, "read_text", lambda path: "MemFree: 1024 kB\nSwapFree: 999999999 kB\n")
    with pytest.raises(CapacityError, match="MemAvailable is required"):
        planning.discover_gpus()
    assert calls[-1] == "shutdown"
