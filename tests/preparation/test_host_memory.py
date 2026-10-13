"""Aggregate compiler concurrency beside mapped model storage."""

import pytest
from types import SimpleNamespace

from ignition.preparation import host_memory
from ignition.preparation.host_memory import worker_limit


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("gib", [16, 24, 64, 128])
def test_compiler_pools_leave_host_headroom(pools, gib):
    workers = worker_limit(8, pools=pools, available=gib << 30)
    assert 1 <= workers <= 8
    assert pools * workers * (2 << 30) + (8 << 30) <= gib << 30


def test_insufficient_host_ram_fails_before_workers_start():
    with pytest.raises(MemoryError, match="available host memory"):
        worker_limit(8, pools=4, available=15 << 30)


def test_unified_memory_budgets_host_and_device_storage_together():
    assert worker_limit(8, available=24 << 30) == 8
    assert worker_limit(8, available=24 << 30, device_bytes=2 << 30) == 4
    assert worker_limit(8, pools=2, available=24 << 30, device_bytes=2 << 30) == 2
    with pytest.raises(MemoryError, match="available host memory"):
        worker_limit(8, available=11 << 30, device_bytes=2 << 30)


@pytest.mark.parametrize("available,expected", [(12 << 30, 12 << 30), (40 << 30, 32 << 30)])
def test_integrated_device_uses_available_ram(monkeypatch, available, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: available)
    assert host_memory.device_memory_info("cuda:0") == (expected, 32 << 30)


def test_discrete_device_budget_is_independent_of_host_ram(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (24 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: pytest.fail("discrete VRAM must not use host capacity"))
    assert host_memory.device_memory_info("cuda:0") == (24 << 30, 32 << 30)
