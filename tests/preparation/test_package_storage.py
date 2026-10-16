"""Package storage keeps only requested allocator bytes and, on unified memory, library
allocations above the addresses a serving context occupies."""

import ctypes
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.reachability import Reachability
from ignition.preparation.runtime_package import Allocation, _without_zero_ranges, _write_allocation


def test_cuda_access_descriptor_array_matches_driver_abi():
    from ignition.preparation.package_storage import _Access, _Location

    access = (_Access * 2)(_Access(_Location(3, 0), 3), _Access(_Location(1, 7), 3))
    assert bytes(access) == struct.pack("<6i", 3, 0, 3, 1, 7, 3)
    assert ctypes.sizeof(_Access) == 12


def test_unused_bytes_include_live_block_padding_and_retired_pools(monkeypatch, tmp_path):
    base = 0x200000000000
    storage = PackageStorage.__new__(PackageStorage)
    storage.device = 0
    storage._preparation_pools = {(0, 1), (0, 2)}

    def block(offset, size, requested, state="active_allocated"):
        return dict(address=base + offset, size=size, requested_size=requested, state=state)

    def segment(device, pool, blocks):
        return dict(device=device, segment_pool_id=pool, blocks=blocks)

    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [
        segment(0, (0, 1), [block(0, 64, 32), block(64, 32, 0, "inactive"), block(96, 32, 32)]),
        segment(0, (0, 2), [block(128, 64, 24, "active_awaiting_free")]),
        segment(0, (0, 3), [block(192, 64, 0, "inactive")]),
        segment(1, (0, 1), [block(256, 64, 0, "inactive")]),
    ])
    unused = storage.unused_ranges()
    assert unused == [(base + 32, base + 64), (base + 64, base + 96), (base + 152, base + 192)]

    live, unrelated = base + 4096, base + 8192
    data = bytearray(192)
    struct.pack_into("<Q", data, 40, unrelated)
    struct.pack_into("<Q", data, 64, unrelated)
    struct.pack_into("<Q", data, 96, live)
    struct.pack_into("<Q", data, 160, unrelated)
    allocation = _without_zero_ranges(Allocation("mixed", base, len(data)),
                                      [(start, end - start) for start, end in unused])
    traversal = Reachability([allocation, Allocation("live", live, 8), Allocation("unrelated", unrelated, 8)],
                             roots=("mixed",), inactive=unused)

    def download(address, size):
        assert all(address + size <= start or address >= end for start, end in unused)
        return bytes(data[address - base:address - base + size])

    _write_allocation(SimpleNamespace(download=download), allocation, tmp_path / "tensor.bin",
                      visit=traversal.contents)
    assert traversal.reached == {"mixed", "live"}


@pytest.mark.skipif(os.name == "nt", reason="Linux package address layout")
def test_linux_device_ranges_avoid_mimalloc_arena_hints():
    # mimalloc hints its aligned arena reservations into 2-6 TiB; PyTorch allocates CPU tensors there.
    hint_start, hint_end = 2 << 40, 6 << 40
    for rank in range(16):
        start = PackageStorage.DEVICE_BASE + rank * PackageStorage.RESERVATION_BYTES
        end = start + PackageStorage.RESERVATION_BYTES
        assert end <= hint_start or start >= hint_end


@pytest.mark.skipif(os.name != "nt", reason="Windows WDDM address layout")
def test_windows_host_range_stays_below_stalling_wddm_addresses():
    # RTX Spark's driver stalls on host-address hints at 16 TiB and 32 TiB.
    assert PackageStorage.DEVICE_BASE + PackageStorage.RESERVATION_BYTES <= PackageStorage.HOST_BASE
    assert PackageStorage.HOST_BASE + PackageStorage.RESERVATION_BYTES <= 16 << 40
