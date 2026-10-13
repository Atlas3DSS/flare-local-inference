"""Package storage keeps only requested allocator bytes and, on unified memory, library
allocations above the addresses a serving context occupies."""

import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.reachability import Reachability
from ignition.preparation.runtime_package import Allocation, _without_zero_ranges, _write_allocation


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


class Driver:
    def __init__(self, allocations):
        self.allocations, self.calls = list(allocations), []

    def call(self, name, *args):
        self.calls.append(name)
        if name == "cuMemAlloc_v2":
            args[0]._obj.value = self.allocations.pop(0)
        elif name == "cuMemAddressReserve":
            pointer, size, alignment, address, _ = args
            self.reservation = (address.value, size.value, alignment.value)
            pointer._obj.value = address.value


def storage(driver):
    return SimpleNamespace(cuda=driver, granularity=2 << 20, EXTERNAL_FLOOR=PackageStorage.EXTERNAL_FLOOR)


def test_unified_memory_reserves_the_window_below_the_external_floor():
    driver = Driver([0x327e00000, 0x411800000])
    window = PackageStorage._reserve_low_window(storage(driver))
    assert window == (0x327e00000, 0x400000000 - 0x327e00000)
    assert driver.reservation == (0x327e00000, 0x400000000 - 0x327e00000, 2 << 20)
    assert driver.calls == ["cuMemAlloc_v2", "cuMemFree_v2", "cuMemAddressReserve", "cuMemAlloc_v2", "cuMemFree_v2"]


def test_allocations_already_above_the_floor_need_no_window():
    driver = Driver([0x500000000])
    assert PackageStorage._reserve_low_window(storage(driver)) is None
    assert "cuMemAddressReserve" not in driver.calls


def test_a_window_that_does_not_move_allocations_is_released():
    driver = Driver([0x327e00000, 0x328000000])
    with pytest.raises(RuntimeError, match="still start at 0x328000000"):
        PackageStorage._reserve_low_window(storage(driver))
    assert driver.calls[-1] == "cuMemAddressFree"
