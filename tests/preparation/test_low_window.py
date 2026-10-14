"""Sealing the low device window keeps every free span below the external floor reserved."""

import ctypes as C

from ignition.preparation.package_storage import PackageStorage

GRANULE = 2 << 20


class AddressSpace:
    """cuMemAddressReserve over occupied spans: a busy hint yields another address, as the driver does."""

    def __init__(self, occupied):
        self.taken = list(occupied)
        self.calls = 0

    def free(self, address, size):
        return all(address + size <= start or start + length <= address for start, length in self.taken)

    def call(self, name, *args):
        self.calls += 1
        if name == "cuMemAlloc_v2":
            # The driver's next device allocation goes to the lowest free granule.
            address = next(a for a in range(PackageStorage.LOW_SCAN_START, 1 << 45, GRANULE)
                           if self.free(a, GRANULE))
            C.cast(args[0], C.POINTER(C.c_uint64))[0] = address
        elif name == "cuMemAddressReserve":
            window, size, _, hint, _ = args
            size, hint = size.value, hint.value
            address = hint if self.free(hint, size) else 1 << 45
            self.taken.append((address, size))
            C.cast(window, C.POINTER(C.c_uint64))[0] = address
        elif name == "cuMemAddressFree":
            self.taken.remove((args[0].value, args[1].value))


def storage(space):
    owner = object.__new__(PackageStorage)
    owner.integrated, owner.granularity, owner.cuda, owner.low_reservations = True, GRANULE, space, []
    return owner


def test_every_free_span_below_the_floor_is_reserved_and_frees_are_resealed():
    floor, start = PackageStorage.EXTERNAL_FLOOR, PackageStorage.LOW_SCAN_START
    context = (0x327c00000, 0x32ee00000 - 0x327c00000)
    scratch = (0x380000000, 64 << 20)
    space = AddressSpace([context, scratch])
    owner = storage(space)
    owner.seal_low_window()
    reserved = sorted(owner.low_reservations)
    # Everything from the scan start to the floor is either occupied or reserved, and nothing more.
    spans = sorted(reserved + [context, scratch])
    assert spans[0][0] == start and spans[-1][0] + spans[-1][1] == floor
    assert all(a + n == b for (a, n), (b, _) in zip(spans, spans[1:]))
    # Storage freed later reopens a span; sealing again reserves just that span.
    space.taken.remove(scratch)
    calls = space.calls
    owner.seal_low_window()
    assert sorted(owner.low_reservations) == sorted(reserved + [scratch])
    # Held spans are skipped, and a taken granule costs one probe (a reserve and a free).
    assert space.calls - calls < 2 * (context[1] // GRANULE) + 50


def test_low_external_allocations_are_named_on_an_integrated_gpu():
    owner = storage(AddressSpace([]))
    assert owner.low_allocations({0x327c00000: 8 << 20, 0x41b400000: 8 << 20}) == [(0x327c00000, 8 << 20)]
    owner.integrated = False
    assert owner.low_allocations({0x327c00000: 8 << 20}) == []


def test_setup_seals_the_window_and_verifies_allocations_land_above_the_floor():
    import pytest

    space = AddressSpace([(0x327c00000, 0x32ee00000 - 0x327c00000)])
    owner = storage(space)
    owner._reserve_low_window()
    assert owner.low_reservations
    # A driver that still places allocations below the floor fails setup.
    owner = storage(AddressSpace([]))
    owner.seal_low_window = lambda: None
    with pytest.raises(RuntimeError, match="still start at 0x100000000"):
        owner._reserve_low_window()
