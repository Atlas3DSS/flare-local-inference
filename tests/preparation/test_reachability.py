"""Typed data must not create spurious CUDA resource dependencies."""

import gc
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.reachability import Reachability
from ignition.preparation.runtime_package import Allocation, Argument, Kernel
from ignition.preparation.tensor_metadata import NumericStorage, tensor_ranges


def test_numeric_scale_and_adjacent_bytes_do_not_form_a_pointer():
    scale = 0x200a295e5000
    target = 0x200a3f800000
    owner = Allocation("data", scale, 256)
    traversal = Reachability([owner, Allocation("unused", target, 256)],
                             roots=("data",), numeric=((scale, scale + 4),))
    data = struct.pack("<Q", target)
    assert struct.unpack("<f", data[:4])[0] == 1.0
    traversal.contents(owner, 0, data)
    assert traversal.reached == {"data"}


def test_pointer_table_is_followed_beside_numeric_tensor_and_across_chunks():
    base = 0x200000000000
    owner = Allocation("mixed", base, 256)
    traversal = Reachability([owner, Allocation("live", base + 256, 256),
                              Allocation("dead", base + 512, 256)],
                             roots=("mixed",), numeric=((base, base + 16),))
    data = struct.pack("<QQ", base + 512, base + 512) + b"padding" + struct.pack("<Q", base + 257)
    traversal.contents(owner, 0, data[:26])
    traversal.contents(owner, 26, data[26:])
    assert traversal.reached == {"mixed", "live"}


def test_numeric_argument_and_pointer_argument_with_identical_bytes():
    allocations = [Allocation("scalar_bits", 0x200000000000, 256),
                   Allocation("input", 0x200000000100, 256)]
    kernel = Kernel(1, "module", (Argument(8, numeric=((0, 8),)), Argument(8)))
    traversal = Reachability(allocations)
    node = {"kind": "kernel", "function": 0, "attributes": [], "arguments": [
        {"data": struct.pack("<Q", allocations[0].address), "relocations": []},
        {"data": struct.pack("<Q", allocations[1].address), "relocations": []},
    ]}
    traversal.graph({"nodes": [node]}, [kernel])
    assert traversal.reached == {"input"}
    node["arguments"][1]["data"] = node["arguments"][0]["data"]
    traversal.graph({"nodes": [node]}, [kernel])
    assert traversal.reached == {"scalar_bits", "input"}


def test_unknown_struct_keeps_unaligned_pointer_but_never_combines_arguments():
    allocation = Allocation("input", 0x200000001000, 256)
    traversal = Reachability([allocation])
    pointer = struct.pack("<Q", allocation.address)
    traversal.blob(pointer[:4])
    traversal.blob(pointer[4:])
    assert not traversal.reached
    traversal.blob(b"abc" + pointer + b"xyz")
    assert traversal.reached == {"input"}


def test_opaque_value_cannot_retain_freed_preparation_storage():
    base = 0x200000000000
    allocation = Allocation("segment", base, 4096)
    free = ((base + 1024, base + 3072),)
    for offset, expected in ((0, True), (1023, True), (1024, False),
                             (2048, False), (3071, False), (3072, True)):
        traversal = Reachability([allocation], inactive=free)
        traversal.blob(b"unaligned" + struct.pack("<Q", base + offset))
        assert bool(traversal.reached) == expected


def test_free_storage_filter_applies_to_pointer_tables_and_not_explicit_bindings():
    base = 0x200000000000
    table = Allocation("table", base, 16)
    target = Allocation("freed", base + 4096, 4096)
    traversal = Reachability([table, target], roots=("table",),
                             inactive=((target.address, target.address + target.nbytes),))
    traversal.contents(table, 0, struct.pack("<Q", target.address))
    assert traversal.reached == {"table"}
    traversal.reference({"allocation": "freed"})
    assert traversal.reached == {"table", "freed"}


def test_tensor_views_exclude_holes_and_padding_and_deduplicate_aliases():
    tensor = torch.empty((5, 6), dtype=torch.float32)
    address = tensor.data_ptr()
    assert tensor_ranges(tensor.T) == [(address, address + 120)]
    assert tensor_ranges(tensor[1:4, 2:4]) == [
        (address + 32, address + 40), (address + 56, address + 64), (address + 80, address + 88)]
    assert tensor_ranges(tensor[0, :1].expand(100)) == [(address, address + 4)]
    assert tensor_ranges(tensor[:, :0]) == []
    table = Allocation("mixed", address, 120, alignment=1)
    traversal = Reachability([table, Allocation("peer", 0x200000000000, 256)],
                             roots=("mixed",), numeric=tensor_ranges(tensor[:, 2:4]))
    data = bytearray(120)
    struct.pack_into("<Q", data, 0, 0x200000000000)
    traversal.contents(table, 0, data)
    assert traversal.reached == {"mixed", "peer"}


def test_checkpoint_role_tracks_storage_lifetime_without_retaining_weights():
    numeric = NumericStorage()
    tensor = torch.empty(128, dtype=torch.uint8)
    address = tensor.data_ptr()
    view = tensor[17:31]
    numeric.record(view)
    del view
    gc.collect()
    assert numeric.ranges() == [(address + 17, address + 31)]
    del tensor
    gc.collect()
    assert numeric.ranges() == []


def test_sampling_indices_do_not_retain_allocations_from_unaligned_token_ids():
    from ignition.components.speculation import PreparedSpeculativeDistributions

    with NumericStorage() as numeric:
        sampling = PreparedSpeculativeDistributions(
            torch.empty((1, 2, 256)),
            SimpleNamespace(vocab_size=256, generation=torch.zeros(1, dtype=torch.int64)),
            torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int32), torch.zeros((1, 1), dtype=torch.int64))
    indices = sampling.vocab_ids
    owner = Allocation("indices", indices.data_ptr(), indices.nbytes)
    unused = Allocation("unrelated", 50 << 16, 8)
    data = indices.numpy().tobytes()
    raw = Reachability([owner, unused], roots=("indices",))
    raw.contents(owner, 0, data)
    assert raw.reached == {"indices", "unrelated"}
    typed = Reachability([owner, unused], roots=("indices",), numeric=numeric.ranges())
    typed.contents(owner, 0, data)
    assert typed.reached == {"indices"}


def test_opaque_bytes_do_not_bridge_noncontiguous_snapshot_spans():
    allocation = Allocation("data", 0x200000000000, 256)
    traversal = Reachability([allocation, Allocation("unused", 0x200000000100, 256)], roots=("data",))
    pointer = struct.pack("<Q", 0x200000000100)
    traversal.contents(allocation, 0, pointer[:4])
    traversal.contents(allocation, 128, pointer[4:])
    assert traversal.reached == {"data"}


def test_resized_checkpoint_storage_invalidates_numeric_metadata():
    numeric = NumericStorage()
    tensor = torch.empty(128, dtype=torch.uint8)
    numeric.record(tensor)
    tensor.resize_(256)
    with pytest.raises(ValueError, match="storage was resized"):
        numeric.ranges()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="explicit CUDA assignment required")
def test_metadata_skips_uninitialized_cache_and_deduplicates_live_views(monkeypatch):
    import ignition.preparation.tensor_metadata as metadata

    cache = torch.empty((32768, 64), dtype=torch.float32, device="cuda")
    weights = torch.empty((4, 4), dtype=torch.float32, device="cuda")
    objects = [cache[:, ::2], weights, weights.view_as(weights), weights.view(torch.uint8)[:2]]
    monkeypatch.setattr(metadata.gc, "get_objects", lambda: objects)
    ranges = metadata.tensor_ranges
    visited = []

    def inspect(tensor):
        assert tensor.data_ptr() != cache.data_ptr()
        visited.append(tensor)
        return ranges(tensor)

    monkeypatch.setattr(metadata, "tensor_ranges", inspect)
    numeric, arguments = metadata.live_metadata([
        Allocation("cache", cache.data_ptr(), cache.nbytes, initialize="zero"),
        Allocation("weights", weights.data_ptr(), weights.nbytes),
    ], NumericStorage(), {})
    assert len(visited) == 2
    assert numeric == [(weights.data_ptr() + 32, weights.data_ptr() + 64)]
    assert arguments == {}
