"""Typed continuation regions preserve alignment, slot isolation, and byte identity."""

import pytest
import torch

from ignition.components.state import StateLayout, StateStorage, save_hidden


def test_mixed_dtype_regions_alias_only_their_slot_bytes():
    layout = StateLayout.from_specs((("count", (1,), torch.int64),
                                    ("history", (3, 5), torch.bfloat16),
                                    ("recurrent", (2, 3, 3), torch.float32)))
    assert [region.offset for region in layout.regions] == [0, 256, 512]
    assert layout.row_bytes == 768
    pool = StateStorage.allocate(layout, 6, "cpu")
    for region in layout.regions:
        view = pool.region(region.name)
        assert view.stride(0) * view.element_size() == layout.row_bytes
        assert view.untyped_storage().data_ptr() == pool.backing.data_ptr()
        view[1].fill_(3)
        assert not view[0].count_nonzero() and not view[2:].count_nonzero()
    pool.backing[5].copy_(pool.backing[1])
    for region in layout.regions:
        torch.testing.assert_close(pool.region(region.name)[5], pool.region(region.name)[1], rtol=0, atol=0)
    assert layout.snapshot_capacity(1535) == 1


@pytest.mark.parametrize("specifications,error", [
    ([], ValueError),
    ([("", (1,), torch.float32)], ValueError),
    ([("x", (1,), torch.float32)] * 2, ValueError),
    ([("x", (0,), torch.float32)], ValueError),
    ([("x", (True,), torch.float32)], ValueError),
    ([("x", (1,), "float32")], TypeError),
])
def test_invalid_region_specifications(specifications, error):
    with pytest.raises(error):
        StateLayout.from_specs(specifications)


def test_hidden_checkpoints_preserve_unselected_physical_slots():
    pool = StateStorage.allocate(StateLayout.from_specs((("hidden", (7,), torch.bfloat16),)), 6, "cpu")
    pool.backing.fill_(0x31)
    before = pool.backing.clone()
    source = torch.arange(28, dtype=torch.bfloat16).reshape(4, 7)
    save_hidden(source, pool.region("hidden")[1:5], torch.tensor([2, -1, 0, 3]), torch.tensor([1, 0, 1, 0]))
    torch.testing.assert_close(pool.region("hidden")[1], source[2])
    torch.testing.assert_close(pool.region("hidden")[3], source[0])
    assert torch.equal(pool.backing[[0, 2, 4, 5]], before[[0, 2, 4, 5]])
