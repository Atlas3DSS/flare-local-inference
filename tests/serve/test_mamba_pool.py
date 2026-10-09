"""Tests for MambaPool — GPU-tensor SSM state pool."""

import pytest
import torch

from serve.cache.mamba_pool import MambaPool


def _make_pool(num_slots=8, num_layers=3):
    return MambaPool(
        num_slots=num_slots,
        num_linear_layers=num_layers,
        num_heads=32,
        head_v_dim=128,
        head_k_dim=128,
        conv_dim=8192,
        conv_kernel=4,
        device="cpu",
    )


def test_alloc_and_free():
    pool = _make_pool(num_slots=4)
    assert pool.num_free == 4

    slots = pool.alloc(1)
    assert slots.shape == (1,)
    assert pool.num_free == 3

    slots2 = pool.alloc(1)
    assert pool.num_free == 2
    assert slots[0].item() != slots2[0].item()

    pool.free(slots)
    assert pool.num_free == 3

    pool.free(slots2)
    assert pool.num_free == 4


def test_slot_zero_reserved():
    """Slot 0 should never be returned by alloc."""
    pool = _make_pool(num_slots=8)
    all_slots = pool.alloc(8)
    assert 0 not in all_slots.tolist()
    # All slots should be in range [1, 8].
    for s in all_slots.tolist():
        assert 1 <= s <= 8


def test_batch_alloc():
    pool = _make_pool(num_slots=8)
    slots = pool.alloc(4)
    assert slots.shape == (4,)
    assert pool.num_free == 4
    # All unique.
    assert len(set(slots.tolist())) == 4


def test_batch_free():
    pool = _make_pool(num_slots=8)
    slots = pool.alloc(4)
    assert pool.num_free == 4
    pool.free(slots)
    assert pool.num_free == 8


def test_alloc_exhaustion():
    pool = _make_pool(num_slots=2)
    pool.alloc(2)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(1)


def test_alloc_over_request():
    pool = _make_pool(num_slots=4)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(5)


def test_free_zeros_state():
    pool = _make_pool(num_slots=4, num_layers=2)
    slots = pool.alloc(1)
    slot = slots[0].item()

    # Write non-zero data.
    pool.state.ssm[0, slot].fill_(1.0)
    pool.state.ssm[1, slot].fill_(2.0)
    pool.state.conv[0][slot].fill_(1.0)
    pool.state.conv[1][slot].fill_(2.0)

    pool.free(slots)

    # Should be zeroed.
    assert pool.state.ssm[0, slot].sum() == 0
    assert pool.state.ssm[1, slot].sum() == 0
    assert pool.state.conv[0][slot].sum() == 0
    assert pool.state.conv[1][slot].sum() == 0


def test_slot_zero_always_zero():
    """Slot 0 should remain zero even after operations."""
    pool = _make_pool(num_slots=4, num_layers=2)
    pool.alloc(4)  # Alloc all.
    assert pool.state.ssm[:, 0].sum() == 0
    assert pool.state.conv[0][0].sum() == 0


def test_copy_from():
    pool = _make_pool(num_slots=4, num_layers=2)
    slots = pool.alloc(2)
    src, dst = slots[0].item(), slots[1].item()

    # Write to src.
    pool.state.ssm[0, src].fill_(42.0)
    pool.state.conv[0][src].fill_(7.0)

    pool.copy_from(src, dst)

    # dst should match src.
    assert torch.equal(pool.state.ssm[0, dst], pool.state.ssm[0, src])
    assert torch.equal(pool.state.conv[0][dst], pool.state.conv[0][src])

    # Verify deep copy — modifying src shouldn't affect dst.
    pool.state.ssm[0, src].fill_(0.0)
    assert pool.state.ssm[0, dst].sum() != 0


def test_zero_all():
    pool = _make_pool(num_slots=4, num_layers=2)
    pool.state.ssm.fill_(1.0)
    pool.state.conv[0].fill_(1.0)
    pool.state.conv[1].fill_(1.0)

    pool.zero_all()

    assert pool.state.ssm.sum() == 0
    assert pool.state.conv[0].sum() == 0
    assert pool.state.conv[1].sum() == 0


def test_state_shapes():
    pool = _make_pool(num_slots=4, num_layers=3)
    # SSM: [num_layers, num_slots+1, num_heads, head_v_dim, head_k_dim].
    assert pool.state.ssm.shape == (3, 5, 32, 128, 128)
    # Conv: per-layer [num_slots+1, conv_dim, kernel-1].
    assert len(pool.state.conv) == 3
    assert pool.state.conv[0].shape == (5, 8192, 3)


def test_ssm_state_for_layer():
    pool = _make_pool(num_slots=4, num_layers=3)
    layer_state = pool.ssm_state_for_layer(1)
    assert layer_state.shape == (5, 32, 128, 128)
    # Should be a view into the unified tensor.
    assert layer_state.data_ptr() == pool.state.ssm[1].data_ptr()


def test_conv_state_for_layer():
    pool = _make_pool(num_slots=4, num_layers=3)
    layer_conv = pool.conv_state_for_layer(0)
    assert layer_conv.shape == (5, 8192, 3)
    assert layer_conv.data_ptr() == pool.state.conv[0].data_ptr()


def test_memory_bytes():
    pool = _make_pool(num_slots=4, num_layers=2)
    assert pool.memory_bytes() > 0
    # SSM: 2 layers * 5 slots * 32 * 128 * 128 * 4 bytes.
    expected_ssm = 2 * 5 * 32 * 128 * 128 * 4
    # Conv: 2 layers * 5 slots * 8192 * 3 * 2 bytes.
    expected_conv = 2 * 5 * 8192 * 3 * 2
    assert pool.memory_bytes() == expected_ssm + expected_conv


def test_alloc_free_reuse():
    """Freed slots can be reallocated."""
    pool = _make_pool(num_slots=2)
    s1 = pool.alloc(2)
    pool.free(s1)
    s2 = pool.alloc(2)
    assert pool.num_free == 0
    # Should have gotten slots back.
    assert set(s2.tolist()) == set(s1.tolist())


def test_scheduler_integration():
    """Scheduler allocates and frees MambaPool slots alongside pages."""
    from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
    from serve.engine.request import Request
    from serve.engine.sampling import SamplingParams
    from serve.engine.scheduler import BatchScheduler

    class MockPagePool:
        def __init__(self):
            self.num_pages = 100
            self._free = list(range(100))
        def alloc(self, n):
            r = self._free[-n:]
            del self._free[-n:]
            return r
        def free(self, ids):
            self._free.extend(ids)
        @property
        def num_free(self):
            return len(self._free)

    page_pool = MockPagePool()
    cache = PrefixCheckpointCache(page_pool)
    ssm_pool = _make_pool(num_slots=4, num_layers=2)

    sched = BatchScheduler(
        cache=cache, pool=page_pool, ssm_pool=ssm_pool,
        captured_bs=[1, 2, 4], max_running=4,
        max_prefill_tokens=4096, device="cpu",
    )

    req = Request(
        rid=1,
        prompt_ids=list(range(10)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert req.ssm_slot >= 1  # Slot 0 is reserved.
    assert ssm_pool.num_free == 3

    sched.process_prefill_chunk([42], batch.requests)
    assert req.is_finished
    assert req.ssm_slot == -1
    assert ssm_pool.num_free == 4
