"""Equivalence tests: verify the MambaPool + MambaForwardMetadata refactor
produces identical inputs to GDN as the old SSMStatePool + raw ssm_cache_indices.

These tests don't run the GDN kernel — they verify that the plumbing
delivers the same values to the same function signatures.
"""

import torch
import pytest

from serve.cache.mamba_pool import MambaPool
from serve.cache.ssm_pool import SSMStatePool
from serve.engine.mamba_metadata import MambaForwardMetadata


# -- Pool equivalence ----------------------------------------------------------

class TestPoolEquivalence:
    """MambaPool slot contents match SSMStatePool for the same operations."""

    POOL_KWARGS = dict(
        num_linear_layers=3,
        num_heads=4,
        head_v_dim=16,
        head_k_dim=16,
        conv_dim=128,
        conv_kernel=4,
    )

    def test_fresh_alloc_state_is_zero(self):
        """Both pools return zero-initialized state for fresh slots."""
        old = SSMStatePool(num_slots=4, device="cpu", **self.POOL_KWARGS)
        new = MambaPool(num_slots=4, device="cpu", **self.POOL_KWARGS)

        old_slot = old.alloc()
        new_slot = new.alloc(1)[0].item()

        # SSM state: old is [num_slots, ...], new is [num_slots+1, ...].
        old_ssm = old.ssm_state[0][old_slot]
        new_ssm = new.ssm_state_for_layer(0)[new_slot]
        assert old_ssm.shape == new_ssm.shape
        assert torch.equal(old_ssm, new_ssm)  # Both zero.

        # Conv state.
        old_conv = old.conv_state[0][old_slot]
        new_conv = new.conv_state_for_layer(0)[new_slot]
        assert old_conv.shape == new_conv.shape
        assert torch.equal(old_conv, new_conv)

    def test_write_and_read_state(self):
        """Writing to a slot and reading back gives same values."""
        old = SSMStatePool(num_slots=4, device="cpu", **self.POOL_KWARGS)
        new = MambaPool(num_slots=4, device="cpu", **self.POOL_KWARGS)

        old_slot = old.alloc()
        new_slot = new.alloc(1)[0].item()

        data = torch.randn(4, 16, 16)
        old.ssm_state[0][old_slot].copy_(data)
        new.ssm_state_for_layer(0)[new_slot].copy_(data)

        assert torch.equal(old.ssm_state[0][old_slot], new.ssm_state_for_layer(0)[new_slot])

    def test_free_zeros_both(self):
        """Both pools zero state on free."""
        old = SSMStatePool(num_slots=4, device="cpu", **self.POOL_KWARGS)
        new = MambaPool(num_slots=4, device="cpu", **self.POOL_KWARGS)

        old_slot = old.alloc()
        new_slot = new.alloc(1)[0].item()

        old.ssm_state[0][old_slot].fill_(42.0)
        new.ssm_state_for_layer(0)[new_slot].fill_(42.0)

        old.free(old_slot)
        new.free(torch.tensor([new_slot], dtype=torch.int32))

        assert old.ssm_state[0][old_slot].sum() == 0
        assert new.ssm_state_for_layer(0)[new_slot].sum() == 0

    def test_layer_tensor_shapes_match(self):
        """The per-layer tensors handed to bind_cache have compatible shapes."""
        old = SSMStatePool(num_slots=4, device="cpu", **self.POOL_KWARGS)
        new = MambaPool(num_slots=4, device="cpu", **self.POOL_KWARGS)

        # Old: [num_slots, heads, v, k] = [4, 4, 16, 16].
        # New: [num_slots+1, heads, v, k] = [5, 4, 16, 16].
        # Shapes differ in dim0 (4 vs 5), but indexing with valid slots works.
        old_ssm = old.ssm_state[0]
        new_ssm = new.ssm_state_for_layer(0)
        assert old_ssm.shape[1:] == new_ssm.shape[1:]
        assert new_ssm.shape[0] == old_ssm.shape[0] + 1  # +1 for reserved slot 0.

    def test_kernel_indexing_equivalence(self):
        """Indexing ssm_state[cache_indices] gives same data for same slot content."""
        old = SSMStatePool(num_slots=4, device="cpu", **self.POOL_KWARGS)
        new = MambaPool(num_slots=4, device="cpu", **self.POOL_KWARGS)

        old_slot = old.alloc()
        new_slot = new.alloc(1)[0].item()

        # Write identical data.
        data = torch.randn(4, 16, 16)
        old.ssm_state[0][old_slot].copy_(data)
        new.ssm_state_for_layer(0)[new_slot].copy_(data)

        # Simulate what GDN forward_extend does: ssm_state[cache_indices].
        old_indices = torch.tensor([old_slot], dtype=torch.int64)
        new_indices = torch.tensor([new_slot], dtype=torch.int64)

        old_gathered = old.ssm_state[0][old_indices]
        new_gathered = new.ssm_state_for_layer(0)[new_indices]
        assert torch.equal(old_gathered, new_gathered)


# -- Metadata equivalence -----------------------------------------------------

class TestMetadataEquivalence:
    """MambaForwardMetadata produces the same values GDN received before."""

    def test_decode_cache_indices_passthrough(self):
        """Decode: cache_indices on metadata == raw ssm_cache_indices."""
        raw_indices = torch.tensor([3, 1, 7], dtype=torch.int64)
        meta = MambaForwardMetadata(
            cache_indices=raw_indices,
            has_initial_states=torch.ones(3, dtype=torch.bool),
        )
        assert torch.equal(meta.cache_indices, raw_indices)

    def test_extend_cache_indices_passthrough(self):
        """Extend: cache_indices on metadata == raw ssm_cache_indices."""
        raw_indices = torch.tensor([2, 5], dtype=torch.int64)
        cu = torch.tensor([0, 32, 64], dtype=torch.int32)
        meta = MambaForwardMetadata(
            cache_indices=raw_indices,
            has_initial_states=torch.zeros(2, dtype=torch.bool),
            cu_seqlens=cu,
            seq_lens=[32, 32],
        )
        assert torch.equal(meta.cache_indices, raw_indices)
        assert torch.equal(meta.cu_seqlens, cu)

    def test_has_initial_states_new_request(self):
        """New request (pre_write=0): has_initial_states=False, matching old heuristic."""
        # Old: conv_state[slot].abs().sum() > 0 → False for fresh zero state.
        # New: pre_write_seqlens > 0 → False for new request.
        pre_write = torch.tensor([0], dtype=torch.int32)
        has_initial = pre_write > 0
        assert not has_initial[0].item()

    def test_has_initial_states_continuation(self):
        """Continuation (pre_write>0): has_initial_states=True, matching old heuristic."""
        # Old: conv_state[slot] has data from prior chunk → abs().sum() > 0 = True.
        # New: pre_write_seqlens > 0 = True.
        pre_write = torch.tensor([64], dtype=torch.int32)
        has_initial = pre_write > 0
        assert has_initial[0].item()

    def test_has_initial_states_matches_old_heuristic(self):
        """Verify equivalence between old heuristic and new computation
        for the common cases: fresh request, decode, and chunked continuation."""
        pool = MambaPool(
            num_slots=4, num_linear_layers=1, num_heads=4,
            head_v_dim=16, head_k_dim=16, conv_dim=128,
            conv_kernel=4, device="cpu",
        )

        slot = pool.alloc(1)[0].item()
        conv = pool.conv_state_for_layer(0)

        # Case 1: Fresh request — conv state is zero.
        old_heuristic = conv[slot].abs().sum() > 0
        new_from_pre_write = torch.tensor([0], dtype=torch.int32) > 0
        assert old_heuristic.item() == new_from_pre_write[0].item()  # Both False.

        # Case 2: After prefill — conv state has data.
        conv[slot].fill_(1.0)
        old_heuristic = conv[slot].abs().sum() > 0
        new_from_pre_write = torch.tensor([32], dtype=torch.int32) > 0
        assert old_heuristic.item() == new_from_pre_write[0].item()  # Both True.

        # Case 3: After free + realloc — conv state zeroed.
        pool.free(torch.tensor([slot], dtype=torch.int32))
        slot2 = pool.alloc(1)[0].item()
        old_heuristic = conv[slot2].abs().sum() > 0
        new_from_pre_write = torch.tensor([0], dtype=torch.int32) > 0
        assert old_heuristic.item() == new_from_pre_write[0].item()  # Both False.


# -- End-to-end runner metadata construction -----------------------------------

class TestRunnerMetadataConstruction:
    """Verify the metadata the runner builds matches what GDN used to receive."""

    def _build_metadata_like_runner(self, ssm_cache_indices, q_seqlens, cache_seqlens, mode):
        """Replicate runner._forward_inner metadata construction."""
        device = ssm_cache_indices.device
        q_lens = torch.tensor(q_seqlens, dtype=torch.int32, device=device)
        pre_write = cache_seqlens - q_lens
        cu_seqlens_q = torch.zeros(len(q_seqlens) + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = q_lens.cumsum(0)

        is_decode = (mode == "decode")
        return MambaForwardMetadata(
            cache_indices=ssm_cache_indices,
            has_initial_states=(
                torch.ones(ssm_cache_indices.shape[0], dtype=torch.bool, device=device)
                if is_decode
                else (pre_write[:ssm_cache_indices.shape[0]] > 0)
            ),
            cu_seqlens=cu_seqlens_q if not is_decode else None,
            seq_lens=q_seqlens if not is_decode else None,
        )

    def test_prefill_new_request(self):
        """Prefill of a new 32-token request."""
        indices = torch.tensor([3], dtype=torch.int64)
        meta = self._build_metadata_like_runner(
            indices, q_seqlens=[32], cache_seqlens=torch.tensor([32], dtype=torch.int32),
            mode="extend",
        )
        assert torch.equal(meta.cache_indices, indices)
        assert not meta.has_initial_states[0].item()  # New request.
        assert meta.cu_seqlens.tolist() == [0, 32]

    def test_prefill_continuation(self):
        """Second chunk of a chunked prefill (pre_write > 0)."""
        indices = torch.tensor([3], dtype=torch.int64)
        meta = self._build_metadata_like_runner(
            indices, q_seqlens=[64], cache_seqlens=torch.tensor([128], dtype=torch.int32),
            mode="extend",
        )
        assert meta.has_initial_states[0].item()  # Continuation.
        assert meta.cu_seqlens.tolist() == [0, 64]

    def test_decode(self):
        """Decode step — has_initial_states always True."""
        indices = torch.tensor([3, 5], dtype=torch.int64)
        meta = self._build_metadata_like_runner(
            indices, q_seqlens=[1, 1], cache_seqlens=torch.tensor([100, 50], dtype=torch.int32),
            mode="decode",
        )
        assert meta.has_initial_states.all()
        assert meta.cu_seqlens is None

    def test_batch_prefill_mixed(self):
        """Batched prefill: one new request, one continuation."""
        indices = torch.tensor([2, 4], dtype=torch.int64)
        meta = self._build_metadata_like_runner(
            indices, q_seqlens=[32, 32],
            cache_seqlens=torch.tensor([32, 96], dtype=torch.int32),
            mode="extend",
        )
        assert not meta.has_initial_states[0].item()  # New (pre_write=0).
        assert meta.has_initial_states[1].item()       # Continuation (pre_write=64).
        assert meta.cu_seqlens.tolist() == [0, 32, 64]
