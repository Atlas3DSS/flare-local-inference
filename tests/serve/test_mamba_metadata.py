"""Tests for MambaForwardMetadata construction."""

import torch

from serve.engine.mamba_metadata import MambaForwardMetadata


def test_decode_metadata():
    """Decode: all requests have initial states."""
    meta = MambaForwardMetadata(
        cache_indices=torch.tensor([1, 3, 5], dtype=torch.int64),
        has_initial_states=torch.ones(3, dtype=torch.bool),
    )
    assert meta.cache_indices.shape == (3,)
    assert meta.has_initial_states.all()
    assert meta.cu_seqlens is None
    assert meta.seq_lens is None


def test_extend_metadata_new_requests():
    """Extend with new requests: has_initial_states should be False."""
    meta = MambaForwardMetadata(
        cache_indices=torch.tensor([2, 4], dtype=torch.int64),
        has_initial_states=torch.zeros(2, dtype=torch.bool),
        cu_seqlens=torch.tensor([0, 10, 25], dtype=torch.int32),
        seq_lens=[10, 15],
    )
    assert not meta.has_initial_states.any()
    assert meta.cu_seqlens.shape == (3,)
    assert meta.seq_lens == [10, 15]


def test_extend_metadata_continuation():
    """Extend with continuation: has_initial_states should be True."""
    # pre_write_seqlens > 0 means this request has prior state.
    pre_write = torch.tensor([64, 0], dtype=torch.int32)
    has_initial = pre_write > 0

    meta = MambaForwardMetadata(
        cache_indices=torch.tensor([1, 2], dtype=torch.int64),
        has_initial_states=has_initial,
        cu_seqlens=torch.tensor([0, 5, 20], dtype=torch.int32),
        seq_lens=[5, 15],
    )
    assert meta.has_initial_states[0].item() is True
    assert meta.has_initial_states[1].item() is False


def test_metadata_from_runner_decode():
    """Simulate what runner._forward_inner builds for decode."""
    ssm_cache_indices = torch.tensor([3, 7], dtype=torch.int64)
    device = ssm_cache_indices.device

    meta = MambaForwardMetadata(
        cache_indices=ssm_cache_indices,
        has_initial_states=torch.ones(2, dtype=torch.bool, device=device),
    )
    assert meta.has_initial_states.all()
    assert meta.cache_indices[0] == 3
    assert meta.cache_indices[1] == 7


def test_metadata_from_runner_extend():
    """Simulate what runner._forward_inner builds for extend."""
    ssm_cache_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    pre_write_seqlens = torch.tensor([0, 128, 0], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 11, 23, 40], dtype=torch.int32)

    meta = MambaForwardMetadata(
        cache_indices=ssm_cache_indices,
        has_initial_states=(pre_write_seqlens[:3] > 0),
        cu_seqlens=cu_seqlens_q,
        seq_lens=[11, 12, 17],
    )
    # Request 0: new (pre_write=0), request 1: continuation (pre_write=128), request 2: new.
    assert not meta.has_initial_states[0]
    assert meta.has_initial_states[1]
    assert not meta.has_initial_states[2]
