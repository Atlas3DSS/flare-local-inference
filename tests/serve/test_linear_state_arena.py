"""Tests for LinearStateArena snapshot capture and restore."""

import torch

from serve.cache.linear_state_arena import LinearStateArena


def _make_arena(live_slots=4, snapshot_slots=2):
    return LinearStateArena(
        live_slots=live_slots,
        snapshot_slots=snapshot_slots,
        num_linear_layers=2,
        num_heads=2,
        head_v_dim=4,
        head_k_dim=4,
        conv_dim=8,
        conv_kernel=4,
        device="cpu",
    )


def test_capture_and_restore_snapshot_round_trip():
    arena = _make_arena()
    live_slots = arena.alloc(2)
    src = live_slots[0].item()
    dst = live_slots[1].item()

    arena.ssm_state_for_layer(0)[src].fill_(5.0)
    arena.ssm_state_for_layer(1)[src].fill_(9.0)
    arena.conv_state_for_layer(0)[src].fill_(3.0)
    arena.conv_state_for_layer(1)[src].fill_(7.0)

    snapshot_slot = arena.capture_snapshot(src)

    arena.ssm_state_for_layer(0)[dst].zero_()
    arena.ssm_state_for_layer(1)[dst].zero_()
    arena.conv_state_for_layer(0)[dst].zero_()
    arena.conv_state_for_layer(1)[dst].zero_()

    arena.restore_snapshot(snapshot_slot, dst)

    assert torch.equal(arena.ssm_state_for_layer(0)[dst], arena.ssm_state_for_layer(0)[src])
    assert torch.equal(arena.ssm_state_for_layer(1)[dst], arena.ssm_state_for_layer(1)[src])
    assert torch.equal(arena.conv_state_for_layer(0)[dst], arena.conv_state_for_layer(0)[src])
    assert torch.equal(arena.conv_state_for_layer(1)[dst], arena.conv_state_for_layer(1)[src])


def test_memory_bytes_includes_live_and_snapshot_storage():
    arena = _make_arena(live_slots=3, snapshot_slots=2)
    expected = LinearStateArena.estimate_memory_bytes(
        live_slots=3,
        snapshot_slots=2,
        num_linear_layers=2,
        num_heads=2,
        head_v_dim=4,
        head_k_dim=4,
        conv_dim=8,
        conv_kernel=4,
    )
    assert arena.memory_bytes() == expected
    assert arena.num_snapshot_slots == 2
    assert arena.num_snapshot_free == 2
