"""LinearStateArena — live and snapshot storage for hybrid linear state."""

from __future__ import annotations

import torch

from serve.cache.tensor_arena import TensorArena


class LinearStateArena:
    """Owns live request state and cached checkpoint snapshots."""

    def __init__(
        self,
        *,
        live_slots: int,
        snapshot_slots: int,
        num_linear_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
        device: torch.device | str = "cuda",
    ):
        self.live = TensorArena(
            num_slots=live_slots,
            num_linear_layers=num_linear_layers,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
            device=device,
        )
        self.snapshot = (
            TensorArena(
                num_slots=snapshot_slots,
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
                device=device,
            )
            if snapshot_slots > 0
            else None
        )

    def alloc(self, n: int = 1) -> torch.Tensor:
        return self.live.alloc(n)

    def free(self, slots: torch.Tensor) -> None:
        self.live.free(slots)

    def free_slot(self, slot: int) -> None:
        self.live.free_slot(slot)

    def capture_snapshot(self, live_slot: int) -> int:
        if self.snapshot is None:
            raise RuntimeError("LinearStateArena has no snapshot capacity")
        snapshot_slot = self.snapshot.alloc(1)[0].item()
        self.snapshot.copy_from_other(self.live, live_slot, snapshot_slot)
        return snapshot_slot

    def restore_snapshot(self, snapshot_slot: int, live_slot: int) -> None:
        if self.snapshot is None:
            raise RuntimeError("LinearStateArena has no snapshot capacity")
        self.live.copy_from_other(self.snapshot, snapshot_slot, live_slot)

    def free_snapshot(self, snapshot_slot: int) -> None:
        if self.snapshot is None:
            return
        self.snapshot.free_slot(snapshot_slot)

    def zero_all(self) -> None:
        self.live.zero_all()
        if self.snapshot is not None:
            self.snapshot.zero_all()

    def ssm_state_for_layer(self, layer_idx: int) -> torch.Tensor:
        return self.live.ssm_state_for_layer(layer_idx)

    def conv_state_for_layer(self, layer_idx: int) -> torch.Tensor:
        return self.live.conv_state_for_layer(layer_idx)

    @property
    def num_free(self) -> int:
        return self.live.num_free

    @property
    def num_snapshot_free(self) -> int:
        return self.snapshot.num_free if self.snapshot is not None else 0

    @property
    def num_slots(self) -> int:
        return self.live.num_slots

    @property
    def num_snapshot_slots(self) -> int:
        return self.snapshot.num_slots if self.snapshot is not None else 0

    def memory_bytes(self) -> int:
        total = self.live.memory_bytes()
        if self.snapshot is not None:
            total += self.snapshot.memory_bytes()
        return total

    @staticmethod
    def estimate_memory_bytes(
        *,
        live_slots: int,
        snapshot_slots: int,
        num_linear_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
    ) -> int:
        total = TensorArena.estimate_memory_bytes(
            num_slots=live_slots,
            num_linear_layers=num_linear_layers,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
        )
        if snapshot_slots > 0:
            total += TensorArena.estimate_memory_bytes(
                num_slots=snapshot_slots,
                num_linear_layers=num_linear_layers,
                num_heads=num_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
            )
        return total

    @staticmethod
    def slot_memory_bytes_for_shape(
        *,
        num_linear_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
    ) -> int:
        return TensorArena.slot_memory_bytes_for_shape(
            num_linear_layers=num_linear_layers,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
        )
