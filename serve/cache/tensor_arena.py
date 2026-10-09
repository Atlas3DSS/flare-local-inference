"""TensorArena — slot allocator over preallocated linear-state tensors.

Each slot indexes the same slice across the arena's backing tensors.
For hybrid linear attention, one slot contains per-layer conv state and
recurrent (SSM) state. Slot 0 is reserved padding and never allocated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class TensorArena:
    """GPU-tensor arena with unified slot allocation and copy semantics."""

    @dataclass(frozen=True)
    class State:
        conv: list[torch.Tensor]
        ssm: torch.Tensor

    def __init__(
        self,
        *,
        num_slots: int,
        num_linear_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
        device: torch.device | str = "cuda",
    ):
        self.num_slots = num_slots
        self.num_linear_layers = num_linear_layers
        self.num_heads = num_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.device = torch.device(device)

        self.state = TensorArena.State(
            conv=[
                torch.zeros(
                    num_slots + 1,
                    conv_dim,
                    conv_kernel - 1,
                    device=device,
                    dtype=torch.bfloat16,
                )
                for _ in range(num_linear_layers)
            ],
            ssm=torch.zeros(
                num_linear_layers,
                num_slots + 1,
                num_heads,
                head_v_dim,
                head_k_dim,
                device=device,
                dtype=torch.float32,
            ),
        )

        self._free_slots = torch.arange(1, num_slots + 1, device=device, dtype=torch.int32)
        self._num_free = num_slots

    def alloc(self, n: int = 1) -> torch.Tensor:
        """Allocate n slots."""
        if n > self._num_free:
            raise RuntimeError(f"TensorArena exhausted: requested {n}, have {self._num_free}")
        start = self._num_free - n
        slots = self._free_slots[start:self._num_free].clone()
        self._num_free -= n
        return slots

    def free(self, slots: torch.Tensor) -> None:
        """Free slots and zero their state."""
        n = slots.shape[0]
        slot_idx = slots.long()
        for layer_idx in range(self.num_linear_layers):
            self.state.conv[layer_idx][slot_idx] = 0
        self.state.ssm[:, slot_idx] = 0
        self._free_slots[self._num_free:self._num_free + n] = slots.to(self._free_slots.dtype)
        self._num_free += n

    def free_slot(self, slot: int) -> None:
        """Free a single slot by index."""
        self.free(torch.tensor([slot], device=self.device, dtype=torch.int32))

    def copy_from(self, src: int, dst: int) -> None:
        """Deep copy all layer state from src slot to dst slot."""
        self.copy_from_other(self, src, dst)

    def copy_from_other(self, other: TensorArena, src: int, dst: int) -> None:
        """Deep copy all layer state from *other.src* into *self.dst*."""
        self._check_compatible(other)
        for layer_idx in range(self.num_linear_layers):
            self.state.conv[layer_idx][dst].copy_(other.state.conv[layer_idx][src])
        self.state.ssm[:, dst].copy_(other.state.ssm[:, src])

    def zero_all(self) -> None:
        """Zero all arena state."""
        for layer_idx in range(self.num_linear_layers):
            self.state.conv[layer_idx].zero_()
        self.state.ssm.zero_()

    def ssm_state_for_layer(self, layer_idx: int) -> torch.Tensor:
        return self.state.ssm[layer_idx]

    def conv_state_for_layer(self, layer_idx: int) -> torch.Tensor:
        return self.state.conv[layer_idx]

    def _check_compatible(self, other: TensorArena) -> None:
        if (
            self.num_linear_layers != other.num_linear_layers
            or self.num_heads != other.num_heads
            or self.head_v_dim != other.head_v_dim
            or self.head_k_dim != other.head_k_dim
            or self.conv_dim != other.conv_dim
            or self.conv_kernel != other.conv_kernel
        ):
            raise ValueError("TensorArena shapes do not match")

    @property
    def num_free(self) -> int:
        return self._num_free

    @classmethod
    def slot_memory_bytes_for_shape(
        cls,
        *,
        num_linear_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
    ) -> int:
        ssm_bytes = num_linear_layers * num_heads * head_v_dim * head_k_dim * 4
        conv_bytes = num_linear_layers * conv_dim * (conv_kernel - 1) * 2
        return ssm_bytes + conv_bytes

    @classmethod
    def estimate_memory_bytes(
        cls,
        *,
        num_slots: int,
        num_linear_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_dim: int,
        conv_kernel: int,
    ) -> int:
        slot_bytes = cls.slot_memory_bytes_for_shape(
            num_linear_layers=num_linear_layers,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            conv_dim=conv_dim,
            conv_kernel=conv_kernel,
        )
        return (num_slots + 1) * slot_bytes

    def slot_memory_bytes(self) -> int:
        return self.slot_memory_bytes_for_shape(
            num_linear_layers=self.num_linear_layers,
            num_heads=self.num_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
        )

    def memory_bytes(self) -> int:
        return self.estimate_memory_bytes(
            num_slots=self.num_slots,
            num_linear_layers=self.num_linear_layers,
            num_heads=self.num_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
        )
