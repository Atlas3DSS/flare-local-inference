"""Fixed-size SSM state pool for linear attention layers.

Each request gets one slot containing per-layer conv state and
recurrent (SSM) state. No paging — states are fixed-size tensors
updated in-place by the Triton kernels.

Much simpler than paged KV cache: allocate on admit, free on finish.
"""

from __future__ import annotations

import torch


class SSMStatePool:
    """Pool of SSM state slots for GDN linear attention layers."""

    def __init__(
        self,
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
        self.device = torch.device(device)

        # Per-layer recurrent state: [num_slots, num_heads, head_v_dim, head_k_dim].
        # Float32 for numerical stability in recurrent updates.
        self.ssm_state = [
            torch.zeros(num_slots, num_heads, head_v_dim, head_k_dim,
                        device=device, dtype=torch.float32)
            for _ in range(num_linear_layers)
        ]

        # Per-layer conv state: [num_slots, conv_dim, conv_kernel - 1].
        self.conv_state = [
            torch.zeros(num_slots, conv_dim, conv_kernel - 1,
                        device=device, dtype=torch.bfloat16)
            for _ in range(num_linear_layers)
        ]

        self._free = list(range(num_slots))

    def alloc(self) -> int:
        """Allocate one slot. Returns slot index."""
        if not self._free:
            raise RuntimeError("SSM state pool exhausted")
        return self._free.pop()

    def free(self, slot: int) -> None:
        """Free a slot and zero its state."""
        for i in range(self.num_linear_layers):
            self.ssm_state[i][slot].zero_()
            self.conv_state[i][slot].zero_()
        self._free.append(slot)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def memory_bytes(self) -> int:
        """Total GPU memory used by the pool."""
        ssm_bytes = sum(s.nelement() * s.element_size() for s in self.ssm_state)
        conv_bytes = sum(s.nelement() * s.element_size() for s in self.conv_state)
        return ssm_bytes + conv_bytes
