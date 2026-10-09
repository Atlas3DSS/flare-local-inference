"""Physical page pool for paged KV cache.

Pre-allocates a flat slab of KV cache pages on one device. Pages are
fixed at 64 tokens (matching b12x's native page size). Allocation and
freeing are CPU-side list operations — no Triton, no radix tree.
"""

from __future__ import annotations

import torch


_PAGE_SIZE = 64


class PagePool:
    """Fixed-size pool of KV cache pages on one device.

    The pool owns two tensors per layer (K and V), each shaped
    ``[num_pages, page_size, kv_heads, head_dim]``.  Pages are handed
    out as integer indices into dimension 0.
    """

    def __init__(
        self,
        num_pages: int,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        device: torch.device | str = "cuda",
    ):
        self.num_pages = num_pages
        self.num_layers = num_layers
        self.page_size = _PAGE_SIZE
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)

        page_shape = (num_pages, _PAGE_SIZE, kv_heads, head_dim)
        self.k_cache = [
            torch.zeros(page_shape, dtype=kv_dtype, device=self.device)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(page_shape, dtype=kv_dtype, device=self.device)
            for _ in range(num_layers)
        ]

        # Freelist — all pages start free.
        self._free: list[int] = list(range(num_pages))

    # -- allocation --------------------------------------------------------

    def alloc(self, n: int) -> list[int]:
        """Allocate *n* pages. Raises RuntimeError if insufficient."""
        if n > len(self._free):
            raise RuntimeError(
                f"PagePool OOM: requested {n} pages, "
                f"only {len(self._free)} free of {self.num_pages} total"
            )
        allocated = self._free[-n:]
        del self._free[-n:]
        return allocated

    def free(self, page_ids: list[int]) -> None:
        """Return pages to the pool."""
        self._free.extend(page_ids)

    # -- queries -----------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def utilization(self) -> float:
        return 1.0 - len(self._free) / self.num_pages if self.num_pages > 0 else 0.0

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def estimate_num_pages(
        memory_bytes: int,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> int:
        """Estimate how many pages fit in *memory_bytes*."""
        elem_size = torch.finfo(kv_dtype).bits // 8 if kv_dtype.is_floating_point else 1
        bytes_per_page = _PAGE_SIZE * kv_heads * head_dim * elem_size
        # K + V per layer.
        bytes_per_page_total = bytes_per_page * 2 * num_layers
        return max(1, memory_bytes // bytes_per_page_total)
