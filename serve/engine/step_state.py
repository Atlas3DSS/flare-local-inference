"""Per-step state passed to every layer during the forward pass.

Layers bind their own cache references (KV or SSM) at init time.
StepState carries only the per-step mutable tensors shared across layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from serve.engine.mamba_metadata import MambaForwardMetadata


@dataclass
class StepState:
    """Mutable per-step state that every layer receives."""
    cos: torch.Tensor
    sin: torch.Tensor
    positions: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor       # Pre-write seqlens for attention KV write.
    cu_seqlens_q: torch.Tensor
    mamba: MambaForwardMetadata | None = None
    is_decode: bool = False
    attn_workspaces_prepared: bool = False
    prepared_attn_workspace_ids: set[int] | None = None
