"""Precomputed metadata for Mamba/GDN forward passes.

Built once per step by the engine, consumed by every GDN layer.
Replaces the ad-hoc ssm_cache_indices tensor and the fragile
conv_state.abs().sum() > 0 heuristic for has_initial_states.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MambaForwardMetadata:
    """Per-step metadata for all GDN linear attention layers."""

    cache_indices: torch.Tensor          # [batch] int64 — slot indices into MambaPool.
    has_initial_states: torch.Tensor     # [batch] bool — True if continuing from prior state.

    # Extend-only (None during decode).
    cu_seqlens: torch.Tensor | None = None   # [batch+1] int32 — cumulative seq lengths.
    seq_lens: list[int] | None = None        # Per-request seq lengths (CPU).
