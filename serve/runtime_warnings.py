"""Targeted import helpers for noisy third-party runtime warnings."""

from __future__ import annotations

import warnings

def import_torch_safely():
    """Import torch while suppressing noisy import-time FutureWarnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import torch
    return torch
