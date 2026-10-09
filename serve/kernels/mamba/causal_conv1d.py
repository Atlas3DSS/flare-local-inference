# Adapted from sglang/srt/layers/attention/mamba/causal_conv1d.py
# Originally from https://github.com/Dao-AILab/causal-conv1d
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from .causal_conv1d_triton import PAD_SLOT_ID
from .causal_conv1d_triton import causal_conv1d_fn as _causal_conv1d_fn_triton
from .causal_conv1d_triton import causal_conv1d_update as _causal_conv1d_update_triton

try:
    from sgl_kernel import causal_conv1d_fwd
    from sgl_kernel import causal_conv1d_update as causal_conv1d_update_kernel

    torch.ops.sgl_kernel.causal_conv1d_update
    _HAS_SGL_KERNEL = True
except (ImportError, AttributeError):
    _HAS_SGL_KERNEL = False


def _get_seq_lens_cpu(query_start_loc, x):
    if query_start_loc is not None:
        return (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()
    return [x.shape[-1]]


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
):
    # Use Triton when: (1) sgl_kernel not available, or (2) input is
    # non-contiguous and seq_lens_cpu is already pre-computed by caller.
    use_triton = not _HAS_SGL_KERNEL or (x.stride(-1) != 1 and "seq_lens_cpu" in kwargs)
    if use_triton:
        if "seq_lens_cpu" not in kwargs:
            kwargs["seq_lens_cpu"] = _get_seq_lens_cpu(query_start_loc, x)
        return _causal_conv1d_fn_triton(
            x,
            weight,
            bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            **kwargs,
        )
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id,
    )
    return x


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    use_triton = not _HAS_SGL_KERNEL
    if use_triton:
        return _causal_conv1d_update_triton(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
            pad_slot_id=pad_slot_id,
        )
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    causal_conv1d_update_kernel(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x
