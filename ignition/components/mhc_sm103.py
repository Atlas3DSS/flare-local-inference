"""mHC post and pre for prefill-sized calls on SM103 through FlashInfer's mHC kernels.

b12x's fused post/pre runs a 4096-row GLM call in about 250 us on GB300, far
from bandwidth. FlashInfer's residual mixing and pre-map big fuse run near it;
between them, a Triton kernel projects the flattened residual onto the 24
mixing logits and sums its squares in four K splits, which the big fuse
reduces. The FP32 projection weights split into BF16 high and low halves, so
two BF16 MMAs reproduce them more closely than TF32 does. The collapsed branch
input then takes the weighted RMSNorm. Arithmetic follows b12x's references:
mixes scaled by the RMS of the flattened residual, pre = sigmoid + hc_eps,
post = 2 sigmoid, comb by Sinkhorn with hc_eps, and the mixing
new[j] = post[j] x + sum_i comb[i, j] residual[i].
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton

SPLITS = 4
_BLOCK_T = 128
_BLOCK_K = 64
_CONFIG = triton.Config({}, num_warps=8, num_stages=3)
# Calls with more rows than this (prefill) compose FlashInfer's kernels; decode keeps b12x's fused launches.
MIN_ROWS = 256
_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_SPLIT_WEIGHTS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _project(Residual, FnHigh, FnLow, Dot, Sqrsum, rows, K: tl.constexpr, SPLITS: tl.constexpr,
             BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr):
    tile, split = tl.program_id(0), tl.program_id(1)
    tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < rows
    mixes = tl.arange(0, 32)
    dot = tl.zeros((BLOCK_T, 32), tl.float32)
    squares = tl.zeros((BLOCK_T,), tl.float32)
    span = K // SPLITS
    for start in range(split * span, (split + 1) * span, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        values = tl.load(Residual + tokens.to(tl.int64)[:, None] * K + k[None, :], mask=live[:, None], other=0.0)
        dot += tl.dot(values, tl.trans(tl.load(FnHigh + mixes[:, None] * K + k[None, :])))
        dot += tl.dot(values, tl.trans(tl.load(FnLow + mixes[:, None] * K + k[None, :])))
        wide = values.to(tl.float32)
        squares += tl.sum(wide * wide, 1)
    tl.store(Dot + (split * rows + tokens.to(tl.int64))[:, None] * 24 + mixes[None, :], dot,
             mask=live[:, None] & (mixes[None, :] < 24))
    tl.store(Sqrsum + split * rows + tokens, squares, mask=live)


def _split_weights(fn):
    """BF16 high and low halves of the FP32 projection, padded to 32 mixes; shared by every call on ``fn``."""
    key = fn.data_ptr()
    if key not in _SPLIT_WEIGHTS:
        padded = torch.zeros((32, fn.shape[1]), dtype=torch.float32, device=fn.device)
        padded[:24] = fn
        high = padded.to(torch.bfloat16)
        _SPLIT_WEIGHTS[key] = (high, (padded - high.float()).to(torch.bfloat16))
    return _SPLIT_WEIGHTS[key]


class ComposedMHC:
    """Post/pre or pre into ``scratch`` (residual, post, comb, normalized) for one prefill capacity.

    Layers run serially, so every call of one capacity on a device shares its logits, square sums
    and collapsed input.
    """

    def __init__(self, operation, weights, scratch, *, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
        from flashinfer.mhc import _mhc_post_impl, _mhc_pre_big_fuse_impl
        from flashinfer.norm import rmsnorm

        self.mix, self.fuse, self.norm = _mhc_post_impl, _mhc_pre_big_fuse_impl, rmsnorm

        rows, _, hidden = scratch.residual.shape
        if operation not in ("pre", "post_pre") or weights.fn.shape != (24, 4 * hidden):
            raise ValueError("composed mHC serves pre and post/pre with 24 mixes")
        self.operation, self.weights, self.scratch = operation, weights, scratch
        self.eps = (float(rms_eps), float(hc_eps), int(sinkhorn_iters), float(norm_eps))
        self.high, self.low = _split_weights(weights.fn)
        key = (scratch.residual.device, rows, hidden)
        if key not in _SCRATCH:
            device = scratch.residual.device
            _SCRATCH[key] = (torch.empty((SPLITS, rows, 24), dtype=torch.float32, device=device),
                             torch.empty((SPLITS, rows), dtype=torch.float32, device=device),
                             torch.empty((rows, hidden), dtype=torch.bfloat16, device=device))
        self.dot, self.squares, self.collapsed = _SCRATCH[key]
        flat = scratch.residual.view(rows, -1)
        self.args = (flat, self.high, self.low, self.dot, self.squares, rows)
        self.options = {"K": 4 * hidden, "SPLITS": SPLITS, "BLOCK_T": _BLOCK_T, "BLOCK_K": _BLOCK_K}
        self.grid = (triton.cdiv(rows, _BLOCK_T), SPLITS, 1)
        self.compiled = _project.fn.warmup(*self.args, grid=self.grid, num_warps=_CONFIG.num_warps,
                                           num_stages=_CONFIG.num_stages, **self.options)

    def __call__(self, *inputs):
        s, w = self.scratch, self.weights
        rms_eps, hc_eps, iterations, norm_eps = self.eps
        if self.operation == "post_pre":
            source, residual, post, comb = inputs
            self.mix(s.residual, source, residual, post, comb)
        else:
            s.residual.copy_(inputs[0])
        launch_triton(_project, self.compiled, self.args, self.grid, **self.options)
        self.fuse(s.post, s.comb, self.collapsed, self.dot, self.squares, s.residual, w.scale, w.base,
                               s.residual.shape[1] * s.residual.shape[2], rms_eps, hc_eps, hc_eps, 2.0, iterations,
                               SPLITS, 0)
        self.norm(self.collapsed, w.norm, norm_eps, out=s.normalized)
        return s.normalized
