"""Fused gated RMS normalization of recurrent attention outputs before their output projection."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton


@triton.jit
def _gated_rms_norm(Hidden, Gate, Weight, Output, Live, eps, HEADS: tl.constexpr, DIM: tl.constexpr,
                    HEAD_BLOCK: tl.constexpr):
    # FP32 RMS normalization per head, then weight, then the sigmoid gate, rounded once to BF16.
    # Rows at or after the live count are written as zeros.
    row = tl.program_id(0)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    column = tl.arange(0, DIM)[None, :]
    mask = head < HEADS
    offset = row * (HEADS * DIM) + head * DIM + column
    hidden = tl.load(Hidden + offset, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=1)[:, None] / DIM
    hidden = hidden * tl.rsqrt(variance + eps)
    hidden = tl.load(Weight + column).to(tl.float32) * hidden
    gate = tl.load(Gate + offset, mask=mask, other=0.0).to(tl.float32)
    value = hidden * tl.sigmoid(gate)
    live = row < tl.load(Live)
    tl.store(Output + offset, tl.where(live, value, 0.0).to(Output.dtype.element_ty), mask=mask)


class PreparedGatedRMSNorm:
    """Normalize [rows, heads, dim] BF16 outputs per head and gate them with sigmoid(gate).

    Matches an FP32 gated RMSNorm (normalize, scale by the weight, multiply by
    the sigmoid gate, round to BF16) such as Transformers' RMSNormGated with a
    sigmoid activation. Rows at or after the device live count become zeros.
    Inputs and output are caller-owned contiguous storage; the output may feed
    a projection directly.
    """

    def __init__(self, hidden, gate, weight, output, live, *, eps):
        rows, heads, dim = hidden.shape
        if (gate.shape != hidden.shape or output.shape != hidden.shape or weight.shape != (dim,)
                or any(t.dtype != torch.bfloat16 for t in (hidden, gate, output))
                or not all(t.is_contiguous() for t in (hidden, gate, output, weight))
                or live.shape != (1,) or dim & (dim - 1)):
            raise ValueError("gated RMS norm requires contiguous BF16 [rows,heads,dim] tensors and a [dim] weight")
        self.head_block = min(8, triton.next_power_of_2(heads))
        self.args = (hidden, gate, weight, output, live, float(eps))
        self.grid = (rows, triton.cdiv(heads, self.head_block), 1)
        self.options = {"HEADS": heads, "DIM": dim, "HEAD_BLOCK": self.head_block, "num_warps": 4}
        self.compiled = _gated_rms_norm.warmup(*self.args, grid=self.grid, **self.options)

    def __call__(self):
        launch_triton(_gated_rms_norm, self.compiled, self.args, self.grid, **self.options)
