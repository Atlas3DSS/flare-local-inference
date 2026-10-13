"""Gated pooling with read-only history and explicit accepted-prefix commit."""

import torch
import triton
import triton.language as tl

from ignition.components.linear import UnquantizedProjection
from ignition.components.rms_norm import PreparedRMSNorm


@triton.jit
def _pool(
    Values,
    Scores,
    Ape,
    StateValues,
    StateScores,
    Requests,
    Positions,
    Starts,
    Begins,
    StateRows,
    Output,
    SS: tl.constexpr,
    TS: tl.constexpr,
    HEAD: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(Requests + token)
    position = tl.load(Positions + token)
    start = tl.load(Starts + request)
    begin = tl.load(Begins + request)
    state = tl.load(StateRows + request)
    emitted = (position >= 0) & ((position + 1) % RATIO == 0) & (state > 0)
    if emitted:
        extent: tl.constexpr = RATIO * (2 if OVERLAP else 1)
        width: tl.constexpr = HEAD * (2 if OVERLAP else 1)
        offsets = tl.arange(0, extent)
        absolute = position + 1 - extent + offsets
        channel = dim[None, :] + (tl.where(offsets[:, None] < RATIO, 0, HEAD) if OVERLAP else 0)
        valid = (absolute[:, None] >= 0) & (dim[None, :] < HEAD)
        local = absolute[:, None] >= start
        source = (begin + absolute[:, None] - start) * TS + channel
        history = state * SS + (absolute[:, None] % extent) * width + channel
        values = tl.load(Values + source, valid & local, other=0)
        values += tl.load(StateValues + history, valid & ~local, other=0)
        scores = tl.load(Scores + source, valid & local, other=0)
        scores += tl.load(StateScores + history, valid & ~local, other=0)
        scores += tl.load(Ape + (absolute[:, None] % RATIO) * width + channel, valid, other=0)
        scores = tl.where(valid, scores, -float("inf"))
        maximum = tl.max(scores, 0)
        probabilities = tl.exp(scores - maximum[None, :])
        pooled = tl.sum(values * probabilities, 0) / tl.sum(probabilities, 0)
        tl.store(Output + token * HEAD + dim, pooled, dim < HEAD)
    else:
        tl.store(Output + token * HEAD + dim, 0, dim < HEAD)


@triton.jit
def _commit(
    Values,
    Scores,
    StateValues,
    StateScores,
    Starts,
    Begins,
    Counts,
    StateRows,
    SS: tl.constexpr,
    TS: tl.constexpr,
    WIDTH: tl.constexpr,
    EXTENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, ring_row = tl.program_id(0), tl.program_id(1)
    dim = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    start, count = tl.load(Starts + request), tl.load(Counts + request)
    begin, state = tl.load(Begins + request), tl.load(StateRows + request)
    end = start + count
    distance = ((end - 1) % EXTENT - ring_row + EXTENT) % EXTENT
    absolute = end - 1 - distance
    valid = (state > 0) & (count > 0) & (absolute >= start) & (dim < WIDTH)
    source = (begin + absolute - start) * TS + dim
    destination = state * SS + ring_row * WIDTH + dim
    values = tl.load(Values + source, valid, other=0)
    scores = tl.load(Scores + source, valid, other=0)
    tl.store(StateValues + destination, values, valid)
    tl.store(StateScores + destination, scores, valid)


def pool_gated(values, scores, ape, state_values, state_scores, metadata, output, *, ratio):
    """Emit each complete group without changing committed compressor history."""
    head = output.shape[1]
    torch.library.wrap_triton(_pool)[(values.shape[0], triton.cdiv(head, 32))](
        values,
        scores,
        ape,
        state_values,
        state_scores,
        metadata.requests,
        metadata.positions,
        metadata.starts,
        metadata.query_start_loc,
        metadata.state_rows,
        output,
        state_values.stride(0),
        values.stride(0),
        head,
        ratio,
        ratio == 4,
        32,
        enable_fp_fusion=False,
    )


def commit_gated(values, scores, state_values, state_scores, metadata, counts):
    """Update ring slots touched by the accepted prefix, preserving inactive rows."""
    torch.library.wrap_triton(_commit)[
        (counts.numel(), state_values.shape[1], triton.cdiv(values.shape[1], 256))
    ](
        values,
        scores,
        state_values,
        state_scores,
        metadata.starts,
        metadata.query_start_loc,
        counts,
        metadata.state_rows,
        state_values.stride(0),
        values.stride(0),
        values.shape[1],
        state_values.shape[1],
        256,
    )


class PreparedGatedCompressor:
    def __init__(self, spec, weights, source, metadata, pool, slots, *, session, name, deferred=False):
        if spec.ratio not in (4, 128) or spec.head_dim not in (128, 512):
            raise ValueError("gated compression requires C4/C128 and a 128/512-dimensional head")
        self.spec, self.metadata, self.deferred = spec, metadata, deferred
        self.state_values = pool.reserved_region(f"{spec.prefix}.values")
        self.state_scores = pool.reserved_region(f"{spec.prefix}.scores")
        self.ape = weights["ape"]
        tokens = source.shape[0]
        self.values = torch.zeros((tokens, spec.state_shape[1]), dtype=torch.float32, device=source.device)
        self.scores = torch.zeros_like(self.values)
        self.pooled = torch.zeros((tokens, spec.head_dim), dtype=torch.bfloat16, device=source.device)
        self.output = torch.zeros_like(self.pooled)
        self.emitted = torch.zeros(tokens, dtype=torch.bool, device=source.device)
        self.slots = torch.full_like(slots, -1)
        self.destinations = slots
        self.project_values = UnquantizedProjection(
            source, weights["values"], self.values, session=session, name=name + ".values"
        )
        self.project_scores = UnquantizedProjection(
            source, weights["gates"], self.scores, session=session, name=name + ".gates"
        )
        self.normalize = PreparedRMSNorm(
            self.pooled,
            weights["norm"],
            self.output,
            eps=pool.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        )

    def __call__(self, source):
        self.project_values(source)
        self.project_scores(source)
        pool_gated(
            self.values,
            self.scores,
            self.ape,
            self.state_values,
            self.state_scores,
            self.metadata,
            self.pooled,
            ratio=self.spec.ratio,
        )
        self.normalize(self.pooled)
        positions = self.metadata.positions
        self.emitted.copy_((positions >= 0) & ((positions + 1) % self.spec.ratio == 0))
        self.slots.copy_(torch.where(self.emitted, self.destinations, -1))
        if not self.deferred:
            self.commit(self.metadata.lengths)
        return self.output

    def commit(self, counts):
        commit_gated(self.values, self.scores, self.state_values, self.state_scores, self.metadata, counts)
