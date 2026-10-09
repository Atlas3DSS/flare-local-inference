"""Fixed device inputs and captured metadata for the hybrid Qwen decoder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import triton
import triton.language as tl

from .attention import QSAMetadata
from .gdn import GDNMetadata
from .ple import PLEMetadata
from .sampling import DeviceRequestState, PreparedTokenState
from .state import HybridStatePool


def compile_kernel(kernel, args, grid):
    compiled = kernel.warmup(*args, grid=grid, num_warps=4)
    return partial(compiled[grid], *args)


@triton.jit
def _validate(Slots, Epochs, Live, Lengths, Generation, Status, Error, Pending, Consumed,
              ROWS: tl.constexpr, TOKENS: tl.constexpr, CONTEXT: tl.constexpr,
              CONSUMED_STRIDE: tl.constexpr, PREFILL: tl.constexpr):
    slot = tl.program_id(0)
    rows = tl.arange(0, 4)
    match = (rows < ROWS) & (rows < tl.load(Live))
    match &= tl.load(Slots + rows, rows < ROWS, other=-1) == slot
    match &= tl.load(Epochs + rows, rows < ROWS, other=-1) == tl.load(Generation + slot)
    count = tl.sum(match.to(tl.int32), 0)
    if (count > 0) & (tl.load(Status + slot) == 1):
        row = tl.max(tl.where(match, rows, 0), 0)
        length = tl.load(Lengths + row) if PREFILL else 1
        consumed = tl.load(Consumed + slot * CONSUMED_STRIDE)
        valid = (count == 1) & (tl.load(Live) <= ROWS) & (length > 0) & (length <= TOKENS)
        valid &= (consumed >= 0) & (consumed + length <= CONTEXT)
        if PREFILL:
            valid &= (consumed % 64 == 0) & (tl.load(Pending + slot) < 0)
        else:
            valid &= tl.load(Pending + slot) >= 0
        if not valid:
            tl.store(Status + slot, 3)
            tl.store(Error + slot, 32)


@dataclass(frozen=True)
class DecoderInputs:
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    prefill_positions: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor
    vision_token_indices: torch.Tensor

    @classmethod
    def allocate(cls, device, *, context: int, tokens: int = 512):
        if (type(context) is not int or type(tokens) is not int
                or context <= 0 or context % 64 or tokens <= 0):
            raise ValueError("decoder capacities require positive tokens and a 64-aligned context")
        return cls(
            torch.full((4,), -1, dtype=torch.int32, device=device),
            torch.zeros(4, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(4, dtype=torch.int32, device=device),
            torch.full((tokens,), 248044, dtype=torch.int64, device=device),
            torch.zeros((tokens, 3), dtype=torch.int64, device=device),
            torch.full((4, context // 64), -1, dtype=torch.int32, device=device),
            torch.zeros(4, dtype=torch.int32, device=device),
            torch.full((tokens,), -1, dtype=torch.int32, device=device),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@triton.jit
def _metadata(Slots, Live, Lengths, Pending, Positions, Active, Delta, Tokens, InputRope,
              Cu, ConvSlots, Fresh, StateIndices, Initial, Final, Checkpoint, Offset,
              NumSeqs, NumTokens, RequestIds, QueryPositions, Rope, SequenceLengths,
              SlotMapping, PhysicalPages, PackedPages, IsPrefilling,
              ROWS: tl.constexpr, TOKENS: tl.constexpr, PAGES: tl.constexpr,
              DELTA_STRIDE: tl.constexpr, PREFILL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    rows = tl.arange(0, 4)
    live = tl.minimum(tl.load(Live), ROWS)
    active = tl.load(Active + rows, rows < ROWS, other=0) != 0
    lengths = tl.load(Lengths + rows, rows < ROWS, other=0) if PREFILL else tl.full((4,), 1, tl.int32)
    lengths = tl.where((rows < live) & active, lengths, 0)
    begin = tl.sum(tl.where(rows < row, lengths, 0), 0)
    length = tl.sum(tl.where(rows == row, lengths, 0), 0)
    total = tl.sum(lengths, 0)
    slot = tl.load(Slots + row)
    valid = (row < live) & (length > 0) & (slot >= 0) & (slot < 4)
    position = tl.load(Positions + row)
    model_slot = tl.where(valid, slot + 1, -1)
    end = position + length
    boundary = end // 64 * 64
    checkpoint = valid & (boundary > position)
    tl.store(Cu + row, begin)
    if row == 0:
        tl.store(Cu + ROWS, total)
        tl.store(NumTokens, total)
        tl.store(NumSeqs, live)
    tl.store(ConvSlots + row, model_slot)
    tl.store(Fresh + row, valid & (position == 0))
    tl.store(StateIndices + row, tl.where(valid, model_slot, 0))
    tl.store(Initial + row, tl.where(valid & (position > 0), model_slot, 0))
    tl.store(Final + row, tl.where(valid, model_slot, 0))
    tl.store(Checkpoint + row, tl.where(checkpoint, slot + 5, 0))
    tl.store(Offset + row, tl.where(checkpoint, boundary - position, 0))
    tl.store(SequenceLengths + row, tl.where(valid, end, 0))
    tl.store(IsPrefilling + row, PREFILL & valid)
    if not PREFILL:
        tl.store(Tokens + begin, tl.load(Pending + row), valid)
    delta = tl.load(Delta + (slot + 1) * DELTA_STRIDE, valid, other=0)
    for block in range(tl.cdiv(TOKENS, BLOCK)):
        local = block * BLOCK + tl.arange(0, BLOCK)
        index = begin + local
        mask = valid & (local < length) & (index < TOKENS)
        logical = position + local
        page = tl.load(PhysicalPages + slot * PAGES + logical // 64, mask, other=-1)
        tl.store(RequestIds + index, row, mask)
        tl.store(QueryPositions + index, logical, mask)
        tl.store(SlotMapping + index, page * 64 + logical % 64, mask)
        for axis in tl.static_range(3):
            rope = tl.load(InputRope + index * 3 + axis, mask, other=0) if PREFILL else logical + delta
            tl.store(Rope + index * 3 + axis, rope, mask)
    for block in range(tl.cdiv(PAGES, BLOCK)):
        page_indices = block * BLOCK + tl.arange(0, BLOCK)
        pages = tl.load(PhysicalPages + slot * PAGES + page_indices, valid & (page_indices < PAGES), other=-1)
        tl.store(PackedPages + row * PAGES + page_indices, pages, page_indices < PAGES)


@triton.jit
def _convolution_checkpoint(Source, State, Cu, Checkpoint, Offset,
                            CHANNELS: tl.constexpr, HISTORY: tl.constexpr,
                            STATE_STRIDE: tl.constexpr, STATE_HISTORY: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot, offset = tl.load(Checkpoint + row), tl.load(Offset + row)
    if (slot > 0) & (offset >= HISTORY):
        end = tl.load(Cu + row) + offset
        index = tile * BLOCK + tl.arange(0, BLOCK)
        channel, history = index // HISTORY, index % HISTORY
        value = tl.load(Source + (end - HISTORY + history) * CHANNELS + channel,
                        channel < CHANNELS, other=0)
        tl.store(State + slot * STATE_STRIDE + channel * STATE_HISTORY + history, value, channel < CHANNELS)


@triton.jit
def _residual_checkpoint(Source, Saved, Rope, SavedRope, Slots, Active, Cu, Checkpoint, Offset,
                         SAVED_STRIDE: tl.constexpr, ROPE_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    begin, end = tl.load(Cu + row), tl.load(Cu + row + 1)
    active = (tl.load(Active + row) != 0) & (slot >= 0) & (slot < 4) & (end > begin)
    value = tl.load(Source + (end - 1) * 10240 + channel, active & (channel < 10240), other=0)
    tl.store(Saved + (slot + 1) * SAVED_STRIDE + channel, value, active & (channel < 10240))
    destination, offset = tl.load(Checkpoint + row), tl.load(Offset + row)
    boundary = active & (destination > 0) & (offset > 0)
    value = tl.load(Source + (begin + offset - 1) * 10240 + channel,
                    boundary & (channel < 10240), other=0)
    tl.store(Saved + destination * SAVED_STRIDE + channel, value, boundary & (channel < 10240))
    if tile == 0:
        axes = tl.arange(0, 4)
        rope = tl.load(Rope + (end - 1) * 3 + axes, active & (axes < 3), other=-1)
        tl.store(SavedRope + (slot + 1) * ROPE_STRIDE + axes, rope, active & (axes < 3))
        rope = tl.load(Rope + (begin + offset - 1) * 3 + axes, boundary & (axes < 3), other=-1)
        tl.store(SavedRope + destination * ROPE_STRIDE + axes, rope, boundary & (axes < 3))


@triton.jit
def _hidden_checkpoint(Hidden, Saved, Delta, Sample, Slots, Active, Cu, Checkpoint, Offset, Positions, Rope,
                       SAVED_STRIDE: tl.constexpr, DELTA_STRIDE: tl.constexpr,
                       PREFILL: tl.constexpr, POSITION_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    valid = (tl.load(Active + row) != 0) & (slot >= 0) & (slot < 4)
    begin, end = tl.load(Cu + row), tl.load(Cu + row + 1)
    valid &= end > begin
    value = tl.load(Hidden + (end - 1) * 2560 + channel, valid & (channel < 2560), other=0)
    tl.store(Sample + row * 2560 + channel, value, channel < 2560)
    tl.store(Saved + (slot + 1) * SAVED_STRIDE + channel, value, valid & (channel < 2560))
    if PREFILL:
        position = tl.load(Positions + row)
        old_delta = tl.load(Delta + (slot + 1) * DELTA_STRIDE, valid, other=0)
        old_next = position + old_delta
        indices = tl.arange(0, POSITION_BLOCK)
        rope = tl.load(Rope + indices, valid & (indices // 3 >= begin) & (indices // 3 < end), other=-1)
        next_position = tl.maximum(old_next, tl.max(rope, 0) + 1)
        tl.store(Delta + (slot + 1) * DELTA_STRIDE, next_position - (position + end - begin), valid)
        destination, offset = tl.load(Checkpoint + row), tl.load(Offset + row)
        if valid & (destination > 0):
            candidate = tl.load(Hidden + (begin + offset - 1) * 2560 + channel, channel < 2560, other=0)
            tl.store(Saved + destination * SAVED_STRIDE + channel, candidate, channel < 2560)
            boundary_rope = tl.where(indices // 3 < begin + offset, rope, -1)
            boundary_next = tl.maximum(old_next, tl.max(boundary_rope, 0) + 1)
            tl.store(Delta + destination * DELTA_STRIDE, boundary_next - (position + offset))


@triton.jit
def _decode_checkpoint(Backing, Slots, Active, Checkpoint, ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot, destination = tl.load(Slots + row), tl.load(Checkpoint + row)
    if (tl.load(Active + row) != 0) & (slot >= 0) & (slot < 4) & (destination > 0):
        offset = tile * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Backing + (slot + 1) * ROW_BYTES + offset, offset < ROW_BYTES, other=0)
        tl.store(Backing + destination * ROW_BYTES + offset, value, offset < ROW_BYTES)


@triton.jit
def _resume_hidden(Saved, Sample, Slots, Epochs, Live, Generation, Status, Error, Consumed, Pending,
                   SAVED_STRIDE: tl.constexpr, CONSUMED_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < 4)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    consumed = tl.load(Consumed + slot * CONSUMED_STRIDE, valid, other=0)
    pending = tl.load(Pending + slot, valid, other=-1)
    ready = (consumed > 0) & (consumed % 64 == 0) & (pending < 0)
    if valid & ~ready:
        tl.store(Status + slot, 3)
        tl.store(Error + slot, 33)
    value = tl.load(Saved + (slot + 1) * SAVED_STRIDE + channel,
                    valid & ready & (channel < 2560), other=0)
    tl.store(Sample + row * 2560 + channel, value, channel < 2560)


class PreparedDecoderState:
    """Derive model metadata from persistent generation state inside each graph.

    Prefill has one sequence and consumes explicit token/position uploads.
    Decode packs the pending GPU samples from up to four physical slots. Epoch
    and terminal-state checks precede any model-state mutation. The scheduler
    validates lengths, vocabulary IDs, context bounds and physical page IDs.
    """

    def __init__(self, inputs: DecoderInputs, requests: DeviceRequestState, pool: HybridStatePool,
                 *, tokens: int, rows: int, mode: str):
        if mode not in ("prefill", "decode") or type(rows) is not int or not 1 <= rows <= 4:
            raise ValueError("decoder mode requires prefill/decode and one through four rows")
        if (type(tokens) is not int or tokens <= 0
                or any(tokens > tensor.shape[0] for tensor in
                       (inputs.token_ids, inputs.prefill_positions, inputs.vision_token_indices))):
            raise ValueError("decoder token capacity exceeds its input buffers")
        if (mode == "prefill" and rows != 1) or (mode == "decode" and tokens != rows):
            raise ValueError("prefill has one row; decode has one token per row")
        self.inputs, self.requests, self.pool = inputs, requests, pool
        self.tokens, self.rows, self.mode = tokens, rows, mode
        device = inputs.token_ids.device
        self.gdn = GDNMetadata.allocate(rows, device)
        self.qsa = QSAMetadata.allocate(tokens=tokens, batch=rows,
            context=inputs.page_tables.shape[1] * 64, device=device)
        self.qsa = QSAMetadata(self.qsa.main_block_table, self.qsa.main_block_table,
            self.gdn.convolution_slots, self.qsa.request_ids, self.qsa.query_positions,
            self.qsa.rope_positions, self.qsa.sequence_lengths, self.gdn.query_start_loc,
            self.gdn.accepted_tokens, self.qsa.is_prefilling, self.qsa.slot_mapping)
        self.token_state = PreparedTokenState(requests, inputs.token_ids[:tokens], self.gdn.query_start_loc,
            inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows)
        self.token_state.prepare()
        self._validate = compile_kernel(_validate,
            (inputs.row_slot_ids, inputs.row_generations, inputs.live_rows, inputs.query_lengths,
             requests.generation, requests.status, requests.error_code, requests.pending_token,
             requests.consumed_tokens, rows, tokens, inputs.page_tables.shape[1] * 64,
             requests.consumed_tokens.stride(0), mode == "prefill"), (4, 1, 1))
        self.ple = PLEMetadata(inputs.token_ids[:tokens], self.gdn.query_start_loc,
            self.token_state.ple_history, self.gdn.convolution_slots, self.gdn.fresh,
            self.gdn.accepted_tokens, self.gdn.num_seqs, self.gdn.num_tokens)
        g, q, t = self.gdn, self.qsa, self.token_state
        args = (inputs.row_slot_ids, inputs.live_rows, inputs.query_lengths,
            t.pending_tokens, t.positions, t.active, pool.rope_delta, inputs.token_ids, inputs.prefill_positions,
            g.query_start_loc, g.convolution_slots, g.fresh, g.state_indices, g.initial_state_indices,
            g.final_state_indices, g.checkpoint_state_indices, g.checkpoint_offsets, g.num_seqs, g.num_tokens,
            q.request_ids, q.query_positions, q.rope_positions, q.sequence_lengths, q.slot_mapping,
            inputs.page_tables, q.main_block_table, q.is_prefilling,
            rows, tokens, inputs.page_tables.shape[1], pool.rope_delta.stride(0), mode == "prefill", 128)
        self._metadata = compile_kernel(_metadata, args, (rows, 1, 1))
        self._convolutions = {}
        self._residual = None
        if mode == "decode":
            self._decode_checkpoint = compile_kernel(_decode_checkpoint,
                (pool.backing, inputs.row_slot_ids, t.active, g.checkpoint_state_indices,
                 pool.layout.row_bytes, 16384), (rows, triton.cdiv(pool.layout.row_bytes, 16384), 1))

    def prepare_hidden(self, hidden, sample):
        self._hidden = compile_kernel(_hidden_checkpoint,
            (hidden, self.pool.last_hidden, self.pool.rope_delta, sample, self.inputs.row_slot_ids,
             self.token_state.active, self.gdn.query_start_loc, self.gdn.checkpoint_state_indices,
             self.gdn.checkpoint_offsets, self.token_state.positions, self.qsa.rope_positions,
             self.pool.last_hidden.stride(0), self.pool.rope_delta.stride(0),
             self.mode == "prefill", triton.next_power_of_2(self.tokens * 3), 4096), (self.rows, 1, 1))

    def prepare_convolution(self, name, source, state):
        if self.mode == "prefill":
            channels, state_history = state.shape[1:]
            history = state_history - self.pool.layout.speculative_tokens
            self._convolutions[name] = compile_kernel(_convolution_checkpoint,
                (source, state, self.gdn.query_start_loc, self.gdn.checkpoint_state_indices,
                 self.gdn.checkpoint_offsets, channels, history, state.stride(0), state_history, 1024),
                (self.rows, triton.cdiv(channels * history, 1024), 1))

    def prepare_residual(self, residual):
        if self.pool.layout.speculative_tokens:
            saved = self.pool.region("target_residual")
            rope = self.pool.region("target_residual_position")
            self._residual = compile_kernel(_residual_checkpoint,
                (residual, saved, self.qsa.rope_positions, rope, self.inputs.row_slot_ids, self.token_state.active,
                 self.gdn.query_start_loc, self.gdn.checkpoint_state_indices,
                 self.gdn.checkpoint_offsets, saved.stride(0), rope.stride(0), 1024), (self.rows, 10, 1))

    def begin(self):
        self._validate()
        self.token_state.gather()
        self.qsa.request_ids.fill_(-1)
        self.qsa.query_positions.fill_(-1)
        self.qsa.rope_positions.fill_(-1)
        self.qsa.slot_mapping.fill_(-1)
        self._metadata()

    def checkpoint_convolution(self, name):
        if self.mode == "prefill":
            self._convolutions[name]()

    def finish(self):
        self._hidden()
        if self._residual is not None:
            self._residual()
        self.token_state.consume()
        if self.mode == "decode":
            self._decode_checkpoint()


class PreparedResume:
    """Sample directly from an aligned restored prefix's final hidden vector."""

    def __init__(self, inputs, requests, pool, sample_hidden, logits, lm_head_weight,
                 completion_records, frame_index, submission_sequence, *, session, name):
        from .linear import BF16Projection
        from .sampling import PreparedSampling

        rows = sample_hidden.shape[0]
        if not 1 <= rows <= 4 or logits.shape != (rows, requests.vocab_size):
            raise ValueError("resume hidden/logit row capacities must agree")
        self._gather = compile_kernel(_resume_hidden,
            (pool.last_hidden, sample_hidden, inputs.row_slot_ids, inputs.row_generations, inputs.live_rows,
             requests.generation, requests.status, requests.error_code, requests.consumed_tokens,
             requests.pending_token, pool.last_hidden.stride(0), requests.consumed_tokens.stride(0), 4096),
            (rows, 1, 1))
        self.sample_hidden = sample_hidden
        self.projection = BF16Projection(sample_hidden, lm_head_weight, logits, session=session, name=name)
        self.sampler = PreparedSampling(logits, requests, inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, completion_records, frame_index, submission_sequence,
            sample_enabled=inputs.sample_enabled[:rows])
        self.sampler.prepare()

    def __call__(self):
        self._gather()
        self.projection(self.sample_hidden)
        self.sampler()
