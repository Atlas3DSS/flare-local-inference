"""Packed request metadata for windowed and compressed-history attention."""

from dataclasses import dataclass

import torch

from ignition.components.sampling import ERROR, RUNNING
from ignition.components.token_state import PreparedTokenState

from .pool import PAGE_TOKENS


@dataclass(frozen=True)
class BatchInputs:
    request_state_rows: torch.Tensor
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, device, *, capacity):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros((capacity, 8), torch.int64), zeros(capacity) - 1,
            zeros(capacity, torch.int64), zeros(1), zeros(capacity), zeros(tokens, torch.int64),
            zeros((capacity, blocks)) - 1, zeros(capacity))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Metadata:
    token_ids: torch.Tensor
    sample_indices: torch.Tensor
    query_start_loc: torch.Tensor
    state_rows: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    requests: torch.Tensor
    positions: torch.Tensor
    page_table: torch.Tensor
    live_counts: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, blocks, device):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros(tokens, torch.int64), zeros(rows, torch.int64), zeros(rows + 1),
            zeros(rows, torch.int64), zeros(rows, torch.int64), zeros(rows), zeros(tokens, torch.int64),
            zeros(tokens, torch.int64) - 1, zeros((rows, blocks)) - 1, zeros(2))


class PreparedBatch:
    def __init__(self, inputs, requests, metadata, *, mode, shard, pool):
        self.inputs, self.requests, self.metadata, self.shard = inputs, requests, metadata, shard
        self.rows, self.tokens, self.mode = metadata.starts.numel(), metadata.token_ids.numel(), mode
        if mode not in ("prefill", "decode") or mode == "prefill" and self.rows != 1:
            raise ValueError("DeepSeek prefill requires one packed sequence")
        self.accept_unsampled = shard.rank != 0
        self.pool = pool
        device, capacity = metadata.token_ids.device, requests.capacity
        self.row_ids = torch.arange(self.rows, device=device)
        self.slot_ids = torch.arange(capacity, device=device)
        self.token_offsets = torch.arange(self.tokens, device=device)
        self.block_ids = torch.arange(inputs.page_tables.shape[1], device=device)
        self.validation_input = torch.zeros((1, max(8, 2 * capacity)), dtype=torch.bfloat16, device=device)
        self.validation_sum = torch.zeros_like(self.validation_input)
        self.present = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.token_state = PreparedTokenState(requests, metadata.token_ids, metadata.query_start_loc,
            inputs.row_slot_ids[:self.rows], inputs.row_generations[:self.rows], inputs.live_rows)
        self.token_state.prepare()

    def validate(self):
        b, r = self.inputs, self.requests
        slots = b.row_slot_ids[:self.rows].long()
        safe = slots.clamp(0, r.capacity - 1)
        resident = (self.row_ids < b.live_rows) & (slots >= 0) & (slots < r.capacity)
        resident &= b.row_generations[:self.rows] == r.generation[safe]
        duplicate = ((slots[:, None] == slots[None, :]) & resident[:, None] & resident[None, :]).sum(1) > 1
        lengths = b.query_lengths[:self.rows] if self.mode == "prefill" else torch.ones_like(safe)
        position = r.consumed_tokens[safe]
        end = position + lengths
        valid = ~duplicate & (b.live_rows <= self.rows) & (lengths > 0) & (lengths <= self.tokens)
        valid &= (position >= 0) & (end <= b.page_tables.shape[1] * PAGE_TOKENS)
        pages = b.page_tables[safe]
        valid &= (((pages >= 0) & (pages < self.pool.memory_capacity[1]))
                  | (self.block_ids >= (end[:, None] + PAGE_TOKENS - 1) // PAGE_TOKENS)).all(1)
        state = b.request_state_rows[safe, 0]
        valid &= (state > 0) & (state < self.pool.memory_capacity[0])
        pending, tokens = r.pending_token[safe], b.token_ids[:self.tokens]
        if self.mode == "prefill":
            invalid = ((tokens < 0) | (tokens >= r.vocab_size)) & (self.token_offsets < lengths[0])
            valid &= (pending < 0) & ~invalid.any()
        else:
            valid &= (tokens >= 0) & (tokens < r.vocab_size)
            valid &= (pending == tokens) | (self.accept_unsampled & (pending < 0))
        present = resident & (r.status[safe] == RUNNING)
        bad = present & ~valid
        bad_slots = ((self.slot_ids[:, None] == slots[None, :]) & bad[None, :]).any(1)
        self.present.copy_(((self.slot_ids[:, None] == slots[None, :]) & present[None, :]).any(1))
        self.validation_input[0, :r.capacity].copy_(bad_slots)
        self.validation_input[0, r.capacity:2 * r.capacity].copy_(self.present)
        return self.validation_input

    def begin(self, totals):
        b, r, m, t = self.inputs, self.requests, self.metadata, self.token_state
        participants = totals[0, r.capacity:2 * r.capacity]
        bad = self.present & ((totals[0, :r.capacity] > 0)
            | ((participants > 0) & (participants < self.shard.world_size)))
        r.status.copy_(torch.where(bad, ERROR, r.status))
        r.error_code.copy_(torch.where(bad, 32, r.error_code))
        t.gather()
        active = t.active.bool()
        slots = b.row_slot_ids[:self.rows].long().clamp(0, r.capacity - 1)
        lengths = b.query_lengths[:self.rows] if self.mode == "prefill" else torch.ones_like(m.lengths)
        lengths = torch.where(active, lengths, 0)
        ends = lengths.cumsum(0, dtype=torch.int32)
        starts = ends - lengths
        m.query_start_loc[0].zero_()
        m.query_start_loc[1:].copy_(ends)
        m.lengths.copy_(lengths)
        m.starts.copy_(t.positions)
        m.state_rows.copy_(torch.where(active, b.request_state_rows[slots, 0], 0))
        m.page_table.copy_(torch.where(active[:, None], b.page_tables[slots], -1))
        m.live_counts[0:1].copy_(ends[-1:])
        m.live_counts[1:2].copy_(b.live_rows.clamp(0, self.rows))
        m.sample_indices.copy_((ends - 1).clamp(min=0))
        request = (self.token_offsets[:, None] >= ends[None, :]).sum(1).clamp(max=self.rows - 1)
        live = self.token_offsets < ends[-1]
        m.requests.copy_(request)
        m.positions.copy_(torch.where(live, t.positions[request] + self.token_offsets - starts[request], -1))
        m.token_ids.copy_(torch.where(live, b.token_ids[:self.tokens]
            if self.mode == "prefill" else b.token_ids[request], 0))

    def finish(self):
        self.token_state.consume()
