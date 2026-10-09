"""MTP feedback alignment, private boundary pages, and immutable draft snapshots."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attention import QSAMetadata
from .decoder_state import compile_kernel


@triton.jit
def _restore_page(Source, Cache, Tables, Consumed, Changed,
                  SOURCE_STRIDE: tl.constexpr, CONSUMED_STRIDE: tl.constexpr,
                  PAGE_ELEMENTS: tl.constexpr, TABLE_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    slot, tile = tl.program_id(0), tl.program_id(1)
    consumed = tl.load(Consumed + slot * CONSUMED_STRIDE)
    active = tl.load(Changed + slot) & (consumed > 0)
    page = tl.load(Tables + slot * TABLE_WIDTH + (consumed - 1) // 64, active, other=-1)
    active &= page >= 0
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Source + (slot + 1) * SOURCE_STRIDE + offsets,
                    active & (offsets < PAGE_ELEMENTS), other=0)
    tl.store(Cache + page * PAGE_ELEMENTS + offsets, value, active & (offsets < PAGE_ELEMENTS))


@triton.jit
def _snapshot_page(Cache, Destination, Tables, Slots, Boundaries,
                   DESTINATION_STRIDE: tl.constexpr, PAGE_ELEMENTS: tl.constexpr,
                   TABLE_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot, boundary = tl.load(Slots + row), tl.load(Boundaries + row)
    active = (slot >= 0) & (slot < 4) & (boundary > 0)
    page = tl.load(Tables + slot * TABLE_WIDTH + (boundary - 1) // 64, active, other=-1)
    active &= page >= 0
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Cache + page * PAGE_ELEMENTS + offsets, active & (offsets < PAGE_ELEMENTS), other=0)
    tl.store(Destination + (slot + 5) * DESTINATION_STRIDE + offsets,
             value, active & (offsets < PAGE_ELEMENTS))


@triton.jit
def _snapshot_ring(Keys, Logical, Rope, Cu, RawKeys, RawPositions, RawRope,
                   SavedKeys, SavedPositions, SavedRope, SavedAnchor, Slots, Boundaries,
                   KEY_STRIDE: tl.constexpr, RING: tl.constexpr, SKS: tl.constexpr,
                   SPS: tl.constexpr, SRS: tl.constexpr, SAS: tl.constexpr):
    row, ring_slot = tl.program_id(0), tl.program_id(1)
    slot, boundary = tl.load(Slots + row), tl.load(Boundaries + row)
    active = (slot >= 0) & (slot < 4) & (boundary > 0)
    position = (boundary - 2) // RING * RING + ring_slot
    position = tl.where(position > boundary - 2, position - RING, position)
    kept = active & (position >= boundary - 4) & (position < boundary - 1)
    begin, end = tl.load(Cu + row), tl.load(Cu + row + 1)
    first = tl.load(Logical + begin, active & (end > begin), other=0)
    index = begin + position - first
    current = kept & (index >= begin) & (index < end)
    channels = tl.arange(0, 128)
    key = tl.load(Keys + index * KEY_STRIDE + channels, current, other=0)
    prior_tag = tl.load(RawPositions + (slot + 1) * RING + ring_slot, kept & ~current, other=-1)
    prior = tl.load(RawKeys + ((slot + 1) * RING + ring_slot) * 128 + channels,
                    kept & ~current & (prior_tag == position), other=0)
    tl.store(SavedKeys + (slot + 5) * SKS + ring_slot * 128 + channels,
             tl.where(current, key, prior), active)
    tl.store(SavedPositions + (slot + 5) * SPS + ring_slot, tl.where(kept, position, -1), active)
    axes = tl.arange(0, 4)
    rope = tl.load(Rope + index * 3 + axes, current & (axes < 3), other=-1)
    prior_rope = tl.load(RawRope + ((slot + 1) * RING + ring_slot) * 3 + axes,
                         kept & ~current & (prior_tag == position) & (axes < 3), other=-1)
    tl.store(SavedRope + (slot + 5) * SRS + ring_slot * 3 + axes,
             tl.where(current, rope, prior_rope), active & (axes < 3))
    if ring_slot == 0:
        tl.store(SavedAnchor + (slot + 5) * SAS, boundary - 2, active)


class PreparedDraftCache:
    """Restore the private open page once per physical-slot generation.

    Draft logical page j uses target physical page j+1. A 64N-token immutable
    prefix owns N target pages and N-1 complete draft pages. Its partial draft
    page is carried in the hybrid snapshot and restored into a private page.
    """

    def __init__(self, pool, requests, cache, tables, previous_accepted):
        self.pool, self.requests, self.cache, self.tables = pool, requests, cache, tables
        self.previous_accepted = previous_accepted
        self.generation = torch.zeros(4, dtype=torch.int64, device=pool.backing.device)
        self.changed = torch.zeros(4, dtype=torch.bool, device=pool.backing.device)
        self.slots = torch.arange(4, device=pool.backing.device)
        self.raw_fields = (("raw_k_ring", "draft_raw_keys"),
            ("raw_logical_positions", "draft_raw_positions"),
            ("raw_rope_positions", "draft_raw_rope"),
            ("raw_interval_start_positions", "draft_raw_anchor"))
        self.page_fields = (("main_k_cache", "draft_boundary_key"),
            ("main_v_cache", "draft_boundary_value"),
            ("compressed_k_cache", "draft_boundary_selector"))
        self._snapshots = []
        self._bind_restore()

    def _bind_restore(self):
        self._restore_pages = []
        for cache_name, region in self.page_fields:
            source, cache = self.pool.region(region), getattr(self.cache, cache_name)
            self._restore_pages.append(compile_kernel(_restore_page,
                (source, cache, self.tables, self.requests.consumed_tokens, self.changed,
                 source.stride(0), self.requests.consumed_tokens.stride(0), cache[0].numel(),
                 self.tables.shape[1], 1024), (4, triton.cdiv(cache[0].numel(), 1024), 1)))

    def ensure(self, slots, epochs, live_rows):
        matches = self.slots[:, None] == slots[None, :]
        matches &= self.slots[:slots.numel()][None, :] < live_rows[0]
        matches &= (live_rows[0] >= 0) & (live_rows[0] <= slots.numel())
        matches &= self.requests.generation[:, None] == epochs[None, :]
        selected = matches.sum(1) == 1
        self.changed.copy_(selected & (self.requests.generation != self.generation)
                           & (self.requests.status == 1))
        restored = self.requests.consumed_tokens > 0
        for cache_name, region in self.raw_fields:
            destination = getattr(self.cache, cache_name)[1:5]
            source = self.pool.region(region)[1:5].view_as(destination)
            shape = (4,) + (1,) * (source.ndim - 1)
            empty = 0 if cache_name == "raw_k_ring" else -1
            value = torch.where(restored.view(shape), source, empty)
            destination.copy_(torch.where(self.changed.view(shape), value, destination))
        for operation in self._restore_pages:
            operation()
        self.previous_accepted.copy_(torch.where(self.changed, 1, self.previous_accepted))
        self.generation.copy_(torch.where(self.changed, self.requests.generation, self.generation))

    def snapshot(self, mtp, slots, boundaries):
        operation = PreparedDraftSnapshot(self, mtp, slots, boundaries)
        self._snapshots.append(operation)
        return operation

    def rebind_pages_for_preparation(self, cache):
        self.cache = cache
        self._bind_restore()
        for snapshot in self._snapshots:
            snapshot.bind()


class PreparedDraftSnapshot:
    def __init__(self, owner, mtp, slots, boundaries):
        self.owner, self.mtp, self.slots, self.boundaries = owner, mtp, slots, boundaries
        self.bind()

    def bind(self):
        owner, mtp = self.owner, self.mtp
        cache, pool = owner.cache, owner.pool
        self._pages = []
        for name, region in owner.page_fields:
            source, destination = getattr(cache, name), pool.region(region)
            self._pages.append(compile_kernel(_snapshot_page,
                (source, destination, owner.tables, self.slots, self.boundaries,
                 destination.stride(0), source[0].numel(), owner.tables.shape[1], 1024),
                (self.slots.numel(), triton.cdiv(source[0].numel(), 1024), 1)))
        keys, positions, rope, anchor = (pool.region(name) for name in (
            "draft_raw_keys", "draft_raw_positions", "draft_raw_rope", "draft_raw_anchor"))
        m = mtp.metadata
        self._ring = compile_kernel(_snapshot_ring,
            (mtp.attention.index_key, m.query_positions, m.rope_positions, m.query_start_loc,
             cache.raw_k_ring, cache.raw_logical_positions, cache.raw_rope_positions,
             keys, positions, rope, anchor, self.slots, self.boundaries,
             mtp.attention.index_key.stride(0), cache.raw_k_ring.shape[1], keys.stride(0),
             positions.stride(0), rope.stride(0), anchor.stride(0)), (self.slots.numel(), cache.raw_k_ring.shape[1], 1))

    def __call__(self):
        for operation in self._pages:
            operation()
        self._ring()


class DraftMetadata:
    """Caller-owned feedback rows and QSA metadata for one prepared MTP shape."""

    def __init__(self, *, tokens, rows, context, device):
        self.tokens, self.rows = tokens, rows
        self.qsa = QSAMetadata.allocate(tokens=tokens, batch=rows, context=context, device=device)
        self.token_ids = torch.zeros(tokens, dtype=torch.int64, device=device)
        self.hidden = torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device)
        self.output_indices = torch.zeros(rows, dtype=torch.int64, device=device)
        self.indices = torch.arange(tokens, device=device)
        self.row_indices = torch.arange(rows, device=device)

    def set(self, slots, lengths, starts, token_ids, hidden, rope, tables, *, capacity):
        """Pack fixed-stride source rows into b12x's dense request intervals."""
        m = self.qsa
        ends = lengths.cumsum(0)
        begins = ends - lengths
        m.query_start_loc[0].zero_()
        m.query_start_loc[1:].copy_(ends)
        row = (self.indices[:, None] >= ends[None, :]).sum(1).clamp_max(self.rows - 1)
        local = self.indices - begins[row]
        active = (self.indices < ends[-1]) & (lengths[row] > 0)
        source = (row * capacity + local).clamp(0, token_ids.numel() - 1)
        self.token_ids.copy_(token_ids.reshape(-1)[source])
        self.hidden.copy_(hidden.reshape(-1, 10240)[source])
        self.hidden.masked_fill_(~active[:, None], 0)
        logical = starts[row] + local
        m.request_ids.copy_(torch.where(active, row, -1))
        m.query_positions.copy_(torch.where(active, logical, -1))
        m.rope_positions.copy_(rope.reshape(-1, 3)[source])
        m.rope_positions.masked_fill_(~active[:, None], -1)
        m.raw_state_slot_ids.copy_(torch.where(lengths > 0, slots.long() + 1, -1))
        m.sequence_lengths.copy_(torch.where(lengths > 0, starts + lengths, 0))
        m.num_accepted_tokens.fill_(1)
        m.is_prefilling.fill_(True)
        safe_slots = slots.long().clamp(0, 3)
        m.main_block_table.copy_(tables[safe_slots])
        m.compressed_block_table.copy_(m.main_block_table)
        page = m.main_block_table[row, (logical // 64).clamp(0, tables.shape[1] - 1)]
        m.slot_mapping.copy_(torch.where(active, page.long() * 64 + logical % 64, -1))
        self.output_indices.copy_((ends - 1).clamp_min(0))
