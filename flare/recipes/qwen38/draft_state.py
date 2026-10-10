"""Packed target-feedback inputs and metadata for prepared MTP execution."""

from __future__ import annotations

import torch

from .attention import QSAMetadata

DRAFT_RAW_FIELDS = ("raw_k_ring", "raw_logical_positions", "raw_rope_positions", "raw_interval_start_positions")


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
