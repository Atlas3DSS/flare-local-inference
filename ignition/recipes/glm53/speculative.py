"""Captured GLM proposals, packed target verification and accepted state commits."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.flashinfer_sampling import sample_probabilities
from ignition.components.recurrent import DeltaMetadata, DeltaState
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.state import gather_rows, save_hidden

from .batch import PreparedBatch
from .mtp import DraftMetadata, PreparedMTP
from .stack import PreparedDecoder
from .state import DecoderMetadata


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, session,
                 workspace, records, frame, sequence, runtime, vocabulary_runtime, coordinator):
        self.weights, self.pool, self.requests, self.inputs = weights, pool, requests, inputs
        self.scratch, self.session, self.workspace = scratch, session, workspace
        self.records, self.frame, self.sequence = records, frame, sequence
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.draft_tables = torch.full_like(inputs.page_tables, -1)
        self.draft_generation = torch.zeros_like(requests.generation)
        self.draft_offset = torch.zeros_like(requests.generation)

    def draft(self, metadata, name, *, cache_only=False, mode="decode"):
        return PreparedMTP(self.weights, self.pool.draft, metadata, mode=mode,
            session=self.session, name=name, workspace=self.workspace, cache_only=cache_only, **self.collectives)


class PreparedPrefill:
    """Absorb every prompt pair, retaining normalized target feedback at its endpoint."""

    def __init__(self, owner, generation, feedback, name):
        self.owner, self.generation, self.feedback = owner, generation, feedback
        tokens = feedback.shape[0]
        device = feedback.device
        self.metadata = DraftMetadata(tokens, 1, owner.inputs.page_tables.shape[1], device)
        self.absorb = owner.draft(self.metadata, f"{name}.absorb", cache_only=True, mode="prefill")
        self.previous = torch.zeros((1, 4096), dtype=torch.bfloat16, device=device)
        self.indices = torch.arange(tokens, device=device)
        self.before = torch.zeros(1, dtype=torch.int64, device=device)

    def __call__(self):
        o, g = self.owner, self.generation
        slot = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[slot])
        gather_rows(o.pool.last_hidden, o.inputs.request_state_rows[slot, 0], self.previous)
        g()
        length = g.batch.metadata.recurrent.num_tokens.long()
        skip = (self.before == 0).long()
        indices = (self.indices + skip[0]).clamp_max(self.indices.numel() - 1)
        hidden = self.feedback[(indices - 1).clamp_min(0)]
        hidden = torch.where(((self.indices == 0) & (self.before[0] > 0))[:, None], self.previous, hidden)
        self.metadata.set(o.inputs.row_slot_ids[:1], (length - skip).clamp_min(0),
            (self.before - 1).clamp_min(0), g.batch.metadata.token_ids[indices], hidden,
            o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=self.indices.numel())
        self.absorb()


class PreparedRound:
    """All ranks replay identical stochastic draws from gathered logits and request seeds."""

    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width = owner, batch, width
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows, self.columns = torch.arange(b, device=device), torch.arange(w, device=device)
        self.offsets = torch.arange(b * w, device=device)
        self.state_columns = torch.arange(8, device=device)
        self.metadata = DecoderMetadata.allocate(b * w, b, blocks, device)
        self.metadata = DecoderMetadata(self.metadata.token_ids, torch.arange(b * w, device=device),
            DeltaMetadata.allocate(b, device, verification_columns=8), self.metadata.indexed)
        self.validator = PreparedBatch(o.inputs, o.requests, DecoderMetadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pages=o.pool.memory_capacity[1:])
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=o.collectives['runtime'], coordinator=o.collectives['coordinator'],
            session=o.session, name=f"{name}.validation")
        self.decoder = PreparedDecoder(o.weights, o.scratch.rows(b * w, "decode", sample_rows=b * w),
            o.pool.decoder, self.metadata, mode="decode", session=o.session, name=f"{name}.target",
            workspace=o.workspace, **o.collectives)
        self.draft_metadata = DraftMetadata(b, b, blocks, device)
        self.draft = o.draft(self.draft_metadata, f"{name}.draft") if w > 1 else None
        self.draft_logits = torch.empty_like(self.draft.logits) if self.draft is not None else None
        self.absorb_metadata = DraftMetadata(b * w, b, blocks, device)
        self.absorb = o.draft(self.absorb_metadata, f"{name}.absorb", cache_only=True)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.pending = torch.zeros_like(self.before)
        self.previous = torch.zeros((b, 4096), dtype=torch.bfloat16, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.probabilities = torch.zeros((b, w - 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (torch.zeros_like(self.before) for _ in range(4))
        self.tail_backup = torch.empty_like(o.pool.draft.index.tail[:b])
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts)
        proposals = tuple(PreparedSpeculativeDistributions(self.draft_logits.view(b, 1, -1), *args,
            self.drafts, prefix_offset=i) for i in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.probabilities, self.drafts, target.probabilities,
            *args, rng=o.rng, draft_valid=self.valid, target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b])
        verify.prepare()
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.records, o.frame, o.sequence, rng=o.rng, token_limits=o.inputs.query_lengths[:b])
        self.operations = target, proposals, verify, commit
        if w > 1:
            dummy = torch.zeros((b, o.requests.vocab_size), dtype=torch.float32, device=self.seed.device)
            dummy[:, 0].fill_(1)
            self.draw(dummy)

    def draw(self, probabilities):
        for row in range(self.batch):
            self.sampled[row:row + 1].copy_(sample_probabilities(probabilities[row:row + 1],
                seed=self.seed[row:row + 1], offset=self.offset[row:row + 1]))

    def target_metadata(self, slots, limits):
        o, b, w, m = self.owner, self.batch, self.width, self.metadata
        g, index = m.recurrent, m.indexed
        lengths = torch.where(self.active, limits, 0)
        ends = lengths.cumsum(0)
        starts = ends - lengths
        g.query_start_loc[0].zero_()
        g.query_start_loc[1:].copy_(ends)
        states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1)]
        g.convolution_slots.copy_(torch.where(self.active, states[:, 0], -1))
        g.fresh.zero_()
        g.accepted_tokens.fill_(1)
        g.state_indices.copy_(torch.where(self.active[:, None], states, 0))
        g.num_seqs.copy_(o.inputs.live_rows.clamp(0, b))
        g.num_tokens.copy_(ends[-1:])
        index.query_start_loc.copy_(g.query_start_loc)
        index.state_slots.copy_(g.convolution_slots)
        index.block_table.copy_(o.inputs.page_tables[slots.clamp(0, o.requests.capacity - 1)])
        row = (self.offsets[:, None] >= ends[None]).sum(1).clamp_max(b - 1)
        column = self.offsets - starts[row]
        active = self.offsets < ends[-1]
        pos = self.before[row] + column
        index.requests.copy_(row)
        index.positions.copy_(torch.where(active, pos, -1))
        page = index.block_table[row, (pos // 256).clamp(0, index.block_table.shape[1] - 1)]
        index.token_slots.copy_(torch.where(active, page.long() * 256 + pos % 256, -1))
        source = (row * w + column).clamp(0, b * w - 1)
        m.token_ids.copy_(torch.where(active, self.candidates.reshape(-1)[source], 0))
        m.sample_indices.copy_((starts[:, None] + self.columns).clamp(0, b * w - 1).reshape(-1))

    def commit_state(self, slots, counts):
        o, w = self.owner, self.width
        states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1)]
        selected = states[self.rows, (counts - 1).clamp_min(0).long()]
        live = torch.where(counts > 0, states[:, 0], 0).long()
        for layer in o.pool.decoder.layers:
            if isinstance(layer, DeltaState):
                values = layer.recurrent[selected]
                values = torch.where((counts > 0)[:, None, None, None], values, layer.recurrent[live])
                layer.recurrent.index_copy_(0, live, values)
                conv = layer.convolution[live]
                indices = ((counts - 1).clamp_min(0)[:, None, None]
                           + torch.arange(3, device=slots.device)[None, None]).expand(-1, conv.shape[1], -1)
                history = conv.gather(2, indices)
                conv[:, :, :3].copy_(history)
                layer.convolution.index_copy_(0, live, conv)
        hidden = self.decoder.scratch.hidden.view(self.batch, w, 4096)
        selected_hidden = hidden[self.rows, (counts - 1).clamp_min(0).long()]
        save_hidden(selected_hidden, o.pool.last_hidden,
                    live, (counts > 0).int())

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        limits = o.inputs.query_lengths[:b]
        self.validator.validate()
        end = o.requests.consumed_tokens[safe] + limits
        bad = (limits < 1) | (limits > w) | (end > o.inputs.page_tables.shape[1] * 256)
        pages = o.inputs.page_tables[safe]
        required = self.validator.block_ids < (end[:, None] + 255) // 256
        bad |= (required & ((pages < 0) | (pages >= self.validator.pages))).any(1)
        bad_slots = ((self.validator.slot_ids[:, None] == slots[None]) & bad[None]).any(1)
        self.validator.validation_input[0, :o.requests.capacity].add_(bad_slots)
        self.validator.begin(self.validation(self.validator.validation_input))
        self.active.copy_(self.validator.token_state.active.bool())
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.pending.copy_(o.requests.pending_token[safe])
        gather_rows(o.pool.last_hidden, o.inputs.request_state_rows[safe, 0], self.previous)
        gather_rows(o.pool.draft.index.tail, o.inputs.request_state_rows[safe, 0], self.tail_backup)
        self.base_offset.copy_(torch.where(o.draft_generation[safe] == o.requests.generation[safe], o.draft_offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        for i, proposal in enumerate(proposals):
            hidden = self.previous if i == 0 else self.draft.hidden[self.draft_metadata.output_indices]
            token = self.pending if i == 0 else self.drafts[:, i - 1]
            proposing = self.active & (i < limits - 1)
            self.draft_metadata.set(slots, proposing.long(), self.before - 1 + i,
                token, hidden, o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=1)
            self.draft()
            torch.index_select(self.draft.logits, 0, self.draft_metadata.output_indices, out=self.draft_logits)
            proposal()
            self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
            self.offset.copy_(self.base_offset + 4 * i)
            self.draw(self.probabilities[:, i])
            self.drafts[:, i].copy_(self.sampled)
            self.valid[:, i].copy_(proposal.valid[:, 0] & proposing)
        for slot in range(o.requests.capacity):
            match = self.active & (slots == slot)
            base = torch.where(match, self.base_offset, 0).sum()
            o.draft_offset[slot].copy_(torch.where(match.any(), base + 4 * (w - 1), o.draft_offset[slot]))
            o.draft_generation[slot].copy_(torch.where(match.any(), o.requests.generation[slot], o.draft_generation[slot]))
        self.candidates[:, 0].copy_(self.pending)
        self.candidates[:, 1:].copy_(self.drafts)
        self.target_metadata(slots, limits)
        self.decoder()
        target()
        verify()
        commit()
        self.commit_state(slots, commit.emitted_count)
        tail = o.pool.draft.index.tail
        save_hidden(self.tail_backup.view(b, -1), tail.view(tail.shape[0], -1),
                    o.inputs.request_state_rows[safe, 0], self.active)
        hidden = torch.cat((self.previous[:, None], self.decoder.scratch.hidden.view(b, w, 4096)[:, :-1]), dim=1)
        self.absorb_metadata.set(slots, commit.emitted_count.long(), self.before - 1,
            self.candidates, hidden, o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=w)
        self.absorb()
