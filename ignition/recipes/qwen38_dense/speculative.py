"""Dense MTP transactions using the shared acceptance and ABI3 completion engine."""

import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.components.recurrent import DeltaMetadata
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.state import gather_rows, scatter_rows, save_hidden
from ignition.components.state_transfer import PreparedStateTransfer
from ignition.preparation.scratch import GraphBufferPool

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder
from .mtp import MTPScratch, PreparedMTP


class DenseMaintenance(PreparedMaintenance):
    """Snapshot draft partial pages at P-1, including target page boundaries."""

    def __init__(self, *args, draft_tables=None, **kwargs):
        self.draft_tables = draft_tables
        super().__init__(*args, **kwargs)

    def _transfer(self):
        enabled = self.draft_tables is not None
        return PreparedStateTransfer(self.pool, self.commands, self.inputs.page_tables,
            pages=((name, value) for name, value in self.pool.pages.items() if not name.startswith("draft.")),
            trailing_pages=((name, value) for name, value in self.pool.pages.items()
                            if enabled and name.startswith("draft.")),
            trailing_tables=self.draft_tables, page_tokens=PAGE_TOKENS,
            state_rows=self.inputs.request_state_rows, requests=self.requests,
            excluded_regions=self.excluded_regions)


class SpeculativePreparation:
    def __init__(self, weights, mtp_weights, pool, requests, inputs, scratch, *, width,
                 session, workspace, completion, frame, sequence):
        self.weights, self.mtp_weights, self.pool = weights, mtp_weights, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.width, self.session, self.workspace = width, session, workspace
        self.completion, self.frame, self.sequence = completion, frame, sequence
        device = pool.backing.device
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.draft_tables = torch.full_like(inputs.page_tables, -1)
        self.generation = torch.zeros_like(requests.generation)
        self.offset = torch.zeros_like(requests.generation)
        self.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        self.owners = []

    def draft(self, metadata, feedback, name, *, cache_only=False):
        scratch = MTPScratch.allocate(metadata.token_ids.numel(), metadata.sample_indices.numel(),
                                      self.weights.config, self.pool.backing.device)
        key, value = self.pool.attention("draft", self.weights.config)
        operation = PreparedMTP(self.mtp_weights, scratch, metadata.token_ids, feedback,
            metadata, key, value, self.workspace, output_indices=metadata.sample_indices,
            session=self.session, name=name, cache_only=cache_only)
        self.owners.append(operation)
        return operation


class PreparedPrefill:
    """Absorb exactly P-1 pairs; last_hidden remains the normalized target tail."""

    def __init__(self, owner, generation, name):
        self.owner, self.generation = owner, generation
        o, g = owner, generation
        device, tokens = o.pool.backing.device, g.batch.tokens
        self.feedback = torch.zeros((tokens, o.weights.config.hidden), dtype=torch.float32, device=device)
        g.decoder.feedback_output = self.feedback
        self.previous = torch.zeros((1, o.weights.config.hidden), dtype=torch.float32, device=device)
        self.before = torch.zeros(1, dtype=torch.int64, device=device)
        self.indices = torch.arange(tokens, device=device)
        self.metadata = Metadata.allocate(tokens, 1, o.inputs.page_tables.shape[1], device)
        self.draft_feedback = torch.zeros_like(self.feedback)
        self.absorb = o.draft(self.metadata, self.draft_feedback, name + ".absorb", cache_only=True)
        g.batch.validate_prefill = self.validate_draft_pages

    def validate_draft_pages(self, tokens, lengths):
        o = self.owner
        slots = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        pages = o.draft_tables[slots]
        endpoint = (self.before + lengths - 1).clamp_min(0)
        needed = self.generation.batch.block_ids < (endpoint[:, None] + PAGE_TOKENS - 1) // PAGE_TOKENS
        return (~needed | ((pages >= 0) & (pages < o.pool.memory_capacity[1]))).all(1)

    def __call__(self):
        o, g = self.owner, self.generation
        slots = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[slots])
        states = o.inputs.request_state_rows[slots, 0]
        states = torch.where((states > 0) & (states < o.pool.memory_capacity[0]), states, 0)
        gather_rows(o.pool.last_hidden, states, self.previous)
        g()
        skip = (self.before == 0).long()
        indices = (self.indices + skip[0]).clamp_max(self.indices.numel() - 1)
        self.draft_feedback.copy_(torch.where(((self.indices == 0) & (self.before[0] > 0))[:, None],
            self.previous, self.feedback[(indices - 1).clamp_min(0)]))
        lengths = (g.batch.metadata.lengths - skip).clamp_min(0)
        self.metadata.set((self.before - 1).clamp_min(0), lengths,
            states, o.draft_tables[slots],
            g.batch.metadata.token_ids[indices][None, :], o.inputs.live_rows)
        self.absorb()


class PreparedRound:
    """Draft, packed target verification, acceptance, exact checkpoint commit, absorption."""

    def __init__(self, owner, batch, width, name):
        o, b, w = owner, batch, width
        self.owner, self.batch, self.width = o, b, w
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.columns = torch.arange(w, device=device)
        self.slot_ids = torch.arange(o.requests.capacity, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.delta = DeltaMetadata.allocate(b, device, verification_columns=w)
        self.validator = PreparedBatch(o.inputs, o.requests, Metadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pool=o.pool)
        self.decoder = PreparedDecoder(o.weights, o.scratch.rows(b * w, b * w), self.metadata,
            o.pool, mode="decode", delta=self.delta, vocabulary_runtime=None, coordinator=None,
            session=o.session, name=name + ".target", workspace=o.workspace)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.previous = torch.zeros((b, o.weights.config.hidden), dtype=torch.float32, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.draft_metadata = Metadata.allocate(b, b, blocks, device)
        self.draft_feedback = torch.zeros_like(self.previous)
        self.draft = o.draft(self.draft_metadata, self.draft_feedback, name + ".draft") if w > 1 else None
        self.draft_logits = torch.zeros((b, 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        self.absorb_metadata = Metadata.allocate(b * w, b, blocks, device)
        self.absorb_feedback = torch.zeros((b * w, o.weights.config.hidden), dtype=torch.float32, device=device)
        self.absorb = o.draft(self.absorb_metadata, self.absorb_feedback, name + ".absorb", cache_only=True)
        self.seed, self.offset, self.base_offset, self.sampled = (torch.zeros_like(self.before) for _ in range(4))
        buffers = o.sampling_buffers.cursor()
        self.sampling_scratch = prepare_sampling_scratch(b, w, o.requests.vocab_size, device, buffers=buffers)
        self.probabilities = buffers.zeros((b, w - 1, o.requests.vocab_size), torch.float32)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.recurrent_copy = torch.zeros((b, 48 * 128 * 128), dtype=torch.float32, device=device)
        from ignition.components.sm70_gdn import PreparedSM70ConvolutionCommit
        self.commit_counts = torch.zeros(b, dtype=torch.int32, device=device)
        self.convolution_commit = PreparedSM70ConvolutionCommit(
            o.pool.reserved_backing,
            {i: o.pool.recurrent(i).convolution for i, kind in enumerate(o.weights.config.layers)
             if kind == "linear_attention"}, self.metadata.state_rows, self.commit_counts)
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1),
            *args, self.drafts, workspace=self.sampling_scratch["target"])
        proposals = tuple(PreparedDraftDistribution(self.draft_logits, *args, self.drafts,
            prefix_offset=i, seed=self.seed, offset=self.offset, sampled=self.sampled,
            workspace=self.sampling_scratch["proposal"]) for i in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.probabilities, self.drafts, target.probabilities,
            *args, rng=o.rng, draft_valid=self.valid, target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b], workspace=self.sampling_scratch["verify"])
        verify.prepare()
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.completion, o.frame, o.sequence, logits=self.decoder.scratch.logits.view(b, w, -1),
            rng=o.rng, token_limits=o.inputs.query_lengths[:b])
        commit.prepare()
        self.operations = target, proposals, verify, commit

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        limits = o.inputs.query_lengths[:b]
        self.validator.validate()
        states = o.inputs.request_state_rows[safe, :w]
        end = o.requests.consumed_tokens[safe] + limits
        pages, draft_pages = o.inputs.page_tables[safe], o.draft_tables[safe]
        needed = self.validator.block_ids < (end[:, None] + PAGE_TOKENS - 1) // PAGE_TOKENS
        draft_needed = self.validator.block_ids < ((end - 1).clamp_min(0)[:, None] + PAGE_TOKENS - 1) // PAGE_TOKENS
        bad = (limits < 1) | (limits > w) | (end > pages.shape[1] * PAGE_TOKENS)
        bad |= (needed & ((pages < 0) | (pages >= o.pool.memory_capacity[1]))).any(1)
        bad |= (draft_needed & ((draft_pages < 0) | (draft_pages >= o.pool.memory_capacity[1]))).any(1)
        bad |= ((states <= 0) | (states >= o.pool.memory_capacity[0])).any(1)
        # Every write column is private, even between two packed requests.
        flat = states.reshape(-1)
        resident = ((self.rows < o.inputs.live_rows) & (slots >= 0) & (slots < o.requests.capacity)
                    & (o.inputs.row_generations[:b] == o.requests.generation[safe]))
        matches = (flat[:, None] == flat[None, :]) & resident.repeat_interleave(w)[None, :]
        bad |= (matches.sum(1).view(b, w) > 1).any(1)
        bad_slots = ((self.slot_ids[:, None] == slots[None, :]) & bad[None, :]).any(1)
        self.validator.validation_input[0, :o.requests.capacity].add_(bad_slots)
        self.validator.begin(self.validator.validation_input)
        self.active.copy_(self.validator.token_state.active.bool()
            & (o.requests.generated_tokens[safe] < o.requests.max_new_tokens[safe]))
        states = torch.where(self.active[:, None], states, 0)
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.candidates[:, 0].copy_(o.requests.pending_token[safe])
        gather_rows(o.pool.last_hidden, states[:, 0], self.previous)
        self.base_offset.copy_(torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        for i, proposal in enumerate(proposals):
            hidden = self.previous if i == 0 else self.draft.hidden[self.draft_metadata.sample_indices]
            token = self.candidates[:, 0] if i == 0 else self.drafts[:, i - 1]
            live = self.active & (i < limits - 1)
            self.draft_metadata.set(self.before - 1 + i, live.int(), states[:, 0], draft_pages,
                                    token[:, None], o.inputs.live_rows)
            self.draft_feedback.copy_(hidden[self.draft_metadata.requests])
            self.draft()
            self.draft_logits[:, 0].copy_(self.draft.logits)
            self.offset.copy_(self.base_offset + 4 * i)
            proposal()
            self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
            self.drafts[:, i].copy_(self.sampled)
            self.valid[:, i].copy_(proposal.valid[:, 0] & live)
        matches = self.active[None, :] & (self.slot_ids[:, None] == slots[None, :])
        base = torch.where(matches, self.base_offset[None, :], 0).sum(1)
        o.offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), o.offset))
        o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))
        self.candidates[:, 1:].copy_(self.drafts)
        lengths = torch.where(self.active, limits, 0)
        self.metadata.set(self.before, lengths, states[:, 0], pages, self.candidates, o.inputs.live_rows)
        d, m = self.delta, self.metadata
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(self.active, states[:, 0], -1))
        d.fresh.copy_(self.before == 0)
        d.accepted_tokens.fill_(1)
        d.state_indices.copy_(torch.where(self.active[:, None], states, 0))
        d.num_seqs.copy_(m.live_counts[1:])
        d.num_tokens.copy_(m.live_counts[:1])
        self.decoder()
        target()
        verify()
        commit()
        counts = commit.emitted_count
        selected = states.gather(1, (counts - 1).long().clamp_min(0)[:, None])[:, 0]
        for i, kind in enumerate(o.weights.config.layers):
            if kind == "linear_attention":
                recurrent = o.pool.recurrent(i).recurrent.flatten(1)
                gather_rows(recurrent, selected, self.recurrent_copy)
                scatter_rows(self.recurrent_copy, states[:, 0], recurrent, counts > 0)
        self.commit_counts.copy_(counts)
        self.convolution_commit()
        # Repair all accepted MTP pairs from true target feedback. Proposal KV
        # outside this logical prefix remains unreachable and will be overwritten.
        hidden = self.decoder.scratch.hidden.view(b, w, -1)
        feedback = torch.cat((self.previous[:, None, :], hidden[:, :-1]), dim=1)
        self.absorb_metadata.set((self.before - 1).clamp_min(0), counts, states[:, 0], draft_pages,
                                 self.candidates, o.inputs.live_rows)
        request = self.absorb_metadata.requests
        starts = self.absorb_metadata.query_start_loc[:-1]
        offsets = torch.arange(b * w, device=counts.device) - starts[request]
        indices = (request * w + offsets).clamp(0, b * w - 1)
        self.absorb_feedback.copy_(feedback.reshape(b * w, -1)[indices])
        self.absorb()
        save_hidden(hidden[self.rows, (counts - 1).long().clamp_min(0)], o.pool.last_hidden,
                    states[:, 0], counts > 0)
