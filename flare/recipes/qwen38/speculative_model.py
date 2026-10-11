"""Captured MTP or copied proposals, target verification, and commit rounds."""

from __future__ import annotations

import torch

from flare.components.draft_sampling import PreparedDraftDistribution

from .decoder import PreparedDecoder
from .draft_state import DRAFT_RAW_FIELDS, DraftMetadata
from .mtp import MTPScratch, PreparedMTP
from flare.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedCopyProposals, PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
)
from .speculative_state import PreparedVerificationState


class SpeculativePreparation:
    """Shared persistent owners for all prepared speculative graph capacities."""

    def __init__(self, *, weights, mtp_weights, pool, requests, inputs, scratch,
                 workspaces, rotary, cache, table, session, completion, frame, sequence,
                 extra_base, context, prefill_capacity, max_cache_pages, proposal_source="mtp",
                 runtime=None, vocabulary_runtime=None, coordinator=None):
        if proposal_source not in ("mtp", "copy"):
            raise ValueError("speculative proposal source must be mtp or copy")
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.proposal_source = proposal_source
        self.weights, self.mtp_weights, self.pool, self.requests = weights, mtp_weights, pool, requests
        self.inputs, self.scratch, self.workspaces, self.rotary = inputs, scratch, workspaces, rotary
        self.cache, self.table, self.session = cache, table, session
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.extra_base, self.context, self.max_cache_pages = extra_base, context, max_cache_pages
        device = pool.backing.device
        self.previous_accepted = torch.ones(requests.capacity, dtype=torch.int32, device=device)
        self.candidates = torch.zeros((requests.capacity, 8), dtype=torch.int64, device=device)
        self.draft_tables = (torch.full((requests.capacity, context // 64), -1, dtype=torch.int32, device=device)
                             if proposal_source == "mtp" else None)
        self.copy_ids = torch.full((requests.capacity, 7), -1, dtype=torch.int64, device=device)
        self.copy_count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.copy_expected = torch.full((requests.capacity,), -1, dtype=torch.int64, device=device)
        self.copy_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.copy_grammar_mask = torch.zeros((requests.capacity, 8, (requests.vocab_size + 31) // 32), dtype=torch.int32, device=device)
        self.copy_grammar_mask_count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.accepted_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.draft_rng_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.draft_rng_offset = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.mtp_scratch = MTPScratch.allocate(prefill_capacity, requests.capacity, device, world_size=weights.shard.world_size) if proposal_source == "mtp" else None
        self.owners, self.decoders, self.mtp_modules = [], [], []

    def buffers(self):
        buffers = {"copy_token_ids": self.copy_ids, "copy_token_count": self.copy_count,
                   "copy_expected_consumed": self.copy_expected, "copy_generation": self.copy_generation,
                   "copy_grammar_mask": self.copy_grammar_mask,
                   "copy_grammar_mask_count": self.copy_grammar_mask_count}
        if self.draft_tables is not None:
            buffers["draft_page_tables"] = self.draft_tables
        return buffers

    def mtp(self, metadata, name, *, cache_only=False):
        if self.proposal_source != "mtp":
            raise RuntimeError("copy proposals do not construct an MTP module")
        module = PreparedMTP(self.mtp_weights, self.mtp_scratch.rows(metadata.tokens, metadata.rows),
            metadata.token_ids, metadata.hidden, self.cache[48], metadata.qsa, self.rotary,
            self.workspaces, output_indices=metadata.output_indices, session=self.session,
            name=name, draft_tokens=7, max_cache_pages=self.max_cache_pages, cache_only=cache_only, **self.collectives)
        self.mtp_modules.append(module)
        return module

    def prefill(self, decoder, name):
        if self.proposal_source == "copy":
            return decoder
        operation = PreparedPrefillAbsorption(self, decoder, name)
        self.owners.append(operation)
        return operation

    def resume(self, resume):
        return resume

    def round(self, batch, width, name):
        operation = PreparedSpeculativeRound(self, batch, width, name)
        self.owners.append(operation)
        return operation

    def rebind_pages_for_preparation(self, cache):
        self.cache = cache
        for module in self.mtp_modules:
            module.rebind_pages_for_preparation(cache[48])


class PreparedPrefillAbsorption:
    """Absorb exactly P-1 target-derived pairs before publishing a P-token prefix."""

    def __init__(self, owner, decoder, name):
        self.owner, self.decoder = owner, decoder
        tokens, device = decoder.state.tokens, owner.pool.backing.device
        self.metadata = DraftMetadata(tokens=tokens, rows=1, context=owner.context, device=device)
        self.mtp = owner.mtp(self.metadata, f"{name}.mtp_absorb", cache_only=True)
        self.target_residual = torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device)
        decoder.residual_output = self.target_residual
        self.previous = torch.zeros((1, 10240), dtype=torch.bfloat16, device=device)
        self.previous_rope = torch.zeros((1, 3), dtype=torch.int64, device=device)
        self.before = torch.zeros(1, dtype=torch.int64, device=device)
        self.indices = torch.arange(tokens, device=device)

    def __call__(self):
        o, state, m = self.owner, self.decoder.state, self.metadata
        slots = o.inputs.row_slot_ids[:1]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.previous.copy_(o.pool.region("target_residual")[safe + 1])
        self.previous_rope.copy_(o.pool.region("target_residual_position")[safe + 1])
        self.decoder()
        active = state.token_state.active.bool()
        length = (state.gdn.query_start_loc[1:] - state.gdn.query_start_loc[:-1]).long()
        shift = (self.before == 0).long()
        indices = (self.indices + shift[0]).clamp_max(self.indices.numel() - 1)
        hidden = self.target_residual[(indices - 1).clamp_min(0)]
        rope = state.qsa.rope_positions[(indices - 1).clamp_min(0)]
        first = (self.indices == 0) & (self.before[0] > 0)
        hidden = torch.where(first[:, None], self.previous, hidden)
        rope = torch.where(first[:, None], self.previous_rope, rope)
        lengths = torch.where(active, (length - shift).clamp_min(0), 0)
        m.set(slots, lengths, (self.before - 1).clamp_min(0), state.ple.token_ids[indices],
              hidden, rope, o.draft_tables, capacity=self.indices.numel())
        self.mtp()



class PreparedSpeculativeRound:
    """One graph contains every draft, all target rows, acceptance and state commit."""

    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width = owner, batch, width
        device = owner.pool.backing.device
        self.token_limits = owner.inputs.query_lengths[:batch]
        self.state = PreparedVerificationState(owner.inputs, owner.requests, owner.pool,
            owner.candidates, owner.previous_accepted, width=width, rows=batch, extra_base=owner.extra_base, token_limits=self.token_limits)
        scratch = owner.scratch.rows(batch * width, batch * width)
        self.target_residual = torch.zeros((batch * width, 10240), dtype=torch.bfloat16, device=device)
        self.decoder = PreparedDecoder(owner.weights, scratch, self.state, owner.cache, owner.rotary,
            owner.table, owner.workspaces, owner.completion, owner.frame, owner.sequence,
            session=owner.session, name=f"{name}.target", max_cache_pages=owner.max_cache_pages,
            residual_output=self.target_residual, verification=True, **owner.collectives)
        owner.decoders.append(self.decoder)
        self.copy_only = owner.proposal_source == "copy"
        self.draft_metadata = (DraftMetadata(tokens=batch, rows=batch, context=owner.context, device=device)
                               if not self.copy_only else None)
        self.draft = owner.mtp(self.draft_metadata, f"{name}.draft") if width > 1 and not self.copy_only else None
        self.absorb_metadata = (DraftMetadata(tokens=batch * width, rows=batch, context=owner.context, device=device)
                                if not self.copy_only else None)
        self.absorb = owner.mtp(self.absorb_metadata, f"{name}.absorb", cache_only=True) if not self.copy_only else None
        self.before = torch.zeros(batch, dtype=torch.int64, device=device)
        self.pending = torch.zeros(batch, dtype=torch.int64, device=device)
        self.previous = torch.zeros((batch, 10240), dtype=torch.bfloat16, device=device)
        self.previous_rope = torch.zeros((batch, 3), dtype=torch.int64, device=device)
        self.active = torch.zeros(batch, dtype=torch.bool, device=device)
        self.drafts = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
        self.draft_probabilities = torch.zeros((batch, width - 1, owner.requests.vocab_size), dtype=torch.float32, device=device)
        self.draft_valid = torch.zeros((batch, width - 1), dtype=torch.bool, device=device)
        self.offered = torch.full((batch,), width - 1, dtype=torch.int32, device=device)
        self.copy_mask = torch.zeros(batch, dtype=torch.int64, device=device)
        self.raw_backup = ({} if self.copy_only else
            {name: torch.empty_like(getattr(owner.cache[48], name)) for name in DRAFT_RAW_FIELDS})
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.seed = torch.zeros(batch, dtype=torch.int64, device=device)
        self.offset = torch.zeros(batch, dtype=torch.int64, device=device)
        self.base_offset = torch.zeros(batch, dtype=torch.int64, device=device)
        self.sampled = torch.zeros(batch, dtype=torch.int64, device=device)
        self.copy_proposals = (PreparedCopyProposals(owner.requests, owner.inputs.row_slot_ids[:batch],
            owner.inputs.row_generations[:batch], owner.inputs.live_rows,
            copy_ids=owner.copy_ids, copy_count=owner.copy_count, copy_expected=owner.copy_expected,
            copy_generation=owner.copy_generation, grammar_mask_count=owner.copy_grammar_mask_count,
            draft_ids=self.drafts, draft_probabilities=self.draft_probabilities, draft_valid=self.draft_valid,
            copy_mask=self.copy_mask, previous_accepted=owner.previous_accepted,
            accepted_generation=owner.accepted_generation, token_limits=self.token_limits) if self.copy_only else None)
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        slots, epochs = o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b]
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1),
            o.requests, slots, epochs, o.inputs.live_rows, self.drafts,
            grammar_masks=o.copy_grammar_mask if self.copy_only else None,
            grammar_mask_count=o.copy_grammar_mask_count if self.copy_only else None)
        proposals = (() if self.copy_only else
            tuple(PreparedDraftDistribution(self.draft.scratch.logits.view(b, 1, -1),
            o.requests, slots, epochs, o.inputs.live_rows, self.drafts, prefix_offset=index,
            seed=self.seed, offset=self.offset, sampled=self.sampled) for index in range(w - 1)))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.draft_probabilities, self.drafts, target.probabilities,
            o.requests, slots, epochs, o.inputs.live_rows, rng=o.rng,
            draft_valid=self.draft_valid, target_valid=target.valid, token_limits=self.token_limits)
        verify.prepare()
        commit = PreparedSpeculativeCommit(o.requests, verify.output,
            self.state.token_ids.view(b, w), slots, epochs, o.inputs.live_rows,
            o.completion, o.frame, o.sequence, rng=o.rng, copy_offered_mask=self.copy_mask,
            draft_cache_available=not self.copy_only, token_limits=self.token_limits)
        self.state.prepare_commit(commit.emitted_count, commit.consumed_before,
                                  self.target_residual, self.decoder.scratch.sample_hidden)
        model_commit = tuple(self.state._commit)
        self.operations = (target, proposals, verify, commit, model_commit)

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        if self.operations is None:
            raise RuntimeError("speculative round prepare() must complete before capture")
        target, proposals, verify, commit, model_commit = self.operations
        slots = o.inputs.row_slot_ids[:b]
        self.state._validate()
        if self.copy_only:
            self.copy_proposals()
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        self.active.copy_((self.rows < o.inputs.live_rows[0]) & (slots >= 0) & (slots < o.requests.capacity)
            & (o.requests.generation[safe] == o.inputs.row_generations[:b]) & (o.requests.status[safe] == 1))
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.pending.copy_(o.requests.pending_token[safe])
        self.previous.copy_(o.pool.region("target_residual")[safe + 1])
        self.previous_rope.copy_(o.pool.region("target_residual_position")[safe + 1])
        for name, backup in self.raw_backup.items():
            backup.copy_(getattr(o.cache[48], name))
        if not self.copy_only:
            self.copy_mask.zero_()
            self.drafts.zero_()
            self.draft_valid.zero_()
        self.base_offset.copy_(torch.where(o.draft_rng_generation[safe] == o.requests.generation[safe],
                                          o.draft_rng_offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        delta = o.pool.rope_delta[safe + 1, 0]
        for index, proposal in enumerate(proposals):
            hidden = self.previous if index == 0 else self.draft.scratch.residual[self.draft_metadata.output_indices]
            tokens = self.pending if index == 0 else self.drafts[:, index - 1]
            rope = self.previous_rope if index == 0 else (self.before + index - 1 + delta)[:, None].expand(-1, 3)
            draft_active = self.active
            if self.token_limits is not None:
                draft_active = draft_active & (index < self.token_limits)
            self.draft_metadata.set(slots, draft_active.long(), self.before - 1 + index,
                tokens, hidden, rope, o.draft_tables, capacity=1)
            self.draft()
            self.offset.copy_(self.base_offset + 4 * index)
            proposal()
            copy = (self.active & (o.copy_generation[safe] == o.requests.generation[safe])
                    & (o.copy_expected[safe] == self.before)
                    & (index < o.copy_count[safe]) & (o.copy_ids[safe, index] >= 0)
                    & (o.copy_ids[safe, index] < o.requests.vocab_size))
            point_mass = (target.vocab_ids == o.copy_ids[safe, index, None]).float()
            self.draft_probabilities[:, index].copy_(torch.where(copy[:, None], point_mass,
                                                                proposal.probabilities[:, 0]))
            self.offset.copy_(self.base_offset + 4 * index)
            self._draw(self.draft_probabilities[:, index])
            self.copy_mask.bitwise_or_(copy.long() * (1 << index))
            self.drafts[:, index].copy_(self.sampled)
            self.draft_valid[:, index].copy_(copy | proposal.valid[:, 0])
        if w > 1 and not self.copy_only:
            for slot in range(o.requests.capacity):
                match = self.active & (slots == slot)
                base = torch.where(match, self.base_offset, 0).sum()
                o.draft_rng_offset[slot].copy_(torch.where(match.any(), base + 4 * (w - 1), o.draft_rng_offset[slot]))
                o.draft_rng_generation[slot].copy_(torch.where(match.any(), o.requests.generation[slot], o.draft_rng_generation[slot]))
        if self.token_limits is not None:
            self.drafts.masked_fill_(self.columns[None, :-1] >= self.token_limits[:, None] - 1, 0)
        o.candidates[:b, 0].copy_(self.pending)
        o.candidates[:b, 1:w].copy_(self.drafts)
        self.decoder()
        target()
        verify()
        commit()
        for operation in model_commit:
            operation()
        if self.copy_only:
            return
        for name, backup in self.raw_backup.items():
            getattr(o.cache[48], name).copy_(backup)
        residual = self.target_residual.view(b, w, 10240)
        hidden = torch.cat((self.previous[:, None, :], residual[:, :-1]), dim=1)
        rope = torch.cat((self.previous_rope[:, None, :], self.state.qsa.rope_positions.view(b, w, 3)[:, :-1]), dim=1)
        self.absorb_metadata.set(slots, commit.emitted_count.long(), self.before - 1,
            self.state.token_ids, hidden, rope, o.draft_tables, capacity=w)
        self.absorb()
