"""Causal MTP proposals, stochastic verification, and accepted-prefix state updates."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import save_hidden

from .decoder import PreparedDecoder
from .mtp import PreparedAcceptedContext, PreparedProposals


class PreparedRound:
    def __init__(self, owner, batch, width, name):
        o, b, w = owner, batch, width
        self.batch, self.width = b, w
        self.inputs, self.requests, self.pool = o.inputs, o.requests, o.pool
        self.records, self.frame, self.sequence = o.records, o.frame, o.sequence
        self.rng, self.draft_offset, self.draft_generation = o.rng, o.offset, o.generation
        self.hidden = o.weights.config.hidden
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.validator = PreparedBatch(o.inputs, o.requests, Metadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pool=o.pool, verification_width=w)
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=o.collectives["runtime"], coordinator=o.collectives["coordinator"],
            session=o.session, name=name + ".validation")
        common = dict(session=o.session, workspace=o.workspace, **o.collectives)
        scratch = o.scratch.rows(b * w, b * w)
        self.decoder = PreparedDecoder(o.weights, scratch, self.metadata, o.pool,
            mode="decode" if w == 1 else "verify", name=name + ".target", overlap=o.overlap, **common)
        self.context = PreparedAcceptedContext(o.weights, self.metadata, o.pool)
        self.draft = PreparedProposals(o.weights, o.pool, o.draft_scratch, b, w - 1,
                                      name=name + ".draft", **common)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.probabilities = torch.zeros((b, w - 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (torch.zeros_like(self.before) for _ in range(4))
        self.sampling_scratch = o.sampling_scratch
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        b, w = self.batch, self.width
        args = (self.requests, self.inputs.row_slot_ids[:b], self.inputs.row_generations[:b], self.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1), *args,
            self.drafts, workspace=self.sampling_scratch["target"])
        proposals = tuple(PreparedDraftDistribution(self.draft.logits.view(b, 1, -1), *args,
            self.drafts, prefix_offset=i, seed=self.seed, offset=self.offset, sampled=self.sampled,
            workspace=self.sampling_scratch["proposal"]) for i in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.probabilities, self.drafts, target.probabilities,
            *args, rng=self.rng, draft_valid=self.valid, target_valid=target.valid,
            token_limits=self.inputs.query_lengths[:b], workspace=self.sampling_scratch["verify"])
        verify.prepare()
        commit = PreparedSpeculativeCommit(self.requests, verify.output, self.candidates,
            self.inputs.row_slot_ids[:b], self.inputs.row_generations[:b], self.inputs.live_rows, self.records, self.frame,
            self.sequence, rng=self.rng, token_limits=self.inputs.query_lengths[:b], draft_context_lag=0)
        self.operations = target, proposals, verify, commit

    def __call__(self):
        b, w = self.batch, self.width
        target, proposals, verify, commit = self.operations
        self.validator.begin(self.validation(self.validator.validate()))
        slots = self.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, self.requests.capacity - 1)
        self.active.copy_(self.validator.token_state.active.bool())
        self.before.copy_(self.requests.consumed_tokens[safe])
        limits = torch.where(self.active, self.inputs.query_lengths[:b], 0)
        states = self.inputs.request_state_rows[safe, 0]
        self.candidates[:, 0].copy_(self.requests.pending_token[safe])
        self.base_offset.copy_(torch.where(self.draft_generation[safe] == self.requests.generation[safe], self.draft_offset[safe], 0))
        self.seed.copy_(self.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        self.draft.begin(torch.where(self.active, states, 0))
        if w > 1:
            for i, proposal in enumerate(proposals):
                tokens = self.candidates[:, 0] if i == 0 else self.drafts[:, i - 1]
                proposing = self.active & (i < limits - 1)
                self.draft.step(i, tokens, self.before, states, self.active, proposing)
                self.offset.copy_(self.base_offset + 4 * i)
                proposal()
                self.drafts[:, i].copy_(self.sampled)
                self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
                self.valid[:, i].copy_(proposal.valid[:, 0] & proposing)
            matches = self.active[None] & (slots[None] == torch.arange(self.requests.capacity, device=slots.device)[:, None])
            base = torch.where(matches, self.base_offset[None], 0).sum(1)
            self.draft_offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), self.draft_offset))
            self.draft_generation.copy_(torch.where(matches.any(1), self.requests.generation, self.draft_generation))
        self.draft.finish(self.before, states, self.active)
        self.candidates[:, 1:].copy_(self.drafts)
        self.metadata.set(self.before, limits, states, self.inputs.page_tables[safe], self.candidates, self.inputs.live_rows)
        self.decoder()
        target()
        verify()
        commit()
        counts = commit.emitted_count
        if w > 1:
            self.decoder.commit(counts)
        hidden = self.decoder.scratch.hidden.view(b, w, self.hidden)[self.rows, (counts - 1).long().clamp_min(0)]
        save_hidden(hidden, self.pool.last_hidden, self.metadata.state_rows, counts > 0)
        self.context(self.decoder.auxiliary, self.decoder.scratch.embedding, counts)

    def scratch_tensors(self):
        return (self.metadata, self.before, self.active, self.candidates, self.drafts, self.probabilities,
                self.valid, self.seed, self.offset, self.base_offset, self.sampled,
                self.draft.scratch_tensors())
