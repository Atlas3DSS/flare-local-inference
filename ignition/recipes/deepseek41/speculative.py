"""Captured DSpark proposals, packed target verification, and accepted-state commits."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.state import save_hidden

from .batch import Metadata, PreparedBatch
from .decoder import PreparedDecoder, Scratch
from .dspark import PreparedContext, PreparedDraft, PreparedMarkov


class SpeculativePreparation:
    def __init__(self, weights, tables, pool, requests, inputs, scratch, *, draft_tokens, records, frame, sequence,
                 session, workspace, runtime, vocabulary_runtime, markov_runtime, coordinator):
        self.weights, self.tables, self.pool = weights, tables, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.records, self.frame, self.sequence = records, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.markov_runtime = markov_runtime
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation = torch.zeros_like(requests.generation)
        self.offset = torch.zeros_like(requests.generation)
        draft = weights.config.draft
        self.block_size = max(1, draft_tokens)
        tokens = requests.capacity * self.block_size
        self.draft_scratch = Scratch.allocate(tokens, tokens, weights.shard.world_size, pool.backing.device,
                                             experts=draft.experts, top_k=draft.top_k)


class PreparedRound:
    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width = owner, batch, width
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.validator = PreparedBatch(o.inputs, o.requests, Metadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pool=o.pool, verification_width=w)
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=o.collectives["runtime"], coordinator=o.collectives["coordinator"],
            session=o.session, name=name + ".validation")
        self.decoder = PreparedDecoder(o.weights, o.tables, o.scratch.rows(b * w, b * w), self.metadata,
            o.pool, mode="decode" if w == 1 else "verify", session=o.session,
            name=name + ".target", workspace=o.workspace, **o.collectives)
        self.context = PreparedContext(o.weights, self.decoder.auxiliary, self.metadata, o.pool,
            session=o.session, name=name + ".context", workspace=o.workspace)
        block = o.block_size
        self.draft_metadata = Metadata.allocate(b * block, b, 1, device, samples=b * block)
        self.draft = None if w == 1 else PreparedDraft(o.weights, o.draft_scratch.rows(b * block, b * block),
            self.draft_metadata, o.pool, session=o.session, name=name + ".draft", workspace=o.workspace, **o.collectives)
        self.draft_inputs = torch.full((b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device)
        self.markov_token = torch.zeros(b, dtype=torch.int64, device=device)
        self.markov = None if w == 1 else PreparedMarkov(o.weights, self.markov_token, runtime=o.markov_runtime,
            coordinator=o.collectives["coordinator"], session=o.session, name=name + ".markov")
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.logits = torch.zeros((b, 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        self.probabilities = torch.zeros((b, w - 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.confidence = torch.zeros((b, w - 1), dtype=torch.float32, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (torch.zeros_like(self.before) for _ in range(4))
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts)
        proposals = tuple(PreparedDraftDistribution(self.logits, *args, self.drafts, prefix_offset=i,
            seed=self.seed, offset=self.offset, sampled=self.sampled) for i in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.probabilities, self.drafts, target.probabilities, *args,
            rng=o.rng, draft_valid=self.valid, target_valid=target.valid, token_limits=o.inputs.query_lengths[:b])
        verify.prepare()
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates, o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b], o.inputs.live_rows, o.records, o.frame, o.sequence,
            rng=o.rng, draft_context_lag=0, token_limits=o.inputs.query_lengths[:b])
        self.operations = target, proposals, verify, commit

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        self.validator.begin(self.validation(self.validator.validate()))
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        self.active.copy_(self.validator.token_state.active.bool())
        self.before.copy_(o.requests.consumed_tokens[safe])
        limits = torch.where(self.active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        self.candidates[:, 0].copy_(o.requests.pending_token[safe])
        self.base_offset.copy_(torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        if self.draft is not None:
            block = self.draft_inputs.shape[1]
            self.draft_inputs[:, 0].copy_(self.candidates[:, 0])
            self.draft_metadata.set(self.before, self.active.int() * block, states,
                o.inputs.page_tables[safe, :1], self.draft_inputs, o.inputs.live_rows)
            self.draft()
            self.markov_token.copy_(torch.where(self.active, self.candidates[:, 0], 0))
            for i, proposal in enumerate(proposals):
                self.logits[:, 0].copy_(self.draft.scratch.logits.view(b, block, -1)[:, i].float() + self.markov())
                self.confidence[:, i].copy_(self.markov.confidence(self.draft.head_hidden.view(b, block, -1)[:, i]))
                self.offset.copy_(self.base_offset + 4 * i)
                proposal()
                self.drafts[:, i].copy_(self.sampled)
                self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
                self.valid[:, i].copy_(proposal.valid[:, 0] & self.active & (i < limits - 1))
                self.markov_token.copy_(self.sampled)
            slot_ids = torch.arange(o.requests.capacity, device=slots.device)
            matches = self.active[None, :] & (slots[None, :] == slot_ids[:, None])
            base = torch.where(matches, self.base_offset[None, :], 0).sum(1)
            o.offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), o.offset))
            o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))
        self.candidates[:, 1:].copy_(self.drafts)
        self.metadata.set(self.before, limits, states, o.inputs.page_tables[safe], self.candidates, o.inputs.live_rows)
        self.decoder()
        target()
        verify()
        commit()
        counts = commit.emitted_count
        if w > 1:
            self.decoder.commit(counts)
        self.context(self.decoder.auxiliary, counts)
        hidden = self.decoder.scratch.hidden.view(b, w, 5120)[self.rows, (counts - 1).long().clamp_min(0)]
        save_hidden(hidden, o.pool.last_hidden, self.metadata.state_rows, counts > 0)
