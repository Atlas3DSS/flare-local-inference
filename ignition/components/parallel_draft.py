"""Parallel draft proposals, stochastic target verification and accepted-state commits."""

import torch

from .collectives import PreparedAllReduce
from .draft_sampling import PreparedDraftDistribution
from .paged_batch import Metadata, PreparedBatch
from .speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from .state import save_hidden


def prepare_sampling_scratch(batch, width, vocab, device):
    """Share sequential shapes while keeping target probabilities live through verification.

    Proposal probabilities must be copied to retained storage before the next
    proposal position. Target filtering and verification have separate workspaces
    because verification still reads the filtered target distribution.
    """
    from ignition.preparation.scratch import bind_scratch

    return {
        "proposal": bind_scratch(PreparedDraftDistribution.scratch_layout(batch, 1, vocab), device=device),
        "target": bind_scratch(PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab), device=device),
        "verify": bind_scratch(PreparedStochasticVerification.scratch_layout(batch, width, vocab), device=device),
    }


class PreparedParallelDraftRound:
    """Prepare verification and optionally its draft generator.

    A verification-only round receives retained drafts from PreparedDraftVerification;
    it allocates no draft decoder, Markov projection, or proposal samplers.
    """

    @staticmethod
    def storage_layout(batch, width, vocab):
        return dict(probabilities=((batch, width - 1, vocab), torch.float32),
                    logits=((batch, 1, vocab), torch.float32))

    def __init__(self, owner, batch, width, name, *, generate_proposals=True, sampling_scratch=None,
                 round_storage=None):
        self.owner, self.batch, self.width = owner, batch, width
        self.generate_proposals = generate_proposals
        self.sampling_scratch = {} if sampling_scratch is None else sampling_scratch
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.validator = PreparedBatch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
            page_tokens=o.page_tokens,
        )
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.collectives["runtime"],
            coordinator=o.collectives["coordinator"],
            session=o.session,
            name=name + ".validation",
        )
        self.decoder = o.forwards.decoder(
            o.weights,
            o.tables,
            o.scratch.rows(b * w, b * w),
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            **o.collectives,
        )
        self.context = o.forwards.context(
            o.weights,
            self.decoder.auxiliary,
            self.metadata,
            o.pool,
            session=o.session,
            name=name + ".context",
            workspace=o.workspace,
        )
        block = o.block_size
        self.draft_metadata = Metadata.allocate(b * block, b, 1, device, samples=b * block)
        self.draft = (
            None
            if w == 1 or not generate_proposals
            else o.forwards.draft(
                o.weights,
                o.draft_scratch.rows(b * block, b * block),
                self.draft_metadata,
                o.pool,
                session=o.session,
                name=name + ".draft",
                workspace=o.workspace,
                **o.collectives,
            )
        )
        self.draft_inputs = torch.full(
            (b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device
        )
        self.markov_token = torch.zeros(b, dtype=torch.int64, device=device)
        self.markov = (
            None
            if self.draft is None or o.forwards.markov is None
            else o.forwards.markov(
                o.weights,
                self.markov_token,
                runtime=o.markov_runtime,
                coordinator=o.collectives["coordinator"],
                session=o.session,
                name=name + ".markov",
            )
        )
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        from ignition.preparation.scratch import bind_scratch

        layout = self.storage_layout(b, w, o.requests.vocab_size)
        if not generate_proposals:
            del layout["logits"]
        retained = bind_scratch(layout, device=device, workspace=round_storage)
        self.logits, self.probabilities = retained.get("logits"), retained["probabilities"]
        for tensor in retained.values():
            tensor.zero_()
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (
            torch.zeros_like(self.before) for _ in range(4)
        )
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts,
            workspace=self.sampling_scratch.get("target")
        )
        proposals = tuple(
            PreparedDraftDistribution(
                self.logits,
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=self.sampling_scratch.get("proposal"),
            )
            for i in range(w - 1 if self.generate_proposals else 0)
        )
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=o.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b],
            workspace=self.sampling_scratch.get("verify"),
        )
        verify.prepare()
        commit = PreparedSpeculativeCommit(
            o.requests,
            verify.output,
            self.candidates,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
            o.records,
            o.frame,
            o.sequence,
            rng=o.rng,
            draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b],
        )
        self.operations = target, proposals, verify, commit

    def __call__(self):
        self.propose()
        self.verify()

    def propose(self, advance_rng=True, observer=None):
        if not self.generate_proposals:
            raise RuntimeError("verification-only preparation cannot generate proposals")
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
        self.base_offset.copy_(
            torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0)
        )
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        if self.draft is not None:
            block = self.draft_inputs.shape[1]
            self.draft_inputs[:, 0].copy_(self.candidates[:, 0])
            self.draft_metadata.set(
                self.before,
                self.active.int() * block,
                states,
                o.inputs.page_tables[safe, :1],
                self.draft_inputs,
                o.inputs.live_rows,
            )
            self.draft()
            self.markov_token.copy_(torch.where(self.active, self.candidates[:, 0], 0))
            for i, proposal in enumerate(proposals):
                self.logits[:, 0].copy_(
                    self.draft.scratch.logits.view(b, block, -1)[:, i + o.logit_offset].float()
                )
                if self.markov is not None:
                    self.logits[:, 0].add_(self.markov())
                if observer is not None:
                    observer.record_embedding(i, self.markov.embedding)
                self.offset.copy_(self.base_offset + 4 * i)
                proposal()
                self.drafts[:, i].copy_(self.sampled)
                self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
                self.valid[:, i].copy_(proposal.valid[:, 0] & self.active & (i < limits - 1))
                self.markov_token.copy_(self.sampled)
            if advance_rng:
                self.commit_draft_rng()
        self.candidates[:, 1:].copy_(self.drafts)

    def commit_draft_rng(self, drafts=None):
        o, b, w = self.owner, self.batch, self.width
        drafts = w - 1 if drafts is None else drafts
        slots = o.inputs.row_slot_ids[:b].long()
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        matches = self.active[None, :] & (slots[None, :] == slot_ids[:, None])
        base = torch.where(matches, self.base_offset[None, :], 0).sum(1)
        o.offset.copy_(torch.where(matches.any(1), base + 4 * drafts, o.offset))
        o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))

    def verify(self, proposal_valid=None):
        o, b, w = self.owner, self.batch, self.width
        target, _, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        active = self.active if proposal_valid is None else self.active & proposal_valid
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        self.metadata.set(
            self.before, limits, states, o.inputs.page_tables[safe], self.candidates, o.inputs.live_rows
        )
        self.decoder()
        target()
        verify()
        if proposal_valid is not None:
            verify.output.valid.logical_and_(proposal_valid)
        commit()
        counts = commit.emitted_count
        if w > 1:
            self.decoder.commit(counts)
        self.context(self.decoder.auxiliary, counts)
        hidden = self.decoder.scratch.hidden.view(b, w, o.weights.config.text["hidden_size"])[
            self.rows, (counts - 1).long().clamp_min(0)
        ]
        save_hidden(hidden, o.pool.last_hidden, self.metadata.state_rows, counts > 0)
