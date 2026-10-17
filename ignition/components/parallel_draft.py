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


def prepare_sampling_scratch(batch, width, vocab, device, *, buffers=None, proposal=True, verification=True):
    """Share sequential sampling stages and retain target results through verification.

    Proposal probabilities must be copied to retained storage before the next
    proposal position. Target probabilities and validity stay disjoint from the
    verification workspace; filtering intermediates are dead before verification.
    """
    from ignition.preparation.scratch import bind_scratch

    layouts = {}
    if proposal and width > 1:
        layouts["proposal"] = PreparedDraftDistribution.scratch_layout(batch, 1, vocab)
    if verification:
        layouts["target"] = PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab)
        layouts["verify"] = PreparedStochasticVerification.scratch_layout(batch, width, vocab)
    if buffers is None:
        return {name: bind_scratch(layout, device=device) for name, layout in layouts.items()}
    proposal_buffers = buffers.fork()
    retained = (bind_scratch({key: layouts["target"].pop(key) for key in ("probabilities", "valid")},
                            device=device, workspace=buffers) if verification else {})
    cursors = {name: proposal_buffers if name == "proposal" else buffers.fork() for name in layouts}
    result = {name: bind_scratch(layout, device=device, workspace=cursors[name])
              for name, layout in reversed(tuple(layouts.items()))}
    if verification:
        result["target"].update(retained)
    buffers.join(*cursors.values())
    return result


class PreparedParallelDraftRound:
    """Prepare verification and optionally its draft generator.

    A verification-only round receives retained drafts from PreparedDraftVerification;
    it allocates no draft decoder, Markov projection, or proposal samplers. An owner
    whose ``draft_argmax`` names a vocabulary-parallel argmax runtime proposes greedy
    drafts from the draft head's and Markov bias's vocabulary shards without gathering
    them; each proposal distribution is then one-hot, so verification stays exact.
    """

    @staticmethod
    def storage_layout(batch, width, vocab):
        return dict(probabilities=((batch, width - 1, vocab), torch.float32),
                    logits=((batch, 1, vocab), torch.float32))

    def __init__(self, owner, batch, width, name, *, generate_proposals=True, buffers=None, proposal_only=False):
        self.owner, self.batch, self.width = owner, batch, width
        self.generate_proposals = generate_proposals
        self.proposal_only = proposal_only
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.sampling_scratch = prepare_sampling_scratch(b, w, o.requests.vocab_size, device, buffers=buffers,
                                                        proposal=generate_proposals, verification=not proposal_only)
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
        self.decoder = None if proposal_only else o.forwards.decoder(
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
        self.context = None if proposal_only else o.forwards.context(
            o.weights,
            self.decoder.auxiliary,
            self.metadata,
            o.pool,
            session=o.session,
            name=name + ".context",
            workspace=o.workspace,
        )
        block = o.block_size
        self.draft_metadata = Metadata.allocate(b * block, b, blocks, device, samples=b * block)
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
        self.selector = getattr(self.draft, "selector", None)
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
        self.argmax = None
        if self.draft is not None and getattr(o, "draft_argmax", None) is not None:
            from .collectives import PreparedVocabularyArgmax

            self.greedy_tokens = torch.zeros(b, dtype=torch.int64, device=device)
            self.local_logits = self.draft.local_logits.view(b, block, -1)
            self.argmax = PreparedVocabularyArgmax(self.local_logits[:, o.logit_offset], self.markov.local_bias,
                self.greedy_tokens, runtime=o.draft_argmax, session=o.session,
                coordinator=o.collectives["coordinator"], name=name + ".draft_argmax")
        from ignition.preparation.scratch import bind_scratch

        layout = self.storage_layout(b, w, o.requests.vocab_size)
        if not generate_proposals:
            del layout["logits"]
        retained = bind_scratch(layout, device=device, workspace=buffers)
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
        target = None if self.proposal_only else PreparedSpeculativeDistributions(
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
            for i in range(w - 1 if self.generate_proposals and self.argmax is None else 0)
        )
        if target is not None:
            target.prepare()
        for proposal in proposals:
            proposal.prepare()
        if self.proposal_only:
            self.operations = None, proposals, None, None
            return
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

    def propose(self, advance_rng=True, observer=None, allowed=None):
        if not self.generate_proposals:
            raise RuntimeError("verification-only preparation cannot generate proposals")
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        self.validator.begin(self.validation(self.validator.validate()))
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        self.active.copy_(self.validator.token_state.active.bool())
        if allowed is not None:
            self.active.logical_and_(allowed)
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
                o.inputs.page_tables[safe],
                self.draft_inputs,
                o.inputs.live_rows,
            )
            self.draft()
            self.markov_token.copy_(torch.where(self.active, self.candidates[:, 0], 0))
            if self.argmax is not None:
                # Greedy drafts: each position's shard logits plus its Markov bias, from the previous draft.
                for i in range(w - 1):
                    self.markov()
                    self.argmax(self.local_logits[:, i + o.logit_offset], self.markov.local_bias, self.greedy_tokens)
                    self.drafts[:, i].copy_(self.greedy_tokens)
                    self.valid[:, i].copy_(self.active & (i < limits - 1))
                    self.markov_token.copy_(self.greedy_tokens)
                self.probabilities.zero_()
                self.probabilities.scatter_(2, self.drafts[:, :, None], 1.0)
            for i, proposal in enumerate(proposals):
                self.logits[:, 0].copy_(
                    self.draft.scratch.logits.view(b, block, -1)[:, i + o.logit_offset].float()
                )
                if self.markov is not None:
                    self.logits[:, 0].add_(self.markov())
                if self.selector is not None:
                    self.selector(self.logits[:, 0], self.markov_token, i + o.logit_offset)
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
