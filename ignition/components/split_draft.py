"""Proposal and verification graphs with persistent proposal ownership.

Proposal generation leaves target state and proposal RNG counters unchanged;
verification consumes one proposal. Native scheduling supports a single lane.
"""

import math

import torch
import triton
import triton.language as tl

from .linear import UnquantizedProjection
from .sampling import PreparedCompletion, _write_completion


@triton.jit
def _reject_stale(Valid, Slots, Epochs, Live, Generation, Consumed, Generated, Records, Frame, Sequence,
                  CS: tl.constexpr, SLOTS: tl.constexpr):
    slot = tl.program_id(0)
    if (tl.load(Live) > 0) & (tl.load(Slots) == slot) & ~tl.load(Valid):
        epoch = tl.load(Epochs)
        current = epoch == tl.load(Generation + slot)
        consumed = tl.load(Consumed + slot * CS, current, other=0)
        generated = tl.load(Generated + slot, current, other=0)
        # A diagnostic rejection reports failure without mutating the request occupying the slot.
        token = tl.full((), -1, tl.int64)
        _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                          token, 3, 256, True, 24, SLOTS)


def causal_prefix(confidences, threshold):
    """Admit position i using only confidence through i and stop at the first failure.

    Confidence i must depend only on history and proposals preceding i. Looking
    ahead to a later confidence when deciding whether to admit i biases sampling.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("confidence threshold must be finite and in (0, 1]")
    product, count = 1., 0
    for confidence in confidences:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        product *= confidence
        if product < threshold:
            break
        count += 1
    return count


def lagged_budget(confidences, costs):
    """Choose a width from a completed previous round's confidence snapshot.

    Costs cover widths one through len(confidences)+1 and include the entire
    proposal, verification and feedback step. Current sampled drafts must not
    influence this decision.
    """
    if len(costs) != len(confidences) + 1 or any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("positive complete-step costs must cover every verification width")
    expected, survival = 1., 1.
    utilities = [expected / costs[0]]
    for confidence, cost in zip(confidences, costs[1:], strict=True):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        survival *= confidence
        expected += survival
        utilities.append(expected / cost)
    return max(range(len(utilities)), key=utilities.__getitem__) + 1


class PreparedDraftConfidence:
    """Preserve checkpoint storage precision while accumulating confidence in FP32."""

    def __init__(self, source, weight, output, *, name):
        o, b, w = source.owner, source.batch, source.width
        hidden = source.draft.head_hidden
        markov = source.markov.embedding
        if weight.shape != (1, hidden.shape[1] + markov.shape[1]):
            raise ValueError("confidence weight must cover hidden and preceding-token Markov features")
        self.hidden = hidden.view(b, o.block_size, -1)[:, o.logit_offset:o.logit_offset + w - 1]
        self.embeddings = torch.empty((b, w - 1, markov.shape[1]), dtype=markov.dtype, device=markov.device)
        self.features = torch.empty((b, w - 1, weight.shape[1]), dtype=weight.dtype, device=markov.device)
        self.logits = torch.empty((b, w - 1), dtype=torch.float32, device=markov.device)
        self.output = output
        self.projection = UnquantizedProjection(self.features.view(b * (w - 1), -1), weight,
            self.logits.view(-1, 1), session=o.session, name=name)

    def record_embedding(self, position, embedding):
        self.embeddings[:, position].copy_(embedding)

    def __call__(self):
        self.features[:, :, :self.hidden.shape[-1]].copy_(self.hidden)
        self.features[:, :, self.hidden.shape[-1]:].copy_(self.embeddings)
        self.projection(self.features.view(-1, self.features.shape[-1]))
        self.output.copy_(self.logits.sigmoid())


class PreparedDraftProposal:
    """Generate a retained block without consuming target state or reserving RNG."""

    def __init__(self, source, confidence_output):
        if source.batch != 1 or source.width <= 1 or source.markov is None:
            raise ValueError("split proposal qualification requires one Markov-conditioned request")
        self.source = source
        o = source.owner
        self.confidence = PreparedDraftConfidence(source, o.weights.draft.confidence_head,
            confidence_output, name="split.proposal.confidence")
        self.ready = torch.zeros_like(source.active)
        self.slots = torch.zeros_like(o.inputs.row_slot_ids[:source.batch])
        self.generations = torch.zeros_like(source.before)
        self.state_rows = torch.zeros_like(source.before)
        self.sequence = torch.zeros_like(o.sequence)
        self.completion = PreparedCompletion(o.requests, o.inputs.row_slot_ids[:source.batch],
            o.inputs.row_generations[:source.batch], o.inputs.live_rows, o.records, o.frame, o.sequence)
        self.completion.prepare()

    def buffers(self):
        source = self.source
        return {"proposal_" + name: tensor for name, tensor in dict(
            ready=self.ready, slots=self.slots, generations=self.generations,
            state_rows=self.state_rows, sequence=self.sequence, before=source.before,
            base_offset=source.base_offset, candidates=source.candidates,
            drafts=source.drafts, probabilities=source.probabilities,
            valid=source.valid, active=source.active).items()}

    def __call__(self):
        s, o = self.source, self.source.owner
        s.propose(False, self.confidence)
        self.confidence()
        self.ready.copy_(s.active)
        self.slots.copy_(o.inputs.row_slot_ids[:s.batch])
        safe = self.slots.long().clamp(0, o.requests.capacity - 1)
        self.generations.copy_(o.requests.generation[safe])
        self.state_rows.copy_(o.inputs.request_state_rows[safe, 0])
        self.sequence.copy_(o.sequence)
        self.completion()


class PreparedDraftVerification:
    """Verify a prefix of the immediately preceding retained proposal exactly once."""

    def __init__(self, proposal, target):
        if target.batch != proposal.source.batch or target.width > proposal.source.width:
            raise ValueError("verification shape must fit the retained proposal")
        self.proposal, self.target = proposal, target
        self.original_live = torch.zeros_like(target.owner.inputs.live_rows)
        self.admitted = torch.zeros_like(proposal.ready)

    def __call__(self, same_graph=False):
        p, t = self.proposal, self.target
        s, o, w = p.source, t.owner, t.width
        slots = o.inputs.row_slot_ids[:t.batch]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        valid = p.ready & (p.sequence + (0 if same_graph else 1) == o.sequence)
        valid &= (o.inputs.live_rows == 1) & (o.requests.status[safe] == 1)
        valid &= (slots == p.slots) & (o.inputs.row_generations[:t.batch] == p.generations)
        valid &= (o.requests.generation[safe] == p.generations)
        valid &= (o.requests.consumed_tokens[safe] == s.before)
        valid &= (o.requests.pending_token[safe] == s.candidates[:, 0])
        valid &= (o.inputs.request_state_rows[safe, 0] == p.state_rows)
        valid &= (o.inputs.query_lengths[:t.batch] >= 1) & (o.inputs.query_lengths[:t.batch] <= w)
        self.admitted.copy_(valid)
        self.original_live.copy_(o.inputs.live_rows)
        # Invalid ownership must not make the ordinary commit fail a replacement request.
        o.inputs.live_rows.copy_(torch.where(valid, self.original_live, 0))
        t.before.copy_(s.before)
        t.base_offset.copy_(s.base_offset)
        t.active.copy_(s.active & valid)
        t.candidates.copy_(s.candidates[:, :w])
        t.drafts.copy_(s.drafts[:, :w - 1])
        t.probabilities.copy_(s.probabilities[:, :w - 1])
        t.valid.copy_(s.valid[:, :w - 1])
        # Confidence can expose discarded proposals to a later width decision. Never
        # reuse any of their uniforms, including after target-only verification.
        t.commit_draft_rng(s.width - 1)
        t.verify(valid)
        o.inputs.live_rows.copy_(self.original_live)
        r = o.requests
        torch.library.wrap_triton(_reject_stale)[(r.capacity,)](
            self.admitted, o.inputs.row_slot_ids, o.inputs.row_generations, self.original_live, r.generation,
            r.consumed_tokens, r.generated_tokens, o.records, o.frame, o.sequence,
            r.consumed_tokens.stride(0), r.capacity)
        p.ready.zero_()


class PreparedCombinedDraftRound:
    """One-graph reference for the retained full-block proposal RNG contract."""

    def __init__(self, proposal, verification):
        self.proposal, self.verification = proposal, verification

    def __call__(self):
        self.proposal()
        self.verification(same_graph=True)
