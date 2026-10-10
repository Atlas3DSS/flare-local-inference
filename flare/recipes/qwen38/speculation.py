"""Captured chain acceptance and request-state commit.

Model verification produces one row for the pending input and each draft.
Commit emits a clipped prefix and consumes the equally long prefix of those
inputs. Model checkpoints and MTP history are selected by the caller using the
returned counts; these operations never reconstruct model state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flare.components.sampling import (
    DUPLICATE_SLOT,
    ERROR,
    FINISHED,
    INVALID_DISTRIBUTION,
    RUNNING,
)

from .sampling import DeviceRequestState

INVALID_SPECULATION = 6
TARGET_RNG_DOMAIN = 0x4D54505F56455249
DRAFT_RNG_DOMAIN = 0x4D54505F44524146
WIDTHS = tuple(range(1, 9))


def _tensor(value, shape, dtype, device, name):
    if value.shape != torch.Size(shape) or value.dtype != dtype or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} requires contiguous {dtype} {tuple(shape)} on {device}")


def _geometry(batch, width):
    if type(batch) is not int or batch not in (1, 2, 4) or type(width) is not int or width not in WIDTHS:
        raise ValueError("speculation requires batch capacity 1/2/4 and width 1 through 8")


def _metadata(state, slots, generations, live_rows, batch, device):
    if state.generation.device != device:
        raise ValueError("request state and speculative operands must share a device")
    _tensor(slots, (batch,), torch.int32, device, "slots")
    _tensor(generations, (batch,), torch.int64, device, "generations")
    _tensor(live_rows, (1,), torch.int32, device, "live_rows")


def _live(state, slots, generations, live_rows, rows):
    safe = slots.long().clamp(0, 3)
    selected = (rows < live_rows[0]) & (live_rows[0] >= 0) & (live_rows[0] <= slots.numel())
    selected &= (slots >= 0) & (slots < 4) & (state.generation[safe] == generations)
    duplicates = ((slots[:, None] == slots[None, :]) & selected[None, :]).sum(-1) > 1
    return safe, selected & ~duplicates & (state.status[safe] == RUNNING)


@dataclass
class SpeculativeAcceptance:
    token_ids: torch.Tensor
    proposed_count: torch.Tensor
    accepted_drafts: torch.Tensor
    offered_drafts: torch.Tensor
    valid: torch.Tensor
    rng_increment: torch.Tensor

    @classmethod
    def allocate(cls, batch, width, device):
        _geometry(batch, width)
        return cls(torch.full((batch, width), -1, dtype=torch.int64, device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device),
            torch.zeros(batch, dtype=torch.int64, device=device))

    def validate(self, batch, width, device):
        for name, shape, dtype in (("token_ids", (batch, width), torch.int64),
                ("proposed_count", (batch,), torch.int32), ("accepted_drafts", (batch,), torch.int32),
                ("offered_drafts", (batch,), torch.int32), ("valid", (batch,), torch.bool),
                ("rng_increment", (batch,), torch.int64)):
            _tensor(getattr(self, name), shape, dtype, device, name)


@dataclass
class SpeculativeRngState:
    """Physical-slot target-verification stream, independent of proposal draws."""

    generation: torch.Tensor
    target_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        return cls(torch.zeros(4, dtype=torch.int64, device=device),
                   torch.zeros(4, dtype=torch.int64, device=device))


class PreparedCopyProposals:
    """Validate physical-slot copy commands before verification mutates model state.

    A copied token is a deterministic proposal distribution, including when the
    target sampler is stochastic. Only target verification consumes RNG draws.
    """

    def __init__(self, requests, slots, generations, live_rows, *, copy_ids,
                 copy_count, copy_expected, copy_generation, grammar_mask_count,
                 draft_ids, draft_probabilities, draft_valid, copy_mask,
                 previous_accepted, accepted_generation):
        batch, drafts = draft_ids.shape
        _geometry(batch, drafts + 1)
        device = draft_ids.device
        _metadata(requests, slots, generations, live_rows, batch, device)
        for value, shape, dtype, name in (
                (copy_ids, (4, 7), torch.int64, "copy_ids"),
                (copy_count, (4,), torch.int32, "copy_count"),
                (copy_expected, (4,), torch.int64, "copy_expected"),
                (copy_generation, (4,), torch.int64, "copy_generation"),
                (grammar_mask_count, (4,), torch.int32, "grammar_mask_count"),
                (draft_ids, (batch, drafts), torch.int64, "draft_ids"),
                (draft_probabilities, (batch, drafts, requests.vocab_size), torch.float32, "draft_probabilities"),
                (draft_valid, (batch, drafts), torch.bool, "draft_valid"),
                (copy_mask, (batch,), torch.int64, "copy_mask"),
                (previous_accepted, (4,), torch.int32, "previous_accepted"),
                (accepted_generation, (4,), torch.int64, "accepted_generation")):
            _tensor(value, shape, dtype, device, name)
        self.state, self.slots, self.generations, self.live = requests, slots, generations, live_rows
        self.ids, self.count, self.expected, self.generation = copy_ids, copy_count, copy_expected, copy_generation
        self.mask_count = grammar_mask_count
        self.drafts, self.probabilities, self.valid, self.copy_mask = draft_ids, draft_probabilities, draft_valid, copy_mask
        self.previous_accepted, self.accepted_generation = previous_accepted, accepted_generation
        self.rows = torch.arange(batch, device=device)
        self.active = torch.zeros(batch, dtype=torch.bool, device=device)

    def __call__(self):
        state, drafts = self.state, self.drafts.shape[1]
        safe, live = _live(state, self.slots, self.generations, self.live, self.rows)
        ids = self.ids[safe, :drafts]
        valid = ((self.generation[safe] == self.generations)
            & (self.expected[safe] == state.consumed_tokens[safe])
            & (self.count[safe] >= drafts) & (self.count[safe] <= 7)
            & ((ids >= 0) & (ids < state.vocab_size)).all(-1))
        constrained = state.grammar_enabled[safe].bool()
        valid &= torch.where(constrained,
            (self.mask_count[safe] >= drafts + 1) & (self.mask_count[safe] <= 8),
            self.mask_count[safe] == 0)
        self.active.copy_(live & valid)
        for slot in range(4):
            failed = (live & ~valid & (self.slots == slot)).any()
            state.status[slot].copy_(torch.where(failed, ERROR, state.status[slot]))
            state.error_code[slot].copy_(torch.where(failed, INVALID_SPECULATION, state.error_code[slot]))
            reset = (self.active & (self.slots == slot)).any() & (self.accepted_generation[slot] != state.generation[slot])
            self.previous_accepted[slot].copy_(torch.where(reset, 1, self.previous_accepted[slot]))
            self.accepted_generation[slot].copy_(torch.where(reset, state.generation[slot], self.accepted_generation[slot]))
        self.drafts.copy_(torch.where(self.active[:, None], ids, 0))
        self.valid.copy_(self.active[:, None].expand_as(self.valid))
        self.probabilities.zero_()
        self.probabilities.scatter_(2, self.drafts[:, :, None], 1.)
        self.copy_mask.copy_(torch.where(self.active, (1 << drafts) - 1, 0))
        return self


class PreparedSpeculativeDistributions:
    """Conditional request filtering using the ordinary FlashInfer policy.

    Row j includes candidate IDs before j+prefix_offset in its penalty counts.
    The committed histogram already contains the incoming pending token.
    Optional physical-slot grammar masks describe each conditional prefix.
    """

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                 *, prefix_offset=0, grammar_masks=None, grammar_mask_count=None):
        batch, width, vocab = logits.shape
        _geometry(batch, width)
        device = logits.device
        if logits.dtype not in (torch.float32, torch.bfloat16) or not logits.is_contiguous() or vocab != requests.vocab_size:
            raise ValueError("conditional logits require contiguous FP32/BF16 [B,W,vocabulary]")
        _metadata(requests, slots, generations, live_rows, batch, device)
        if candidate_prefix_ids.ndim != 2 or candidate_prefix_ids.shape[0] != batch:
            raise ValueError("candidate_prefix_ids requires [B,K]")
        _tensor(candidate_prefix_ids, candidate_prefix_ids.shape, torch.int64, device, "candidate_prefix_ids")
        if type(prefix_offset) is not int or prefix_offset < 0 or width - 1 + prefix_offset > candidate_prefix_ids.shape[1]:
            raise ValueError("prefix_offset must keep every conditional prefix within candidate IDs")
        self.logits, self.state, self.slots, self.generations, self.live_rows = logits, requests, slots, generations, live_rows
        self.candidates, self.prefix_offset = candidate_prefix_ids, prefix_offset
        if (grammar_masks is None) != (grammar_mask_count is None):
            raise ValueError("conditional grammar masks and counts must be provided together")
        if grammar_masks is not None:
            _tensor(grammar_masks, (4, 8, (vocab + 31) // 32), torch.int32, device, "grammar_masks")
            _tensor(grammar_mask_count, (4,), torch.int32, device, "grammar_mask_count")
            if width + prefix_offset > 8:
                raise ValueError("conditional grammar prefix exceeds eight mask rows")
        self.grammar_masks, self.grammar_mask_count = grammar_masks, grammar_mask_count
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.vocab_ids = torch.arange(vocab, device=device)
        self.scores = torch.empty((batch, width, vocab), dtype=torch.float32, device=device)
        self.probabilities = torch.empty_like(self.scores)
        self.counts = torch.empty((batch, width, vocab), dtype=torch.int32, device=device)
        self.point_mass_ids = torch.empty((batch, width), dtype=torch.int64, device=device)
        self.maximum = torch.empty((batch, width), dtype=torch.float32, device=device)
        self.valid = torch.empty((batch, width), dtype=torch.bool, device=device)
        self.packed_top_k = torch.empty(batch * width, dtype=torch.int32, device=device)
        self.packed_top_p = torch.empty(batch * width, dtype=torch.float32, device=device)
        self._prepared = False

    def prepare_scores(self):
        state, columns = self.state, self.columns
        safe, live = _live(state, self.slots, self.generations, self.live_rows, self.rows)
        self.counts.copy_(state.histogram[safe, None, :].expand_as(self.counts))
        prefix = columns + self.prefix_offset
        prefix_valid = torch.ones_like(self.valid)
        for index in range(self.candidates.shape[1]):
            ids = self.candidates[:, index]
            used = index < prefix
            valid = (ids >= 0) & (ids < state.vocab_size)
            prefix_valid &= valid[:, None] | ~used[None, :]
            indices = ids.clamp(0, state.vocab_size - 1)[:, None, None].expand(-1, columns.numel(), 1)
            additions = (valid[:, None] & used[None, :]).to(torch.int32).unsqueeze(-1)
            self.counts.scatter_add_(-1, indices, additions)
        params = state.parameters[safe]
        scores = self.logits.float()
        penalty = params[:, None, 3:4]
        adjusted = torch.where(scores > 0, scores / penalty, scores * penalty)
        scores = torch.where(self.counts > 0, adjusted, scores)
        scores = scores - (self.counts > 0) * params[:, None, 4:5] - self.counts * params[:, None, 5:6]
        temperature = params[:, None, :1]
        scores = scores / torch.where(temperature > 0, temperature, 1)
        if self.grammar_masks is None:
            words = state.grammar_mask[safe][:, None, self.vocab_ids // 32]
        else:
            words = self.grammar_masks[safe][:, prefix][:, :, self.vocab_ids // 32]
        allowed = ((words >> (self.vocab_ids % 32)) & 1).bool() | ~state.grammar_enabled[safe, None, None].bool()
        self.scores.copy_(torch.where(allowed, scores, -torch.inf))
        torch.max(self.scores, -1, out=(self.maximum, self.point_mass_ids))
        self.valid.copy_(live[:, None] & prefix_valid & torch.isfinite(self.maximum))
        if self.grammar_masks is not None:
            count = self.grammar_mask_count[safe, None]
            self.valid.logical_and_(~state.grammar_enabled[safe, None].bool()
                                    | ((prefix[None, :] < count) & (count <= 8)))
        elif columns.numel() > 1:
            self.valid.logical_and_(~state.grammar_enabled[safe, None].bool())
        self.packed_top_k.copy_(torch.where(state.top_k[safe] > 0, state.top_k[safe], state.vocab_size)[:, None].expand_as(self.valid).reshape(-1))
        self.packed_top_p.copy_(params[:, 1:2].expand_as(self.valid).reshape(-1))
        return safe, params

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("FlashInfer probability filtering requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        self._prepared = True
        self()
        self.library_scratch = (
            sampling._get_cache_buf(f"top_k_mask_logits_row_states_{self.logits.device}", 1024 * 1024, self.logits.device, zero_init=True),
            sampling._get_cache_buf("softmax_workspace", 1024 * 1024, self.logits.device),
        )

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("probability filtering requires prepare() before capture")
        _, params = self.prepare_scores()
        dummy = torch.where(self.vocab_ids == 0, 0., -torch.inf)
        self.scores.copy_(torch.where(self.valid[:, :, None], self.scores, dummy))
        flat = self.scores.view(-1, self.state.vocab_size)
        masked = self._flashinfer.top_k_mask_logits(flat, self.packed_top_k)
        probabilities = self._flashinfer.softmax(masked)
        probabilities = self._flashinfer.top_p_renorm_probs(probabilities, self.packed_top_p, is_deterministic=True).view_as(self.probabilities)
        maximum = probabilities.amax(-1, keepdim=True)
        eligible = (probabilities > 0) & (probabilities >= maximum * params[:, None, 2:3])
        filtered = torch.where(eligible, probabilities, 0)
        mass = filtered.sum(-1, keepdim=True)
        self.valid.logical_and_(torch.isfinite(mass.squeeze(-1)) & (mass.squeeze(-1) > 0))
        normalized = filtered / mass.clamp_min(torch.finfo(torch.float32).tiny)
        point_mass = (self.vocab_ids == self.point_mass_ids[:, :, None]).float()
        probabilities = torch.where(params[:, None, :1] == 0, point_mass, normalized)
        self.probabilities.copy_(torch.where(self.valid[:, :, None], probabilities, (self.vocab_ids == 0).float()))
        return self.probabilities


class PreparedStochasticVerification:
    """Fixed-width one-row FlashInfer calls with device-owned RNG offsets.

    K+2 bounds scalar Philox draws, including the independent-acceptance metric.
    Reservations are rounded to four draws. Functional library outputs remain
    graph-private and are copied into the persistent acceptance buffers.
    """

    def __init__(self, draft_probs, draft_token_ids, target_probs, requests, slots, generations,
                 live_rows, *, rng, draft_valid=None, target_valid=None, output=None):
        batch, width, vocab = target_probs.shape
        _geometry(batch, width)
        device = target_probs.device
        for value, shape, dtype, name in ((draft_probs, (batch, width - 1, vocab), torch.float32, "draft_probs"),
                (draft_token_ids, (batch, width - 1), torch.int64, "draft_token_ids"),
                (target_probs, (batch, width, vocab), torch.float32, "target_probs")):
            _tensor(value, shape, dtype, device, name)
        _metadata(requests, slots, generations, live_rows, batch, device)
        if vocab != requests.vocab_size:
            raise ValueError("verification vocabulary must match request state")
        for name in ("generation", "target_offset"):
            _tensor(getattr(rng, name), (4,), torch.int64, device, f"rng.{name}")
        self.draft_probs, self.drafts, self.target_probs = draft_probs, draft_token_ids, target_probs
        self.state, self.slots, self.generations, self.live_rows, self.rng = requests, slots, generations, live_rows, rng
        self.draft_valid = torch.ones((batch, width - 1), dtype=torch.bool, device=device) if draft_valid is None else draft_valid
        self.target_valid = torch.ones((batch, width), dtype=torch.bool, device=device) if target_valid is None else target_valid
        _tensor(self.draft_valid, (batch, width - 1), torch.bool, device, "draft_valid")
        _tensor(self.target_valid, (batch, width), torch.bool, device, "target_valid")
        self.output = SpeculativeAcceptance.allocate(batch, width, device) if output is None else output
        self.output.validate(batch, width, device)
        self.rows, self.columns = torch.arange(batch, device=device), torch.arange(width, device=device)
        self.dummy = (torch.arange(vocab, device=device) == 0).float()
        self.safe_draft_probs, self.safe_target_probs = torch.empty_like(draft_probs), torch.empty_like(target_probs)
        self.safe_ids = torch.empty_like(draft_token_ids, dtype=torch.int32)
        self.seed_bits = torch.empty(batch, dtype=torch.int64, device=device)
        self.offset = torch.empty(batch, dtype=torch.int64, device=device)
        self.independent_accepted = torch.zeros(batch, dtype=torch.int32, device=device)
        self.emitted_drafts = torch.zeros(batch, dtype=torch.int32, device=device)
        self.draw_reservation = 4 * ((width + 1 + 3) // 4)
        self._prepared = False

    def prepare(self):
        if self.target_probs.device.type != "cuda":
            raise ValueError("FlashInfer chain verification requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        self._prepared = True
        self()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("chain verification requires prepare() before capture")
        out = self.output
        safe, live = _live(self.state, self.slots, self.generations, self.live_rows, self.rows)
        valid = live & self.draft_valid.all(-1) & self.target_valid.all(-1)
        valid &= ((self.drafts >= 0) & (self.drafts < self.state.vocab_size)).all(-1)
        for probabilities in (self.draft_probs, self.target_probs):
            valid &= (torch.isfinite(probabilities) & (probabilities >= 0)).all(dim=(-1, -2))
            mass = probabilities.sum(-1)
            valid &= (torch.isfinite(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)
        self.safe_draft_probs.copy_(torch.where(valid[:, None, None], self.draft_probs, self.dummy))
        self.safe_target_probs.copy_(torch.where(valid[:, None, None], self.target_probs, self.dummy))
        self.safe_ids.copy_(torch.where(valid[:, None], self.drafts, 0))
        self.seed_bits.copy_(self.state.seed.view(torch.int64)[safe] ^ TARGET_RNG_DOMAIN)
        self.offset.copy_(torch.where(self.rng.generation[safe] == self.generations, self.rng.target_offset[safe], 0))
        self.independent_accepted.zero_()
        self.emitted_drafts.zero_()
        for row in range(self.drafts.shape[0]):
            tokens, _, _ = self._flashinfer.chain_speculative_sampling(
                self.safe_draft_probs[row:row + 1], self.safe_ids[row:row + 1], self.safe_target_probs[row:row + 1],
                self.independent_accepted[row:row + 1], self.emitted_drafts[row:row + 1],
                deterministic=True, seed=self.seed_bits[row:row + 1], offset=self.offset[row:row + 1])
            out.token_ids[row:row + 1].copy_(tokens)
        out.accepted_drafts.copy_(torch.where(valid, self.emitted_drafts, 0))
        out.proposed_count.copy_(torch.where(valid, self.emitted_drafts + 1, 0))
        out.offered_drafts.fill_(self.drafts.shape[1])
        out.valid.copy_(valid)
        out.rng_increment.copy_(torch.where(valid, self.draw_reservation, 0))
        out.token_ids.masked_fill_(~valid[:, None], -1)
        return out


class PreparedSpeculativeCommit:
    """Commit a verified chain prefix and emit physical-slot ABI3 records.

    A round emitting t tokens commits verification input t-1, leaving emitted
    token t-1 pending. Exact snapshot maintenance selects the committed state
    column after the request's model submissions have drained.
    """

    def __init__(self, requests: DeviceRequestState, acceptance: SpeculativeAcceptance,
                 input_ids, slots, generations, live_rows, completion_records, frame_index,
                 submission_sequence, *, rng=None, copy_offered_mask=None, draft_cache_available=True):
        batch, width = input_ids.shape
        _geometry(batch, width)
        device = input_ids.device
        if type(draft_cache_available) is not bool:
            raise ValueError("draft_cache_available must be boolean")
        self.draft_cache_available = draft_cache_available
        _tensor(input_ids, (batch, width), torch.int64, device, "input_ids")
        _metadata(requests, slots, generations, live_rows, batch, device)
        acceptance.validate(batch, width, device)
        _tensor(completion_records, (2, 4, 24), torch.int64, device, "completion_records")
        _tensor(frame_index, (1,), torch.int32, device, "frame_index")
        _tensor(submission_sequence, (1,), torch.int64, device, "submission_sequence")
        self.copy_offered_mask = torch.zeros(batch, dtype=torch.int64, device=device) if copy_offered_mask is None else copy_offered_mask
        _tensor(self.copy_offered_mask, (batch,), torch.int64, device, "copy_offered_mask")
        if rng is not None:
            for name in ("generation", "target_offset"):
                _tensor(getattr(rng, name), (4,), torch.int64, device, f"rng.{name}")
        self.state, self.acceptance, self.inputs, self.slots = requests, acceptance, input_ids, slots
        self.generations, self.live, self.records, self.frame, self.sequence = generations, live_rows, completion_records, frame_index, submission_sequence
        self.rng = rng
        self.rows, self.columns = torch.arange(batch, device=device), torch.arange(width, device=device)
        self.eos_columns = torch.arange(requests.eos_token_ids.shape[1], device=device)
        self.history_columns = torch.arange(2, device=device)
        self.emitted_ids = torch.full_like(input_ids, -1)
        self.committed_input_ids = torch.full_like(input_ids, -1)
        self.emitted_count = torch.zeros(batch, dtype=torch.int32, device=device)
        self.accepted_count = torch.zeros_like(self.emitted_count)
        self.accepted_drafts = torch.zeros_like(self.emitted_count)
        self.commit_column = torch.full_like(self.emitted_count, -1)
        self.consumed_before = torch.zeros(batch, dtype=torch.int64, device=device)
        self.generated_before = torch.zeros_like(self.consumed_before)
        self.history_before = torch.zeros((batch, 2), dtype=torch.int64, device=device)
        self.consumed_after = torch.empty(4, dtype=torch.int64, device=device)
        self.history_after = torch.empty((4, 2), dtype=torch.int64, device=device)
        self._records = torch.zeros((4, 24), dtype=torch.int64, device=device)

    def __call__(self):
        state, acceptance, columns = self.state, self.acceptance, self.columns
        safe = self.slots.long().clamp(0, 3)
        selected = ((self.rows < self.live[0]) & (self.live[0] >= 0) & (self.live[0] <= self.rows.numel())
            & (self.slots >= 0) & (self.slots < 4) & (state.generation[safe] == self.generations))
        self.consumed_before.copy_(state.consumed_tokens[safe])
        self.generated_before.copy_(state.generated_tokens[safe])
        self.history_before.copy_(state.ple_history[safe])
        self.emitted_ids.fill_(-1)
        self.committed_input_ids.fill_(-1)
        self.emitted_count.zero_()
        self.accepted_count.zero_()
        self.accepted_drafts.zero_()
        self.commit_column.fill_(-1)
        self._records.zero_()
        self._records[:, 4].fill_(-1)
        self._records[:, 12].fill_(-1)
        self._records[:, 16:].fill_(-1)

        for slot in range(4):
            matches = selected & (self.slots == slot)
            count = matches.sum()
            row = torch.where(matches, self.rows, 0).amax().view(1)
            ids = acceptance.token_ids.index_select(0, row)[0]
            inputs = self.inputs.index_select(0, row)[0]
            proposed = acceptance.proposed_count.index_select(0, row)[0].long()
            accepted = acceptance.accepted_drafts.index_select(0, row)[0].long()
            offered = acceptance.offered_drafts.index_select(0, row)[0].long()
            valid = acceptance.valid.index_select(0, row)[0]
            increment = acceptance.rng_increment.index_select(0, row)[0]
            remaining = (state.max_new_tokens[slot] - state.generated_tokens[slot]).clamp_min(0)
            enabled = (count == 1) & (state.status[slot] == RUNNING) & (remaining > 0)
            shape_valid = ((proposed >= 1) & (proposed <= columns.numel()) & (accepted == proposed - 1)
                & (offered >= accepted) & (offered <= columns.numel() - 1))
            shape_valid &= (((ids >= 0) & (ids < state.vocab_size)) | ~(columns < proposed)).all()
            shape_valid &= (((inputs >= 0) & (inputs < state.vocab_size)) | ~(columns < proposed)).all()
            shape_valid &= (state.pending_token[slot] >= 0) & (inputs[0] == state.pending_token[slot])
            shape_valid &= ((ids[:-1] == inputs[1:]) | ~(columns[:-1] < accepted)).all()
            if self.rng is None:
                shape_valid &= increment == 0
            else:
                shape_valid &= increment >= 0
            valid = valid & shape_valid
            eos = ((ids[:, None] == state.eos_token_ids[slot])
                & (self.eos_columns < state.eos_count[slot])).any(-1)
            eos_position = torch.where(eos & (columns < proposed), columns + 1, columns.numel()).amin()
            emitted = torch.minimum(torch.minimum(proposed, remaining), eos_position)
            emitted = torch.where(enabled & valid, emitted, 0)
            emitted_drafts = torch.minimum(accepted.clamp_min(0), emitted)
            take = columns < emitted
            pending = ids.gather(0, (emitted - 1).clamp(0, columns.numel() - 1).view(1))[0]
            consumed = state.consumed_tokens[slot] + emitted
            generated = state.generated_tokens[slot] + emitted
            history = torch.cat((state.ple_history[slot], inputs))
            tail = history.gather(0, emitted + self.history_columns)
            terminal = (generated >= state.max_new_tokens[slot]) | (eos & take).any()
            failed = enabled & ~valid
            duplicate = count > 1
            status = torch.where(duplicate | failed, ERROR,
                torch.where((count == 1) & (state.status[slot] == RUNNING) & (terminal | (remaining == 0)), FINISHED, state.status[slot]))
            error = torch.where(duplicate, DUPLICATE_SLOT,
                torch.where(failed, torch.where(shape_valid, INVALID_DISTRIBUTION, INVALID_SPECULATION), state.error_code[slot]))

            state.histogram[slot].scatter_add_(0, ids.clamp(0, state.vocab_size - 1), take.to(torch.int32))
            self.consumed_after[slot].copy_(consumed)
            state.generated_tokens[slot].copy_(generated)
            state.pending_token[slot].copy_(torch.where(emitted > 0, pending, state.pending_token[slot]))
            state.rng_counter.view(torch.int64)[slot].add_(emitted)
            self.history_after[slot].copy_(torch.where(emitted > 0, tail, state.ple_history[slot]))
            state.status[slot].copy_(status)
            state.error_code[slot].copy_(error)
            if self.rng is not None:
                base = torch.where(self.rng.generation[slot] == state.generation[slot], self.rng.target_offset[slot], 0)
                self.rng.target_offset[slot].copy_(torch.where((emitted > 0) & (increment > 0), base + increment, self.rng.target_offset[slot]))
                self.rng.generation[slot].copy_(torch.where((emitted > 0) & (increment > 0), state.generation[slot], self.rng.generation[slot]))

            self.emitted_ids.copy_(torch.where(matches[:, None] & take[None, :], ids, self.emitted_ids))
            self.committed_input_ids.copy_(torch.where(matches[:, None] & take[None, :], inputs, self.committed_input_ids))
            for output, value in ((self.emitted_count, emitted), (self.accepted_count, emitted),
                    (self.accepted_drafts, emitted_drafts), (self.commit_column, emitted - 1)):
                output.copy_(torch.where(matches, value, output))
            record = self._records[slot]
            participating = count > 0
            record[0].copy_(torch.where(participating, state.generation[slot], 0))
            record[1].copy_(torch.where(participating, self.sequence[0], 0))
            record[2].copy_(torch.where(participating, consumed, 0))
            record[3].copy_(torch.where(participating, generated, 0))
            record[4].copy_(torch.where(emitted > 0, pending, -1))
            record[5].copy_(torch.where(participating, status, 0))
            record[6].copy_(torch.where(participating, error, 0))
            record[8].copy_(emitted)
            record[9].copy_(torch.where(enabled & valid, offered, 0))
            record[10].copy_(emitted_drafts)
            record[11].copy_(torch.where(enabled & valid, self.copy_offered_mask.index_select(0, row)[0], 0))
            if self.draft_cache_available:
                record[13].copy_(torch.where(participating & (consumed > 0), consumed - 1, 0))
            else:
                record[13].fill_(-1)
            record[15].copy_(torch.where(participating, columns.numel(), 0))
            record[16:16 + columns.numel()].copy_(torch.where(take, ids, -1))
        state.consumed_tokens.copy_(self.consumed_after)
        state.ple_history.copy_(self.history_after)
        self.records.index_copy_(0, self.frame.long(), self._records.unsqueeze(0))
        return self
