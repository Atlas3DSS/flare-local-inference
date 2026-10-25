"""Dense MTP transactions using the shared acceptance and ABI3 completion engine."""

import torch
import triton
import triton.language as tl

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.components.recurrent import DeltaMetadata
from ignition.components.sampling import _check_tensor, _compile
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.state import gather_rows, save_hidden
from ignition.components.state_transfer import PreparedStateTransfer
from ignition.preparation.scratch import GraphBufferPool

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder
from .mtp import MTPScratch, PreparedMTP


@triton.jit
def _validate_draft_checkpoint(Sources, Destinations, Operations, Expected, Count, Errors,
                               Tables, RequestSlots, StateRows, MemoryCapacity,
                               WIDTH: tl.constexpr, PAGE: tl.constexpr, LIVE: tl.constexpr):
    row = tl.program_id(0)
    count, pages = tl.load(Count), tl.load(MemoryCapacity + 1)
    slot, endpoint = tl.load(RequestSlots + row), tl.load(Expected + row)
    state = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < LIVE), other=0)
    restore = tl.load(Destinations + row) == state
    error = tl.load(Errors + row)
    active = (row < count) & (error == 0) & (tl.load(Operations + row) == 0) & (endpoint > 1)
    logical = endpoint // PAGE - 1
    first = active & (logical >= 0) & (logical < WIDTH)
    second = active & (endpoint % PAGE >= 2) & (logical + 1 < WIDTH)
    page0 = tl.load(Tables + slot * WIDTH + logical, first, other=-1)
    page1 = tl.load(Tables + slot * WIDTH + logical + 1, second, other=-1)
    invalid = (first & ((page0 < 0) | (page0 >= pages))) | (second & ((page1 < 0) | (page1 >= pages)))

    others = tl.arange(0, LIVE)
    other_slot, other_endpoint = tl.load(RequestSlots + others), tl.load(Expected + others)
    other_state = tl.load(StateRows + other_slot * 8, (other_slot >= 0) & (other_slot < LIVE), other=0)
    other_restore = tl.load(Destinations + others) == other_state
    other_active = (others < count) & (others != row) & (tl.load(Operations + others) == 0)
    other_active &= (other_slot >= 0) & (other_slot < LIVE) & (other_endpoint > 1)
    other_logical = other_endpoint // PAGE - 1
    other_first = other_active & (other_logical >= 0) & (other_logical < WIDTH)
    other_second = other_active & (other_endpoint % PAGE >= 2) & (other_logical + 1 >= 0) & (other_logical + 1 < WIDTH)
    other_page0 = tl.load(Tables + other_slot * WIDTH + other_logical, other_first, other=-1)
    other_page1 = tl.load(Tables + other_slot * WIDTH + other_logical + 1, other_second, other=-1)
    overlap = (first & other_first & (page0 == other_page0)) | (first & other_second & (page0 == other_page1))
    overlap |= (second & other_first & (page1 == other_page0)) | (second & other_second & (page1 == other_page1))
    conflict = tl.sum((overlap & (restore | other_restore)).to(tl.int32), 0) > 0
    conflict |= first & second & (page0 == page1)
    error = tl.where(active & conflict, 4, error)
    error = tl.where(active & invalid, 6, error)
    tl.store(Errors + row, error)


@triton.jit
def _copy_draft_checkpoint(Pointers, Backing, Tables, Sources, Destinations, Operations,
                           Expected, Count, Errors, RequestSlots, StateRows,
                           ROW_BYTES: tl.constexpr, PAGE_BYTES: tl.constexpr, WIDTH: tl.constexpr,
                           PAGE: tl.constexpr, LIVE: tl.constexpr, BLOCK: tl.constexpr):
    row, field_part, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    field, part = field_part // 2, field_part % 2
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    slot, endpoint = tl.load(RequestSlots + row), tl.load(Expected + row)
    state = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < LIVE), other=0)
    snapshot = source == state
    logical = endpoint // PAGE + part - 1
    active = (row < tl.load(Count)) & (tl.load(Errors + row) == 0) & (tl.load(Operations + row) == 0)
    active &= (endpoint > 1) & (logical >= 0) & (logical < WIDTH) & ((part == 0) | (endpoint % PAGE >= 2))
    page = tl.load(Tables + slot * WIDTH + logical, active, other=0)
    cache = tl.load(Pointers + field * 3).to(tl.pointer_type(tl.uint8))
    offset = tl.load(Pointers + field * 3 + part + 1)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    mask = active & (index < PAGE_BYTES)
    saved_slot = tl.where(snapshot, destination, source)
    saved = Backing + saved_slot * ROW_BYTES + offset + index
    external = cache + page.to(tl.int64) * PAGE_BYTES + index
    if snapshot:
        value = tl.load(external, mask, other=0)
        tl.store(saved, value, mask)
    else:
        value = tl.load(saved, mask, other=0)
        tl.store(external, value, mask)


class PreparedDenseStateTransfer(PreparedStateTransfer):
    """Save both private draft pages beyond the shared target page prefix.

    With native draft page_offset=1 and N=P//128, draft logical N-1 is
    private even when fully consumed. Logical N also needs saving at P%128>=2.
    This is the two-part Qwen38 checkpoint rule with the dense page size.
    """

    def __init__(self, *args, draft_tables, **kwargs):
        super().__init__(*args, **kwargs)
        pool, c, device = self.pool, self.commands, self.pool.backing.device
        _check_tensor(draft_tables, self.tables.shape, torch.int32, device, "draft_page_tables")
        values, page_bytes = [], None
        for name in ("draft.key", "draft.value"):
            cache = pool.pages[name]
            region, extra = pool.layout.region(name), pool.layout.region(name + ".extra")
            if (region.dtype != torch.uint8 or extra.dtype != torch.uint8 or extra.nbytes != region.nbytes
                    or cache.shape != (self.pages[0][1].shape[0], region.nbytes)
                    or (page_bytes is not None and page_bytes != region.nbytes)):
                raise ValueError("dense draft checkpoint pages must match their two byte backup regions")
            _check_tensor(cache, cache.shape, torch.uint8, device, name)
            values.append((cache.data_ptr(), region.offset, extra.offset))
            page_bytes = region.nbytes
        self.draft_pointers = torch.tensor(values, dtype=torch.int64, device=device)
        self.draft_validate_args = (c.source_slots, c.destination_slots, c.operation, c.expected_consumed,
            c.num_ops, c.errors, draft_tables, c.request_slots, self.state_rows, pool.memory_capacity,
            self.tables.shape[1], PAGE_TOKENS, self.capacity)
        self.draft_copy_args = (self.draft_pointers, pool.backing, draft_tables, c.source_slots,
            c.destination_slots, c.operation, c.expected_consumed, c.num_ops, c.errors, c.request_slots,
            self.state_rows, pool.layout.row_bytes, page_bytes, self.tables.shape[1], PAGE_TOKENS,
            self.capacity, 1024)
        self._draft_validate, self._draft_copy = None, None

    def prepare(self):
        super().prepare()
        self._draft_validate = _compile(_validate_draft_checkpoint, self.draft_validate_args, (self.capacity, 1, 1))
        self._draft_copy = _compile(_copy_draft_checkpoint, self.draft_copy_args,
            (self.capacity, 4, triton.cdiv(self.draft_copy_args[12], 1024)))

    def validate(self):
        super().validate()
        self._draft_validate()

    def apply(self):
        super().apply()
        self._draft_copy()


class DenseMaintenance(PreparedMaintenance):
    """Snapshot the target partial page and both shifted private draft pages."""

    def __init__(self, *args, draft_tables=None, **kwargs):
        self.draft_tables = draft_tables
        super().__init__(*args, **kwargs)

    def _transfer(self):
        enabled = self.draft_tables is not None
        transfer = PreparedDenseStateTransfer if enabled else PreparedStateTransfer
        return transfer(self.pool, self.commands, self.inputs.page_tables,
            pages=((name, value) for name, value in self.pool.pages.items() if not name.startswith("draft.")),
            **({"draft_tables": self.draft_tables} if enabled else {}), page_tokens=PAGE_TOKENS,
            state_rows=self.inputs.request_state_rows, requests=self.requests,
            excluded_regions=self.excluded_regions)


class SpeculativePreparation:
    def __init__(self, weights, mtp_weights, pool, requests, inputs, scratch, *, width,
                 session, workspace, completion, frame, sequence, steering=None):
        self.weights, self.mtp_weights, self.pool = weights, mtp_weights, pool
        self.steering = steering
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.width, self.session, self.workspace = width, session, workspace
        self.completion, self.frame, self.sequence = completion, frame, sequence
        device = pool.backing.device
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        # Replay-commit: every verification round stores the FP32 recurrence operands of its packed tokens here (one
        # plane per linear-attention layer, shared by all round shapes, which never run concurrently) and the commit
        # replays the accepted prefix from the request's single committed row. 33,152 B per token and layer.
        from ignition.components.sm70_gdn import operand_floats
        first = next(layer.linear_attn for layer in weights.tree.layers if layer.block_type == "linear_attention")
        self.key_heads, self.value_heads = first.num_k_heads, first.num_v_heads
        linear_layers = sum(kind == "linear_attention" for kind in weights.config.layers)
        self.replay_operands = torch.zeros((linear_layers, requests.capacity * width,
                                            operand_floats(self.key_heads, self.value_heads)),
                                           dtype=torch.float32, device=device)
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
    """Draft, packed target verification, acceptance, exact replay commit, absorption.

    Each request owns one committed continuation row. Verification reads it as the initial state, writes no state
    and stores every step's recurrence operands; the commit replays the accepted prefix into the row.
    """

    def __init__(self, owner, batch, width, name):
        o, b, w = owner, batch, width
        self.owner, self.batch, self.width = o, b, w
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.columns = torch.arange(w, device=device)
        self.slot_ids = torch.arange(o.requests.capacity, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.delta = DeltaMetadata.allocate(b, device)
        self.replay_operands = o.replay_operands[:, :b * w]
        self.validator = PreparedBatch(o.inputs, o.requests, Metadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pool=o.pool)
        self.decoder = PreparedDecoder(o.weights, o.scratch.rows(b * w, b * w), self.metadata,
            o.pool, mode="decode", delta=self.delta, vocabulary_runtime=None, coordinator=None,
            session=o.session, name=name + ".target", workspace=o.workspace, row_invariant=True,
            replay=self.replay_operands,
            steering=None if o.steering is None else o.steering.bind(o.inputs, o.requests, self.metadata))
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
        from ignition.components.sm70_gdn import PreparedSM70ConvolutionCommit, PreparedSM70ReplayCommit
        self.commit_counts = torch.zeros(b, dtype=torch.int32, device=device)
        linear = {i: o.pool.recurrent(i) for i, kind in enumerate(o.weights.config.layers) if kind == "linear_attention"}
        self.replay_commit = PreparedSM70ReplayCommit(
            o.pool.reserved_backing, {i: state.recurrent for i, state in linear.items()}, self.replay_operands,
            self.metadata.state_rows, self.commit_counts, self.delta.query_start_loc, self.delta.fresh,
            key_heads=o.key_heads, value_heads=o.value_heads)
        self.convolution_commit = PreparedSM70ConvolutionCommit(
            o.pool.reserved_backing, {i: state.convolution for i, state in linear.items()},
            self.metadata.state_rows, self.commit_counts)
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
        states = o.inputs.request_state_rows[safe, 0]
        end = o.requests.consumed_tokens[safe] + limits
        pages, draft_pages = o.inputs.page_tables[safe], o.draft_tables[safe]
        needed = self.validator.block_ids < (end[:, None] + PAGE_TOKENS - 1) // PAGE_TOKENS
        draft_needed = self.validator.block_ids < ((end - 1).clamp_min(0)[:, None] + PAGE_TOKENS - 1) // PAGE_TOKENS
        bad = (limits < 1) | (limits > w) | (end > pages.shape[1] * PAGE_TOKENS)
        bad |= (needed & ((pages < 0) | (pages >= o.pool.memory_capacity[1]))).any(1)
        bad |= (draft_needed & ((draft_pages < 0) | (draft_pages >= o.pool.memory_capacity[1]))).any(1)
        bad |= (states <= 0) | (states >= o.pool.memory_capacity[0])
        # Every committed row is private, even between two packed requests.
        resident = ((self.rows < o.inputs.live_rows) & (slots >= 0) & (slots < o.requests.capacity)
                    & (o.inputs.row_generations[:b] == o.requests.generation[safe]))
        matches = (states[:, None] == states[None, :]) & resident[None, :]
        bad |= matches.sum(1) > 1
        bad_slots = ((self.slot_ids[:, None] == slots[None, :]) & (bad & resident)[None, :]).any(1)
        self.validator.validation_input[0, :o.requests.capacity].add_(bad_slots)
        self.validator.begin(self.validator.validation_input)
        self.active.copy_(self.validator.token_state.active.bool()
            & (o.requests.generated_tokens[safe] < o.requests.max_new_tokens[safe]))
        states = torch.where(self.active, states, 0)
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.candidates[:, 0].copy_(o.requests.pending_token[safe])
        gather_rows(o.pool.last_hidden, states, self.previous)
        self.base_offset.copy_(torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        for i, proposal in enumerate(proposals):
            hidden = self.previous if i == 0 else self.draft.hidden[self.draft_metadata.sample_indices]
            token = self.candidates[:, 0] if i == 0 else self.drafts[:, i - 1]
            live = self.active & (i < limits - 1)
            self.draft_metadata.set(self.before - 1 + i, live.int(), states, draft_pages,
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
        self.metadata.set(self.before, lengths, states, pages, self.candidates, o.inputs.live_rows)
        d, m = self.delta, self.metadata
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(self.active, states, -1))
        d.fresh.copy_(self.before == 0)
        d.accepted_tokens.fill_(1)
        # The committed row is the initial state of every verification position and is never written by it.
        d.state_indices[:, 0].copy_(states)
        d.num_seqs.copy_(m.live_counts[1:])
        d.num_tokens.copy_(m.live_counts[:1])
        self.decoder()
        target()
        verify()
        commit()
        counts = commit.emitted_count
        # Replay the accepted prefix (stored operands, same sequential arithmetic) into the committed row of every
        # linear-attention layer, then normalize the convolution windows.
        self.commit_counts.copy_(counts)
        self.replay_commit()
        self.convolution_commit()
        # Repair all accepted MTP pairs from true target feedback. Proposal KV
        # outside this logical prefix remains unreachable and will be overwritten.
        hidden = self.decoder.scratch.hidden.view(b, w, -1)
        feedback = torch.cat((self.previous[:, None, :], hidden[:, :-1]), dim=1)
        self.absorb_metadata.set((self.before - 1).clamp_min(0), counts, states, draft_pages,
                                 self.candidates, o.inputs.live_rows)
        request = self.absorb_metadata.requests
        starts = self.absorb_metadata.query_start_loc[:-1]
        offsets = torch.arange(b * w, device=counts.device) - starts[request]
        indices = (request * w + offsets).clamp(0, b * w - 1)
        self.absorb_feedback.copy_(feedback.reshape(b * w, -1)[indices])
        self.absorb()
        save_hidden(hidden[self.rows, (counts - 1).long().clamp_min(0)], o.pool.last_hidden,
                    self.metadata.state_rows, counts > 0)
