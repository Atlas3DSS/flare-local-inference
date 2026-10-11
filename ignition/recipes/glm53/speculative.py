"""Captured GLM proposals, packed target verification and accepted state commits."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.recurrent import DeltaMetadata, DeltaState
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.state import save_hidden

from .batch import PreparedBatch
from .mtp import DraftMetadata, MTPScratch, PreparedMTP
from .stack import PreparedDecoder
from .state import DecoderMetadata


@triton.jit
def _commit_delta_rows(Words, Slots, Counts, Recurrent, Convolution, row_words, row_halves, extra_base,
                       RECURRENT_WORDS: tl.constexpr, CHANNELS: tl.constexpr, COLUMNS: tl.constexpr,
                       WORD_BLOCK: tl.constexpr, CHANNEL_BLOCK: tl.constexpr):
    # One program copies a slice of one layer's accepted recurrent state into the live row and shifts the
    # same slice of its convolution history. Rows that kept at most one token already hold their state.
    layer, row, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + row).to(tl.int64)
    if count > 1:
        slot = tl.load(Slots + row).to(tl.int64)
        live, selected = slot + 1, extra_base + slot * 7 + count - 2
        recurrent = tl.load(Recurrent + layer)
        words = chunk * WORD_BLOCK + tl.arange(0, WORD_BLOCK)
        inside = words < RECURRENT_WORDS
        values = tl.load(Words + selected * row_words + recurrent + words, inside)
        tl.store(Words + live * row_words + recurrent + words, values, inside)
        channels = chunk * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
        present = channels < CHANNELS
        halves = Words.to(tl.pointer_type(tl.bfloat16))
        history = halves + live * row_halves + tl.load(Convolution + layer) + channels * COLUMNS
        first = tl.load(history + count - 1, present)
        second = tl.load(history + count, present)
        third = tl.load(history + count + 1, present)
        tl.store(history, first, present)
        tl.store(history + 1, second, present)
        tl.store(history + 2, third, present)


def delta_regions(pool):
    """Word offsets of every KDA recurrent region, half offsets of its convolution, and their extents."""
    delta = [i for i, layer in enumerate(pool.decoder.layers) if isinstance(layer, DeltaState)]
    recurrent = [pool.layout.region(f"layer{i}.recurrent") for i in delta]
    convolution = [pool.layout.region(f"layer{i}.convolution") for i in delta]
    if (len({(r.shape, r.dtype) for r in recurrent}) != 1 or len({(r.shape, r.dtype) for r in convolution}) != 1
            or recurrent[0].dtype != torch.float32 or convolution[0].dtype != torch.bfloat16):
        raise ValueError("KDA commits require uniform FP32 recurrent and BF16 convolution regions")
    device = pool.backing.device
    return (torch.tensor([r.offset // 4 for r in recurrent], dtype=torch.int64, device=device),
            torch.tensor([r.offset // 2 for r in convolution], dtype=torch.int64, device=device),
            recurrent[0].nbytes // 4, *convolution[0].shape)


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, extra_base, session,
                 workspace, records, frame, sequence, runtime, vocabulary_runtime, coordinator, max_cache_pages=None,
                 prefill_tokens=None, decode_tokens=None):
        self.weights, self.pool, self.requests, self.inputs = weights, pool, requests, inputs
        self.max_cache_pages = max_cache_pages
        # Draft passes and absorptions run serially, so each mode shares one scratch at its largest capacity.
        blocks, device = inputs.page_tables.shape[1], pool.backing.device
        self.draft_scratch = {}
        if prefill_tokens:
            self.draft_scratch["prefill"] = MTPScratch.absorbing(scratch.rows(prefill_tokens, "prefill", sample_rows=1))
        if decode_tokens:
            self.draft_scratch["decode"] = MTPScratch.allocate(decode_tokens, weights.shard.world_size, blocks,
                                                               "decode", device)
        self.scratch, self.extra_base, self.session, self.workspace = scratch, extra_base, session, workspace
        self.records, self.frame, self.sequence = records, frame, sequence
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.draft_tables = torch.full_like(inputs.page_tables, -1)
        self.draft_generation = torch.zeros_like(requests.generation)
        self.draft_offset = torch.zeros_like(requests.generation)

    def draft(self, metadata, name, *, cache_only=False, mode="decode"):
        shared = self.draft_scratch.get(mode)
        scratch = None if shared is None else shared.rows(metadata.token_ids.numel(), mode)
        return PreparedMTP(self.weights, self.pool.draft, metadata, mode=mode,
            session=self.session, name=name, workspace=self.workspace, cache_only=cache_only,
            max_cache_pages=self.max_cache_pages, scratch=scratch, **self.collectives)


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

    def rebind_pages(self, pages):
        self.absorb.rebind_pages()
        self.generation.rebind_pages(pages)

    def __call__(self):
        o, g = self.owner, self.generation
        slot = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[slot])
        self.previous.copy_(o.pool.last_hidden[slot + 1])
        g()
        length = g.batch.metadata.recurrent.num_tokens.long()
        skip = (self.before == 0).long()
        indices = (self.indices + skip[0]).clamp_max(self.indices.numel() - 1)
        hidden = self.feedback[(indices - 1).clamp_min(0)]
        hidden = torch.where(((self.indices == 0) & (self.before[0] > 0))[:, None], self.previous, hidden)
        self.metadata.set(o.inputs.row_slot_ids[:1], (length - skip).clamp_min(0),
            (self.before - 1).clamp_min(0), g.batch.metadata.token_ids[indices], hidden,
            o.draft_tables, capacity=self.indices.numel())
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
        self.delta_regions = delta_regions(o.pool)
        self.metadata = DecoderMetadata.allocate(b * w, b, blocks, device)
        self.metadata = DecoderMetadata(self.metadata.token_ids, torch.arange(b * w, device=device),
            DeltaMetadata.allocate(b, device, verification_columns=8), self.metadata.indexed)
        self.validator = PreparedBatch(o.inputs, o.requests, DecoderMetadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pages=o.pool.draft.index.cache.shape[0])
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=o.collectives['runtime'], coordinator=o.collectives['coordinator'],
            session=o.session, name=f"{name}.validation")
        self.decoder = PreparedDecoder(o.weights, o.scratch.rows(b * w, "decode", sample_rows=b * w),
            o.pool.decoder, self.metadata, mode="decode", session=o.session, name=f"{name}.target",
            workspace=o.workspace, max_cache_pages=o.max_cache_pages, **o.collectives)
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
        self.tail_backup = torch.empty_like(o.pool.draft.index.tail[:o.requests.capacity + 1])
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts)
        proposals = tuple(PreparedDraftDistribution(self.draft_logits.view(b, 1, -1), *args,
            self.drafts, prefix_offset=i, seed=self.seed, offset=self.offset, sampled=self.sampled)
            for i in range(w - 1))
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

    def rebind_pages(self, pages):
        self.decoder.rebind_pages()
        for draft in (self.draft, self.absorb):
            if draft is not None:
                draft.rebind_pages()
        self.validator.pages = pages

    def target_metadata(self, slots, limits):
        o, b, w, m = self.owner, self.batch, self.width, self.metadata
        g, index = m.recurrent, m.indexed
        lengths = torch.where(self.active, limits, 0)
        ends = lengths.cumsum(0)
        starts = ends - lengths
        g.query_start_loc[0].zero_()
        g.query_start_loc[1:].copy_(ends)
        g.convolution_slots.copy_(torch.where(self.active, slots + 1, -1))
        g.fresh.zero_()
        g.accepted_tokens.fill_(1)
        states = torch.where(self.state_columns == 0, slots[:, None] + 1,
            o.extra_base + slots[:, None] * 7 + self.state_columns - 1)
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
        """Keep each live KDA row's state after its last emitted token in one launch for all layers."""
        o, w = self.owner, self.width
        backing = o.pool.backing
        words, halves, extent, channels, columns = self.delta_regions
        args = (backing.view(torch.int32), slots, counts, words, halves, backing.shape[1] // 4,
                backing.shape[1] // 2, o.extra_base, extent, channels, columns, 4096, 128)
        grid = (words.numel(), self.batch, max(triton.cdiv(extent, 4096), triton.cdiv(channels, 128)))
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(_commit_delta_rows)[grid](*args, num_warps=4)
        else:
            _commit_delta_rows[grid](*args, num_warps=4)
        hidden = self.decoder.scratch.hidden.view(self.batch, w, 4096)
        selected_hidden = hidden[self.rows, (counts - 1).clamp_min(0).long()]
        save_hidden(selected_hidden, o.pool.last_hidden[1:1 + o.requests.capacity],
                    o.inputs.row_slot_ids[:self.batch], (counts > 0).int())

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
        self.previous.copy_(o.pool.last_hidden[safe + 1])
        self.tail_backup.copy_(o.pool.draft.index.tail[:o.requests.capacity + 1])
        self.base_offset.copy_(torch.where(o.draft_generation[safe] == o.requests.generation[safe], o.draft_offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        for i, proposal in enumerate(proposals):
            hidden = self.previous if i == 0 else self.draft.hidden[self.draft_metadata.output_indices]
            token = self.pending if i == 0 else self.drafts[:, i - 1]
            proposing = self.active & (i < limits - 1)
            self.draft_metadata.set(slots, proposing.long(), self.before - 1 + i,
                token, hidden, o.draft_tables, capacity=1)
            self.draft()
            torch.index_select(self.draft.logits, 0, self.draft_metadata.output_indices, out=self.draft_logits)
            self.offset.copy_(self.base_offset + 4 * i)
            proposal()
            self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
            self.drafts[:, i].copy_(self.sampled)
            self.valid[:, i].copy_(proposal.valid[:, 0] & proposing)
        # One masked update per tensor instead of one storage write per request slot.
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        match = self.active[None, :] & (slots[None, :] == slot_ids[:, None])
        matched = match.any(1)
        base = torch.where(match, self.base_offset[None, :], 0).sum(1)
        o.draft_offset.copy_(torch.where(matched, base + 4 * (w - 1), o.draft_offset))
        o.draft_generation.copy_(torch.where(matched, o.requests.generation, o.draft_generation))
        self.candidates[:, 0].copy_(self.pending)
        self.candidates[:, 1:].copy_(self.drafts)
        self.target_metadata(slots, limits)
        self.decoder()
        target()
        verify()
        commit()
        self.commit_state(slots, commit.emitted_count)
        o.pool.draft.index.tail[:o.requests.capacity + 1].copy_(self.tail_backup)
        hidden = torch.cat((self.previous[:, None], self.decoder.scratch.hidden.view(b, w, 4096)[:, :-1]), dim=1)
        self.absorb_metadata.set(slots, commit.emitted_count.long(), self.before - 1,
            self.candidates, hidden, o.draft_tables, capacity=w)
        self.absorb()
