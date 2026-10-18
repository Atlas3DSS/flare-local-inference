"""MLA DFlash2 context injection, bidirectional blocks, and conditional proposals.

As vLLM keeps a replicated DFlash2 cache beside a context-parallel target, every rank holds the draft's
E4M3 latent records of every position: its full-attention layer in pages that cover all of their
positions, and each sliding layer in a ring per continuation row that holds its window.
"""

from dataclasses import dataclass
from functools import partial
import math

import torch
from b12x.attention import dense_mla

from ignition.components.candidate_selector import PreparedCandidateSelector
from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.kda_speculation import KDARecords
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.rotary import _rotate
from ignition.components.speculation import SpeculativeRngState
from ignition.components.vocabulary import PreparedVocabularyProjection

from .config import PAGE_TOKENS
from .dcp import write_records
from .decoder import PreparedDecoder, Scratch
from .model import rms_norm
from .pool import DRAFT_RING, draft_layers

FP8 = torch.float8_e4m3fn


class DraftRotary:
    def __init__(self, config, device):
        c = config.text["rope_parameters"]
        dims = torch.arange(32, dtype=torch.float32, device=device)
        base, original, factor = c["rope_theta"], c["original_max_position_embeddings"], c["factor"]
        def boundary(rotations):
            return 64 * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(boundary(c["beta_fast"])), 0), min(math.ceil(boundary(c["beta_slow"])), 63)
        smooth = 1 - ((dims - low) / max(high - low, 1e-3)).clamp(0, 1)
        frequencies = 1 / base ** (dims / 32)
        self.frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
        self.scale = (1 + .1 * c["mscale_all_dim"] * math.log(factor)) ** 2 / math.sqrt(192)

    def __call__(self, tensor, positions):
        heads = tensor.shape[1] if tensor.ndim == 3 else 1
        torch.library.wrap_triton(_rotate)[(tensor.shape[0], heads)](
            tensor, positions, self.frequencies, tensor.stride(0), tensor.stride(1) if tensor.ndim == 3 else 0,
            tensor.shape[-1] - 64, 1, None, False)


class PreparedContext:
    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 runtime, coordinator, buffers):
        self.metadata, self.pool = metadata, pool
        c, device, tokens = weights.config.draft.text, auxiliary.device, auxiliary.shape[0]
        def buffer(key, width):
            if key not in buffers:
                buffers[key] = torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
            return buffers[key][:tokens]
        self.local = buffer("local", c["hidden_size"] // weights.shard.world_size)
        self.hidden = buffer("hidden", c["hidden_size"])
        self.kv = buffer("kv", 576)
        self.latent = buffer("latent", 576)
        self.project = UnquantizedProjection(auxiliary, weights.draft.projection, self.local,
                                             session=session, name=name + ".project")
        self.gather = PreparedChunkedAllGather(self.local, self.hidden, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".gather")
        self.norm = rms_norm(weights.draft.hidden_norm, c["rms_norm_eps"])
        self.rotary = DraftRotary(weights.config.draft, device)
        # Page slots of the full-attention layers' records and ring slots of the sliding layers'.
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.ring_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        sliding, full = draft_layers(weights.config)
        self.page_records = pool.draft_attention(full[0]).shape[1] if full else PAGE_TOKENS
        self.layers = []
        for i, layer in enumerate(weights.draft.layers):
            a = layer["attention"]
            kv = UnquantizedProjection(self.hidden, a["context"], self.kv, session=session, name=f"{name}.{i}.kv")
            norm = rms_norm(a["kv_norm"], c["rms_norm_eps"])
            self.layers.append((kv, norm, pool.draft_attention(i), i in sliding))

    def __call__(self, auxiliary, counts):
        m, records = self.metadata, self.page_records
        columns = (m.positions.clamp_min(0) // records).clamp_max(m.page_table.shape[1] - 1)
        pages = m.token_page_table.gather(1, columns[:, None]).squeeze(1)
        live = (m.positions >= 0) & (m.positions - m.starts[m.requests] < counts[m.requests])
        self.slots.copy_(torch.where(live, pages.long() * records + m.positions % records, -1))
        # A ring keeps each request's last positions; earlier positions of a longer chunk would share their slots.
        recent = m.positions - m.starts[m.requests] >= counts[m.requests] - DRAFT_RING
        rows = m.state_rows[m.requests].long()
        self.ring_slots.copy_(torch.where(live & recent, rows * DRAFT_RING + m.positions % DRAFT_RING, -1))
        self.hidden.copy_(self.norm(self.gather(self.project(auxiliary))))
        for kv, norm, cache, ring in self.layers:
            kv(self.hidden)
            self.latent[:, :512].copy_(norm(self.kv[:, :512]))
            self.latent[:, 512:].copy_(self.kv[:, 512:])
            self.rotary(self.latent, m.positions)
            write_records(self.latent, self.ring_slots if ring else self.slots, cache)


@dataclass(frozen=True)
class DraftScratch:
    scratch: Scratch
    buffers: dict
    token_capacity: int

    @property
    def logits(self):
        return self.scratch.logits

    @property
    def hidden(self):
        return self.scratch.hidden

    def rows(self, tokens, samples):
        if not 0 < tokens <= self.token_capacity or samples != tokens:
            raise ValueError("DFlash2 scratch requires complete draft rows within its token capacity")
        return type(self)(self.scratch.rows(tokens, samples), self.buffers, self.token_capacity)

    def scratch_tensors(self):
        return self.scratch, self.buffers


class DraftAttention:
    """One draft layer: the block's queries attend the committed context's E4M3 records, a sliding layer's
    through its continuation ring, then their own block's BF16 latents, merged by log-sum-exp."""

    def __init__(self, weights, source, metadata, cache, config, window, *, session, name, workspace, buffers,
                 page_tokens=PAGE_TOKENS, token_capacity=None):
        self.weights, self.metadata = weights, metadata
        tokens, hidden = source.shape
        capacity = tokens if token_capacity is None else token_capacity
        if capacity < tokens:
            raise ValueError("DFlash2 attention scratch capacity must cover its draft rows")
        heads = weights["key"].shape[0]
        def buffer(key, *shape, dtype=torch.bfloat16):
            if key not in buffers:
                buffers[key] = torch.zeros((capacity, *shape), dtype=dtype, device=source.device)
            return buffers[key][:tokens]
        self.input, self.qnorm = buffer("input", 2112), buffer("qnorm", 1536)
        self.query, self.latent = buffer("query", heads, 192), buffer("latent", 576)
        self.absorbed, self.attended = buffer("absorbed", heads, 576), buffer("attended", heads, 512)
        self.lse = buffer("lse", heads, dtype=torch.float32)
        self.expanded, self.output = buffer("expanded", heads * 128), buffer("output", hidden)
        self.projections = {key: UnquantizedProjection(x, weights[key], y, session=session, name=name + "." + key)
            for key, x, y in (("input", source, self.input), ("q_b", self.qnorm, self.query.flatten(1)),
                             ("output", self.expanded, self.output))}
        self.query_norm = rms_norm(weights["query_norm"], config.text["rms_norm_eps"])
        self.kv_norm = rms_norm(weights["kv_norm"], config.text["rms_norm_eps"])
        self.rotary = DraftRotary(config, source.device)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.block = config.block_size
        # A ring's every logical page is its continuation row's single page of ring records.
        records, context = cache.shape[1], metadata.page_table.shape[1] * page_tokens
        width = -(-context // records)
        self.ring_table = (torch.zeros((metadata.lengths.numel(), width), dtype=torch.int32, device=source.device)
                           if window else None)
        # The context cache ends before the block. Right-aligned window bounds
        # therefore subtract the block width from the model's left window.
        caps = dense_mla.Caps(device=source.device, mode="decode", q_dtype=torch.bfloat16, kv_dtype=FP8,
            num_q_heads=heads, page_size=records, max_total_q=tokens, max_batch=metadata.lengths.numel(),
            max_cache_tokens=width * records, max_page_table_width=width, num_cache_pages=cache.shape[0],
            window_size=window - self.block if window else None, causal=False,
            use_cuda_graph=True, partial_dtype=torch.float32)
        self.attention = PreparedDenseMLA(self.absorbed, cache, self.attended,
            metadata.page_table if self.ring_table is None else self.ring_table, self.lengths,
            metadata.query_start_loc, caps=caps, scale=self.rotary.scale, session=session, name=name + ".context",
            workspace=workspace, lse=self.lse)

    def __call__(self, source):
        m, p = self.metadata, self.projections
        p["input"](source)
        self.qnorm.copy_(self.query_norm(self.input[:, :1536]))
        self.latent[:, :512].copy_(self.kv_norm(self.input[:, 1536:2048]))
        self.latent[:, 512:].copy_(self.input[:, 2048:])
        p["q_b"](self.qnorm)
        self.rotary(self.query, m.positions)
        self.rotary(self.latent, m.positions)
        self.absorbed[:, :, :512].copy_(torch.bmm(self.query[:, :, :128].transpose(0, 1), self.weights["key"]).transpose(0, 1))
        self.absorbed[:, :, 512:].copy_(self.query[:, :, 128:])
        if self.ring_table is not None:
            self.ring_table.copy_(m.state_rows[:, None].expand_as(self.ring_table))
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts, 0))
        self.attention()
        # Block K/V stays in scratch; proposals never overwrite committed context
        # or require pages beyond the runtime's selected verification width.
        rows, block, heads = m.lengths.numel(), self.block, self.query.shape[1]
        query = self.absorbed.view(rows, block, heads, 576).transpose(1, 2).float()
        latent = self.latent.view(rows, block, 576).float()
        scores = torch.matmul(query, latent[:, None].transpose(-1, -2)) * self.rotary.scale
        local_lse = scores.logsumexp(-1).transpose(1, 2).reshape_as(self.lse)
        local = torch.matmul(scores.softmax(-1), latent[:, None, :, :512]).transpose(1, 2).reshape_as(self.attended)
        combined = torch.logaddexp(self.lse, local_lse)
        context_weight = (self.lse - combined).exp()
        block_weight = (local_lse - combined).exp()
        self.attended.copy_(self.attended.float() * context_weight[:, :, None] + local * block_weight[:, :, None])
        self.expanded.copy_(torch.bmm(self.attended.transpose(0, 1), self.weights["value"]).transpose(0, 1).flatten(1))
        return p["output"](self.expanded)


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, embedding_runtime,
                 router_runtime, vision_state, coordinator, session, name, workspace, buffers=None,
                 page_tokens=PAGE_TOKENS):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c, s = weights.config.draft, scratch.scratch
        source, storage = s.activations, scratch.buffers
        tokens, hidden = source.shape
        capacity = scratch.token_capacity
        local = torch.zeros((tokens, hidden // weights.shard.world_size), dtype=source.dtype, device=source.device)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, local, source,
            runtime=embedding_runtime, coordinator=coordinator, num_rows=metadata.live_counts[:1],
            session=session, name=name + ".embedding")
        if "partial" not in storage:
            storage["partial"] = torch.zeros((capacity, hidden), dtype=source.dtype, device=source.device)
            storage["total"] = torch.zeros_like(storage["partial"])
            storage["mlp"] = MLPScratch.allocate(capacity, hidden, c.text["intermediate_size"] // weights.shard.world_size, source.device)
        self.layers = []
        for i, w in enumerate(weights.draft.layers):
            prefix = f"{name}.{i}"
            attention = DraftAttention(w["attention"], source, metadata, pool.draft_attention(i), c,
                c.text["sliding_window"] if c.text["layer_types"][i] == "sliding_attention" else 0,
                session=session, name=prefix + ".attention", workspace=workspace, buffers=storage.setdefault("attention", {}),
                page_tokens=page_tokens, token_capacity=capacity)
            conv = {kind: PreparedGroupedConvolution(source, w[kind], block_size=c.block_size, group_size=16,
                session=session, name=prefix + "." + kind, buffers=storage.setdefault(kind, {}), token_capacity=capacity)
                for kind in ("attention_conv", "mlp_conv")}
            mlp_s = storage["mlp"].rows(tokens)
            mlp = PreparedMLP(source, w["mlp"], mlp_s.gate_up, mlp_s.intermediate, mlp_s.output,
                session=session, name=prefix + ".mlp", workspace=workspace)
            reduce = PreparedAllReduce(storage["partial"][:tokens], storage["total"][:tokens], runtime=runtime,
                coordinator=coordinator, session=session, name=prefix + ".reduce")
            self.layers.append((rms_norm(w["input_norm"], c.text["rms_norm_eps"]), conv["attention_conv"], attention,
                rms_norm(w["post_norm"], c.text["rms_norm_eps"]), conv["mlp_conv"], mlp, reduce))
        self.norm = rms_norm(weights.draft.norm, c.text["rms_norm_eps"])
        self.head = PreparedVocabularyProjection(s.hidden, weights.head, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        self.selector = PreparedCandidateSelector(s.hidden.view(-1, c.block_size, hidden), weights.draft.selector,
            top_k=c.text["dflash_config"]["selector_top_k"], session=session, name=name + ".selector")

    def __call__(self):
        hidden = self.embedding.gather(self.embedding.lookup())
        for norm, conv, attention, post_norm, mlp_conv, mlp, reduce in self.layers:
            hidden = hidden + conv.finish(reduce(attention(conv.prepare(norm(hidden)))))
            hidden = hidden + mlp_conv.finish(reduce(mlp(mlp_conv.prepare(post_norm(hidden)))))
        self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        self.head(self.scratch.hidden)
        self.selector.prepare(self.scratch.hidden.view(-1, self.weights.config.draft.block_size, hidden.shape[-1]))


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object = PreparedDraft
    markov: object = None


def target_decoder(weights, tables, scratch, metadata, pool, **kwargs):
    return PreparedDecoder(weights, scratch, metadata, pool, **kwargs)


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, records, frame, sequence,
                 session, workspace, collectives, decoder_buffers, context_buffers, page_tokens=PAGE_TOKENS):
        self.page_tokens = page_tokens
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.records, self.frame, self.sequence = records, frame, sequence
        self.session, self.workspace = session, workspace
        # The target decoder takes every collective; DFlash2's draft modules take the shared reductions and gathers.
        target = {key: collectives[key] for key in ("route_runtime", "context", "latent_runtimes") if key in collectives}
        self.collectives = {key: value for key, value in collectives.items() if key not in target}
        self.markov_runtime = None
        self.context_buffers = context_buffers
        kda_layers = {i: w["attention"] for i, w in enumerate(weights.layers) if weights.config.layers[i] == "kda"}
        heads = weights.config.text["linear_attn_config"]["num_heads"] // weights.shard.world_size
        self.kda_records = KDARecords.allocate(kda_layers, requests.capacity + 1, heads, pool.backing.device)
        self.forwards = DraftForwards(partial(target_decoder, buffers=decoder_buffers, kda_records=self.kda_records,
                                              **target),
            partial(PreparedContext, runtime=collectives["embedding_runtime"], coordinator=collectives["coordinator"],
                    buffers=self.context_buffers),
            partial(PreparedDraft, page_tokens=page_tokens))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        draft_rows = requests.capacity * self.block_size
        self.draft_scratch = DraftScratch(Scratch.allocate(draft_rows,
            draft_rows, weights.config, weights.shard.world_size, pool.backing.device), {}, draft_rows)
