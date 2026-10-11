"""GLM head-sharded latent attention with replicated C4 index and KV caches."""

from dataclasses import dataclass

import torch
from b12x.attention import sparse_mla
from flashinfer.triton.norm import rms_norm

from ignition.components.linear import BatchedQuantizedProjection, QuantizedProjection
from ignition.components.sparse_mla import PreparedLatentCacheWriter, PreparedSparseMLA

from .indexer import IndexMetadata, IndexScratch, IndexState, PreparedGlmIndexer


@dataclass(frozen=True)
class MLAState:
    cache: torch.Tensor
    index: IndexState

    @classmethod
    def allocate(cls, pages, slots, device, *, ring=4):
        index = IndexState.allocate(pages, slots, device, ring=ring)
        return cls(torch.zeros((pages * 4, 64, 528), dtype=torch.uint8, device=device), index)


@dataclass(frozen=True)
class MLAScratch:
    projected_input: torch.Tensor
    query_input: torch.Tensor
    query_residual: torch.Tensor
    latent_input: torch.Tensor
    latent: torch.Tensor
    query: torch.Tensor
    absorbed_query: torch.Tensor
    attended: torch.Tensor
    expanded: torch.Tensor
    output: torch.Tensor
    visible: torch.Tensor
    index: IndexScratch
    # Decode context parallel attention over one slice of rows: all heads' queries and partial outputs.
    gathered: torch.Tensor
    partial: torch.Tensor
    lse: torch.Tensor

    @classmethod
    def allocate(cls, tokens, heads, blocks, mode, device, *, context_ranks=1):
        if heads not in (16, 32):
            raise ValueError("GLM MLA scratch requires TP2 or TP4 head partitions")
        index = IndexScratch.allocate(tokens, blocks, mode, device, context_ranks=context_ranks)
        def buffer(*shape):
            return torch.zeros((tokens, *shape), dtype=torch.bfloat16, device=device)
        projected = buffer(2048)
        from .dcp import SLICE_ROWS

        rows = min(tokens, SLICE_ROWS) if context_ranks > 1 else 0
        return cls(projected, projected[:, :1536], buffer(1536), projected[:, 1536:], buffer(512),
            buffer(heads, 256), buffer(heads, 512), buffer(heads, 512), buffer(heads, 256),
            buffer(4096), torch.zeros(tokens, dtype=torch.int32, device=device), index,
            torch.zeros((rows, heads * context_ranks, 512), dtype=torch.bfloat16, device=device),
            torch.zeros((rows, heads * context_ranks, 512), dtype=torch.bfloat16, device=device),
            torch.zeros((rows, heads * context_ranks), dtype=torch.float32, device=device))

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("GLM MLA rows exceed scratch capacity")
        return MLAScratch(self.projected_input[:tokens], self.query_input[:tokens],
            self.query_residual[:tokens], self.latent_input[:tokens],
            self.latent[:tokens], self.query[:tokens], self.absorbed_query[:tokens], self.attended[:tokens],
            self.expanded[:tokens], self.output[:tokens], self.visible[:tokens], self.index.rows(tokens, mode),
            self.gathered[:tokens], self.partial[:tokens], self.lse[:tokens])


class PreparedGlmMLA:
    """One rank's sparse attention before the row-parallel output reduction.

    Each rank holds identical low-rank projections, latent records, and C4
    indexing state. Query and value expansion partition the 64 heads. Packed
    KV weights remain MXFP8 and supply strided key/value views to native BMMs.
    Physical model pages contain 256 tokens, or four 64-record latent pages.
    """

    def __init__(self, module, weights, source, scratch: MLAScratch, state: MLAState,
                 metadata: IndexMetadata, *, mode, session, name, workspace, max_cache_pages=None, context=None):
        rows, heads = scratch.query.shape[:2]
        if (source.shape != (rows, 4096) or source.dtype != torch.bfloat16
                or source.device != scratch.query.device):
            raise ValueError("GLM MLA requires BF16 [token_capacity, 4096] input on the scratch device")
        if (module.num_heads, module.q_lora_rank, module.kv_lora_rank, module.qk_nope_head_dim,
                module.qk_rope_head_dim, module.v_head_dim) != (64, 1536, 512, 256, 0, 256):
            raise ValueError("canonical MLA geometry differs from GLM 5.3 Flash")
        if module.indexer is None or module.skip_topk or heads not in (16, 32):
            raise ValueError("GLM MLA requires a full indexer and TP2 or TP4 head partitions")
        if (state.cache.shape != (state.index.cache.shape[0] * 4, 64, 528)
                or weights["q_b_proj"].out_features != heads * 256
                or weights["kv_b_proj"].out_features != heads * 512):
            raise ValueError("GLM MLA cache or head partition geometry mismatch")
        self.module, self.scratch, self.state, self.metadata = module, scratch, state, metadata
        for key in ("q_a_layernorm", "kv_a_layernorm"):
            module.get_submodule(key).weight = torch.nn.Parameter(weights[f"{key}.weight"], requires_grad=False)
        self.projections = {}
        for key, input, output in (
            ("qkv_a_proj", source, scratch.projected_input),
            ("q_b_proj", scratch.query_residual, scratch.query.view(rows, -1)),
            ("o_proj", scratch.expanded.view(rows, -1), scratch.output),
        ):
            self.projections[key] = QuantizedProjection(input, weights[key], output,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto")
        packed = weights["kv_b_proj"].weight
        values, scales = packed.values.view(heads, 512, 512), packed.scale_rows.view(heads, 512, 16)
        self.query_absorption = BatchedQuantizedProjection(scratch.query.transpose(0, 1),
            (values[:, :256], scales[:, :256]), scratch.absorbed_query.transpose(0, 1), major="n",
            session=session, name=f"{name}.absorb_query")
        self.value_expansion = BatchedQuantizedProjection(scratch.attended.transpose(0, 1),
            (values[:, 256:], scales[:, 256:]), scratch.expanded.transpose(0, 1), major="k",
            session=session, name=f"{name}.expand_value")
        self.indexer = PreparedGlmIndexer(module.indexer, weights, source, scratch.query_residual,
            scratch.index, state.index, metadata, mode=mode, session=session, name=f"{name}.indexer",
            workspace=workspace, max_cache_pages=max_cache_pages, context=context)
        self.writer = PreparedLatentCacheWriter(scratch.latent, state.cache, metadata.token_slots,
            session=session, name=f"{name}.cache_write")
        self.context = context
        ranks = 1 if context is None else context.shards.world
        slices = [(0, rows)] if context is None else context.slices(rows)
        caps = sparse_mla.Caps(device=source.device, num_q_heads=heads * ranks, max_q_rows=slices[0][1],
            max_width=2051, softmax_scale=module.scaling, max_batch=slices[0][1],
            max_kv_rows=max(state.cache.shape[0], 4 * (max_cache_pages or 0)) * 64,
            kv_dtype=torch.uint8, head_dim=512, v_head_dim=512, page_size=64,
            model_type=sparse_mla.ModelType.GLM_NEXT, cache_record_bytes=528,
            mode="decode" if mode == "decode" else "extend",
            **({} if context is None else {"return_lse": True, "lse_scale": "natural"}))
        if context is None:
            self.attention = PreparedSparseMLA(scratch.absorbed_query, state.cache, scratch.index.physical,
                scratch.visible, scratch.index.selected_lengths, scratch.attended, caps=caps,
                session=session, name=f"{name}.attention", workspace=workspace)
            return
        # Each slice gathers every rank's query heads, attends to local records, and keeps this rank's heads.
        from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter

        self.slices = []
        for start, end in slices:
            count = end - start
            gathered, partial, lse = scratch.gathered[:count], scratch.partial[:count], scratch.lse[:count]
            gather = PreparedAllGather(scratch.absorbed_query[start:end].view(count, -1), gathered.view(count, -1),
                runtime=context.heads, session=session, coordinator=context.coordinator,
                name=f"{name}.gather_heads{start}")
            attention = PreparedSparseMLA(gathered, state.cache, scratch.index.physical[start:end],
                scratch.visible[start:end], scratch.index.selected_lengths[start:end], partial, caps=caps,
                session=session, name=f"{name}.attention{start}", workspace=workspace, lse=lse)
            combine = PreparedLseReduceScatter(partial, lse, scratch.attended[start:end], runtime=context.heads,
                session=session, coordinator=context.coordinator, name=f"{name}.combine{start}")
            self.slices.append((start, end, gather, attention, combine))

    def rebind_pages(self):
        """Write and attend through the state's current page pools after they were replaced."""
        self.writer.cache = self.state.cache
        self.indexer.rebind_pages()
        for attention in ([self.attention] if self.context is None else [a for _, _, _, a, _ in self.slices]):
            attention.rebind_cache(self.state.cache)

    def __call__(self, source):
        s, p = self.scratch, self.projections
        p["qkv_a_proj"](source)
        rms_norm(s.query_input, self.module.q_a_layernorm.weight, s.query_residual,
            self.module.q_a_layernorm.variance_epsilon, round_before_weight=True)
        p["q_b_proj"](s.query_residual)
        self.query_absorption(s.query.transpose(0, 1))
        rms_norm(s.latent_input, self.module.kv_a_layernorm.weight, s.latent,
            self.module.kv_a_layernorm.variance_epsilon, round_before_weight=True)
        self.writer()
        self.indexer()
        if self.context is None:
            s.visible.copy_(self.metadata.positions + 1)
            self.attention()
        else:
            s.visible.copy_(self.context.shards.local_tokens(self.metadata.positions + 1))
            for start, end, gather, attention, combine in self.slices:
                gather(s.absorbed_query[start:end].view(end - start, -1))
                attention()
                combine(attention.output, attention.lse)
        self.value_expansion(s.attended.transpose(0, 1))
        return p["o_proj"](s.expanded.view(s.output.shape[0], -1))
