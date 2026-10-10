"""GLM head-sharded latent attention with replicated C4 index and KV caches."""

from dataclasses import dataclass

import torch
from b12x.attention import sparse_mla

from flare.components.linear import BatchedQuantizedProjection, QuantizedProjection
from flare.components.sparse_mla import PreparedLatentCacheWriter, PreparedSparseMLA

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

    @classmethod
    def allocate(cls, tokens, heads, blocks, mode, device):
        if heads not in (16, 32):
            raise ValueError("GLM MLA scratch requires TP2 or TP4 head partitions")
        index = IndexScratch.allocate(tokens, blocks, mode, device)
        def buffer(*shape):
            return torch.zeros((tokens, *shape), dtype=torch.bfloat16, device=device)
        return cls(buffer(1536), buffer(1536), buffer(512), buffer(512),
            buffer(heads, 256), buffer(heads, 512), buffer(heads, 512), buffer(heads, 256),
            buffer(4096), torch.zeros(tokens, dtype=torch.int32, device=device), index)

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("GLM MLA rows exceed scratch capacity")
        return MLAScratch(self.query_input[:tokens], self.query_residual[:tokens], self.latent_input[:tokens],
            self.latent[:tokens], self.query[:tokens], self.absorbed_query[:tokens], self.attended[:tokens],
            self.expanded[:tokens], self.output[:tokens], self.visible[:tokens], self.index.rows(tokens, mode))


class PreparedGlmMLA:
    """One rank's sparse attention before the row-parallel output reduction.

    Each rank holds identical low-rank projections, latent records, and C4
    indexing state. Query and value expansion partition the 64 heads. Packed
    KV weights remain MXFP8 and supply strided key/value views to native BMMs.
    Physical model pages contain 256 tokens, or four 64-record latent pages.
    """

    def __init__(self, module, weights, source, scratch: MLAScratch, state: MLAState,
                 metadata: IndexMetadata, *, mode, session, name, workspace):
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
            ("q_a_proj", source, scratch.query_input),
            ("q_b_proj", scratch.query_residual, scratch.query.view(rows, -1)),
            ("kv_a_proj_with_mqa", source, scratch.latent_input),
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
            scratch.index, state.index, metadata, mode=mode, session=session, name=f"{name}.indexer", workspace=workspace)
        self.writer = PreparedLatentCacheWriter(scratch.latent, state.cache, metadata.token_slots,
            session=session, name=f"{name}.cache_write")
        caps = sparse_mla.Caps(device=source.device, num_q_heads=heads, max_q_rows=rows, max_width=2051,
            softmax_scale=module.scaling, max_batch=rows, max_kv_rows=state.cache.shape[0] * 64,
            kv_dtype=torch.uint8, head_dim=512, v_head_dim=512, page_size=64,
            model_type=sparse_mla.ModelType.GLM_NEXT, cache_record_bytes=528,
            mode="decode" if mode == "decode" else "extend")
        self.attention = PreparedSparseMLA(scratch.absorbed_query, state.cache, scratch.index.physical,
            scratch.visible, scratch.index.selected_lengths, scratch.attended, caps=caps,
            session=session, name=f"{name}.attention", workspace=workspace)

    def __call__(self, source):
        s, p = self.scratch, self.projections
        s.query_residual.copy_(self.module.q_a_layernorm(p["q_a_proj"](source)))
        p["q_b_proj"](s.query_residual)
        self.query_absorption(s.query.transpose(0, 1))
        s.latent.copy_(self.module.kv_a_layernorm(p["kv_a_proj_with_mqa"](source)))
        self.writer()
        self.indexer()
        s.visible.copy_(self.metadata.positions + 1)
        self.attention()
        self.value_expansion(s.attended.transpose(0, 1))
        return p["o_proj"](s.expanded.view(s.output.shape[0], -1))
