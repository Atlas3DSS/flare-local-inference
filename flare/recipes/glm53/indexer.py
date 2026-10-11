"""GLM C4 index projections, sequence pooling and sparse physical selection."""

from dataclasses import dataclass

import torch
from b12x.attention import dsa_indexer, sparse_mla
from b12x.attention.dsa_indexer import glm_pool
from b12x.preparation import PreparedCall

from flare.components.indexer import PreparedSparseIndexer
from flare.components.linear import QuantizedProjection, UnquantizedProjection
from flare.preparation.tuning import declare


@dataclass(frozen=True)
class IndexState:
    cache: torch.Tensor
    tail: torch.Tensor

    @classmethod
    def allocate(cls, pages, slots, device, *, ring=4):
        if pages <= 0 or slots <= 0 or ring < 4 or ring % 4:
            raise ValueError("GLM index state requires positive pages/slots and a ring divisible by four")
        return cls(torch.zeros((pages, 64, 132), dtype=torch.uint8, device=device),
                   torch.zeros((slots, 2, ring, 128), dtype=torch.bfloat16, device=device))


@dataclass(frozen=True)
class IndexMetadata:
    positions: torch.Tensor
    token_slots: torch.Tensor
    requests: torch.Tensor
    state_slots: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor

    @classmethod
    def allocate(cls, tokens, sequences, blocks, device):
        if min(tokens, sequences, blocks) <= 0:
            raise ValueError("GLM index metadata requires positive capacities")
        return cls(torch.full((tokens,), -1, dtype=torch.int64, device=device),
            torch.full((tokens,), -1, dtype=torch.int64, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            torch.zeros(sequences + 1, dtype=torch.int32, device=device),
            torch.full((sequences, blocks), -1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class IndexScratch:
    query: torch.Tensor
    query_fp8: torch.Tensor
    query_scales: torch.Tensor
    key: torch.Tensor
    normalized_key: torch.Tensor
    gate: torch.Tensor
    projected_weights: torch.Tensor
    weights: torch.Tensor
    pool_lengths: torch.Tensor
    active_width: torch.Tensor
    pools: torch.Tensor
    physical: torch.Tensor
    selected_lengths: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, mode, device):
        if tokens <= 0 or blocks <= 0 or mode not in ("decode", "prefill"):
            raise ValueError("GLM index scratch requires positive capacities and decode/prefill mode")
        def tensor(width, dtype=torch.bfloat16):
            return torch.zeros((tokens, width), dtype=dtype, device=device)
        return cls(tensor(4096), tensor(4096, torch.float8_e4m3fn), tensor(32, torch.float32),
            tensor(128), tensor(128), tensor(128), tensor(32), tensor(32, torch.float32),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.tensor([blocks * 64], dtype=torch.int32, device=device), tensor(512, torch.int32),
            tensor(2051, torch.int32), torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.zeros((tokens if mode == "decode" else 1, blocks), dtype=torch.int32, device=device))

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.query.shape[0] or mode not in ("decode", "prefill"):
            raise ValueError("GLM index rows require valid capacity and decode/prefill mode")
        sequences = tokens if mode == "decode" else 1
        if sequences > self.page_table.shape[0]:
            raise ValueError("GLM index request rows exceed page-table capacity")
        return IndexScratch(self.query[:tokens], self.query_fp8[:tokens], self.query_scales[:tokens],
            self.key[:tokens], self.normalized_key[:tokens], self.gate[:tokens],
            self.projected_weights[:tokens], self.weights[:tokens], self.pool_lengths[:tokens],
            self.active_width, self.pools[:tokens], self.physical[:tokens], self.selected_lengths[:tokens],
            self.page_table[:sequences])


class PreparedGlmIndexer:
    """Replicated C4 selection for GLM sparse MLA with 256-token model pages.

    Position -1 masks an inactive query; negative state slots mask inactive
    requests. Token slots and page tables address the same physical model pages.
    Prefill admits one request, while decode supports packed per-request rows.
    Tails and cache outlive graph variants; intermediates can be shared by serial
    layers. Returned physical indices address a latent cache of 64-record pages.
    """

    def __init__(self, module, weights, source, query_residual, scratch: IndexScratch,
                 state: IndexState, metadata: IndexMetadata, *, mode, session, name, workspace):
        rows = source.shape[0]
        if (source.shape != (rows, 4096) or query_residual.shape != (rows, 1536)
                or source.dtype != torch.bfloat16 or query_residual.dtype != source.dtype
                or source.device != query_residual.device):
            raise ValueError("GLM index inputs require BF16 hidden[rows,4096] and query[rows,1536]")
        if (module.n_heads, module.head_dim, module.index_kpool, module.index_topk) != (32, 128, 4, 2048):
            raise ValueError("GLM indexing requires 32 heads, 128 channels, C4 pools and 2048 history tokens")
        if mode not in ("decode", "prefill") or (mode == "prefill" and metadata.state_slots.numel() != 1):
            raise ValueError("GLM index prefill requires one request")
        if (scratch.query.shape != (rows, 4096) or metadata.positions.shape != (rows,)
                or metadata.block_table.shape[1] != scratch.page_table.shape[1]
                or scratch.page_table.shape[0] != (1 if mode == "prefill" else rows)):
            raise ValueError("GLM index scratch and metadata capacities differ")
        self.module, self.scratch, self.state, self.metadata, self.mode = module, scratch, state, metadata, mode
        self.source, self.query_residual = source, query_residual
        module.k_norm.weight = torch.nn.Parameter(weights["indexer.k_norm.weight"], requires_grad=False)
        module.k_norm.bias = torch.nn.Parameter(weights["indexer.k_norm.bias"], requires_grad=False)
        self.ape = weights["indexer.index_kpool_compress_ape"]
        self.query_projection = QuantizedProjection(query_residual, weights["indexer.wq_b"], scratch.query,
            session=session, name=f"{name}.query", workspace=workspace, activation_mode="auto")
        self.key_projection = QuantizedProjection(source, weights["indexer.wk"], scratch.key,
            session=session, name=f"{name}.key", workspace=workspace, activation_mode="auto")
        self.weight_projection = QuantizedProjection(source, weights["indexer.weights_proj"], scratch.projected_weights,
            session=session, name=f"{name}.weights", workspace=workspace, activation_mode="auto")
        self.gate_projection = UnquantizedProjection(source, weights["indexer.index_kpool_compress_gate"], scratch.gate,
            session=session, name=f"{name}.gate")
        self.query_arguments = (scratch.query.view(-1, 128), scratch.query_fp8.view(-1, 128),
                                scratch.query_scales.view(-1), scratch.weights.view(-1))
        self.query_plan = glm_pool.plan_queries(*self.query_arguments)
        self.pool_arguments = (state.cache, state.tail, metadata.state_slots, metadata.query_start_loc,
            scratch.normalized_key, scratch.gate, self.ape, metadata.token_slots, metadata.positions)
        self.pool_plan = glm_pool.plan_pools(*self.pool_arguments, mode=mode)
        declare(session, (
            self.query_plan.request(name=f"{name}.rotate", prepare_call=lambda state:
                PreparedCall(run=lambda: state.quantize(*self.query_arguments))),
            self.pool_plan.request(name=f"{name}.pool", prepare_call=lambda state:
                PreparedCall(run=lambda: state.update(*self.pool_arguments))),
        ))
        page_table = scratch.page_table.expand(rows, -1) if mode == "prefill" else scratch.page_table
        caps = dsa_indexer.Caps(device=source.device, num_q_heads=32, max_q_rows=rows,
            max_page_table_width=page_table.shape[1], topk=512, mode=mode)
        self.selector = PreparedSparseIndexer(scratch.query_fp8.view(rows, 32, 128), scratch.weights,
            state.cache.view(state.cache.shape[0], -1), page_table, scratch.pool_lengths,
            scratch.active_width, scratch.pools, caps=caps, session=session, name=f"{name}.select", workspace=workspace)
        self.expand_arguments = (scratch.pools, metadata.positions, metadata.requests,
                                 metadata.block_table, scratch.physical, scratch.selected_lengths)
        self.expand_options = {"pool_size": 4, "block_size": 256, "block_stride_rows": 256,
                               "num_cache_blocks": state.cache.shape[0]}
        self.expand_plan = sparse_mla.plan_pooled_selection(device=source.device, max_rows=rows,
            page_size=256, max_page_table_width=metadata.block_table.shape[1], num_cache_blocks=state.cache.shape[0])
        declare(session, (self.expand_plan.request(name=f"{name}.expand", prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.expand_arguments, **self.expand_options))),))

    def __call__(self):
        s, m = self.scratch, self.metadata
        self.query_projection(self.query_residual)
        s.normalized_key.copy_(self.module.k_norm(self.key_projection(self.source)))
        self.gate_projection(self.source)
        s.weights.copy_(self.weight_projection(self.source))
        glm_pool.quantize_queries(*self.query_arguments, plan=self.query_plan)
        glm_pool.update_pools(*self.pool_arguments, plan=self.pool_plan)
        s.pool_lengths.copy_((m.positions + 1) // 4)
        if self.mode == "prefill":
            s.page_table.copy_(m.block_table)
        else:
            s.page_table.copy_(m.block_table[m.requests.long()])
        self.selector()
        sparse_mla.expand_pooled_topk_to_physical_slots(*self.expand_arguments,
            **self.expand_options, plan=self.expand_plan)
        return s.physical
