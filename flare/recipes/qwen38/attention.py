"""Prepared Qwen sparse attention with paged KV and selector state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from b12x.attention import paged_kv_write, qsa
from b12x.preparation import PreparationSession, PreparedCall
from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

from flare.components.linear import QuantizedProjection, UnquantizedProjection
from flare.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from flare.preparation.tuning import declare, register_finalizer

from .attention_overlap import ConcurrentAttention


@dataclass(frozen=True)
class RotaryTables:
    cos: torch.Tensor
    sin: torch.Tensor
    axes: torch.Tensor

    @classmethod
    def allocate(cls, config, capacity: int, device: torch.device) -> RotaryTables:
        if capacity <= 0:
            raise ValueError("rotary position capacity must be positive")
        rope = config.rope_parameters
        if (config.head_dim != 256 or rope["partial_rotary_factor"] != .25
                or tuple(rope["mrope_section"]) != (11, 11, 10)
                or rope["rope_theta"] != 10000000):
            raise ValueError("Qwen rotary geometry differs from the checkpoint contract")
        inverse = (1.0 / (rope["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).to(device)
        frequency = torch.arange(capacity, device=device, dtype=torch.float32)[:, None] * inverse
        axes = torch.zeros(32, device=device, dtype=torch.int64)
        axes[1:33:3] = 1
        axes[2:30:3] = 2
        return cls(frequency.cos(), frequency.sin(), axes)

    def gather(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Negative metadata marks inactive rows; their arithmetic uses position zero.
        rows = positions.clamp_min(0).index_select(1, self.axes)
        columns = torch.arange(32, device=positions.device)
        cos, sin = self.cos[rows, columns], self.sin[rows, columns]
        return torch.cat((cos, cos), -1).bfloat16(), torch.cat((sin, sin), -1).bfloat16()


@dataclass(frozen=True)
class QSAState:
    main_k_cache: torch.Tensor
    main_v_cache: torch.Tensor
    compressed_k_cache: torch.Tensor
    raw_k_ring: torch.Tensor
    raw_logical_positions: torch.Tensor
    raw_rope_positions: torch.Tensor
    raw_interval_start_positions: torch.Tensor

    @classmethod
    def allocate(cls, *, pages: int, slots: int, device: torch.device,
                 page_size: int = 64, draft_tokens: int = 0) -> QSAState:
        if pages <= 0 or slots <= 0:
            raise ValueError("QSA page and state capacities must be positive")
        geometry = qsa.cache_requirements(main_page_size=page_size, position_axes=3,
                                         max_speculative_tokens=draft_tokens)
        def zeros(shape):
            return torch.zeros(shape, device=device, dtype=torch.bfloat16)
        def invalid(shape):
            return torch.full(shape, -1, device=device, dtype=torch.int64)
        return cls(zeros((pages, *geometry.main_k_page_shape)),
                   zeros((pages, *geometry.main_v_page_shape)),
                   zeros((pages, *geometry.compressed_page_shape)),
                   zeros((slots, *geometry.raw_k_ring_shape)),
                   invalid((slots, *geometry.raw_logical_positions_shape)),
                   invalid((slots, *geometry.raw_rope_positions_shape)), invalid((slots,)))

    def resize_pages_for_preparation(self, pages: int) -> QSAState:
        """Allocate final page pools while retaining fixed raw selector slots.

        No graph referencing the returned page pools may have been handed to
        native execution. Rebind each prepared attention operator and recapture
        its graphs before use. Page contents are empty after this operation.
        """
        if type(pages) is not int or pages <= 0:
            raise ValueError("QSA page capacity must be positive")
        def pages_like(value):
            return torch.zeros((pages, *value.shape[1:]), dtype=value.dtype, device=value.device)
        return type(self)(pages_like(self.main_k_cache), pages_like(self.main_v_cache),
            pages_like(self.compressed_k_cache), self.raw_k_ring, self.raw_logical_positions,
            self.raw_rope_positions, self.raw_interval_start_positions)


@dataclass(frozen=True)
class QSAMetadata:
    main_block_table: torch.Tensor
    compressed_block_table: torch.Tensor
    raw_state_slot_ids: torch.Tensor
    request_ids: torch.Tensor
    query_positions: torch.Tensor
    rope_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    is_prefilling: torch.Tensor
    slot_mapping: torch.Tensor

    @classmethod
    def allocate(cls, *, tokens: int, batch: int, context: int,
                 device: torch.device, page_size: int = 64) -> QSAMetadata:
        if tokens < batch or min(batch, context) <= 0 or context % page_size:
            raise ValueError("QSA requires positive, page-aligned capacity and tokens >= batch")
        def full(shape, value=-1, dtype=torch.int64):
            return torch.full(shape, value, dtype=dtype, device=device)
        return cls(full((batch, context // page_size), dtype=torch.int32),
                   full((batch, context // page_size), dtype=torch.int32), full((batch,)),
                   full((tokens,)), full((tokens,)), full((tokens, 3)),
                   full((batch,), 0, torch.int32), full((batch + 1,), 0, torch.int32),
                   full((batch,), 0, torch.int32), full((batch,), False, torch.bool), full((tokens,)))


@dataclass(frozen=True)
class QSAScratch:
    projected_q: torch.Tensor
    projected_k: torch.Tensor
    projected_v: torch.Tensor
    projected_index: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    attended: torch.Tensor
    selected_positions: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor
    projected_qkv: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device) -> QSAScratch:
        if tokens <= 0:
            raise ValueError("QSA token capacity must be positive")
        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=torch.bfloat16)
        qkv = zeros(tokens, 13312)
        return cls(qkv[:, :12288], qkv[:, 12288:12800], qkv[:, 12800:],
                   zeros(tokens, 640), zeros(tokens, 24, 256), zeros(tokens, 2, 256),
                   zeros(tokens, 24, 256),
                   torch.full((tokens, 2051), -1, device=device, dtype=torch.int32),
                   zeros(tokens, 6144), zeros(tokens, 2560), qkv)


class PreparedAttention:
    """All Python tensor composition executes during warmup and graph capture.

    Metadata begins inactive during construction. The scheduler supplies dense
    request intervals, page mappings, and accepted-token counts before replay.
    It initializes each raw interval anchor to consumed length minus accepted
    count on restore and invalidates raw tags at aligned checkpoints.
    """

    def __init__(self, module: torch.nn.Module, weights: Mapping, source: torch.Tensor,
                 scratch: QSAScratch, state: QSAState, metadata: QSAMetadata,
                 rotary: RotaryTables, *, session: PreparationSession, name: str,
                 draft_tokens: int = 0, workspace: ScratchArena | torch.Tensor | None = None,
                 projection_workspace: ScratchArena | torch.Tensor | None = None,
                 max_cache_pages: int | None = None, cache_only: bool = False) -> None:
        self.cache_only = cache_only
        self.module, self.scratch, self.state = module, scratch, state
        self.metadata, self.rotary = metadata, rotary
        self.tokens = source.shape[0]
        if source.shape != scratch.output.shape or source.dtype != torch.bfloat16:
            raise ValueError("QSA source must match BF16 [token_capacity, 2560] output")
        if bool((metadata.request_ids != -1).any().item()) or bool((metadata.slot_mapping != -1).any().item()):
            raise ValueError("QSA preparation requires inactive request and cache-write rows")
        batch, table_width = metadata.main_block_table.shape
        page_size = state.main_k_cache.shape[1]
        context = table_width * page_size
        cache_pages = state.main_k_cache.shape[0] if max_cache_pages is None else max_cache_pages
        if type(cache_pages) is not int or cache_pages < state.main_k_cache.shape[0]:
            raise ValueError("planned QSA page capacity must cover the bound page pools")
        self.cache_page_capacity = cache_pages
        caps = qsa.Caps(device=source.device, max_batch=batch,
                        max_raw_state_slots=state.raw_k_ring.shape[0], max_q_rows=self.tokens,
                        max_seq_len=context, num_main_cache_pages=cache_pages,
                        num_compressed_cache_pages=cache_pages,
                        main_page_size=page_size, compressed_page_size=page_size // 4,
                        max_speculative_tokens=draft_tokens, position_axes=3,
                        mrope_sections=(11, 11, 10), mrope_interleaved=True,
                        rms_norm_eps=module.config.rms_norm_eps,
                        attention_splits=16)
        if rotary.cos.shape[0] < context:
            raise ValueError("rotary table capacity must cover the admitted context")
        self.projections = {}
        def projection(source, weight, output, key):
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(source, weight, output, session=session, name=f"{name}.{key}")
            return QuantizedProjection(source, weight, output, session=session,
                                       name=f"{name}.{key}", workspace=projection_workspace)

        for key, output in (("qkv_proj", scratch.projected_qkv),
                            ("indexer.index_qk_proj", scratch.projected_index)):
            self.projections[key] = projection(source, weights[key], output, key)
        if not cache_only:
            self.projections["o_proj"] = projection(scratch.gated, weights["o_proj"], scratch.output, "o_proj")
        self.index_query = scratch.projected_index[:, :512].view(self.tokens, 4, 128)
        self.index_key = scratch.projected_index[:, 512:]
        self.binding_kwargs = {
            **{field: getattr(state, field) for field in state.__dataclass_fields__},
            **{field: getattr(metadata, field) for field in
               ("main_block_table", "compressed_block_table", "raw_state_slot_ids")},
            "index_q_norm_weight": module.indexer.q_layernorm.weight.detach().float(),
            "index_k_norm_weight": module.indexer.k_layernorm.weight.detach().float(),
            "rope_cos": rotary.cos, "rope_sin": rotary.sin,
        }
        operands = {**self.binding_kwargs, "request_ids": metadata.request_ids,
                    "rope_positions": metadata.rope_positions, "index_query": self.index_query,
                    "raw_index_key": self.index_key}
        self.plan = qsa.plan(caps, invocation=qsa.invocation_from_tensors(caps, **operands))
        self.binding_kwargs.update(output=scratch.attended, selected_positions=scratch.selected_positions)
        self.workspace = workspace

        def prepare(materialized):
            storage = planned_scratch(materialized.scratch_specs(), workspace)
            binding = materialized.bind_for_preparation(scratch=storage, **self.binding_kwargs)
            self.workspace = storage
            return PreparedCall(run=lambda: materialized.run_for_preparation(binding, **self.dynamic()),
                                owners=prime_owners(workspace, storage, binding))

        declare(session, (self.plan.request(name=f"{name}.qsa", prepare_call=prepare),), stage="state")
        def refresh():
            self.workspace = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = qsa.bind(self.plan, scratch=self.workspace, **self.binding_kwargs)

        register_finalizer(session, refresh, workspace=workspace)
        self.writer = paged_kv_write.plan(paged_kv_write.Caps(device=source.device,
            num_kv_heads=2, head_dim_k=256, head_dim_v=256, page_size=page_size))
        self.write_kwargs = {"key": scratch.key, "value": scratch.projected_v.view(self.tokens, 2, 256),
                             "k_cache": state.main_k_cache, "v_cache": state.main_v_cache,
                             "slot_mapping": metadata.slot_mapping}
        declare(session, (self.writer.request(name=f"{name}.kv_write", prepare_call=lambda materialized:
            PreparedCall(run=lambda: materialized.run(**self.write_kwargs))),), stage="state")
        self.concurrent = None
        if (not cache_only and self.tokens <= 16 and source.is_cuda and isinstance(workspace, ScratchArena)
                and workspace is projection_workspace
                and all(isinstance(self.projections[key], QuantizedProjection)
                        for key in ("qkv_proj", "indexer.index_qk_proj"))):
            self.concurrent = ConcurrentAttention(self, session=session, workspace=workspace)

    def dynamic(self) -> dict:
        return {"query": self.scratch.query, "index_query": self.index_query,
                "raw_index_key": self.index_key,
                **{field: getattr(self.metadata, field) for field in
                   ("request_ids", "query_positions", "rope_positions", "sequence_lengths",
                    "query_start_loc", "num_accepted_tokens", "is_prefilling")}}

    def rebind_pages_for_preparation(self, state: QSAState) -> None:
        """Replace only cache page pools before final CUDA graph capture."""
        if state.main_k_cache.shape[0] > self.cache_page_capacity:
            raise ValueError("replacement cache exceeds prepared QSA page capacity")
        for name in ("raw_k_ring", "raw_logical_positions", "raw_rope_positions", "raw_interval_start_positions"):
            if getattr(state, name).data_ptr() != getattr(self.state, name).data_ptr():
                raise ValueError("page rebinding must preserve raw selector state storage")
        for name in ("main_k_cache", "main_v_cache", "compressed_k_cache"):
            before, after = getattr(self.state, name), getattr(state, name)
            if (before.shape[1:] != after.shape[1:] or before.stride() != after.stride()
                    or before.dtype != after.dtype or before.device != after.device):
                raise ValueError("page rebinding must preserve the prepared cache ABI")
            self.binding_kwargs[name] = after
        self.binding = qsa.bind(self.plan, scratch=self.workspace, **self.binding_kwargs)
        self.write_kwargs.update(k_cache=state.main_k_cache, v_cache=state.main_v_cache)
        self.state = state

    def __call__(self, source: torch.Tensor) -> torch.Tensor | None:
        s = self.scratch
        if self.concurrent is None:
            self.projections["indexer.index_qk_proj"](source)
            self.projections["qkv_proj"](source)
        else:
            self.concurrent.project(source)
        projected = s.projected_q.view(self.tokens, 24, 512)
        query, gate = projected[..., :256], projected[..., 256:]
        key = s.projected_k.view(self.tokens, 2, 256)
        cos, sin = self.rotary.gather(self.metadata.rope_positions)
        query, key = apply_rotary_pos_emb(self.module.q_norm(query), self.module.k_norm(key), cos, sin,
                                         unsqueeze_dim=1)
        s.query.copy_(query)
        s.key.copy_(key)
        paged_kv_write.run(self.writer, **self.write_kwargs)
        if self.cache_only:
            qsa.run(self.binding, **self.dynamic(), cache_only=True)
            return None
        attention = qsa.run(self.binding, **self.dynamic()) if self.concurrent is None else self.concurrent.attend(source)
        s.gated.copy_(attention.flatten(1) * torch.sigmoid(gate.reshape(self.tokens, 6144)))
        output = self.projections["o_proj"](s.gated)
        output.masked_fill_((self.metadata.request_ids < 0)[:, None], 0)
        return output
