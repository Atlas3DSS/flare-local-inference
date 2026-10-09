"""b12x paged attention module.

Wraps QKV projection, QK norm, RoPE, KV cache write, and paged
attention into a single module that manages explicit decode/extend
workspaces for the current primary paged backend.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from b12x.integration.attention import PagedAttentionWorkspace, paged_attention_forward

from serve.model.ops import apply_partial_rope, rms_norm, write_kv_to_cache


class B12xPagedAttention(torch.nn.Module):
    """Paged attention module backed by b12x kernels.

    Handles the full attention path: QKV proj → QK norm → RoPE →
    KV cache write → paged attention → O proj.
    """

    def __init__(
        self,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rotary_dim: int,
        rms_norm_eps: float,
        qkv_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        tp_group=None,
        max_num_splits: int = 32,
        gemma_norm: bool = False,
        output_gate: bool = False,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.rotary_dim = rotary_dim
        self.rms_norm_eps = rms_norm_eps
        self.tp_group = tp_group
        self.max_num_splits = max_num_splits
        self.gemma_norm = gemma_norm
        self.output_gate = output_gate

        # Cache refs bound by bind_cache().
        self._k_cache = None
        self._v_cache = None

        self.register_buffer("qkv_weight", qkv_weight)
        self.register_buffer("o_proj_weight", o_proj_weight)
        self.register_buffer("q_norm_weight", q_norm_weight)
        self.register_buffer("k_norm_weight", k_norm_weight)

    def workspace_cache_key(
        self,
        *,
        kv_dtype: torch.dtype,
        page_size: int,
        num_cache_pages: int,
        max_total_q: int,
        max_batch: int,
        max_page_table_width: int,
        use_cuda_graph: bool,
    ) -> tuple[object, ...]:
        return (
            self.qkv_weight.dtype,
            kv_dtype,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            page_size,
            num_cache_pages,
            max_total_q,
            max_batch,
            max_page_table_width,
            use_cuda_graph,
        )

    def _supports_native_decode_workspace(
        self,
        *,
        kv_dtype: torch.dtype,
        page_size: int,
    ) -> bool:
        return (
            self.qkv_weight.dtype == torch.bfloat16
            and kv_dtype in (torch.bfloat16, torch.float8_e4m3fn)
            and self.head_dim == 256
            and page_size == 64
            and self.num_kv_heads > 0
            and self.num_q_heads % self.num_kv_heads == 0
            and (self.num_q_heads // self.num_kv_heads) == 8
        )

    def allocate_workspaces(
        self,
        *,
        device: torch.device | str,
        kv_dtype: torch.dtype,
        page_size: int,
        num_cache_pages: int,
        max_total_q: int,
        max_batch: int | None = None,
        max_page_table_width: int | None = None,
        use_cuda_graph: bool = False,
    ) -> dict[str, PagedAttentionWorkspace]:
        """Allocate decode/extend workspaces for this attention module."""
        common = dict(
            device=device,
            dtype=self.qkv_weight.dtype,
            kv_dtype=kv_dtype,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            page_size=page_size,
            max_total_q=max_total_q,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=use_cuda_graph,
        )
        if max_batch is None:
            max_batch = max_total_q
        if max_page_table_width is None:
            max_page_table_width = num_cache_pages
        extend_workspace = (
            PagedAttentionWorkspace.for_contract(mode="extend", **common)
            if use_cuda_graph
            else PagedAttentionWorkspace.for_eager_extend_capacity(
                mode="extend",
                max_batch=max_batch,
                max_page_table_width=max_page_table_width,
                **common,
            )
        )
        decode_workspace = (
            PagedAttentionWorkspace.for_contract(mode="decode", **common)
            if self._supports_native_decode_workspace(kv_dtype=kv_dtype, page_size=page_size)
            else extend_workspace
        )
        return {
            "decode": decode_workspace,
            "extend": extend_workspace,
        }

    def set_workspace(self, workspace: dict[str, PagedAttentionWorkspace]) -> None:
        """Bind decode/extend workspaces. Call once after layer construction."""
        self._workspace = workspace

    def set_output_buffer(self, output: torch.Tensor) -> None:
        """Set a pre-allocated output buffer for CUDA graph capture."""
        self._output_buffer = output

    @torch.compiler.disable
    def _kv_write_and_attend(
        self, q, k, v, k_cache, v_cache, page_table,
        cache_seqlens, cu_seqlens_q, post_write_seqlens,
        k_descale, v_descale,
        is_decode: bool,
        prepare_workspace: bool,
    ):
        """KV cache write + paged attention. Opaque to torch.compile."""
        write_kv_to_cache(
            k, v, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
        )
        output = getattr(self, '_output_buffer', None)
        if output is None:
            output = torch.empty(
                (q.shape[0], self.num_q_heads, self.head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        workspace = self._workspace["decode" if is_decode else "extend"]
        if prepare_workspace:
            workspace.prepare(page_table, post_write_seqlens, cu_seqlens_q)
        return workspace.run(
            q, k_cache, v_cache,
            k_descale=k_descale, v_descale=v_descale,
            output=output,
        )

    def bind_cache(self, *, k_cache=None, v_cache=None, **_kwargs):
        """Bind per-layer KV cache references."""
        self._k_cache = k_cache
        self._v_cache = v_cache

    def forward_from_state(self, hidden_states: torch.Tensor, state) -> torch.Tensor:
        """Forward using StepState. Called by TransformerLayer."""
        prepare_workspace = not state.attn_workspaces_prepared
        if prepare_workspace:
            prepared = state.prepared_attn_workspace_ids
            if prepared is None:
                prepared = set()
                state.prepared_attn_workspace_ids = prepared
            workspace = self._workspace["decode" if state.is_decode else "extend"]
            workspace_id = id(workspace)
            if workspace_id in prepared:
                prepare_workspace = False
            else:
                prepared.add(workspace_id)
        return self.forward(
            hidden_states, state.cos, state.sin, state.positions,
            self._k_cache, self._v_cache, state.page_table,
            state.cache_seqlens, state.cu_seqlens_q,
            output_gate=self.output_gate,
            is_decode=state.is_decode,
            prepare_workspace=prepare_workspace,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        output_gate: bool = False,
        is_decode: bool = False,
        prepare_workspace: bool = True,
    ) -> torch.Tensor:
        """Full attention: QKV → norm → RoPE → KV write → attention → O proj."""
        total_q = hidden_states.shape[0]
        q_size = self.num_q_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        # QKV projection.
        qkv = F.linear(hidden_states, self.qkv_weight)

        if output_gate:
            # QKV has [Q_real + Q_gate, K, V] — Q is doubled.
            q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
            q_gate = q_gate.view(total_q, self.num_q_heads, 2, self.head_dim)
            q = q_gate[:, :, 0, :].reshape(total_q, -1)
            gate = q_gate[:, :, 1, :].reshape(total_q, -1)
        else:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            gate = None

        # QK RMSNorm — applied before or after reshape depending on weight shape.
        if self.q_norm_weight.shape[0] == self.head_dim:
            # Per-head norm (Qwen3.5): reshape first, then norm.
            q = q.view(total_q, self.num_q_heads, self.head_dim)
            k = k.view(total_q, self.num_kv_heads, self.head_dim)
            v = v.view(total_q, self.num_kv_heads, self.head_dim)
            q = rms_norm(q, self.q_norm_weight, self.rms_norm_eps, gemma_style=self.gemma_norm)
            k = rms_norm(k, self.k_norm_weight, self.rms_norm_eps, gemma_style=self.gemma_norm)
        else:
            # Packed norm (MiniMax): norm on flat, then reshape.
            q = rms_norm(q, self.q_norm_weight, self.rms_norm_eps, gemma_style=self.gemma_norm)
            k = rms_norm(k, self.k_norm_weight, self.rms_norm_eps, gemma_style=self.gemma_norm)
            q = q.view(total_q, self.num_q_heads, self.head_dim)
            k = k.view(total_q, self.num_kv_heads, self.head_dim)
            v = v.view(total_q, self.num_kv_heads, self.head_dim)

        # Partial RoPE.
        pos_cos = cos[positions]
        pos_sin = sin[positions]
        q = q.contiguous()
        k = k.contiguous()
        q, k = apply_partial_rope(q, k, pos_cos, pos_sin, self.rotary_dim)

        # KV write + attention.
        post_write_seqlens = cache_seqlens + (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        attn_output, _lse = self._kv_write_and_attend(
            q, k, v, k_cache, v_cache, page_table,
            cache_seqlens, cu_seqlens_q, post_write_seqlens,
            k_descale, v_descale,
            is_decode,
            prepare_workspace,
        )

        # Apply output gate if present.
        attn_output = attn_output[:total_q].contiguous().view(total_q, -1)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        # O projection + TP allreduce.
        attn_output = F.linear(attn_output, self.o_proj_weight)
        if self.tp_group is not None:
            self.tp_group.allreduce_sum_(attn_output)

        return attn_output
