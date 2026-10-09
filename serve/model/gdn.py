"""Gated Delta Network (GDN) linear attention module.

Wraps the FLA Triton kernels with the projections and conv1d state
management needed for Qwen3.5's linear attention layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from serve.model.ops import rms_norm


class GDNLinearAttention(nn.Module):
    """GDN linear attention for one layer.

    Performs: hidden → proj(Q,K,V,Z,A,B) → conv1d → GDN recurrent → norm+gate → out_proj.
    """

    def __init__(
        self,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        hidden_size: int,
        conv_kernel: int,
        # Weights loaded by recipe.
        in_proj_qkv_weight: torch.Tensor,   # [2*K_total + V_total, hidden].
        in_proj_z_weight: torch.Tensor,      # [V_total, hidden].
        in_proj_a_weight: torch.Tensor,      # [num_v_heads, hidden].
        in_proj_b_weight: torch.Tensor,      # [num_v_heads, hidden].
        conv1d_weight: torch.Tensor,         # [QKV_total, 1, kernel].
        out_proj_weight: torch.Tensor,       # [hidden, V_total].
        norm_weight: torch.Tensor,           # [head_v_dim].
        A_log: torch.Tensor,                 # [num_v_heads].
        dt_bias: torch.Tensor,              # [num_v_heads].
        rms_norm_eps: float = 1e-6,
        tp_group=None,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.hidden_size = hidden_size
        self.conv_kernel = conv_kernel
        self.tp_group = tp_group

        self.k_total = num_k_heads * head_k_dim
        self.v_total = num_v_heads * head_v_dim
        self.qkv_total = 2 * self.k_total + self.v_total

        self.register_buffer("in_proj_qkv_weight", in_proj_qkv_weight)
        self.register_buffer("in_proj_z_weight", in_proj_z_weight)
        self.register_buffer("in_proj_a_weight", in_proj_a_weight)
        self.register_buffer("in_proj_b_weight", in_proj_b_weight)
        self.register_buffer("conv1d_weight", conv1d_weight)
        self.register_buffer("out_proj_weight", out_proj_weight)
        self.register_buffer("norm_weight", norm_weight)
        self.register_buffer("A_log", A_log.float())
        self.register_buffer("dt_bias", dt_bias.float())
        self.rms_norm_eps = rms_norm_eps

        # Cache refs bound by bind_cache().
        self._ssm_state = None
        self._conv_state = None

    def bind_cache(self, *, ssm_state=None, conv_state=None, **_kwargs):
        """Bind per-layer SSM/conv cache references."""
        self._ssm_state = ssm_state
        self._conv_state = conv_state

    def forward_from_state(self, hidden_states: torch.Tensor, state) -> torch.Tensor:
        """Forward using StepState. Called by TransformerLayer."""
        mamba = state.mamba
        if state.is_decode:
            return self.forward_decode(
                hidden_states, self._ssm_state, self._conv_state,
                mamba.cache_indices,
            )
        else:
            return self.forward_extend(
                hidden_states, self._ssm_state, self._conv_state,
                mamba.cache_indices, mamba.cu_seqlens,
                mamba.has_initial_states,
            )

    @torch.compiler.disable
    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token decode step. Updates SSM and conv state in-place."""
        from serve.kernels.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from serve.kernels.mamba.causal_conv1d import causal_conv1d_update

        bs = hidden_states.shape[0]

        # Projections.
        mixed_qkv = F.linear(hidden_states, self.in_proj_qkv_weight)
        z = F.linear(hidden_states, self.in_proj_z_weight)
        a = F.linear(hidden_states, self.in_proj_a_weight)
        b = F.linear(hidden_states, self.in_proj_b_weight)

        # Conv1d update (single token).
        mixed_qkv = causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d_weight.squeeze(1),
            conv_state_indices=cache_indices.int(),
            activation="silu",
        )

        # GDN recurrent decode.
        out = mixed_qkv.new_empty(bs, 1, self.num_v_heads, self.head_v_dim)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a, b=b,
            A_log=self.A_log, dt_bias=self.dt_bias,
            scale=self.head_k_dim ** -0.5,
            initial_state=ssm_state,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # out is [bs, 1, HV, V] → [bs, HV, V].
        out = out.squeeze(1)

        # Gated output norm.
        z = z.view(bs, self.num_v_heads, self.head_v_dim)
        out = self._gated_rms_norm(out, z)

        # Output projection.
        out = out.reshape(bs, -1)
        out = F.linear(out, self.out_proj_weight)
        if self.tp_group is not None:
            self.tp_group.allreduce_sum_(out)

        return out

    @torch.compiler.disable
    def forward_extend(
        self,
        hidden_states: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        cache_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill/extend with variable-length sequences."""
        from serve.kernels.fla.chunk import chunk_gated_delta_rule
        from serve.kernels.mamba.causal_conv1d import causal_conv1d_fn

        total_q = hidden_states.shape[0]

        # Projections.
        mixed_qkv = F.linear(hidden_states, self.in_proj_qkv_weight)
        z = F.linear(hidden_states, self.in_proj_z_weight)
        a = F.linear(hidden_states, self.in_proj_a_weight)
        b = F.linear(hidden_states, self.in_proj_b_weight)

        # Causal conv1d over full sequences.
        bs = cu_seqlens.shape[0] - 1
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        has_initial_state = has_initial_states
        mixed_qkv_t = mixed_qkv.t().contiguous()
        mixed_qkv_t = causal_conv1d_fn(
            mixed_qkv_t, self.conv1d_weight.squeeze(1), bias=None,
            conv_states=conv_state,
            query_start_loc=cu_seqlens,
            seq_lens_cpu=seq_lens,
            cache_indices=cache_indices.int(),
            has_initial_state=has_initial_state,
            activation="silu",
        )
        mixed_qkv = mixed_qkv_t.t().contiguous()

        # Split into Q, K, V and reshape for kernel.
        q, k, v = mixed_qkv.split([self.k_total, self.k_total, self.v_total], dim=-1)
        q = q.view(1, total_q, self.num_k_heads, self.head_k_dim)
        k = k.view(1, total_q, self.num_k_heads, self.head_k_dim)
        v = v.view(1, total_q, self.num_v_heads, self.head_v_dim)

        # Compute gating values.
        from serve.kernels.fla.fused_gdn_gating import fused_gdn_gating
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        # Chunk GDN.
        out, _, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=ssm_state,
            cu_seqlens=cu_seqlens.long(),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            initial_state_indices=cache_indices,
        )
        out = out.squeeze(0)

        # Gated output norm.
        z = z.view(total_q, self.num_v_heads, self.head_v_dim)
        out = self._gated_rms_norm(out, z)

        # Output projection.
        out = out.reshape(total_q, -1)
        out = F.linear(out, self.out_proj_weight)
        if self.tp_group is not None:
            self.tp_group.allreduce_sum_(out)

        return out

    def _gated_rms_norm(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """RMSNorm with SiLU gating: norm(x) * silu(z)."""
        x = rms_norm(x, self.norm_weight, self.rms_norm_eps)
        return x * F.silu(z)
