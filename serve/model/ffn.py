"""FFN blocks for transformer layers.

MoEFFN: routed sparse MoE with optional shared expert and TP allreduce.
Encapsulates routing, expert dispatch, and communication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """Sparse MoE FFN block.

    Routing and expert dispatch are handled by b12x_sparse_moe_fp4 internally
    when possible (passing gate_weight directly). Falls back to Python routing
    for models with non-standard routing (sigmoid, bias correction).
    """

    def __init__(
        self,
        *,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor | None = None,
        experts,                    # B12XFP4ExpertWeights.
        top_k: int,
        routing_fn: str = "softmax",   # "softmax" or "sigmoid".
        renormalize_topk: bool = True,
        tp_group=None,
        shared_expert: dict | None = None,
        shared_expert_gate_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.routing_fn = routing_fn
        self.renormalize_topk = renormalize_topk
        self.tp_group = tp_group
        self.experts = experts
        self.shared_expert = shared_expert

        self.register_buffer("gate_weight", gate_weight)
        if gate_bias is not None:
            self.register_buffer("gate_bias", gate_bias)
        else:
            self.gate_bias = None
        if shared_expert_gate_weight is not None:
            self.register_buffer("shared_expert_gate_weight", shared_expert_gate_weight)
        else:
            self.shared_expert_gate_weight = None

    def set_moe_workspace(self, workspace):
        self._moe_workspace = workspace

    def set_moe_output_buffer(self, buf):
        self._moe_output_buffer = buf

    def forward(self, hidden_states: torch.Tensor, state) -> torch.Tensor:
        if self.routing_fn == "sigmoid":
            routing = self._route(hidden_states)
            moe_out = self._run_moe_routed(hidden_states, routing)
        else:
            moe_out = self._run_moe_gated(hidden_states)

        # Shared expert (TP-sharded partial output, combined before allreduce).
        if self.shared_expert is not None:
            shared_out = self._run_shared_expert(hidden_states)
            if self.shared_expert_gate_weight is not None:
                gate = torch.sigmoid(F.linear(hidden_states, self.shared_expert_gate_weight))
                shared_out = shared_out * gate
            moe_out = moe_out + shared_out

        if self.tp_group is not None:
            self.tp_group.allreduce_sum_(moe_out)

        return moe_out

    def _route(self, hidden_states: torch.Tensor):
        """Compute top-k expert routing."""
        from b12x.integration.tp_moe import B12XTopKRouting
        router_logits = F.linear(hidden_states, self.gate_weight)

        if self.routing_fn == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
            if self.gate_bias is not None:
                scores_for_choice = scores + self.gate_bias.unsqueeze(0).float()
            else:
                scores_for_choice = scores
            topk_weights, topk_ids = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
            if self.gate_bias is not None:
                topk_weights = scores.gather(1, topk_ids)
            if self.renormalize_topk:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_logits, topk_ids = torch.topk(router_logits, k=self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_logits.float(), dim=-1)

        return B12XTopKRouting(
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            flat_ids=topk_ids.to(torch.int32).view(-1),
            flat_weights=topk_weights.view(-1),
        )

    @torch.compiler.disable
    def _run_moe_routed(self, hidden_states: torch.Tensor, routing) -> torch.Tensor:
        """MoE with pre-computed routing — opaque to torch.compile."""
        from b12x.integration.tp_moe import b12x_sparse_moe_fp4
        return b12x_sparse_moe_fp4(
            hidden_states, experts=self.experts,
            workspace=self._moe_workspace, routing=routing,
            renormalize_topk=False,
            output=getattr(self, '_moe_output_buffer', None),
        )

    @torch.compiler.disable
    def _run_moe_gated(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE with kernel-internal routing — opaque to torch.compile."""
        from b12x.integration.tp_moe import b12x_sparse_moe_fp4
        return b12x_sparse_moe_fp4(
            hidden_states, experts=self.experts,
            workspace=self._moe_workspace,
            top_k=self.top_k,
            gate_weight=self.gate_weight,
            gate_bias=self.gate_bias,
            renormalize_topk=self.renormalize_topk,
            output=getattr(self, '_moe_output_buffer', None),
            input_scales_static=True,
        )

    @torch.compiler.disable
    def _run_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the shared expert (dense BF16, TP-sharded).

        Uses fused gate_up matmul for numerical consistency with sglang's
        MergedColumnParallelLinear.
        """
        gate_up = F.linear(hidden_states, self.shared_expert["gate_up_proj"])
        mid = gate_up.shape[-1] // 2
        x = F.silu(gate_up[..., :mid]) * gate_up[..., mid:]
        return F.linear(x, self.shared_expert["down_proj"])
