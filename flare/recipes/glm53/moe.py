"""GLM sigmoid routing, NVFP4 experts, and the MXFP8 shared expert."""

from dataclasses import dataclass

import torch

from flare.components.linear import UnquantizedProjection
from flare.components.mlp import MLPScratch, PreparedMLP
from flare.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch


@dataclass(frozen=True)
class MoeScratch:
    router_logits: torch.Tensor
    router: RouterScratch
    shared: MLPScratch
    routed: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world_size, device):
        if world_size not in (2, 4):
            raise ValueError("GLM experts require TP2 or TP4")
        return cls(torch.zeros((tokens, 288), dtype=torch.float32, device=device),
                   RouterScratch.allocate(tokens, device, experts=288, top_k=8),
                   MLPScratch.allocate(tokens, 4096, 2048 // world_size, device),
                   torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=device),
                   torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=device))


class PreparedGlmMoe:
    """One rank's intermediate-sharded expert contribution before all-reduce."""

    def __init__(self, module, source, weights, scratch: MoeScratch, *, session, name, workspace):
        gate = module.gate
        if (gate.num_group != 1 or gate.topk_group != 1 or gate.top_k != 8 or gate.num_experts != 288
                or not gate.norm_topk_prob or gate.routed_scaling_factor != 2.5):
            raise ValueError("GLM routing requires one group, sigmoid top-8 of 288, normalization and scale 2.5")
        if (weights.experts.num_experts != 288 or weights.experts.hidden_size != 4096
                or weights.experts.intermediate_size not in (512, 1024)):
            raise ValueError("GLM MoE requires all 288 experts with a TP2/TP4 intermediate partition")
        if source.dtype != torch.bfloat16 or source.shape != scratch.output.shape or source.shape[1] != 4096:
            raise ValueError("GLM MoE input must match BF16 [tokens,4096] scratch")
        self.source, self.scratch, self.output = source, scratch, scratch.output
        gate.weight = torch.nn.Parameter(weights.router, requires_grad=False)
        gate.e_score_correction_bias = weights.correction
        self.router_projection = UnquantizedProjection(source, weights.router, scratch.router_logits,
            session=session, name=f"{name}.router")
        self.routing = PreparedRouter(scratch.router_logits, scratch.router, session=session,
            name=f"{name}.route_topk", score_func="sigmoid", renormalize=True,
            routed_scaling_factor=2.5, correction_bias=weights.correction)
        self.shared = PreparedMLP(source, weights.shared, scratch.shared.gate_up, scratch.shared.intermediate,
            scratch.shared.output, session=session, name=f"{name}.shared", workspace=workspace,
            swiglu_limit=module.config.swiglu_limit)
        self.experts = PreparedRoutedExperts(source, weights.experts, scratch.router.ids,
            scratch.router.weights, scratch.routed, session=session, name=f"{name}.experts", workspace=workspace,
            router_weight=weights.router, router_plan=self.routing.plan, correction_bias=weights.correction)

    def __call__(self, source):
        if source is not self.source and (source.data_ptr() != self.source.data_ptr() or source.shape != self.source.shape):
            raise ValueError("prepared GLM MoE requires its fixed source storage")
        self.router_projection(source)
        self.routing(self.scratch.router_logits)
        routed = self.experts()
        shared = self.shared(source)
        torch.add(routed, shared, out=self.output)
        return self.output
