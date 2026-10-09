"""Private activation and routing trials for prepared fused expert execution."""

from functools import partial

import torch
from b12x.moe import fused_moe
from b12x.preparation import PreparedCall

from sparkgap.preparation.scratch import planned_scratch
from sparkgap.preparation.tuning import benchmark_resource


class ExpertBenchmarkInputs:
    """Deterministic activation and routing cases outside serving storage."""

    def __init__(self, source, *, experts, top_k, allocate_router, router_weight=None, router_plan=None):
        from b12x.moe.fused_moe.workloads import make_tuning_routes

        tokens, hidden = source.shape
        generator = torch.Generator(device="cpu").manual_seed(7919)
        fixture = torch.randn((tokens, hidden), generator=generator)
        fixture /= fixture.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        self.fixture = fixture.to(device=source.device, dtype=source.dtype)
        self.source = torch.empty_like(source)
        self.output = torch.empty_like(source)
        self.routes = make_tuning_routes(tokens, top_k, experts, device=source.device)
        self.router = allocate_router(tokens, source.device, experts=experts, top_k=top_k)
        self.route_weights = torch.softmax(torch.arange(top_k, dtype=torch.float32) / 8, -1)
        self.route_weights = self.route_weights.to(torch.bfloat16).float().to(source.device)
        self.router_weight = router_weight
        self.router_plan = router_plan
        self.logits = None
        producers = [partial(self._synthetic_routes, index) for index in range(len(self.routes))]
        if tokens > 8 and router_weight is not None:
            if router_plan is None:
                raise ValueError("model-route benchmark producers require an already-prepared route plan")
            if router_weight.shape != (experts, hidden) or router_weight.dtype != torch.bfloat16:
                raise ValueError("benchmark router must preserve BF16 model geometry")
            self.logits = torch.empty((tokens, experts), dtype=torch.bfloat16, device=source.device)
            producers.extend((partial(self._model_routes, False), partial(self._model_routes, True)))
        self.producers = tuple(producers)

    @property
    def nbytes(self):
        tensors = (self.fixture, self.source, self.output, self.routes,
            *(getattr(self.router, field) for field in self.router.__dataclass_fields__),
            self.route_weights, self.logits)
        storage = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                   for value in tensors if value is not None}
        return sum(storage.values())

    def _synthetic_routes(self, index):
        self.source.copy_(self.fixture)
        self.router.ids.copy_(self.routes[index])
        self.router.weights.copy_(self.route_weights.expand_as(self.router.weights))

    def _model_routes(self, correlated):
        fixture = self.fixture[:1].expand_as(self.fixture) if correlated else self.fixture
        self.source.copy_(fixture)
        torch.mm(self.source, self.router_weight.T, out=self.logits)
        fused_moe.route_topk(self.router_plan, self.logits, self.router.selected_logits,
                             self.router.ids, self.router.weights)


def benchmark_call(state, *, source, experts, top_k, session, name, workspace,
                   allocate_router, router_weight, router_plan):
    operands = benchmark_resource(session, ("qwen_moe_inputs", name),
        lambda: ExpertBenchmarkInputs(source, experts=experts.num_experts,
            top_k=top_k, allocate_router=allocate_router, router_weight=router_weight, router_plan=router_plan))
    scratch = planned_scratch(state.scratch.scratch_specs(), workspace)
    binding = state.bind(a=operands.source, topk_ids=operands.router.ids,
        topk_weights=operands.router.weights, output=operands.output,
        scratch=scratch, input_scales_static=True)
    return PreparedCall(run=lambda: state.run(binding), output=operands.output,
        produce=operands.producers[0], benchmark_producers=operands.producers,
        owners=(operands, scratch, binding, experts))
