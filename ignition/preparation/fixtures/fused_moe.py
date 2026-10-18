"""Private activation and routing trials for prepared fused expert execution."""

from functools import partial

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


class ExpertBenchmarkInputs:
    """Deterministic activation and routing cases outside serving storage."""

    def __init__(self, source, *, experts, top_k, allocate_router, router_plan=None):
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
        if router_plan is not None:
            self.route_weights.mul_(router_plan.query.routed_scaling_factor)
        self.producers = tuple(partial(self._synthetic_routes, index) for index in range(len(self.routes)))

    @property
    def nbytes(self):
        tensors = (self.fixture, self.source, self.output, self.routes,
            *(getattr(self.router, field) for field in self.router.__dataclass_fields__),
            self.route_weights)
        storage = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                   for value in tensors if value is not None}
        return sum(storage.values())

    def _synthetic_routes(self, index):
        self.source.copy_(self.fixture)
        self.router.ids.copy_(self.routes[index])
        self.router.weights.copy_(self.route_weights.expand_as(self.router.weights))


def benchmark_call(state, *, source, experts, top_k, session, name, workspace,
                   allocate_router, router_plan, precision=None):
    operands = benchmark_resource(session, ("moe_inputs", name),
        lambda: ExpertBenchmarkInputs(source, experts=experts.num_experts,
            top_k=top_k, allocate_router=allocate_router, router_plan=router_plan))
    scratch = planned_scratch(state.scratch.scratch_specs(), workspace)
    binding = state.bind(a=operands.source, topk_ids=operands.router.ids,
        topk_weights=operands.router.weights, output=operands.output,
        scratch=scratch, input_scales_static=True, **(precision or {}))
    call = PreparedCall(run=lambda: state.run(binding), output=operands.output,
        produce=operands.producers[0], benchmark_producers=operands.producers,
        owners=(operands, scratch, binding, experts))
    if experts.has_csf_blocks:
        from b12x.preparation._measurement import capture_benchmark_call

        call = capture_benchmark_call(call)
    return call


def scale_expansion_call(state, *, ids, experts):
    active = torch.empty(experts.num_experts, dtype=torch.int32, device=ids.device)
    binding = state.bind(ids=ids, out13=experts._impl.w1_blockscale,
        out2=experts._impl.w2_blockscale, scratch=active)
    return PreparedCall(run=binding.run, owners=(binding, active, experts))


def scale_expansion_benchmark_call(state, *, source, experts, top_k, session, name,
                                 allocate_router, router_plan):
    from dataclasses import replace
    from b12x.preparation._measurement import capture_benchmark_call

    operands = benchmark_resource(session, ("moe_inputs", name),
        lambda: ExpertBenchmarkInputs(source, experts=experts.num_experts,
            top_k=top_k, allocate_router=allocate_router, router_plan=router_plan))
    call = scale_expansion_call(state, ids=operands.router.ids, experts=experts)
    return capture_benchmark_call(replace(call,
        produce=operands.producers[0], benchmark_producers=operands.producers,
        owners=(*call.owners, operands)))
