"""Prepared expert routing and packed expert execution with caller-owned storage."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from b12x.moe import fused_moe
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from ignition.preparation.fixtures import fused_moe as moe_fixtures
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class RouterScratch:
    """Caller-owned selected logits, expert IDs, and FP32 route weights."""

    selected_logits: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, experts: int, top_k: int) -> RouterScratch:
        if (any(type(value) is not int or value <= 0 for value in (tokens, experts, top_k))
                or top_k > experts or experts > 1024):
            raise ValueError("router requires positive integer dimensions and top_k <= experts <= 1024")
        return cls(
            torch.zeros((tokens, top_k), dtype=torch.float32, device=device),
            torch.zeros((tokens, top_k), dtype=torch.int32, device=device),
            torch.full((tokens, top_k), 1 / top_k, dtype=torch.float32, device=device),
        )

    def rows(self, tokens: int) -> RouterScratch:
        if type(tokens) is not int or not 0 < tokens <= self.ids.shape[0]:
            raise ValueError("router rows exceed scratch capacity")
        return RouterScratch(*(getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


class PreparedRouter:
    """Native top-k routing, FP32 weights, and descending-ID ties.

    Route weights retain the library's FP32 precision without a BF16 roundtrip.
    The fixed route plan has no opaque scratch or mutable binding.
    A prepared image mask can select an image-specific correction bias per row
    within the same routing operation.
    """

    def __init__(self, logits: torch.Tensor, scratch: RouterScratch, *,
                 session: PreparationSession, name: str, score_func="softmax",
                 renormalize=True, routed_scaling_factor=1.0, correction_bias=None,
                 image_correction_bias=None, image_mask=None):
        if logits.ndim != 2 or logits.dtype not in (torch.bfloat16, torch.float32) or logits.stride(1) != 1:
            raise ValueError("router logits must be BF16 or FP32 [tokens, experts] with contiguous expert columns")
        tokens, experts = logits.shape
        shape = scratch.ids.shape
        if len(shape) != 2 or shape[0] != tokens or not 0 < shape[1] <= experts <= 1024:
            raise ValueError("router output geometry must match tokens and top_k <= experts <= 1024")
        for tensor, dtype in ((scratch.selected_logits, torch.float32),
                              (scratch.ids, torch.int32), (scratch.weights, torch.float32)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != logits.device or tensor.stride(1) != 1:
                raise ValueError("router outputs require matched shapes, native dtypes/device, and contiguous top-k columns")
        self.scratch, self.device = scratch, logits.device
        self.dtype = logits.dtype
        self.options = {key: value for key, value in (
            ("correction_bias", correction_bias), ("image_correction_bias", image_correction_bias),
            ("image_mask", image_mask)) if value is not None}
        self.shape, self.stride = logits.shape, logits.stride()
        self.plan = fused_moe.plan_route_topk(fused_moe.RouteTopKInvocation(
            num_tokens=tokens, num_experts=experts, top_k=shape[1], logits_dtype=str(logits.dtype).removeprefix("torch."),
            score_func=score_func, renormalize=renormalize, routed_scaling_factor=routed_scaling_factor,
            has_correction_bias=correction_bias is not None,
            has_image_correction_bias=image_correction_bias is not None,
            has_image_mask=image_mask is not None,
            logits_row_stride=logits.stride(0), topk_row_stride=scratch.selected_logits.stride(0),
            ids_row_stride=scratch.ids.stride(0), weights_row_stride=scratch.weights.stride(0)))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(logits, scratch.selected_logits, scratch.ids, scratch.weights, **self.options),
            owners=(logits, scratch))),))

    def __call__(self, logits: torch.Tensor) -> None:
        if (logits.dtype != self.dtype or logits.shape != self.shape
                or logits.stride() != self.stride or logits.device != self.device):
            raise ValueError("router logits must preserve the prepared dtype, shape, strides, and device")
        fused_moe.route_topk(self.plan, logits, self.scratch.selected_logits,
                             self.scratch.ids, self.scratch.weights, **self.options)


class PreparedRoutedExperts:
    """Prepared native expert execution with fixed source and route storage.

    Recipes supply expert geometry and enforce their complete model contract.
    Reduction order follows the execution plan selected by b12x.
    """

    def __new__(cls, source=None, experts=None, *args, **kwargs):
        # Experts loaded for SM103 execute through FlashInfer's native kernels. Unpickling, as compiler
        # workers do with a forward's metadata, creates the instance without arguments.
        from ignition.components.moe_sm103 import Sm103Experts, Sm103RoutedExperts

        if isinstance(experts, Sm103Experts):
            return Sm103RoutedExperts(source, experts, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, source, experts, ids, weights, output, *, session: PreparationSession, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None, router_weight=None, router_plan=None,
                 correction_bias=None, a4_prefill=False):
        # A16 experts prepared with A4 capability run large prefill calls with quantized activations.
        precision = {"a4_prefill": True} if a4_prefill else {}
        self.source = source
        self.output = output
        self.scratch = None
        invocation = FrozenMapping({"tuning_execution": "cuda_graph"}) if experts.has_csf_blocks else FrozenMapping()
        self.scale_plan = fused_moe.plan_scale_expansion(
            experts, routes=ids.numel(), ids_dtype=ids.dtype, invocation=invocation,
        ) if experts.has_csf_blocks else None
        self.plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=source.shape[0], top_k=ids.shape[1], route_num_experts=0),
            scale_expansion=self.scale_plan, invocation=invocation,
        )

        def prepare_call(state):
            scratch = planned_scratch(state.scratch.scratch_specs(), workspace)
            self.scratch = scratch
            binding = state.bind(a=source, topk_ids=ids, topk_weights=weights,
                output=output, scratch=scratch, input_scales_static=True, **precision)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        scale_requests = () if self.scale_plan is None else (self.scale_plan.request(
            name=f"{name}/scales",
            prepare_call=partial(moe_fixtures.scale_expansion_call, ids=ids, experts=experts),
            benchmark_call=partial(moe_fixtures.scale_expansion_benchmark_call,
                source=source, experts=experts, top_k=ids.shape[1], session=session, name=name,
                allocate_router=RouterScratch.allocate, router_weight=router_weight,
                router_plan=router_plan, correction_bias=correction_bias)),)
        declare(session, (*scale_requests, self.plan.request(name=name, prepare_call=prepare_call,
            dependencies=tuple(request.name for request in scale_requests),
            benchmark_call=partial(moe_fixtures.benchmark_call, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate, router_weight=router_weight,
                router_plan=router_plan, correction_bias=correction_bias, precision=precision)),))

        def finalize():
            self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = fused_moe.bind(self.plan, a=source, topk_ids=ids, topk_weights=weights,
                output=output, scratch=self.scratch, input_scales_static=True, **precision)

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self) -> torch.Tensor:
        fused_moe.run(binding=self.binding)
        return self.output


