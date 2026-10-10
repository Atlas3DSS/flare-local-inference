"""Prepared Qwen routed NVFP4 experts and the shared MXFP8 expert."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from b12x.gemm import blockscaled
from b12x.moe import fused_moe
from b12x.norm import hyperconnection
from b12x.preparation import PreparationSession, PreparedCall

from flare.preparation.qwen38 import TEXT_PREFIX
from flare.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from flare.preparation.tuning import register_finalizer
from flare.preparation.fixtures import fused_moe as moe_fixtures
from .linear import BF16Projection, QuantizedProjection
from .weights import QwenWeightLoader


@dataclass(frozen=True)
class SharedExpertWeights:
    gate_up: blockscaled.Weight | torch.Tensor
    down: blockscaled.Weight | torch.Tensor
    gate: torch.Tensor

    @classmethod
    def load(cls, loader: QwenWeightLoader, layer: int) -> SharedExpertWeights:
        if type(layer) is not int or not 0 <= layer < 48:
            raise ValueError("decoder layer must be an integer in [0, 48)")
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp"
        gate, up = (loader.load_linear(f"{prefix}.shared_expert.{projection}")
                    for projection in ("gate_proj", "up_proj"))
        values = torch.cat((gate.weight.values, up.weight.values))
        scales = torch.cat((gate.weight.scale_rows[0], up.weight.scale_rows[0]))
        return cls(
            blockscaled.pack_weight(values, scales, recipe="mxfp8"),
            loader.load_linear(f"{prefix}.shared_expert.down_proj"),
            loader.load_tensor(f"{prefix}.shared_expert_gate.weight"),
        )


@dataclass(frozen=True)
class QwenMoeWeights:
    router: torch.Tensor
    experts: fused_moe.PreparedExperts
    shared: SharedExpertWeights

    @classmethod
    def load(cls, loader: QwenWeightLoader, layer: int, *, scale_scratch=None) -> QwenMoeWeights:
        shared = SharedExpertWeights.load(loader, layer)
        return cls(
            loader.load_tensor(f"{TEXT_PREFIX}.layers.{layer}.mlp.gate.weight"),
            loader.load_experts(layer, scale_scratch=scale_scratch),
            shared,
        )


@dataclass(frozen=True)
class RouterScratch:
    """Caller-owned selected logits, expert IDs, and FP32 route weights."""

    selected_logits: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, experts: int = 512, top_k: int = 10) -> RouterScratch:
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
    """Native selected-logit softmax, FP32 weights, and descending-ID ties.

    Route weights retain the library's FP32 precision without a BF16 roundtrip.
    The fixed route plan has no opaque scratch or mutable binding.
    """

    def __init__(self, logits: torch.Tensor, scratch: RouterScratch, *,
                 session: PreparationSession, name: str):
        if logits.ndim != 2 or logits.dtype != torch.bfloat16 or logits.stride(1) != 1:
            raise ValueError("router logits must be BF16 [tokens, experts] with contiguous expert columns")
        tokens, experts = logits.shape
        shape = scratch.ids.shape
        if len(shape) != 2 or shape[0] != tokens or not 0 < shape[1] <= experts <= 1024:
            raise ValueError("router output geometry must match tokens and top_k <= experts <= 1024")
        for tensor, dtype in ((scratch.selected_logits, torch.float32),
                              (scratch.ids, torch.int32), (scratch.weights, torch.float32)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != logits.device or tensor.stride(1) != 1:
                raise ValueError("router outputs require matched shapes, native dtypes/device, and contiguous top-k columns")
        self.scratch, self.device = scratch, logits.device
        self.shape, self.stride = logits.shape, logits.stride()
        self.plan = fused_moe.plan_route_topk(fused_moe.RouteTopKInvocation(
            num_tokens=tokens, num_experts=experts, top_k=shape[1], logits_dtype="bfloat16",
            score_func="softmax", renormalize=True, routed_scaling_factor=1.0,
            logits_row_stride=logits.stride(0), topk_row_stride=scratch.selected_logits.stride(0),
            ids_row_stride=scratch.ids.stride(0), weights_row_stride=scratch.weights.stride(0)))
        session.prepare((self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(logits, scratch.selected_logits, scratch.ids, scratch.weights),
            owners=(logits, scratch))),))

    def __call__(self, logits: torch.Tensor) -> None:
        if (logits.dtype != torch.bfloat16 or logits.shape != self.shape
                or logits.stride() != self.stride or logits.device != self.device):
            raise ValueError("router logits must preserve the prepared BF16 shape, strides, and device")
        fused_moe.route_topk(self.plan, logits, self.scratch.selected_logits,
                             self.scratch.ids, self.scratch.weights)


@dataclass(frozen=True)
class SharedExpertScratch:
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    projected: torch.Tensor
    gate_logits: torch.Tensor
    gate_probability: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device) -> SharedExpertScratch:
        if tokens <= 0:
            raise ValueError("shared expert capacity must be positive")
        return cls(*(torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
                     for width in (1280, 640, 2560, 1, 1, 2560)))

    def rows(self, tokens: int) -> SharedExpertScratch:
        if not 0 < tokens <= self.gate_up.shape[0]:
            raise ValueError("shared expert rows exceed scratch capacity")
        return SharedExpertScratch(*(getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


@dataclass(frozen=True)
class QwenMoeScratch:
    """Capacity-backed intermediates shared by serial decoder layers.

    The shared-expert output remains live while routed experts execute, so
    these are distinct buffers. Views for smaller graph buckets share storage;
    graphs using this scratch must execute in one ordered stream.
    """

    router_logits: torch.Tensor
    router: RouterScratch
    routed_output: torch.Tensor
    output: torch.Tensor
    shared: SharedExpertScratch

    @classmethod
    def allocate(cls, tokens: int, device) -> QwenMoeScratch:
        if tokens <= 0:
            raise ValueError("MoE capacity must be positive")
        return cls(torch.zeros((tokens, 512), dtype=torch.bfloat16, device=device),
            RouterScratch.allocate(tokens, device),
            torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device),
            torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device),
            SharedExpertScratch.allocate(tokens, device))

    def rows(self, tokens: int) -> QwenMoeScratch:
        if not 0 < tokens <= self.router_logits.shape[0]:
            raise ValueError("MoE rows exceed scratch capacity")
        return QwenMoeScratch(self.router_logits[:tokens], self.router.rows(tokens),
            self.routed_output[:tokens], self.output[:tokens], self.shared.rows(tokens))


class PreparedSharedExpert:
    """MXFP8 or stored BF16 SwiGLU with BF16 SiLU and sigmoid boundaries."""

    def __init__(self, source, weights: SharedExpertWeights, *, session: PreparationSession, name: str,
                 scratch: SharedExpertScratch | None = None, projection_workspace: ScratchArena | torch.Tensor | None = None):
        if source.ndim != 2 or source.shape[1] != 2560 or source.dtype != torch.bfloat16:
            raise ValueError("shared expert input must be BF16 [tokens, 2560]")
        def geometry(weight):
            return tuple(reversed(weight.shape)) if isinstance(weight, torch.Tensor) else (weight.in_features, weight.out_features)

        if geometry(weights.gate_up) != (2560, 1280):
            raise ValueError("shared gate/up projection must have geometry [1280, 2560]")
        if geometry(weights.down) != (640, 2560):
            raise ValueError("shared down projection must have geometry [2560, 640]")
        if tuple(weights.gate.shape) != (1, 2560):
            raise ValueError("shared expert gate must have geometry [1, 2560]")
        tokens, device = source.shape[0], source.device
        scratch = SharedExpertScratch.allocate(tokens, device) if scratch is None else scratch.rows(tokens)
        for field, width in zip(scratch.__dataclass_fields__, (1280, 640, 2560, 1, 1, 2560), strict=True):
            tensor = getattr(scratch, field)
            if tensor.shape != (tokens, width) or tensor.dtype != torch.bfloat16 or tensor.device != device or not tensor.is_contiguous():
                raise ValueError(f"shared expert {field} requires contiguous BF16 [{tokens}, {width}] on {device}")
            setattr(self, field, tensor)
        self.scratch = scratch
        def projection(source, weight, output, key):
            if isinstance(weight, torch.Tensor):
                return BF16Projection(source, weight, output, session=session, name=f"{name}.{key}")
            return QuantizedProjection(source, weight, output, session=session,
                                       name=f"{name}.{key}", workspace=projection_workspace)

        self.up_projection = projection(source, weights.gate_up, self.gate_up, "gate_up")
        self.down_projection = projection(self.intermediate, weights.down, self.projected, "down")
        self.gate_projection = BF16Projection(source, weights.gate, self.gate_logits,
            session=session, name=f"{name}.shared_gate")
        self.activation_plan = hyperconnection.plan(
            hyperconnection.Caps(device=device, max_tokens=tokens, hidden_size=640),
            invocation={"operation": "swiglu", "limit": float("inf"), "round_silu": True},
        )
        session.prepare((self.activation_plan.request(
            name=f"{name}.swiglu",
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.launch(self.gate_up, self.gate_up, self.intermediate),
            ),
        ),))

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        self.up_projection(source)
        hyperconnection.run_swiglu(self.gate_up, limit=float("inf"),
            round_silu=True, out=self.intermediate, plan=self.activation_plan)
        self.down_projection(self.intermediate)
        self.gate_projection(source)
        torch.sigmoid(self.gate_logits, out=self.gate_probability)
        torch.mul(self.projected, self.gate_probability, out=self.output)
        return self.output


class PreparedRoutedExperts:
    """Prepared native NVFP4 execution with fixed source and route storage.

    This component also admits explicit subsets for numerical qualification.
    ``PreparedQwenMoe`` enforces the complete 512-expert model contract.
    Reduction order follows the execution plan selected by b12x.
    """

    def __init__(self, source, experts, ids, weights, output, *, session: PreparationSession, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None, router_weight=None, router_plan=None):
        self.source = source
        self.output = output
        self.scratch = None
        self.plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=source.shape[0], top_k=ids.shape[1], route_num_experts=0),
        )

        def prepare_call(state):
            scratch = planned_scratch(state.scratch.scratch_specs(), workspace)
            self.scratch = scratch
            binding = state.bind(a=source, topk_ids=ids, topk_weights=weights,
                output=output, scratch=scratch, input_scales_static=True)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        session.prepare((self.plan.request(name=name, prepare_call=prepare_call,
            benchmark_call=partial(moe_fixtures.benchmark_call, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate, router_weight=router_weight,
                router_plan=router_plan)),))

        def finalize():
            self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = fused_moe.bind(self.plan, a=source, topk_ids=ids, topk_weights=weights,
                output=output, scratch=self.scratch, input_scales_static=True)

        finalize()
        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self) -> torch.Tensor:
        fused_moe.run(binding=self.binding)
        return self.output


class PreparedQwenMoe:
    """Native FP32 routing and shared-expert composition around routed experts.

    Construction allocates and prepares all buffers and plans. Calls are
    recorded into the model CUDA graph; native replay executes its CUDA nodes
    without calling this Python object. Source storage is fixed per graph.
    """

    def __init__(self, source, weights: QwenMoeWeights, *, session: PreparationSession, name: str,
                 scratch: QwenMoeScratch | None = None, workspace: ScratchArena | torch.Tensor | None = None,
                 projection_workspace: ScratchArena | torch.Tensor | None = None):
        if (weights.experts.num_experts, weights.experts.hidden_size,
            weights.experts.intermediate_size) != (512, 2560, 640):
            raise ValueError("Qwen MoE requires all 512 experts with hidden/intermediate sizes 2560/640")
        if tuple(weights.router.shape) != (512, 2560):
            raise ValueError("Qwen router weight must have geometry [512, 2560]")
        if source.ndim != 2 or source.shape[1] != 2560 or source.dtype != torch.bfloat16:
            raise ValueError("Qwen MoE source must be BF16 [tokens, 2560]")
        self.source = source
        scratch = QwenMoeScratch.allocate(source.shape[0], source.device) if scratch is None else scratch.rows(source.shape[0])
        for tensor, width in ((scratch.router_logits, 512), (scratch.routed_output, 2560), (scratch.output, 2560)):
            if tensor.shape != (source.shape[0], width) or tensor.dtype != torch.bfloat16 or tensor.device != source.device or not tensor.is_contiguous():
                raise ValueError("MoE scratch geometry, BF16 dtype and device must match source")
        self.scratch = scratch
        self.router_logits, self.router = scratch.router_logits, scratch.router
        self.routed_output, self.output = scratch.routed_output, scratch.output
        self.router_projection = BF16Projection(source, weights.router, self.router_logits,
            session=session, name=f"{name}.router")
        self.routing = PreparedRouter(self.router_logits, self.router, session=session, name=f"{name}.route_topk")
        self.shared = PreparedSharedExpert(source, weights.shared, session=session, name=f"{name}.shared",
            scratch=scratch.shared, projection_workspace=projection_workspace)
        self.experts = PreparedRoutedExperts(source, weights.experts, self.router.ids,
            self.router.weights, self.routed_output, session=session, name=f"{name}.experts", workspace=workspace,
            router_weight=weights.router, router_plan=self.routing.plan)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source is not self.source and (
                source.data_ptr() != self.source.data_ptr() or source.shape != self.source.shape):
            raise ValueError("prepared Qwen MoE requires its fixed source storage")
        shared = self.shared(source)
        self.router_projection(source)
        self.routing(self.router_logits)
        routed = self.experts()
        torch.add(routed, shared, out=self.output)
        return self.output
