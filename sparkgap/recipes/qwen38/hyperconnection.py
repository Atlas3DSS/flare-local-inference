"""Prepared Qwen hyperconnection mixing and residual injection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.norm import hyperconnection as hc
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from sparkgap.preparation.tuning import register_finalizer

from .linear import BF16Projection


class _HyperconnectionBenchmarkInputs:
    """Private finite activations with the canonical RMS normalization gain."""

    def __init__(self, operation, arguments):
        weight_index = 1 if operation == "grouped_rmsnorm" else None
        self.arguments = tuple(
            value if index == weight_index else torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device)
            for index, value in enumerate(arguments)
        )
        generator = torch.Generator(device="cpu").manual_seed(65537)
        self.inputs = []
        for index, target in enumerate(self.arguments[:-1]):
            if index == weight_index:
                continue
            fixture = torch.randn(target.shape, generator=generator)
            self.inputs.append((target, fixture.to(device=target.device, dtype=target.dtype)))
        self.arguments[-1].zero_()

    def produce(self):
        for target, fixture in self.inputs:
            target.copy_(fixture)


def _benchmark_call(state, operation, arguments, kwargs):
    operands = _HyperconnectionBenchmarkInputs(operation, arguments)
    return PreparedCall(
        run=lambda: state.launch(*operands.arguments, **kwargs),
        output=operands.arguments[-1],
        produce=operands.produce,
        owners=(operands,),
    )


@dataclass(frozen=True)
class HyperconnectionScratch:
    normalized: torch.Tensor
    projected_down: torch.Tensor
    bottleneck: torch.Tensor
    gate_logits: torch.Tensor
    injection_logits: torch.Tensor
    block_input: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device) -> HyperconnectionScratch:
        if tokens <= 0:
            raise ValueError("hyperconnection token capacity must be positive")
        return cls(*(torch.zeros((tokens, width), device=device, dtype=torch.bfloat16)
                     for width in (10240, 320, 320, 10240, 4, 2560)))

    def rows(self, tokens: int) -> HyperconnectionScratch:
        if not 0 < tokens <= self.normalized.shape[0]:
            raise ValueError("hyperconnection rows exceed scratch capacity")
        return HyperconnectionScratch(*(getattr(self, field)[:tokens] for field in self.__dataclass_fields__))

    def priming_arguments(self, source: torch.Tensor, norm_gain: torch.Tensor):
        """Borrow a zero-initialized source kept unchanged until preparation ends."""
        if (source.shape != self.normalized.shape or source.stride() != self.normalized.stride()
                or source.dtype != torch.bfloat16 or source.device != self.normalized.device
                or not source.is_contiguous()):
            raise ValueError("hyperconnection priming source must match normalized shape, stride, dtype and device")
        source_storage = source.untyped_storage().data_ptr()
        if any(source_storage == getattr(self, field).untyped_storage().data_ptr()
               for field in self.__dataclass_fields__):
            raise ValueError("hyperconnection priming source requires storage separate from scratch outputs")
        return {
            "grouped_rmsnorm": (source, norm_gain, self.normalized),
            "scaled_silu": (self.projected_down, self.bottleneck),
            "gate_mean": (self.normalized, self.gate_logits, self.block_input),
            "combine": (source, self.block_input, self.injection_logits, self.normalized),
        }


class PreparedHyperconnection:
    """Use one shared scratch region during serialized decoder execution.

    ``mix`` returns the next block's input. Its injection logits remain live
    until ``combine``; another mixer must not overwrite the scratch in between.
    The final model mixer has no residual injection projection.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        scratch: HyperconnectionScratch,
        *,
        priming_state: torch.Tensor,
        session: PreparationSession,
        name: str,
    ) -> None:
        if module.hc_count != 4 or module.hidden_size != 2560:
            raise ValueError("Qwen hyperconnection requires four streams of width 2560")
        self.scratch = scratch
        self.eps = module.hc_norm.eps
        self.norm_gain = (1.0 + module.hc_norm.weight.detach().float()).contiguous()
        device = scratch.normalized.device
        if self.norm_gain.device != device or device.type != "cuda":
            raise ValueError("hyperconnection weights and scratch must share a CUDA device")
        if tuple(self.norm_gain.shape) != (10240,):
            raise ValueError("hyperconnection norm weight must have 10240 elements")
        self.tokens = scratch.normalized.shape[0]
        caps = hc.Caps(device=device, max_tokens=self.tokens, hidden_size=2560, streams=4, lowrank=320)
        self.plans = {}
        self.bindings = {}
        priming_arguments = scratch.priming_arguments(priming_state, self.norm_gain)
        invocations = {
            "grouped_rmsnorm": {"eps": self.eps, "zero_centered": False, "weight_dtype": "float32"},
            "scaled_silu": {},
            "gate_mean": {},
        }
        if module.block_inject_weight is not None:
            invocations["combine"] = {}
        for operation, options in invocations.items():
            plan = hc.plan(caps, invocation=FrozenMapping({"operation": operation, **options}))
            arguments = priming_arguments[operation]
            kwargs = {"eps": self.eps} if operation == "grouped_rmsnorm" else {}
            session.prepare((plan.request(
                name=f"{name}.{operation}",
                prepare_call=lambda state, args=arguments, kw=kwargs: PreparedCall(run=lambda: state.launch(*args, **kw)),
                benchmark_call=lambda state, op=operation, args=arguments, kw=kwargs: _benchmark_call(state, op, args, kw),
            ),))
            self.plans[operation] = plan
            if operation == "combine":
                continue

            def finalize(op=operation, selected_plan=plan):
                self.bindings[op] = hc.bind(selected_plan, tokens=self.tokens,
                    normalized=scratch.normalized, bottleneck=scratch.bottleneck, block_input=scratch.block_input)

            finalize()
            register_finalizer(session, finalize)

        self.down = BF16Projection(scratch.normalized, module.input_mix_weight_down.weight,
            scratch.projected_down, session=session, name=f"{name}.down")
        self.up = BF16Projection(scratch.bottleneck, module.input_mix_weight_up.weight,
            scratch.gate_logits, session=session, name=f"{name}.up")
        self.injection = None if module.block_inject_weight is None else BF16Projection(
            scratch.normalized, module.block_inject_weight.weight, scratch.injection_logits,
            session=session, name=f"{name}.injection")

    def mix(self, residual: torch.Tensor) -> torch.Tensor:
        normalized = hc.run_grouped_rmsnorm(residual, self.norm_gain, eps=self.eps,
            zero_centered=False, binding=self.bindings["grouped_rmsnorm"])
        projected = self.down(normalized)
        bottleneck = hc.run_scaled_silu(projected, binding=self.bindings["scaled_silu"])
        gate_logits = self.up(bottleneck)
        result = hc.run_gate_mean(normalized, gate_logits, binding=self.bindings["gate_mean"])
        if self.injection is not None:
            self.injection(normalized)
        return result

    def combine(self, residual: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        """Inject in FP32 and return a BF16 intermediate inside the captured graph."""
        if self.injection is None:
            raise ValueError("the final hyperconnection mixer has no residual injection")
        return hc.run_combine(residual, block_output, self.scratch.injection_logits,
                              plan=self.plans["combine"])
