"""Finite varlen-attention operands with one full-capacity sequence."""

from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


@dataclass(frozen=True)
class AttentionBenchmarkInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    seed_q: torch.Tensor
    seed_k: torch.Tensor
    seed_v: torch.Tensor
    cu_seqlens_q: torch.Tensor

    @property
    def nbytes(self):
        return sum(value.untyped_storage().nbytes() for value in vars(self).values())

    def produce(self):
        self.q.copy_(self.seed_q)
        self.k.copy_(self.seed_k)
        self.v.copy_(self.seed_v)


def attention_benchmark_inputs(arguments):
    """Private finite QKV and one full-capacity image for attention tuning."""
    query = arguments["q"]
    generator = torch.Generator(device=query.device).manual_seed(89017)
    result = {}
    for name in ("q", "k", "v"):
        source = arguments[name]
        seed = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                   device=source.device).normal_(generator=generator)
        result[f"seed_{name}"] = seed
        result[name] = torch.empty_strided(source.shape, source.stride(),
                                           dtype=source.dtype, device=source.device).copy_(seed)
    lengths = torch.full_like(arguments["cu_seqlens_q"], query.shape[0])
    lengths[0] = 0
    return AttentionBenchmarkInputs(**result, cu_seqlens_q=lengths)


def benchmark_call(state, *, arguments, session, workspace):
    identity = tuple((name, tuple(arguments[name].shape), arguments[name].stride(),
                      arguments[name].dtype, arguments[name].device)
                     for name in ("q", "k", "v", "cu_seqlens_q"))
    inputs = benchmark_resource(session, ("vision_attention_inputs", identity),
                                lambda: attention_benchmark_inputs(arguments))
    arguments = {**arguments, **{key: getattr(inputs, key)
                                  for key in ("q", "k", "v", "cu_seqlens_q")}}
    scratch = planned_scratch(state.scratch_plan.scratch_specs(), workspace)
    binding = state.bind(scratch=scratch, **arguments)
    return PreparedCall(run=lambda: state.run(binding), produce=inputs.produce,
                        owners=(inputs, scratch, binding))
