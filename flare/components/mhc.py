"""Prepared manifold hyperconnections with shared scratch and caller-owned outputs."""

from dataclasses import dataclass

import torch
from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import PreparedCall

from flare.preparation.scratch import planned_scratch, prime_owners
from flare.preparation.tuning import declare, benchmark_resource, register_finalizer


@dataclass(frozen=True)
class MHCWeights:
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    norm: torch.Tensor


@dataclass(frozen=True)
class MHCScratch:
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    normalized: torch.Tensor

    @classmethod
    def allocate(cls, tokens, hidden, device):
        if type(tokens) is not int or tokens <= 0 or hidden not in (4096, 5120, 7168):
            raise ValueError("mHC requires positive token capacity and a supported hidden size")
        return cls(torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device),
                   torch.empty((tokens, 4), dtype=torch.float32, device=device),
                   torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
                   torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device))

    def bind_args(self):
        return {"out": self.residual, "post": self.post, "comb": self.comb, "y": self.normalized}

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.residual.shape[0]:
            raise ValueError("mHC rows exceed scratch capacity")
        return MHCScratch(*(getattr(self, name)[:tokens] for name in self.__dataclass_fields__))


class _BenchmarkInputs:
    def __init__(self, residual, operation):
        tokens, _, hidden = residual.shape
        generator = torch.Generator(device=residual.device).manual_seed(3187)
        values = torch.randn(residual.shape, dtype=residual.dtype, device=residual.device,
                             generator=generator) * .1
        self.inputs = (values,)
        if operation == "post_pre":
            source = torch.randn((tokens, hidden), dtype=residual.dtype, device=residual.device,
                                 generator=generator) * .1
            post = torch.ones((tokens, 4), dtype=torch.float32, device=residual.device)
            comb = torch.eye(4, device=residual.device).expand(tokens, -1, -1).contiguous()
            self.inputs = (source, values, post, comb)
        self.seeds = self.inputs
        self.inputs = tuple(torch.empty_like(value) for value in self.seeds)
        self.scratch = MHCScratch.allocate(tokens, hidden, residual.device)
        self.nbytes = sum(t.nbytes for t in (*self.seeds, *self.inputs, *self.scratch.bind_args().values()))

    def produce(self):
        for target, seed in zip(self.inputs, self.seeds, strict=True):
            target.copy_(seed)


class PreparedMHC:
    """mHC pre or fused post/pre, including the following weighted RMSNorm.

    Pre consumes a BF16 residual of shape [tokens, 4, hidden]. Post/pre also
    consumes the preceding branch output and its FP32 expansion/mixing weights.
    The output residual and normalized branch input retain BF16 storage; mixing
    coefficients and projection weights use FP32. Runtime calls use fixed shapes.
    """

    def __init__(self, inputs, weights: MHCWeights, scratch: MHCScratch, *,
                 session, name, workspace, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
        self.operation = "pre" if len(inputs) == 1 else "post_pre" if len(inputs) == 4 else None
        if self.operation is None:
            raise ValueError("mHC requires residual or branch/residual/post/comb inputs")
        residual = inputs[0 if self.operation == "pre" else 1]
        if residual.ndim != 3 or residual.shape[1] != 4 or residual.dtype != torch.bfloat16:
            raise ValueError("mHC residual requires BF16 [tokens, 4, hidden]")
        tokens, _, hidden = residual.shape
        if (weights.fn.shape != (24, 4 * hidden) or weights.fn.dtype != torch.float32
                or weights.scale.shape != (3,) or weights.scale.dtype != torch.float32
                or weights.base.shape != (24,) or weights.base.dtype != torch.float32
                or weights.norm.shape != (hidden,) or weights.norm.dtype != torch.bfloat16):
            raise ValueError("mHC weights require FP32 projection/mixes and a BF16 RMSNorm vector")
        self.weights, self.scratch = weights, scratch
        self.options = {"rms_eps": rms_eps, "hc_eps": hc_eps, "sinkhorn_iters": sinkhorn_iters,
                            "norm_weight": weights.norm, "norm_eps": norm_eps}
        self.plan = mhc.plan(mhc.Caps(device=residual.device, max_tokens=tokens, hidden_size=hidden),
            invocation={"operation": self.operation, "output_mode": "provided", "has_norm_weight": True,
                "norm_weight_dtype": "bfloat16", "expanded_residual": self.operation == "pre",
                "rms_eps": rms_eps, "hc_eps": hc_eps, "sinkhorn_iters": sinkhorn_iters, "norm_eps": norm_eps})
        entry = getattr(_impl, f"_b12x_mhc_{self.operation}_impl")
        parameters = (weights.fn, weights.scale, weights.base)

        def call(state, values, buffers, fixture=None):
            storage = planned_scratch(state.scratch_specs(), workspace)
            binding = state.bind(scratch=storage, tokens=tokens, **buffers.bind_args())
            return PreparedCall(
                run=lambda: entry(*values, *parameters, **self.options, binding=binding, _state=state),
                produce=None if fixture is None else fixture.produce,
                output=tuple(buffers.bind_args().values()),
                owners=(*prime_owners(workspace, storage, binding), fixture))

        def benchmark(state):
            fixture = benchmark_resource(session, ("mhc", str(residual.device), tokens, hidden, self.operation),
                                         lambda: _BenchmarkInputs(residual, self.operation))
            return call(state, fixture.inputs, fixture.scratch, fixture)

        declare(session, (self.plan.request(name=name, prepare_call=lambda state: call(state, inputs, scratch),
                                          benchmark_call=benchmark),))

        def finalize():
            self.binding = mhc.bind(self.plan, scratch=planned_scratch(self.plan.scratch_specs(), workspace),
                                    tokens=tokens, **scratch.bind_args())

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self, *inputs):
        entry = mhc.run_pre if self.operation == "pre" else mhc.run_post_pre
        entry(*inputs, self.weights.fn, self.weights.scale, self.weights.base,
              **self.options, binding=self.binding)
        return self.scratch.normalized


class PreparedMHCPost:
    def __init__(self, source, residual, post, comb, output, *, session, name):
        self.output = output
        self.plan = mhc.plan(mhc.Caps(device=residual.device, max_tokens=residual.shape[0],
                                     hidden_size=residual.shape[-1]),
                             invocation={"operation": "post", "output_mode": "provided"})
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: _impl._b12x_mhc_post_impl(source, residual, post, comb, out=output, _state=state))),))

    def __call__(self, source, residual, post, comb):
        return mhc.run_post(source, residual, post, comb, out=self.output, plan=self.plan)
