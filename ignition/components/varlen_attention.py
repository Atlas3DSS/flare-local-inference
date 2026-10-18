"""Prepared variable-length attention with explicit compiler-visible storage."""

from functools import partial

import torch
from b12x.attention import varlen
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle

from ignition.preparation.fixtures import varlen_attention as fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


@torch.library.custom_op("flare::varlen_attention", mutates_args=("out", "scratch"))
def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor,
    sink: torch.Tensor | None,
    handle: int,
    max_seqlen: int,
) -> None:
    binding = varlen.bind(
        plan_from_handle(handle),
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        attention_sink_bias=sink,
        scratch=scratch,
    )
    values, _ = varlen.run(binding)
    out.copy_(values)


@_attention.register_fake
def _fake_attention(q, k, v, cu, out, scratch, sink, handle, max_seqlen):
    pass


class PreparedVarlenAttention:
    def __init__(self, q, k, v, cu, out, *, session, workspace, name, window=None, sink=None, max_seqlen=None,
                 math_mode="flash"):
        self.q, self.k, self.v, self.cu, self.out = q, k, v, cu, out
        self.workspace = workspace
        self.sink = sink
        self.max_seqlen = q.shape[0] if max_seqlen is None else max_seqlen
        if type(self.max_seqlen) is not int or not 0 < self.max_seqlen <= q.shape[0]:
            raise ValueError("maximum attention sequence must fit the prepared query capacity")
        arithmetic = {} if math_mode == "flash" else {"math_mode": math_mode}
        self.plan = varlen.plan(q, k, v, cu, max_seqlen_q=self.max_seqlen, max_seqlen_k=self.max_seqlen,
                               causal=False, window_size=window, attention_sink_bias=sink, **arithmetic)
        arguments = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu,
            "max_seqlen_q": self.max_seqlen,
            "max_seqlen_k": self.max_seqlen,
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": False,
            "attention_sink_bias": sink,
        }

        def prepare(state):
            scratch = planned_scratch(state.scratch_plan.scratch_specs(), workspace)
            binding = state.bind(scratch=scratch, **{**arguments, "cu_seqlens_q": torch.zeros_like(cu)})
            return PreparedCall(
                run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding)
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=prepare,
                    benchmark_call=partial(fixtures.benchmark_call, arguments=arguments, workspace=workspace,
                        inputs=fixtures.input_factory(arguments, storage=getattr(session, "benchmark_storage", None))),
                ),
            ),
        )
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def __call__(self):
        _attention(self.q, self.k, self.v, self.cu, self.out, self.scratch, self.sink, self.plan.handle, self.max_seqlen)
        return self.out
