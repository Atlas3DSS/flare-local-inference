"""Prepared variable-length attention with explicit compiler-visible storage."""

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
    handle: int,
) -> None:
    binding = varlen.bind(
        plan_from_handle(handle),
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        scratch=scratch,
    )
    values, _ = varlen.run(binding)
    out.copy_(values)


@_attention.register_fake
def _fake_attention(q, k, v, cu, out, scratch, handle):
    pass


class PreparedVarlenAttention:
    def __init__(self, q, k, v, cu, out, *, session, workspace, name):
        self.q, self.k, self.v, self.cu, self.out = q, k, v, cu, out
        self.workspace = workspace
        self.plan = varlen.plan(q, k, v, cu, max_seqlen_q=q.shape[0], max_seqlen_k=k.shape[0], causal=False)
        arguments = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu,
            "max_seqlen_q": q.shape[0],
            "max_seqlen_k": k.shape[0],
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": False,
        }

        def prepare(state):
            scratch = planned_scratch(state.scratch_plan.scratch_specs(), workspace)
            binding = state.bind(scratch=scratch, **arguments)
            return PreparedCall(
                run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding)
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=prepare,
                    benchmark_call=lambda state: fixtures.benchmark_call(
                        state, arguments=arguments, session=session, workspace=workspace
                    ),
                ),
            ),
        )
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def __call__(self):
        _attention(self.q, self.k, self.v, self.cu, self.out, self.scratch, self.plan.handle)
        return self.out
