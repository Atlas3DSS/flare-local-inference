"""Prepared grouped low-rank attention output projection through b12x."""

from dataclasses import replace

import torch
from b12x.gemm import wo_projection
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


@torch.library.custom_op("flare::grouped_attention_output", mutates_args=("out", "scratch"))
def _project(source: torch.Tensor, weights: list[torch.Tensor], out: torch.Tensor,
             scratch: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    q = plan.query
    def matrix(parts):
        return wo_projection.MXFP8Rows(parts[0], parts[1].view(torch.float8_e8m0fnu),
                                      parts[2].view(torch.float8_e8m0fnu))
    packed = wo_projection.Weights(matrix(weights[:3]), matrix(weights[3:]),
                                  q.groups, q.group_width, q.rank, q.hidden)
    binding = wo_projection.bind(plan, source_tgd=source, weights=packed, scratch=scratch)
    binding = replace(binding, output=out.as_strided((source.shape[0], q.hidden, 1),
                                                   (q.hidden, 1, source.shape[0] * q.hidden)))
    wo_projection.run(binding=binding, plan=plan)


@_project.register_fake
def _project_fake(source, weights, out, scratch, handle):
    pass


class PreparedGroupedOutputProjection:
    def __init__(self, source, weights, output, *, session, name, workspace):
        self.weights, self.output, self.workspace = weights, output, workspace
        if source.shape[1:] != (weights.groups, weights.group_width):
            raise ValueError("attention output must match the packed projection groups and head width")
        self.plan = wo_projection.plan(wo_projection.Caps(device=source.device, max_tokens=source.shape[0],
            groups=weights.groups, group_width=weights.group_width, rank=weights.rank, hidden=weights.hidden))
        self.operands = [tensor for packed in (weights.wo_a, weights.wo_b)
                         for tensor in (packed.values, packed.scale_rows.view(torch.uint8),
                                        packed.scale_mma.view(torch.uint8))]

        def call(state, values, out, *, produce=None, owners=()):
            scratch = planned_scratch(state._scratch_state.scratch_specs(), workspace)
            binding = state.bind(source_tgd=values, weights=weights, scratch=scratch)
            binding = replace(binding, output=out.as_strided((values.shape[0], weights.hidden, 1),
                (weights.hidden, 1, values.shape[0] * weights.hidden)))
            return PreparedCall(run=lambda: state.run(binding), output=out, produce=produce,
                owners=(*owners, weights, *prime_owners(workspace, scratch, binding)))

        def benchmark(state):
            generator = torch.Generator(device=source.device).manual_seed(4121)
            seed = torch.randn(source.shape, dtype=source.dtype, device=source.device, generator=generator) * .125
            values, out = torch.empty_like(source), torch.empty_like(output)
            return call(state, values, out, produce=lambda: values.copy_(seed), owners=(values, out, seed))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: call(state, source, output), benchmark_call=benchmark),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def __call__(self, source):
        _project(source, self.operands, self.output, self.scratch, self.plan.handle)
        return self.output
