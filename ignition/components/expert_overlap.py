"""Capture shared-expert MLPs beside routing and routed-expert execution."""

import torch

from ignition.components.linear import QuantizedProjection
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


def _run_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                       workspace, plans, limit, round_silu, auxiliary):
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        if scales[index] is None:
            bf16_out(value, weights[index], output, plans[index])
        else:
            packed_out_workspace(
                value, weights[index], scales[index], gains[index], None, output,
                workspace, ("none", "multiplier", "reciprocal")[scale_kinds[index]], None, plans[index], "any")

    gate_up, intermediate, projected = outputs[:3]
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        torch.ops.b12x.hyperconnection_swiglu(gate_up, intermediate, limit, round_silu, plans[-1])
        project(intermediate, 1, projected)
        if len(weights) == 3:
            project(source, 2, outputs[3])


@torch.library.custom_op("flare::start_shared_expert", mutates_args=("outputs", "workspace"))
def _start_shared_expert(
    source: torch.Tensor, weights: list[torch.Tensor],
    scales: list[torch.Tensor | None], gains: list[torch.Tensor | None],
    scale_kinds: list[int], outputs: list[torch.Tensor], workspace: torch.Tensor,
    plans: list[int], limit: float, round_silu: bool, auxiliary_stream: int,
    defer_launch: bool,
) -> None:
    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    if not defer_launch:
        _run_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                           workspace, plans, limit, round_silu, auxiliary)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                             workspace, plans, limit, round_silu, auxiliary_stream, defer_launch):
    pass


@torch.library.custom_op("flare::join_shared_expert", mutates_args=("outputs", "workspace"))
def _join_shared_expert(
    source: torch.Tensor, routed: torch.Tensor, weights: list[torch.Tensor],
    scales: list[torch.Tensor | None], gains: list[torch.Tensor | None],
    scale_kinds: list[int], outputs: list[torch.Tensor], workspace: torch.Tensor,
    plans: list[int], limit: float, round_silu: bool, auxiliary_stream: int,
    defer_launch: bool,
) -> None:
    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    if defer_launch:
        _run_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                           workspace, plans, limit, round_silu, auxiliary)
    main.wait_stream(auxiliary)


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, weights, scales, gains, scale_kinds, outputs,
                            workspace, plans, limit, round_silu, auxiliary_stream, defer_launch):
    pass


class ConcurrentSharedExpert:
    """Fork a prepared MLP onto an auxiliary stream and join before consumption.

    The recipe supplies every plan that can run concurrently on the main stream.
    Main-stream scratch occupies the arena prefix. MLP projections reuse one
    disjoint suffix, sized after autotuning. Explicit mutation operands keep the
    fork, join, and storage lifetimes ordered through forward compilation.
    An optional source projection computes a shared-expert gate on the same fork;
    the recipe applies its activation and combines outputs after ``finish``.
    With ``defer_launch``, ``start`` records the fork and ``finish`` enqueues the
    MLP before joining. The GPU branches still overlap, while main-stream kernels
    precede shared-expert kernels in graph construction order.
    """

    def __init__(self, mlp, *, parallel_plans, session, workspace: ScratchArena,
                 gate_projection=None, defer_launch=False):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        if not isinstance(workspace, ScratchArena) or not parallel_plans:
            raise ValueError("shared-expert overlap requires an arena and concurrent main-stream plans")
        self.mlp, self.parallel_plans, self.arena = mlp, tuple(parallel_plans), workspace
        self.defer_launch = bool(defer_launch)
        self.projections = (mlp.up_projection, mlp.down_projection)
        self.outputs = [mlp.up_projection.output, mlp.intermediate, mlp.down_projection.output]
        if gate_projection is not None:
            self.projections += (gate_projection,)
            self.outputs.append(gate_projection.output)
        self.weights, self.scales, self.gains, self.scale_kinds = [], [], [], []
        for projection in self.projections:
            if projection.bias is not None:
                raise ValueError("shared-expert overlap requires unbiased projections")
            if isinstance(projection, QuantizedProjection):
                values, scales, gain, fp4 = _weight_parts(projection.weight)
                if scales.dtype == torch.float8_e8m0fnu:
                    scales = scales.view(torch.uint8)
                kind = projection.weight.global_scale_kind if fp4 else "none"
            else:
                values, scales, gain, kind = projection.weight, None, None, "none"
            self.weights.append(values)
            self.scales.append(scales)
            self.gains.append(gain)
            self.scale_kinds.append(("none", "multiplier", "reciprocal").index(kind))
            for main_plan in self.parallel_plans:
                workspace.register_parallel(main_plan, projection.plan)
        self.plans = [projection.plan.handle for projection in self.projections] + [mlp.activation_plan.handle]
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        shared_bytes = max(projection.plan.memory_requirements().scratch_nbytes for projection in self.projections)
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((shared_bytes,), torch.uint8))

    def start(self, source):
        _start_shared_expert(source, self.weights, self.scales, self.gains, self.scale_kinds,
                            self.outputs, self.workspace, self.plans, self.mlp.swiglu_limit,
                            self.mlp.round_silu, self.stream_handle, self.defer_launch)

    def finish(self, source, routed):
        _join_shared_expert(source, routed, self.weights, self.scales, self.gains, self.scale_kinds,
                           self.outputs, self.workspace, self.plans, self.mlp.swiglu_limit,
                           self.mlp.round_silu, self.stream_handle, self.defer_launch)
        return self.mlp.down_projection.output
