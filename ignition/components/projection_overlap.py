"""Capture a chain of MXFP8 or MXFP4 projections alongside independent main-stream work."""

import torch

from ignition.components.linear import QuantizedProjection
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


@torch.library.custom_op("flare::start_projection_chain", mutates_args=("outputs", "workspace"))
def _start(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor],
           outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], stream: int) -> None:
    from b12x.gemm.blockscaled._ops import _out_workspace

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        value = source
        for weight, scale, output, plan in zip(weights, scales, outputs, plans, strict=True):
            _out_workspace(value, weight, scale, None, None, output, workspace, "none", None, plan, "any")
            value = output


@_start.register_fake
def _fake_start(source, weights, scales, outputs, workspace, plans, stream):
    pass


@torch.library.custom_op("flare::join_projection_chain", mutates_args=("outputs", "workspace"))
def _join(source: torch.Tensor, dependency: torch.Tensor, outputs: list[torch.Tensor],
          workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join.register_fake
def _fake_join(source, dependency, outputs, workspace, stream):
    pass


class ConcurrentProjectionChain:
    """Run dependent projections on an auxiliary stream with one scratch suffix.

    Each projection consumes its predecessor's complete output. The recipe
    declares every concurrent main-stream plan, starts the chain, and joins
    before consuming its output or reusing source and scratch storage.
    """

    def __init__(self, projections, *, parallel_plans, session, workspace: ScratchArena):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.projections, self.parallel_plans = tuple(projections), tuple(parallel_plans)
        if not isinstance(workspace, ScratchArena) or not self.projections or not self.parallel_plans:
            raise ValueError("projection overlap requires an arena, a projection chain, and concurrent plans")
        self.arena = workspace
        self.weights, self.scales, self.outputs, self.plans = [], [], [], []
        for projection in self.projections:
            if not isinstance(projection, QuantizedProjection) or projection.bias is not None:
                raise ValueError("projection overlap requires unbiased MXFP8 or MXFP4 projections")
            values, scales, gain, fp4 = _weight_parts(projection.weight)
            if values.dtype not in (torch.float8_e4m3fn, torch.uint8) or gain is not None or fp4:
                raise ValueError("projection overlap requires MXFP8 or MXFP4 weights")
            if self.outputs and self.outputs[-1].shape[1] != projection.weight.in_features:
                raise ValueError("projection chain output width must match the following input width")
            self.weights.append(values)
            self.scales.append(scales.view(torch.uint8))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
            for main in self.parallel_plans:
                workspace.register_parallel(main, projection.plan)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in self.projections)
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def start(self, source):
        _start(source, self.weights, self.scales, self.outputs, self.workspace, self.plans, self.stream_handle)

    def finish(self, source, dependency):
        _join(source, dependency, self.outputs, self.workspace, self.stream_handle)
        return self.outputs[-1]
