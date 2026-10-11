"""Prepared b12x projections with caller-owned output and scratch storage."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch
from b12x.gemm import bf16_gemv, blockscaled
from b12x.preparation import PreparationSession, PreparedCall

from flare.preparation.fixtures import bf16_gemv as bf16_fixtures
from flare.preparation.fixtures import blockscaled_gemm as gemm_fixtures
from flare.preparation.scratch import ScratchArena, prime_owners, scratch_storage
from flare.preparation.tuning import declare, register_finalizer


class UnquantizedProjection:
    def __init__(
        self,
        source: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        *,
        session: PreparationSession,
        name: str,
        bias: torch.Tensor | None = None,
    ) -> None:
        if any(t.dtype not in (torch.bfloat16, torch.float32) for t in (source, weight, output)):
            raise TypeError("unquantized projections require BF16 or FP32 operands")
        query = bf16_gemv.query_from_call(source, weight, out=output, bias=bias)
        self.plan = bf16_gemv.plan(query)
        self.weight = weight
        self.bias = bias
        self.output = output
        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, weight, out=output, bias=bias)),
            benchmark_call=partial(bf16_fixtures.benchmark_call, source=source,
                                   weight=weight, output=output, bias=bias),
        ),))

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        return bf16_gemv.mm(source, self.weight, out=self.output, plan=self.plan, bias=self.bias)


class QuantizedProjection:
    def __init__(
        self,
        source: torch.Tensor,
        weight: blockscaled.Weight,
        output: torch.Tensor,
        *,
        session: PreparationSession,
        name: str,
        activation_mode: str = "quantized",
        bias: torch.Tensor | None = None,
        workspace: ScratchArena | torch.Tensor | None = None,
    ) -> None:
        from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight, _weight_parts

        self.weight = weight
        self.output = output
        self.activation_scale = getattr(weight, "activation_global_scale", None)
        if isinstance(weight, NVFP4LinearWeight) and self.activation_scale is None:
            activation_mode = "a16"
        self.activation_mode = activation_mode
        self.bias = bias
        self.workspace = workspace
        query = blockscaled.query_from_call(source, weight, out=output, activation_mode=activation_mode,
                                           workspace=None if isinstance(workspace, ScratchArena) else workspace)
        if isinstance(workspace, ScratchArena):
            query = replace(query, workspace_form="provided", workspace_nbytes=None)
        self.plan = blockscaled.plan(query)
        values, scales, gain, _ = _weight_parts(weight)

        def selected_workspace(nbytes=None):
            if workspace is None:
                return None
            if nbytes is None:
                nbytes = self.plan.memory_requirements().scratch_nbytes
            return scratch_storage(workspace, nbytes=nbytes, device=source.device)

        def prepare(state):
            storage = selected_workspace(state.required_workspace)
            return PreparedCall(run=lambda: state.run(source, values, scales, gain,
                                                       out=output, workspace=storage, activation_scale=self.activation_scale),
                                owners=prime_owners(workspace, storage) if storage is not None else ())

        declare(session, (self.plan.request(
            name=name,
            prepare_call=prepare,
            benchmark_call=partial(gemm_fixtures.benchmark_call, source=source, output=output,
                values=values, scales=scales, gain=gain, bias=bias, weight=weight,
                activation_scale=self.activation_scale,
                selected_workspace=selected_workspace),
        ),))
        register_finalizer(session, lambda: setattr(self, "workspace", selected_workspace()), workspace=workspace)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        return blockscaled.mm(source, self.weight, out=self.output, plan=self.plan, bias=self.bias,
                              workspace=self.workspace)


class BatchedQuantizedProjection:
    """Prepared BF16-by-rowwise-MXFP8 BMM over caller-owned strided tensors.

    The RHS remains packed E4M3 values and per-32 E8M0 scales. The native b12x
    kernel has a fixed configuration and requires no scratch. The major axis
    describes the RHS layout and the axis compressed by its scales.
    """

    def __init__(self, source, weight, output, *, major, session, name):
        from b12x import gemm
        from b12x.gemm._bmm._tuning import BmmQuery

        if source.ndim != 3 or output.ndim != 3 or source.shape[:2] != output.shape[:2]:
            raise ValueError("batched projections require matching batch and row dimensions")
        self.weight, self.output, self.major = weight, output, major
        self.plan = gemm.plan_bmm(BmmQuery(batch=source.shape[0], max_rows=source.shape[1],
            in_features=source.shape[2], out_features=output.shape[2], b_major=major, sf_axis=major))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, weight, output, b_major=major, sf_axis=major))),))

    def __call__(self, source):
        from b12x import gemm

        return gemm.bmm(source, self.weight, self.output, plan=self.plan,
            a_dtype="bfloat16", b_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32, b_major=self.major, sf_axis=self.major)


@torch.library.custom_op("flare::parallel_mx_projections", mutates_args=("outputs", "workspaces"))
def _parallel_mx_projections(
    source: torch.Tensor, values: list[torch.Tensor], scales: list[torch.Tensor],
    outputs: list[torch.Tensor], workspaces: list[torch.Tensor],
    plans: list[int], auxiliary_stream: int, join: bool = True,
) -> None:
    from b12x.gemm.blockscaled._ops import _out_workspace

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)

    def project(index):
        _out_workspace(
            source, values[index], scales[index], None, None, outputs[index],
            workspaces[index], "none", None, plans[index], "any")

    with torch.cuda.stream(auxiliary):
        project(1)
    project(0)
    if join:
        main.wait_stream(auxiliary)


@_parallel_mx_projections.register_fake
def _fake_parallel_mx_projections(source, values, scales, outputs, workspaces, plans, auxiliary_stream, join=True):
    pass


class ParallelProjections:
    """Two prepared MX projections with explicit storage and a captured join."""

    def __init__(self, main: QuantizedProjection, auxiliary: QuantizedProjection, *, session,
                 workspace: ScratchArena):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.projections = (main, auxiliary)
        self.arena = workspace
        self.values, self.scales, self.outputs, self.plans = [], [], [], []
        for projection in self.projections:
            values, scales, gain, fp4 = _weight_parts(projection.weight)
            if (values.dtype not in (torch.float8_e4m3fn, torch.uint8) or gain is not None or fp4
                    or projection.activation_mode != "quantized" or projection.bias is not None):
                raise ValueError("parallel projections require unbiased quantized MXFP8 or MXFP4 weights")
            self.values.append(values)
            self.scales.append(scales.view(torch.uint8))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
        workspace.register_parallel(*(projection.plan for projection in self.projections))
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        self.workspaces = list(self.arena.views(*(
            ((projection.plan.memory_requirements().scratch_nbytes,), torch.uint8)
            for projection in self.projections)))

    def __call__(self, source):
        _parallel_mx_projections(source, self.values, self.scales, self.outputs,
                                   self.workspaces, self.plans, self.stream_handle)
