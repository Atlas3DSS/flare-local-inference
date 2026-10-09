"""Prepared native BF16 projections with caller-owned output storage."""

from __future__ import annotations

from dataclasses import replace

import torch
from b12x.gemm import bf16_gemv
from b12x.gemm import blockscaled
from b12x.preparation import PreparationSession, PreparedCall

from sparkgap.preparation.scratch import ScratchArena, scratch_storage, prime_owners
from sparkgap.preparation.tuning import register_finalizer


class BF16Projection:
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
        if source.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("Qwen BF16 projections require BF16 source and weight")
        if output.dtype != torch.bfloat16:
            raise TypeError("Qwen BF16 projections require BF16 output")
        query = bf16_gemv.query_from_call(source, weight, out=output, bias=bias)
        backend = "mma" if query.max_rows > 8 and query.out_features >= 256 else "simt"
        self.plan = bf16_gemv.plan(query, override=bf16_gemv.GemvConfig(backend=backend))
        self.weight = weight
        self.bias = bias
        self.output = output
        session.prepare((self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, weight, out=output, bias=bias)),
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
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.weight = weight
        self.output = output
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
                                                       out=output, workspace=storage),
                                owners=prime_owners(workspace, storage) if storage is not None else ())

        def benchmark(state):
            trial_source = torch.empty_strided(source.shape, source.stride(),
                                              dtype=source.dtype, device=source.device)
            trial_output = torch.empty_strided(output.shape, output.stride(),
                                              dtype=output.dtype, device=output.device)
            generator = torch.Generator(device=source.device).manual_seed(0xB12)
            seed = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                               generator=generator).mul_(0.125)
            storage = selected_workspace(state.required_workspace)

            def run():
                state.run(trial_source, values, scales, gain, out=trial_output, workspace=storage)
                if bias is not None:
                    trial_output.add_(bias)
                return trial_output

            return PreparedCall(run=run, produce=lambda: trial_source.copy_(seed),
                                output=trial_output, owners=(seed, trial_source, trial_output, storage, weight))

        session.prepare((self.plan.request(
            name=name,
            prepare_call=prepare,
            benchmark_call=benchmark,
        ),))
        self.workspace = selected_workspace()
        register_finalizer(session, lambda: setattr(self, "workspace", selected_workspace()), workspace=workspace)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        return blockscaled.mm(source, self.weight, out=self.output, plan=self.plan, bias=self.bias,
                              workspace=self.workspace)
