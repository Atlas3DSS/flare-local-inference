"""Prepared RMS normalization reading projection slices with explicit row strides."""

from functools import partial

import torch
from b12x.norm.hyperconnection import api as hyperconnection
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures.hyperconnection import benchmark_call
from ignition.preparation.tuning import declare


class PreparedRMSNorm:
    def __init__(self, source, weight, output, *, eps, session, name):
        if (source.ndim != 2 or source.shape != output.shape or weight.shape != source.shape[1:]
                or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                or weight.dtype not in (torch.bfloat16, torch.float32)
                or source.stride(1) != 1 or source.stride(0) < source.shape[1]
                or not output.is_contiguous() or not weight.is_contiguous()):
            raise ValueError("RMS normalization requires BF16 matrices and contiguous output and affine weights")
        self.weight, self.output, self.eps = weight, output, float(eps)
        self.plan = hyperconnection.plan(hyperconnection.Caps(device=source.device,
            max_tokens=source.shape[0], hidden_size=source.shape[1], streams=1, lowrank=1),
            invocation={"operation": "grouped_rmsnorm", "zero_centered": False, "eps": self.eps,
                        "weight_dtype": str(weight.dtype).removeprefix("torch.")})
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(source, weight, output, eps=self.eps)),
            benchmark_call=partial(benchmark_call, operation="grouped_rmsnorm",
                arguments=(source, weight, output), kwargs={"eps": self.eps})),))

    def __call__(self, source):
        torch.ops.b12x.hyperconnection_grouped_rmsnorm(
            source, self.weight, self.output, self.eps, self.plan.handle, False)
        return self.output
