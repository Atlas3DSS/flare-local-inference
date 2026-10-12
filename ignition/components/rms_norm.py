"""Prepared RMS normalization with caller-owned output and strided input staging."""

from functools import partial

import torch
from b12x.norm.hyperconnection import api as hyperconnection
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures.hyperconnection import benchmark_call
from ignition.preparation.tuning import declare


class PreparedRMSNorm:
    def __init__(self, source, weight, output, *, eps, session, name, staging=None):
        if (source.ndim != 2 or source.shape != output.shape or weight.shape != source.shape[1:]
                or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                or weight.dtype not in (torch.bfloat16, torch.float32)
                or not output.is_contiguous() or not weight.is_contiguous()):
            raise ValueError("RMS normalization requires BF16 matrices and contiguous output and affine weights")
        self.staging = None if source.is_contiguous() else staging
        if not source.is_contiguous() and (staging is None or staging.shape != source.shape
                or staging.dtype != source.dtype or staging.device != source.device or not staging.is_contiguous()):
            raise ValueError("strided RMS normalization requires caller-owned contiguous input staging")
        self.weight, self.output, self.eps = weight, output, float(eps)
        self.plan = hyperconnection.plan(hyperconnection.Caps(device=source.device,
            max_tokens=source.shape[0], hidden_size=source.shape[1], streams=1, lowrank=1),
            invocation={"operation": "grouped_rmsnorm", "zero_centered": False, "eps": self.eps,
                        "weight_dtype": str(weight.dtype).removeprefix("torch.")})
        values = source if self.staging is None else self.staging
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(values, weight, output, eps=self.eps)),
            benchmark_call=partial(benchmark_call, operation="grouped_rmsnorm",
                arguments=(values, weight, output), kwargs={"eps": self.eps})),))

    def __call__(self, source):
        if self.staging is not None:
            self.staging.copy_(source)
            source = self.staging
        torch.ops.b12x.hyperconnection_grouped_rmsnorm(
            source, self.weight, self.output, self.eps, self.plan.handle, False)
        return self.output
