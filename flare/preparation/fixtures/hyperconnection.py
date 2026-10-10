"""Private operands for hyperconnection candidate measurements."""

import torch
from b12x.preparation import PreparedCall


class HyperconnectionBenchmarkInputs:
    """Private finite activations with the canonical RMS normalization gain."""

    def __init__(self, operation, arguments):
        weight_index = 1 if operation == "grouped_rmsnorm" else None
        self.arguments = tuple(
            value if index == weight_index else torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device)
            for index, value in enumerate(arguments)
        )
        generator = torch.Generator(device="cpu").manual_seed(65537)
        self.inputs = []
        for index, target in enumerate(self.arguments[:-1]):
            if index == weight_index:
                continue
            fixture = torch.randn(target.shape, generator=generator)
            self.inputs.append((target, fixture.to(device=target.device, dtype=target.dtype)))
        self.arguments[-1].zero_()

    def produce(self):
        for target, fixture in self.inputs:
            target.copy_(fixture)


def benchmark_call(state, operation, arguments, kwargs):
    operands = HyperconnectionBenchmarkInputs(operation, arguments)
    return PreparedCall(
        run=lambda: state.launch(*operands.arguments, **kwargs),
        output=operands.arguments[-1],
        produce=operands.produce,
        owners=(operands,),
    )
