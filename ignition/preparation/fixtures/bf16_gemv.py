"""Private activation trials for BF16 projection backend selection."""

from b12x.preparation import PreparedCall


def benchmark_call(state, *, inputs, weight, bias):
    operands = inputs()
    return PreparedCall(
        run=lambda: state.run(operands.source, weight, out=operands.output, bias=bias),
        produce=operands.produce, output=operands.output,
        owners=(operands, weight, bias),
    )
