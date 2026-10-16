"""Prepared gate projection and stream mean with caller-owned outputs."""

from b12x.gemm import hyperconnection_gate
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures.hyperconnection_gate import HyperconnectionGateInputs
from ignition.preparation.tuning import benchmark_resource, declare


class PreparedHyperconnectionGate:
    def __init__(self, source, weight, normalized, output, workspace, *, session, name):
        rows, rank = source.shape
        hidden = output.shape[1]
        streams, remainder = divmod(normalized.shape[1], hidden)
        if remainder:
            raise ValueError("normalized width must contain complete residual streams")
        self.plan = hyperconnection_gate.plan(hyperconnection_gate.Query(
            max_rows=rows, rank=rank, hidden_size=hidden, streams=streams))
        self.weight, self.output, self.workspace = weight, output, workspace

        def benchmark(state):
            inputs = benchmark_resource(session,
                ("hyperconnection_gate", str(source.device), rows, rank, hidden, streams),
                lambda: HyperconnectionGateInputs(source, normalized, output, workspace))
            return inputs.call(state, weight)

        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(source, weight, normalized, output, workspace)),
            benchmark_call=benchmark,
        ),))

    def __call__(self, source, normalized):
        return hyperconnection_gate.run(source, self.weight, normalized, out=self.output,
                                       workspace=self.workspace, plan=self.plan)
