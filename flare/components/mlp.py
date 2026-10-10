"""Prepared SwiGLU MLPs using b12x projections and caller-owned buffers."""

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall

from flare.components.linear import BF16Projection, QuantizedProjection
from flare.components.weights import MLPWeights


class PreparedMLP:
    """Dense or shared-expert MLP, including rank-local intermediate partitions.

    Gate/up projection rows are gate first, then up. The output is a partial
    hidden vector for sharded weights; the caller owns its collective reduction.
    All output and intermediate buffers must remain live through graph replay.
    """

    def __init__(self, source, weights: MLPWeights, gate_up, intermediate, output, *,
                 session, name, workspace, swiglu_limit=float("inf"), round_silu=True):
        self.gate_up, self.intermediate, self.output = gate_up, intermediate, output
        self.swiglu_limit, self.round_silu = swiglu_limit, round_silu
        if source.ndim != 2 or intermediate.ndim != 2:
            raise ValueError("MLP input and intermediate must be matrices")
        tokens, hidden = source.shape
        width = intermediate.shape[1]
        for tensor, shape in ((source, (tokens, hidden)), (gate_up, (tokens, 2 * width)),
                              (intermediate, (tokens, width)), (output, (tokens, hidden))):
            if (tensor.dtype != torch.bfloat16 or tensor.shape != shape
                    or tensor.device != source.device or not tensor.is_contiguous()):
                raise ValueError("MLP buffers require contiguous BF16 storage with matching geometry and device")

        def projection(input, weight, out, suffix):
            if isinstance(weight, torch.Tensor):
                return BF16Projection(input, weight, out, session=session, name=f"{name}.{suffix}")
            return QuantizedProjection(input, weight, out, session=session,
                                       name=f"{name}.{suffix}", workspace=workspace)

        self.up_projection = projection(source, weights.gate_up, gate_up, "gate_up")
        self.down_projection = projection(intermediate, weights.down, output, "down")
        self.activation_plan = hyperconnection.plan(
            hyperconnection.Caps(device=source.device, max_tokens=tokens, hidden_size=width),
            invocation={"operation": "swiglu", "limit": swiglu_limit, "round_silu": round_silu},
        )
        session.prepare((self.activation_plan.request(
            name=f"{name}.swiglu", prepare_call=lambda state: PreparedCall(
                run=lambda: state.launch(gate_up, gate_up, intermediate),
                owners=(gate_up, intermediate),
            ),
        ),))

    def __call__(self, source):
        self.up_projection(source)
        hyperconnection.run_swiglu(self.gate_up, limit=self.swiglu_limit,
                                  round_silu=self.round_silu, out=self.intermediate,
                                  plan=self.activation_plan)
        return self.down_projection(self.intermediate)
