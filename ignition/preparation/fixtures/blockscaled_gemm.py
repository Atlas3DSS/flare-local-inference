"""Finite strided activation trials for blockscaled GEMM preparation."""

import torch
from b12x.preparation import PreparedCall


def benchmark_call(state, *, source, output, values, scales, gain, bias, weight,
                   selected_workspace, activation_scale=None):
    trial_source = torch.empty_strided(source.shape, source.stride(),
                                      dtype=source.dtype, device=source.device)
    trial_output = torch.empty_strided(output.shape, output.stride(),
                                      dtype=output.dtype, device=output.device)
    generator = torch.Generator(device=source.device).manual_seed(0xB12)
    seed = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                       generator=generator).mul_(0.125)
    storage = selected_workspace(state.required_workspace)

    def run():
        state.run(trial_source, values, scales, gain, out=trial_output, workspace=storage,
                  activation_scale=activation_scale)
        if bias is not None:
            trial_output.add_(bias)
        return trial_output

    return PreparedCall(run=run, produce=lambda: trial_source.copy_(seed),
                        output=trial_output, owners=(seed, trial_source, trial_output, storage, weight))
