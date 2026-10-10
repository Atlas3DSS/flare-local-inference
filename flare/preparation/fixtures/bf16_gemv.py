"""Private activation trials for BF16 projection backend selection."""

import torch
from b12x.preparation import PreparedCall


def benchmark_call(state, *, source, weight, output, bias):
    trial_source = torch.empty_strided(source.shape, source.stride(),
                                      dtype=source.dtype, device=source.device)
    trial_output = torch.empty_strided(output.shape, output.stride(),
                                      dtype=output.dtype, device=output.device)
    generator = torch.Generator(device=source.device).manual_seed(0xB12)
    seed = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                       generator=generator).mul_(0.125)
    return PreparedCall(
        run=lambda: state.run(trial_source, weight, out=trial_output, bias=bias),
        produce=lambda: trial_source.copy_(seed), output=trial_output,
        owners=(seed, trial_source, trial_output, weight, bias),
    )
