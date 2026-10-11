"""Fused sigmoid-gated RMSNorm against the canonical FP32 computation."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("heads", [16, 32])
def test_gated_norm_matches_canonical_math_and_zeroes_inactive_rows(heads):
    from ignition.components.gated_norm import PreparedGatedRMSNorm

    generator = torch.Generator(device="cuda").manual_seed(heads)
    rows, dim = 300, 128
    hidden, gate = (torch.randn((rows, heads, dim), generator=generator, device="cuda").bfloat16() for _ in range(2))
    weight = torch.randn(dim, generator=generator, device="cuda").bfloat16()
    hidden[260:] = float("nan")
    output = torch.full_like(hidden, float("nan"))
    PreparedGatedRMSNorm(hidden, gate, weight, output, torch.tensor([250], dtype=torch.int32, device="cuda"),
                         eps=1e-6)()
    x = hidden[:250].float()
    expected = weight.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * torch.sigmoid(gate[:250].float())
    torch.testing.assert_close(output[:250], expected.bfloat16(), rtol=0, atol=4e-3)
    assert (output[:250] == expected.bfloat16()).float().mean() > 0.999
    assert not output[250:].any()
