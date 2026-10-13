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
    # One BF16 ulp of rounding difference is allowed at any magnitude.
    torch.testing.assert_close(output[:250], expected.bfloat16(), rtol=2**-7, atol=4e-3)
    assert (output[:250] == expected.bfloat16()).float().mean() > 0.999
    assert not output[250:].any()


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_rounded_gated_norm_matches_a_bf16_module_with_a_strided_gate(activation, weight_dtype):
    from ignition.components.gated_norm import PreparedGatedRMSNorm

    generator = torch.Generator(device="cuda").manual_seed(48)
    rows, heads, dim = 300, 48, 128
    hidden = torch.randn((rows, heads, dim), generator=generator, device="cuda").bfloat16()
    packed = torch.randn((rows, 16384), generator=generator, device="cuda").bfloat16()
    gate = packed[:, 10240:].view(rows, heads, dim)
    weight = (1 + 0.3 * torch.randn(dim, generator=generator, device="cuda")).to(weight_dtype)
    output = torch.full_like(hidden, float("nan"))
    PreparedGatedRMSNorm(hidden, gate, weight, output, torch.tensor([250], dtype=torch.int32, device="cuda"),
                         eps=1e-6, round_normalized=True, activation=activation)()
    # Qwen's RMSNormGated: normalize in FP32, round to the BF16 input dtype, weight, then gate in FP32.
    x = hidden[:250].float()
    weighted = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()
    g = gate[:250].float()
    expected = (weighted * (g * torch.sigmoid(g) if activation == "silu" else torch.sigmoid(g))).bfloat16()
    # The rounded intermediate can carry one ulp into the weighted product: allow two.
    torch.testing.assert_close(output[:250], expected, rtol=2**-6, atol=4e-3)
    assert (output[:250] == expected).float().mean() > 0.9999
    assert not output[250:].any()
