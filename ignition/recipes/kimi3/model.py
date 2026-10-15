"""Kimi K3 residual composition and canonical normalization."""

import torch


def rms_norm(weight, eps):
    from transformers.models.kimi_linear.modeling_kimi_linear import KimiLinearRMSNorm

    with torch.device("meta"):
        module = KimiLinearRMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module


def attention_residual(prefix, blocks, norm_weight, projection, eps):
    values = (*blocks, prefix)
    weight = norm_weight.float() * projection.float()
    scores = []
    for value in values:
        value = value.float()
        variance = value.square().sum(-1, keepdim=True) / value.shape[-1]
        key = value * torch.rsqrt(variance + eps)
        scores.append((key * weight).sum(-1))
    probabilities = torch.stack(scores, dim=-1).softmax(-1)
    mixed = values[0].float() * probabilities[:, :1]
    for index, value in enumerate(values[1:], 1):
        mixed = torch.addcmul(mixed, value.float(), probabilities[:, index:index + 1])
    return mixed.to(prefix.dtype)


def situ(gate_up):
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (4. * (gate / 4.).tanh() * gate.sigmoid() * (25. * (up / 25.).tanh())).to(gate_up.dtype)
