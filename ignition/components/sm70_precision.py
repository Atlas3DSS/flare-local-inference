"""BF16 arithmetic boundaries held in FP32 tensors on Volta.

These helpers never create a BF16 tensor. Integer rounding preserves BF16's
range, including values outside FP16, while keeping Inductor's graph FP32.
"""

import torch


def round_bf16(value):
    """Round FP32 to nearest-even BF16 and return its exact FP32 value."""
    if value.dtype != torch.float32:
        raise TypeError("software BF16 rounding requires FP32 storage")
    bits = value.view(torch.int32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    # Rounding a NaN payload must not turn it into infinity or overflow its sign.
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = torch.where(nan, (bits & -65536) | 0x00400000, rounded)
    return rounded.view(torch.float32)


def rms_norm(value, weight, epsilon):
    """Qwen3.5 zero-centered RMSNorm's FP32 reduction and final BF16 cast."""
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    return round_bf16(normalized * (1.0 + weight))


def apply_rotary(q, k, cos, sin):
    """Canonical partial RoPE, rounding each BF16 product and then the sum."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    width = cos.shape[-1]

    def rotate(value):
        active, rest = value[..., :width], value[..., width:]
        first, second = active.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        result = round_bf16(round_bf16(active * cos) + round_bf16(rotated * sin))
        return torch.cat((result, rest), dim=-1)

    return rotate(q), rotate(k)
