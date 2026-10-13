"""Paired RoPE with YaRN frequencies for compressed latent attention."""

import math

import torch


class CompressedRotary:
    def __init__(self, config, device):
        text = config.text
        scaling = text["rope_scaling"]
        self.frequencies = {}
        dims = torch.arange(32, device=device, dtype=torch.float32)
        for compressed in (False, True):
            base = text["compress_rope_theta"] if compressed else text["rope_theta"]
            frequencies = 1 / base ** (dims / 32)
            if compressed:
                original = scaling["original_max_position_embeddings"]

                def boundary(rotations, original=original, base=base):
                    return 64 * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

                low = max(math.floor(boundary(scaling["beta_fast"])), 0)
                high = min(math.ceil(boundary(scaling["beta_slow"])), 63)
                smooth = 1 - ((dims - low) / max(high - low, 1e-3)).clamp(0, 1)
                frequencies = frequencies / scaling["factor"] * (1 - smooth) + frequencies * smooth
            self.frequencies[compressed] = frequencies

    def apply(self, tensor, positions, *, compressed, inverse=False):
        angles = positions.float()[:, None] * self.frequencies[compressed][None]
        cos, sin = angles.cos(), angles.sin() * (-1 if inverse else 1)
        tail = tensor[..., -64:].float().unflatten(-1, (32, 2))
        if tensor.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        a, b = tail[..., 0], tail[..., 1]
        tensor[..., -64:].copy_(torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2))
