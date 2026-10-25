"""Bounded diagnostic observations copied by retained prefill and ordinary B1 decode graphs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationCapture:
    token_indices: torch.Tensor
    token_enabled: torch.Tensor
    token_values: torch.Tensor
    residual_last: torch.Tensor
    pre_mlp_last: torch.Tensor
    mlp_input_last: torch.Tensor
    residual_l2: torch.Tensor
    positions: torch.Tensor
    last_index: torch.Tensor
    live: torch.Tensor

    @classmethod
    def allocate(cls, layers, tokens, hidden, device, *, intermediate):
        return cls(torch.zeros((layers, 32), dtype=torch.int64, device=device),
                   torch.zeros((layers, 32), dtype=torch.bool, device=device),
                   torch.zeros((layers, tokens, 32), dtype=torch.float32, device=device),
                   torch.zeros((layers, hidden), dtype=torch.float32, device=device),
                   torch.zeros((layers, hidden), dtype=torch.float32, device=device),
                   torch.zeros((layers, intermediate), dtype=torch.float32, device=device),
                   torch.zeros((layers, tokens), dtype=torch.float32, device=device),
                   torch.full((tokens,), -1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.bool, device=device))

    def rows(self, tokens):
        return type(self)(self.token_indices, self.token_enabled, self.token_values[:, :tokens], self.residual_last, self.pre_mlp_last, self.mlp_input_last, self.residual_l2[:, :tokens],
                          self.positions[:tokens], self.last_index, self.live)

    def begin(self, metadata):
        self.positions.copy_(metadata.positions)
        self.last_index.copy_((metadata.live_counts[:1] - 1).clamp(0, self.positions.numel() - 1))
        self.live.copy_(metadata.live_counts[:1] > 0)

    def record_pre_mlp(self, value, layer):
        self.pre_mlp_last[layer].copy_(torch.where(self.live, value[self.last_index].reshape(-1), 0))

    def record_mlp_input(self, value, layer):
        selected = value.index_select(-1, self.token_indices[layer])
        valid = (self.positions >= 0)[:, None] & self.token_enabled[layer][None, :]
        self.token_values[layer].copy_(torch.where(valid, selected, 0))
        self.mlp_input_last[layer].copy_(torch.where(self.live, value[self.last_index].reshape(-1), 0))

    def record_residual(self, value, layer):
        self.residual_last[layer].copy_(torch.where(self.live, value[self.last_index].reshape(-1), 0))
        self.residual_l2[layer].copy_(torch.where(self.positions >= 0, value.square().sum(-1).sqrt(), 0))
