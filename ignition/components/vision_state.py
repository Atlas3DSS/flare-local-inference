"""Generation-scoped image features injected into causal text prefill."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillVisionInputs:
    vision_token_indices: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device):
        return cls(torch.full((tokens,), -1, dtype=torch.int32, device=device))

    def buffers(self):
        return {"vision_token_indices": self.vision_token_indices}


class VisionState:
    def __init__(self, inputs, prefill, requests, *, image_token, features, hidden):
        self.inputs, self.prefill, self.requests = inputs, prefill, requests
        self.image_token = image_token
        device = inputs.token_ids.device
        self.arena = torch.zeros((requests.capacity, features, hidden), dtype=torch.bfloat16, device=device)
        self.generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)

    def validate_prefill(self, tokens, lengths):
        b = self.inputs
        slot = b.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        index = self.prefill.vision_token_indices[:tokens.numel()]
        live = torch.arange(tokens.numel(), device=tokens.device) < lengths[0]
        image = tokens == self.image_token
        valid = (~image & (index == -1)) | (image & (index >= 0) & (index < self.count[slot]))
        valid &= ~image | (self.generation[slot] == b.row_generations[0])
        return (~live | valid).all()

    def inject(self, metadata, embedding):
        slot = self.inputs.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        indices = self.prefill.vision_token_indices[:embedding.shape[0]].long().clamp(0, self.arena.shape[1] - 1)
        embedding.copy_(torch.where((metadata.token_ids == self.image_token)[:, None],
                                    self.arena[slot, indices], embedding))
