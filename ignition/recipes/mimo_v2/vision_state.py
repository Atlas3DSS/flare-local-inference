"""Generation-scoped image features injected into MiMo's causal text prefill."""

from dataclasses import dataclass

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication

from .vision import FEATURES, HIDDEN


@dataclass(frozen=True)
class PrefillVisionInputs:
    vision_token_indices: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device):
        return cls(torch.full((tokens,), -1, dtype=torch.int32, device=device))

    def buffers(self):
        return {"vision_token_indices": self.vision_token_indices}


class VisionState:
    def __init__(self, config, inputs, prefill, requests):
        self.inputs, self.prefill, self.requests = inputs, prefill, requests
        self.image_token = config["image_token_id"]
        device = inputs.token_ids.device
        self.arena = torch.zeros((requests.capacity, FEATURES, HIDDEN), dtype=torch.bfloat16, device=device)
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
        indices = self.prefill.vision_token_indices[:embedding.shape[0]].long().clamp(0, FEATURES - 1)
        embedding.copy_(torch.where((metadata.token_ids == self.image_token)[:, None],
                                    self.arena[slot, indices], embedding))


class PreparedVisionGeneration:
    def __init__(self, encoder, state, records, frame, sequence):
        self.encoder, self.state = encoder, state
        self.errors = torch.zeros(1, dtype=torch.int32, device=records.device)
        self.publish = PreparedFeaturePublication(state, encoder.inputs, records, frame, sequence)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = e.scratch.patches.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features, offset = i.vision_num_patches, i.vision_num_features, i.vision_feature_offset
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (count == features * 4)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        patch_live = torch.arange(capacity, device=count.device) < count
        feature_ids = torch.arange(capacity // 4, device=count.device)
        feature_live = feature_ids < features
        columns, rows = i.vision_column_indices[:capacity // 4], i.vision_row_indices[:capacity // 4]
        valid &= (~feature_live | ((columns >= 0) & (columns < features) & (rows >= 0) & (rows < features))).all()
        valid &= (~feature_live | (rows[columns.long().clamp(0, capacity // 4 - 1)] == feature_ids)).all()
        valid &= (~patch_live[:, None] | torch.isfinite(i.vision_pixel_values[:capacity])).all()
        positions = i.vision_position_ids[:capacity]
        valid &= (~patch_live[:, None] | ((positions >= 0) & (positions < capacity))).all()
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & patch_live[:, None], i.vision_pixel_values[:capacity], 0))
        e.positions.copy_(torch.where(valid & patch_live[:, None], positions, 0))
        e.columns.copy_(torch.where(valid & feature_live, columns, feature_ids))
        e.rows.copy_(torch.where(valid & feature_live, rows, feature_ids))
        e.cu.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        e.num_patches.copy_(torch.where(valid, count, 0))
        e.num_features.copy_(torch.where(valid, features, 0))
        output = e()
        self.errors.copy_(self.errors | (~((~feature_live[:, None]) | torch.isfinite(output)).all()).int() * 64)
        self.publish(output, e.num_features, self.errors)
