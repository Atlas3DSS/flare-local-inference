"""Bounded image feature delivery, publication, and prefill injection."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.sampling import _completion_fields, _write_completion

from .vision import FEATURES, HIDDEN


@dataclass(frozen=True)
class PrefillVisionInputs:
    vision_token_indices: torch.Tensor
    vision_left: torch.Tensor
    vision_right: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device):
        return cls(
            torch.full((tokens,), -1, dtype=torch.int32, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class VisionState:
    def __init__(self, weights, inputs, prefill, requests):
        self.inputs, self.prefill, self.requests = inputs, prefill, requests
        device = inputs.token_ids.device
        self.arena = torch.zeros((requests.capacity, FEATURES, HIDDEN), dtype=torch.bfloat16, device=device)
        self.generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.sentinels = torch.stack(
            [
                weights[name]
                for name in ("image_start", "image_pad", "image_pad", "image_newline", "image_end")
            ]
        )

    def visible(self, metadata):
        count = metadata.token_ids.numel()
        return self.prefill.vision_left[:count], self.prefill.vision_right[:count]

    def validate_prefill(self, tokens, lengths):
        b, p = self.inputs, self.prefill
        slot = b.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        index = p.vision_token_indices[: tokens.numel()]
        offsets = torch.arange(tokens.numel(), device=tokens.device)
        live, image = offsets < lengths[0], tokens >= 129280
        feature = tokens == 129282
        valid = (p.vision_left[: tokens.numel()] >= 0) & (p.vision_left[: tokens.numel()] <= offsets)
        valid &= (p.vision_right[: tokens.numel()] >= 0) & (
            p.vision_right[: tokens.numel()] + offsets < lengths[0]
        )
        valid &= (~feature & (index == -1)) | (feature & (index >= 0) & (index < self.count[slot]))
        valid &= ~image | (self.generation[slot] == b.row_generations[0])
        return (~live | valid).all()

    def inject(self, metadata, embedding):
        slot = self.inputs.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        tokens = metadata.token_ids
        index = self.prefill.vision_token_indices[: tokens.numel()].long().clamp(0, FEATURES - 1)
        value = torch.where(
            (tokens == 129282)[:, None],
            self.arena[slot, index],
            self.sentinels[(tokens - 129280).clamp(0, 4)],
        )
        embedding.copy_(torch.where((tokens >= 129280)[:, None], value, embedding))


@triton.jit
def _publish(Output, Arena, Slot, Offset, Count, Error, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (index < tl.load(Count) * 4096) & (tl.load(Error) == 0)
    values = tl.load(Output + index, valid, other=0)
    target = (tl.load(Slot).to(tl.int64) * 1536 + tl.load(Offset)) * 4096 + index
    tl.store(Arena + target, values, valid)


@triton.jit
def _finish(
    Slot,
    Epoch,
    Count,
    Offset,
    Error,
    FeatureEpoch,
    FeatureCount,
    Generation,
    Consumed,
    Generated,
    Status,
    StateError,
    Records,
    Frame,
    Sequence,
    CS: tl.constexpr,
    RECORD_FIELDS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    slot = tl.program_id(0)
    active = tl.load(Slot) == slot
    epoch = tl.load(Epoch)
    current = active & (epoch == tl.load(Generation + slot))
    error = tl.load(Error)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        if error == 0:
            tl.store(FeatureEpoch + slot, epoch)
            tl.store(FeatureCount + slot, tl.load(Offset) + tl.load(Count))
            status = 1
        else:
            status, code = 3, 128 + error
            if current & (tl.load(Status + slot) == 1):
                tl.store(Status + slot, status)
                tl.store(StateError + slot, code)
    consumed = tl.load(Consumed + slot * CS, current, other=0)
    generated = tl.load(Generated + slot, current, other=0)
    _write_completion(
        Records,
        Frame,
        Sequence,
        slot,
        epoch,
        consumed,
        generated,
        -1,
        status,
        code,
        active,
        RECORD_FIELDS,
        SLOTS,
    )


class PreparedVisionGeneration:
    def __init__(self, encoder, state, records, frame, sequence):
        self.encoder, self.state = encoder, state
        self.records, self.frame, self.sequence = records, frame, sequence
        self.errors = torch.zeros(1, dtype=torch.int32, device=records.device)
        self.record_fields = _completion_fields(records, records.device, state.requests.capacity)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = e.scratch.patches.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features = i.vision_num_patches, i.vision_num_features
        offset = i.vision_feature_offset
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (features > 0) & (features <= capacity // 9)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        patch_live = torch.arange(capacity, device=count.device) < count
        indices = i.vision_merge_indices[: capacity // 9]
        feature_live = torch.arange(capacity // 9, device=count.device) < features
        valid &= (~feature_live[:, None] | ((indices >= -1) & (indices < count))).all()
        valid &= (~patch_live[:, None] | torch.isfinite(i.vision_pixel_values[:capacity])).all()
        valid &= (
            ~patch_live[:, None]
            | ((i.vision_position_ids[:capacity] >= 0) & (i.vision_position_ids[:capacity] < capacity))
        ).all()
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & patch_live[:, None], i.vision_pixel_values[:capacity], 0))
        e.positions.copy_(torch.where(valid & patch_live[:, None], i.vision_position_ids[:capacity], 0))
        e.indices.copy_(torch.where(valid & feature_live[:, None], indices, -1))
        e.cu.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        e.num_patches.copy_(torch.where(valid, count, 0))
        e.num_features.copy_(torch.where(valid, features, 0))
        output = e()
        finite = (~feature_live[:, None] | torch.isfinite(output)).all()
        self.errors.copy_(self.errors | (~finite).int() * 64)
        torch.library.wrap_triton(_publish)[(triton.cdiv(output.numel(), 1024),)](
            output, s.arena, i.vision_slot, offset, e.num_features, self.errors, capacity, 1024
        )
        torch.library.wrap_triton(_finish)[(r.capacity,)](
            i.vision_slot,
            i.vision_generation,
            e.num_features,
            offset,
            self.errors,
            s.generation,
            s.count,
            r.generation,
            r.consumed_tokens,
            r.generated_tokens,
            r.status,
            r.error_code,
            self.records,
            self.frame,
            self.sequence,
            r.consumed_tokens.stride(0),
            self.record_fields,
            r.capacity,
        )
