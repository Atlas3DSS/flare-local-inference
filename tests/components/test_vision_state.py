"""Image feature delivery remains scoped to request generations and token positions."""

from types import SimpleNamespace

import torch

from ignition.components.vision_state import PrefillVisionInputs, VisionState


def test_prefill_features_require_matching_generation_and_bounds():
    inputs = SimpleNamespace(token_ids=torch.tensor([7, 99, 99, 0]),
                             row_slot_ids=torch.tensor([1]), row_generations=torch.tensor([12]))
    prefill = PrefillVisionInputs.allocate(4, "cpu")
    state = VisionState(inputs, prefill, SimpleNamespace(capacity=2),
                        image_token=99, features=8, hidden=4)
    state.generation[1] = 12
    state.count[1] = 2
    prefill.vision_token_indices.copy_(torch.tensor([-1, 0, 1, -1]))
    lengths = torch.tensor([3])
    assert state.validate_prefill(inputs.token_ids, lengths)
    state.generation[1] = 11
    assert not state.validate_prefill(inputs.token_ids, lengths)
    state.generation[1] = 12
    prefill.vision_token_indices[2] = 2
    assert not state.validate_prefill(inputs.token_ids, lengths)
    prefill.vision_token_indices[2] = 1
    prefill.vision_token_indices[0] = 0
    assert not state.validate_prefill(inputs.token_ids, lengths)


def test_injection_preserves_text_and_selects_request_features():
    inputs = SimpleNamespace(token_ids=torch.tensor([7, 99, 99]),
                             row_slot_ids=torch.tensor([1]), row_generations=torch.tensor([12]))
    prefill = PrefillVisionInputs.allocate(3, "cpu")
    state = VisionState(inputs, prefill, SimpleNamespace(capacity=2),
                        image_token=99, features=8, hidden=4)
    state.arena[0].fill_(100)
    state.arena[1, 0].fill_(3)
    state.arena[1, 1].fill_(5)
    prefill.vision_token_indices.copy_(torch.tensor([-1, 1, 0]))
    embedding = torch.ones((3, 4), dtype=torch.bfloat16)
    state.inject(inputs, embedding)
    torch.testing.assert_close(embedding, torch.tensor([[1] * 4, [5] * 4, [3] * 4], dtype=torch.bfloat16))
