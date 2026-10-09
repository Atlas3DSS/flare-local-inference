"""The parity reset records shared scratch storage once, including padding."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from qualify import graph_execution
from sparkgap.preparation.scratch import ScratchArena
from sparkgap.recipes.qwen38.vision_state import VisionFeatureState


@dataclass
class RequestFixture:
    epoch: torch.Tensor


@pytest.mark.parametrize("registered_view", [False, True])
def test_mutable_reset_retains_complete_arena_once(registered_view):
    arena = ScratchArena("cpu")
    arena.reserve(513)
    arena.tensor.copy_(torch.arange(arena.nbytes, dtype=torch.int64).to(torch.uint8))
    arena.freeze()
    features = VisionFeatureState(torch.empty(4, 2, 3), torch.zeros(4, dtype=torch.int64),
                                  torch.zeros(4, dtype=torch.int32))
    buffers = {"scratch_alias": arena.tensor[512:513]} if registered_view else {}
    model = SimpleNamespace(_resources=[features, arena], buffers=buffers)
    state = SimpleNamespace(backing=torch.empty((2, 16), dtype=torch.uint8))
    request = RequestFixture(torch.zeros(4, dtype=torch.int64))
    reset = graph_execution.MutableReset(model, state, request, [], [], {})
    key = arena.tensor.untyped_storage().data_ptr()
    name, raw, saved = reset.storage[key]
    assert name == ("input.scratch_alias" if registered_view else "workspace.arena")
    assert raw.storage_offset() == 0
    assert raw.shape == saved.shape == (768,)
    assert torch.equal(raw, arena.tensor) and torch.equal(saved, arena.tensor)
    assert sum(storage.untyped_storage().data_ptr() == key
               for _, storage, _ in reset.storage.values()) == 1
    assert state.backing.untyped_storage().data_ptr() not in reset.storage
    assert features.arena.untyped_storage().data_ptr() not in reset.storage
