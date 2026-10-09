import hashlib

import pytest
import torch

from qualify.replay_floor import _byte_view, _hash


@pytest.mark.parametrize('dtype', [torch.int64, torch.uint64, torch.float32, torch.bfloat16])
def test_single_request_strided_state_has_packed_byte_readback(dtype):
    value = torch.ones(1, dtype=dtype).as_strided((1,), (14456032,))
    packed = torch.ones(1, dtype=dtype).view(torch.uint8)
    assert torch.equal(_byte_view(value), packed)
    assert _hash(value) == hashlib.sha256(packed.numpy().tobytes()).hexdigest()


def test_state_comparison_distinguishes_signed_zero():
    assert torch.equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert not torch.equal(_byte_view(torch.tensor([0.])), _byte_view(torch.tensor([-0.])))


def test_state_readback_respects_noncontiguous_layout():
    value = torch.arange(12).reshape(3, 4).T
    assert torch.equal(_byte_view(value), torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]).view(torch.uint8))
