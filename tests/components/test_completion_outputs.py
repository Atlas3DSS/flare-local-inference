"""Completion storage remains shared across serial mixed graph variants."""

import pytest
import torch

from ignition.components.completion import (
    CompletionOutputs, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS, MAX_OUTPUT_TOKENS,
    validate_completion_buffers,
)


def test_mixed_roles_merge_only_the_active_frame_and_reuse_the_catalog_pool():
    outputs = CompletionOutputs.allocate((2, 4, 24), device='cpu')
    cursor = outputs.private_storage.cursor()
    prompt, decode = outputs.private(cursor), outputs.private(cursor)
    assert prompt.requested is decode.requested is outputs.requested
    prompt.records.fill_(11)
    prompt.logprobs.fill_(12)
    decode.records.fill_(21)
    decode.logprobs.fill_(22)
    outputs.records.fill_(-7)
    outputs.logprobs.fill_(-8)
    outputs.merge(prompt, decode, torch.tensor(2), torch.tensor([1], dtype=torch.int32))
    assert torch.all(outputs.records[0] == -7)
    assert torch.all(outputs.logprobs[0] == -8)
    assert torch.all(outputs.records[1, 2] == 11)
    assert torch.all(outputs.logprobs[1, 2] == 12)
    assert torch.all(outputs.records[1, [0, 1, 3]] == 21)
    assert torch.all(outputs.logprobs[1, [0, 1, 3]] == 22)
    size = outputs.private_storage.nbytes
    for _ in range(20):
        cursor = outputs.private_storage.cursor()
        a, b = outputs.private(cursor), outputs.private(cursor)
        assert a.logprobs.data_ptr() == prompt.logprobs.data_ptr()
        assert b.logprobs.data_ptr() == decode.logprobs.data_ptr()
    assert outputs.private_storage.nbytes == size
    assert outputs.zero_ranges()


@pytest.mark.parametrize('change', ['missing_output', 'missing_flags', 'wrong_dtype', 'wrong_frame', 'wrong_slots', 'wrong_k'])
def test_malformed_logprob_declarations_are_rejected(change):
    outputs = CompletionOutputs.allocate((2, 4, 24), device='meta')
    buffers = outputs.buffers()
    validate_completion_buffers(buffers, 4)
    if change == 'missing_output':
        del buffers['completion_logprobs']
    elif change == 'missing_flags':
        del buffers['logprob_requested']
    elif change == 'wrong_dtype':
        buffers['completion_logprobs'] = buffers['completion_logprobs'].float()
    else:
        shape = {
            'wrong_frame': (1, 4, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS),
            'wrong_slots': (2, 2, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS),
            'wrong_k': (2, 4, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS + LOGPROB_PAIR_FIELDS),
        }[change]
        buffers['completion_logprobs'] = torch.empty(shape, dtype=torch.int32, device='meta')
    with pytest.raises(ValueError) as raised:
        validate_completion_buffers(buffers, 4)
    message = str(raised.value)
    if change.startswith('missing_'):
        missing = 'completion_logprobs' if change == 'missing_output' else 'logprob_requested'
        assert missing in message
    else:
        assert 'completion_logprobs' in message
        assert 'expected shape' in message and 'dtype torch.int32' in message
        assert 'got shape' in message and 'dtype torch.' in message
    # Optional named buffers preserve the older export format.
    validate_completion_buffers({'completion_records': outputs.records}, 4)
