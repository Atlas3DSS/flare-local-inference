"""The sampler measurement oracle follows live state across page boundaries."""

import torch

from scripts.qualify_sampling_paths import SLOTS, cpu_case


def test_cpu_sampler_oracle_crosses_page_boundary_without_rolling_state():
    initialized = torch.cuda.is_initialized()
    logits = torch.full((248320,), -torch.inf, dtype=torch.bfloat16)
    logits[42] = 1
    case = cpu_case({'prompt': [17] * 63, 'tokens': [42], 'logits': [logits]},
                    rows=1, block_steps=3, policy='checkpoint_defaults')
    expected = case['expected']
    slot = SLOTS[0]
    assert expected['state.consumed_tokens'][slot].item() == 66
    assert expected['state.ple_history'][slot].tolist() == [42, 42]
    assert expected['sampled_ids'][:, slot].tolist() == [42, 42, 42]
    assert not any('boundary' in name for name in expected)
    assert torch.cuda.is_initialized() == initialized
