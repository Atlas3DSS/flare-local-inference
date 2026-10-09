"""Exact token agreement rejects changed contexts and incomplete observations."""

from types import SimpleNamespace

import pytest

from qualify.generation_agreement import common_prefix, compare_cases
from qualify.vllm_generation import validate_completion


def test_first_divergence_and_complete_agreement():
    reference = list(range(64))
    assert common_prefix(reference, reference) == 64
    for position in (0, 1, 31, 63):
        actual = reference.copy()
        actual[position] += 100
        assert common_prefix(actual, reference) == position


@pytest.mark.parametrize('tokens', [list(range(63)), list(range(65)),
    [True] * 64, [-1] * 64, [248320] * 64])
def test_incomplete_or_invalid_tokens_fail(tokens):
    with pytest.raises(ValueError):
        common_prefix(tokens, list(range(64)))


def fixtures():
    prompts = [{'id': str(i), 'category': 'chat' if i < 2 else 'tools', 'prompt_tokens': [3, i]}
               for i in range(4)]
    reference = {p['id']: dict(p, generated_tokens=list(range(64)), finish_reason='length') for p in prompts}
    actual = {p['id']: dict(reference[p['id']], generated_tokens=list(range(64))) for p in prompts}
    for prompt, divergence in zip(prompts, (0, 20, 40, 64)):
        if divergence < 64:
            actual[prompt['id']]['generated_tokens'][divergence] += 100
    return prompts, actual, reference


def test_metrics_use_leading_agreement_and_include_every_category():
    result = compare_cases(*fixtures())
    assert result['summary']['median_agreement_length'] == 30
    assert result['summary']['full_64_token_agreement_fraction'] == .25
    assert result['categories']['chat']['median_agreement_length'] == 10
    assert result['categories']['tools']['median_agreement_length'] == 52
    assert result['cases'][-1]['first_divergence_position'] is None


@pytest.mark.parametrize('change', ['missing', 'context', 'category', 'stopped'])
def test_no_prompt_can_be_dropped_or_reconditioned(change):
    prompts, actual, reference = fixtures()
    if change == 'missing':
        actual.pop('0')
    elif change == 'context':
        actual['0']['prompt_tokens'] = [4, 0]
    elif change == 'category':
        actual['0']['category'] = 'tools'
    else:
        actual['0']['finish_reason'] = 'stop'
    with pytest.raises(ValueError):
        compare_cases(prompts, actual, reference)


def test_vllm_observation_retains_actual_ids():
    prompt = {'id': 'case', 'category': 'chat', 'prompt_tokens': [42]}
    output = SimpleNamespace(prompt_token_ids=[42], outputs=[SimpleNamespace(
        token_ids=list(range(64)), finish_reason='length', stop_reason=None)])
    result = validate_completion(output, prompt)
    assert result['generated_tokens'] == list(range(64))
    output.prompt_token_ids = [43]
    with pytest.raises(ValueError, match='different prompt'):
        validate_completion(output, prompt)


@pytest.mark.parametrize('length,reason', [(63, 'length'), (64, 'stop'), (65, 'length')])
def test_vllm_observation_rejects_early_or_wrong_termination(length, reason):
    prompt = {'id': 'case', 'category': 'chat', 'prompt_tokens': [42]}
    output = SimpleNamespace(prompt_token_ids=[42], outputs=[SimpleNamespace(
        token_ids=list(range(length)), finish_reason=reason, stop_reason=None)])
    with pytest.raises(ValueError, match='expected 64'):
        validate_completion(output, prompt)
