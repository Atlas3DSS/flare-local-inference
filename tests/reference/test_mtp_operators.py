import copy

import pytest
import torch

from qualify.mtp_operators import (
    REPRODUCTION, STAGES, _input_records, _operands, common_rope_input,
    compare, load_inputs, padded, tensor_record, validate_inputs,
)
from qualify.provenance import file_record


def sample(rows=5):
    generator = torch.Generator().manual_seed(4149)
    saved = {'step': 0, 'stages': {
        name: torch.randn(rows, width, generator=generator).bfloat16()
        for name, width in STAGES.items()},
        'attention_constants': {name: torch.randn(256, generator=generator).bfloat16()
                                for name in ('q_norm', 'k_norm')}}
    case = {'step': 0, 'positions': torch.arange(rows), 'token_ids': torch.arange(rows),
            'is_prefilling': True}
    return saved, case


def test_actual_projection_strides_and_padding_preserve_every_live_value():
    saved, case = sample()
    x = _operands(saved, case, 8, 'cpu', vllm=True, eps=1e-6)
    assert x['query_raw'].stride() == (13312, 512, 1)
    assert x['key_raw'].stride() == (13312, 256, 1)
    assert x['attn_projected_down'].stride() == (336, 1)
    for name in STAGES:
        assert torch.equal(x[name][:5], saved['stages'][name])
        assert torch.count_nonzero(x[name][5:]) == 0
    sg = _operands(saved, case, 8, 'cpu', vllm=False, eps=1e-6)
    assert sg['query_raw'].stride() == (12288, 512, 1)
    assert sg['attn_projected_down'].stride() == (320, 1)
    assert torch.equal(sg['q_rope_input'], x['q_rope_input'])
    assert torch.equal(sg['k_rope_input'], x['k_rope_input'])
    assert not torch.cuda.is_initialized()


def test_cpu_rope_stimulus_excludes_gate_and_does_not_modify_operand():
    saved, _ = sample(2)
    query = saved['stages']['q_projected']
    weight = saved['attention_constants']['q_norm']
    original = query.clone()
    before = common_rope_input(query, weight, query=True, eps=1e-6)
    changed = query.clone().view(2, 24, 512)
    changed[..., 256:] = 8192
    after = common_rope_input(changed.flatten(1), weight, query=True, eps=1e-6)
    assert torch.equal(before, after)
    assert torch.equal(query, original)
    assert before.shape == (2, 6144) and before.dtype == torch.bfloat16


@pytest.mark.parametrize('change,match', [
    (lambda x: x.update(step=1), 'step'),
    (lambda x: x['stages'].pop('attn_mix'), 'attn_mix'),
    (lambda x: x['stages'].__setitem__('feedback', torch.zeros(4, 10240, dtype=torch.bfloat16)), 'feedback'),
    (lambda x: x['stages'].__setitem__('feedback', x['stages']['feedback'].float()), 'feedback'),
    (lambda x: x['stages']['k_projected'].__setitem__((2, 13), float('nan')), 'k_projected'),
    (lambda x: x['attention_constants'].__setitem__('q_norm', torch.zeros(255, dtype=torch.bfloat16)), 'q_norm'),
])
def test_rejects_wrong_or_incomplete_recorded_operands(change, match):
    saved, case = sample()
    change(saved)
    with pytest.raises(ValueError, match=match):
        validate_inputs([saved], [case])


def test_fixture_identity_and_normalization_weights_are_checked(tmp_path):
    saved, case = sample()
    fixture = tmp_path / 'fixture.pt'
    torch.save({'cases': [case]}, fixture)
    for name in ('sparkgap', 'vllm'):
        torch.save([saved], tmp_path / f'{name}.pt')
    values, _ = load_inputs(tmp_path, fixture)
    assert len(values) == 2
    alternate = tmp_path / 'alternate.pt'
    torch.save({'cases': [{**case, 'token_ids': torch.arange(5) + 1}]}, alternate)
    with pytest.raises(ValueError, match='immutable MTP fixture'):
        load_inputs(tmp_path, alternate)
    saved['attention_constants']['q_norm'][1] += 1
    torch.save([saved], tmp_path / 'vllm.pt')
    with pytest.raises(ValueError, match='normalization weights differ'):
        load_inputs(tmp_path, fixture)


def test_padding_rejects_truncation_and_retains_signed_zero():
    value = torch.tensor([[0., -0.], [2., 3.]], dtype=torch.bfloat16)
    result = padded(value, 4, 'cpu', stride=(8, 1))
    assert tensor_record(result[:2]) == tensor_record(value)
    assert result.stride() == (8, 1)
    with pytest.raises(ValueError, match='smaller'):
        padded(value, 1, 'cpu')


def paired_probe(tmp_path):
    saved, case = sample()
    torch.save({'cases': [case]}, tmp_path / 'fixture.pt')
    for name in ('sparkgap', 'vllm'):
        torch.save([saved], tmp_path / f'{name}.pt')
    records = [file_record(tmp_path / name) for name in ('fixture.pt', 'sparkgap.pt', 'vllm.pt')]
    payloads = {}
    for backend in ('sparkgap', 'vllm'):
        cases = []
        for operand_source in ('sparkgap', 'vllm'):
            for capacity in ((8,) if backend == 'sparkgap' else (5, 8)):
                cases.append({'step': 0, 'operand_source': operand_source, 'capacity': capacity,
                    'live_rows': 5, 'inputs': _input_records(saved, case, eps=1e-6),
                    'outputs': {name: saved['stages'][stage].clone() for name, stage in REPRODUCTION.items()}})
        payloads[backend] = {'schema_version': 1, 'backend': backend, 'completed': True,
                            'operand_sources': records, 'cases': cases,
                            'weights': {'q_proj': {'sha256': 'same'}}}
    return payloads


def save_probe(tmp_path, payloads):
    for backend, payload in payloads.items():
        torch.save(payload, tmp_path / f'{backend}-operators.pt')


def test_comparison_retains_both_row_geometries_and_own_reproduction(tmp_path):
    payloads = paired_probe(tmp_path)
    payloads['sparkgap']['cases'][0]['outputs']['hc_down'][2, 13] += 1
    save_probe(tmp_path, payloads)
    result = compare(tmp_path, tmp_path)
    assert len(result['identical_live_inputs']) == 4
    assert len(result['recorded_own_input_reproduction']) == 2
    assert [x['vllm_capacity'] for x in result['identical_live_inputs']] == [5, 8, 5, 8]
    assert not result['identical_live_inputs'][0]['operators']['hc_down']['fp32_value_bits_equal']
    assert result['identical_live_inputs'][0]['operators']['hc_norm']['fp32_value_bits_equal']
    assert not result['recorded_own_input_reproduction'][0]['operators']['hc_down']['fp32_value_bits_equal']
    assert result['acceptance_threshold'] is None


@pytest.mark.parametrize('kind,match', [
    ('incomplete', 'incomplete'), ('input_change', 'identical live input'),
    ('missing_operator', 'coverage'), ('source_change', 'inputs changed'),
])
def test_comparison_fails_closed_on_incomparable_payloads(tmp_path, kind, match):
    payloads = paired_probe(tmp_path)
    value = payloads['vllm']
    if kind == 'incomplete':
        value['completed'] = False
    elif kind == 'input_change':
        value['cases'][0]['inputs']['positions']['sha256'] = 'different'
    elif kind == 'missing_operator':
        value['cases'][0]['outputs'].pop('k_projection')
    else:
        value['operand_sources'] = copy.deepcopy(value['operand_sources'])
        value['operand_sources'][0]['sha256'] = 'different'
    save_probe(tmp_path, payloads)
    with pytest.raises(ValueError, match=match):
        compare(tmp_path, tmp_path)
