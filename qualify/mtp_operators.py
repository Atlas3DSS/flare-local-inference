"""Identical-input probes of the loaded MTP operators used by ``qualify.mtp``.

Saved operands are replayed independently at each operator boundary. These
measurements separate upstream input differences from operator differences;
they do not substitute an alternative forward pass or define a tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from qualify.provenance import file_record, plain, write_json


STAGES = {
    'feedback': 10240, 'attn_normalized': 10240, 'attn_projected_down': 320,
    'attn_bottleneck': 320, 'attn_gate_logits': 10240, 'attn_mix': 2560,
    'q_projected': 12288, 'k_projected': 512, 'v_projected': 512,
    'qsa_query': 6144, 'qsa_key': 512, 'qsa_gated': 6144, 'attn_output': 2560,
}
REPRODUCTION = {
    'hc_norm': 'attn_normalized', 'hc_down': 'attn_projected_down',
    'hc_silu': 'attn_bottleneck', 'hc_up': 'attn_gate_logits', 'hc_gate': 'attn_mix',
    'q_projection': 'q_projected', 'k_projection': 'k_projected',
    'v_projection': 'v_projected', 'q_norm_rope': 'qsa_query',
    'k_norm_rope': 'qsa_key', 'o_projection': 'attn_output',
}


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def load_inputs(directory, fixture):
    directory = Path(directory)
    if file_record(directory / 'fixture.pt')['sha256'] != file_record(fixture)['sha256']:
        raise ValueError('operator operands require the identical immutable MTP fixture')
    cases = torch.load(fixture, map_location='cpu', weights_only=True)['cases']
    sources = {}
    for backend in ('flare', 'vllm'):
        values = torch.load(directory / f'{backend}.pt', map_location='cpu', weights_only=True)
        validate_inputs(values, cases)
        sources[backend] = values
    for step in range(len(cases)):
        for name in ('q_norm', 'k_norm'):
            left = sources['flare'][step]['attention_constants'][name]
            right = sources['vllm'][step]['attention_constants'][name]
            if tensor_record(left) != tensor_record(right):
                raise ValueError(f'{name}: recorded normalization weights differ')
    return sources, [file_record(directory / name) for name in ('fixture.pt', 'flare.pt', 'vllm.pt')]


def validate_inputs(values, cases):
    if len(values) != len(cases):
        raise ValueError('operator operands must contain every fixture step')
    for saved, case in zip(values, cases, strict=True):
        if saved['step'] != case['step']:
            raise ValueError('operator operand step differs from fixture')
        rows = len(case['token_ids'])
        for name, width in STAGES.items():
            value = saved.get('stages', {}).get(name)
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.bfloat16 or value.shape != (rows, width)
                    or not torch.isfinite(value).all()):
                raise ValueError(f'{name}: expected finite CPU BF16 [{rows},{width}]')
        for name in ('q_norm', 'k_norm'):
            value = saved.get('attention_constants', {}).get(name)
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.shape != (256,) or value.dtype != torch.bfloat16
                    or not torch.isfinite(value).all()):
                raise ValueError(f'{name}: expected finite CPU BF16 normalization weight')


def common_rope_input(projected, weight, *, query, eps):
    """Construct an identical BF16 stimulus, not an expected RoPE output."""
    rows = projected.shape[0]
    value = projected.view(rows, 24, 512)[..., :256] if query else projected.view(rows, 2, 256)
    value = value.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (normalized * (1 + weight.float())).bfloat16().flatten(1)


def padded(value, capacity, device, *, stride=None):
    if capacity < value.shape[0]:
        raise ValueError('operator capacity is smaller than its live rows')
    shape = (capacity, *value.shape[1:])
    target = (torch.empty_strided(shape, stride, dtype=value.dtype, device=device)
              if stride is not None else torch.empty(shape, dtype=value.dtype, device=device))
    target.zero_()
    target[:value.shape[0]].copy_(value)
    return target


def _measure(actual, reference):
    from qualify.layer_comparison import _measurement
    result = _measurement(actual, reference)
    result['actual_storage_dtype'] = result.pop('flare_storage_dtype')
    result['reference_storage_dtype'] = result.pop('vllm_storage_dtype')
    return result


def _run(operations, rows, *, capture):
    buffers = {}
    def invoke():
        for name, operation in operations.items():
            result = operation()
            result = result.reshape(result.shape[0], -1)
            if name not in buffers:
                buffers[name] = torch.empty_like(result)
            buffers[name].copy_(result)
    invoke()
    torch.cuda.synchronize()
    first = {name: value[:rows].cpu().clone() for name, value in buffers.items()}
    graph = None
    try:
        if capture:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                invoke()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                invoke()
            run = graph.replay
        else:
            run = invoke
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()['allocation.all.allocated']
        run(); torch.cuda.synchronize()
        after = torch.cuda.memory_stats()['allocation.all.allocated']
        second = {name: value[:rows].cpu().clone() for name, value in buffers.items()}
        repeat = {name: _measure(second[name], first[name]) for name in first}
        return second, {'execution': 'CUDA graph replay' if capture else 'eager serving operators',
            'allocation_before': before, 'allocation_after': after,
            'replay_allocation_stable': before == after if capture else None,
            'repeat': repeat}
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()


def _operands(saved, case, capacity, device, *, vllm, eps):
    stages = saved['stages']
    result = {name: padded(value, capacity, device) for name, value in stages.items() if name in STAGES}
    result['attn_projected_down'] = padded(stages['attn_projected_down'], capacity, device, stride=(336, 1))
    if vllm:
        qkv = torch.zeros((capacity, 13312), dtype=torch.bfloat16, device=device)
        for target, name in zip(qkv.split((12288, 512, 512), dim=-1),
                                ('q_projected', 'k_projected', 'v_projected'), strict=True):
            target[:len(case['token_ids'])].copy_(stages[name])
        result['q_projected'], result['k_projected'], result['v_projected'] = qkv.split((12288, 512, 512), dim=-1)
    result['query_raw'] = result['q_projected'].view(capacity, 24, 512)[..., :256]
    result['key_raw'] = result['k_projected'].view(capacity, 2, 256)
    for name, query in (('q', True), ('k', False)):
        value = common_rope_input(stages[f'{name}_projected'], saved['attention_constants'][f'{name}_norm'],
                                  query=query, eps=eps)
        result[f'{name}_rope_input'] = padded(value, capacity, device)
    positions = torch.zeros(capacity, dtype=torch.int64, device=device)
    positions[:len(case['positions'])].copy_(case['positions'])
    result['positions'] = positions
    return result


def _input_records(saved, case, *, eps):
    values = {name: tensor_record(saved['stages'][name]) for name in STAGES}
    values['positions'] = tensor_record(case['positions'])
    for name, query in (('q', True), ('k', False)):
        values[f'{name}_rope_input'] = tensor_record(common_rope_input(saved['stages'][f'{name}_projected'],
            saved['attention_constants'][f'{name}_norm'], query=query, eps=eps))
    return values


def run_flare(args, owners, cases):
    from b12x.norm import hyperconnection as hc
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
    sources, records = load_inputs(args.operator_inputs, args.output / 'fixture.pt')
    results, plans = [], {}
    for case in cases:
        op = owners[0 if case['is_prefilling'] else 1][0]
        mixer, attention = op.attn_hc, op.attention
        rows, capacity = len(case['token_ids']), op.scratch.residual.shape[0]
        device, eps = op.scratch.residual.device, attention.module.q_norm.eps
        plans[str(capacity)] = {'hc': {name: plain(plan.selection) for name, plan in mixer.plans.items()},
            'projections': {name: plain(projection.plan.selection) for name, projection in
                {'hc_down': mixer.down, 'hc_up': mixer.up, **attention.projections}.items()}}
        for source, saved_cases in sources.items():
            saved = saved_cases[case['step']]
            x = _operands(saved, case, capacity, device, vllm=False, eps=eps)
            def rope(query, key):
                cos, sin = attention.rotary.gather(x['positions'][:, None].expand(-1, 3))
                return apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
            operations = {
                'hc_norm': lambda: hc.run_grouped_rmsnorm(x['feedback'], mixer.norm_gain, eps=mixer.eps,
                    zero_centered=False, binding=mixer.bindings['grouped_rmsnorm']),
                'hc_down': lambda: mixer.down(x['attn_normalized']),
                'hc_silu': lambda: hc.run_scaled_silu(x['attn_projected_down'], binding=mixer.bindings['scaled_silu']),
                'hc_up': lambda: mixer.up(x['attn_bottleneck']),
                'hc_gate': lambda: hc.run_gate_mean(x['attn_normalized'], x['attn_gate_logits'],
                                                   binding=mixer.bindings['gate_mean']),
                'q_projection': lambda: attention.projections['q_proj'](x['attn_mix']),
                'k_projection': lambda: attention.projections['k_proj'](x['attn_mix']),
                'v_projection': lambda: attention.projections['v_proj'](x['attn_mix']),
                'q_norm': lambda: attention.module.q_norm(x['query_raw']),
                'k_norm': lambda: attention.module.k_norm(x['key_raw']),
                'q_norm_rope': lambda: rope(attention.module.q_norm(x['query_raw']), attention.module.k_norm(x['key_raw']))[0],
                'k_norm_rope': lambda: rope(attention.module.q_norm(x['query_raw']), attention.module.k_norm(x['key_raw']))[1],
                'q_rope': lambda: rope(x['q_rope_input'].view(capacity, 24, 256), x['k_rope_input'].view(capacity, 2, 256))[0],
                'k_rope': lambda: rope(x['q_rope_input'].view(capacity, 24, 256), x['k_rope_input'].view(capacity, 2, 256))[1],
                'o_projection': lambda: attention.projections['o_proj'](x['qsa_gated']),
            }
            values, replay = _run(operations, rows, capture=True)
            results.append({'step': case['step'], 'operand_source': source, 'capacity': capacity,
                'live_rows': rows, 'inputs': _input_records(saved, case, eps=eps),
                'input_strides': {name: list(value.stride()) for name, value in x.items()},
                'outputs': values, 'replay': replay})
            _partial(args, 'flare', results, records)
    return _save(args, 'flare', results, records, plans, {
        'hc_down': tensor_record(mixer.down.weight), 'hc_up': tensor_record(mixer.up.weight),
        'hc_gain': tensor_record(mixer.norm_gain),
        **{name: tensor_record(projection.weight) for name, projection in attention.projections.items()}})


def run_vllm(args, model, cases):
    from b12x.norm import hyperconnection as hc
    sources, records = load_inputs(args.operator_inputs, args.output / 'fixture.pt')
    mixer, attention = model.model.layers[0].attn_hyper_connection, model.model.layers[0].self_attn
    device, eps = mixer.hc_norm.weight.device, attention.q_norm.variance_epsilon
    results = []
    for case in cases:
        rows = len(case['token_ids'])
        for capacity in sorted({rows, 8 if case['is_prefilling'] else 1}):
            for source, saved_cases in sources.items():
                saved = saved_cases[case['step']]
                x = _operands(saved, case, capacity, device, vllm=True, eps=eps)
                def projected():
                    value = attention.qkv_proj(x['attn_mix'])
                    value = value[0] if isinstance(value, tuple) else value
                    return value.split((12288, 512, 512), dim=-1)
                def rope(query, key):
                    return attention.rotary_emb(x['positions'][None].expand(3, -1).contiguous(),
                        query.flatten(1).clone(), key.flatten(1).clone())
                def output_projection():
                    value = attention.o_proj(x['qsa_gated'])
                    return value[0] if isinstance(value, tuple) else value
                operations = {
                    'hc_norm': lambda: hc.run_grouped_rmsnorm(x['feedback'], mixer.hc_norm.weight,
                        eps=mixer.config.rms_norm_eps, binding=mixer._binding(x['feedback'], 'grouped_rmsnorm')),
                    'hc_down': lambda: mixer.input_mix_weight_down_block_inject(x['attn_normalized'])[:, :320],
                    'hc_silu': lambda: hc.run_scaled_silu(x['attn_projected_down'],
                        binding=mixer._binding(x['attn_normalized'], 'scaled_silu')),
                    'hc_up': lambda: mixer.input_mix_weight_up(x['attn_bottleneck']),
                    'hc_gate': lambda: hc.run_gate_mean(x['attn_normalized'], x['attn_gate_logits'],
                        binding=mixer._binding(x['attn_normalized'], 'gate_mean')),
                    'q_projection': lambda: projected()[0], 'k_projection': lambda: projected()[1],
                    'v_projection': lambda: projected()[2],
                    'q_norm': lambda: attention.q_norm(x['query_raw']),
                    'k_norm': lambda: attention.k_norm(x['key_raw']),
                    'q_norm_rope': lambda: rope(attention.q_norm(x['query_raw']), attention.k_norm(x['key_raw']))[0],
                    'k_norm_rope': lambda: rope(attention.q_norm(x['query_raw']), attention.k_norm(x['key_raw']))[1],
                    'q_rope': lambda: rope(x['q_rope_input'], x['k_rope_input'])[0],
                    'k_rope': lambda: rope(x['q_rope_input'], x['k_rope_input'])[1],
                    'o_projection': output_projection,
                }
                values, replay = _run(operations, rows, capture=False)
                results.append({'step': case['step'], 'operand_source': source, 'capacity': capacity,
                    'live_rows': rows, 'inputs': _input_records(saved, case, eps=eps),
                    'input_strides': {name: list(value.stride()) for name, value in x.items()},
                    'outputs': values, 'replay': replay})
                _partial(args, 'vllm', results, records)
    q, k, v = attention.qkv_proj.weight.split((12288, 512, 512), dim=0)
    return _save(args, 'vllm', results, records,
        {'hc': {name: plain(mixer._plan_for(name).selection) for name in ('grouped_rmsnorm', 'scaled_silu', 'gate_mean')}},
        {'hc_down': tensor_record(mixer.input_mix_weight_down_block_inject.weight[:320]),
         'hc_up': tensor_record(mixer.input_mix_weight_up.weight), 'hc_raw_norm': tensor_record(mixer.hc_norm.weight),
         'q_proj': tensor_record(q), 'k_proj': tensor_record(k), 'v_proj': tensor_record(v),
         'o_proj': tensor_record(attention.o_proj.weight),
         'indexer.index_qk_proj': tensor_record(attention.indexer.index_qk_proj.weight)})


def _partial(args, backend, cases, records):
    torch.save({'schema_version': 1, 'backend': backend, 'completed': False,
                'cases': cases, 'operand_sources': records, 'script': file_record(__file__)},
               args.output / f'{backend}-operators.pt')


def _save(args, backend, cases, records, plans, weights):
    payload = {'schema_version': 1, 'backend': backend, 'completed': True, 'cases': cases,
               'operand_sources': records, 'plans': plans, 'weights': weights,
               'script': file_record(__file__), 'acceptance_threshold': None,
               'rope_stimulus': 'CPU FP32 grouped-independent RMS normalization of saved raw Q/K, BF16 rounded; identical input, not an expected output'}
    path = args.output / f'{backend}-operators.pt'
    torch.save(payload, path)
    allocation_checks = [case['replay']['replay_allocation_stable'] for case in cases
                         if case['replay']['replay_allocation_stable'] is not None]
    if allocation_checks and not all(allocation_checks):
        raise RuntimeError(f'{backend}: operator CUDA graph replay allocated; outputs retained at {path}')
    return {'result': file_record(path), 'script': payload['script'], 'case_count': len(cases),
            'all_repeats_value_exact': all(item['fp32_value_bits_equal'] for case in cases for item in case['replay']['repeat'].values()),
            'all_graph_replays_allocation_stable': all(allocation_checks) if allocation_checks else None}


def compare(directory, operands):
    directory, operands = Path(directory), Path(operands)
    left = torch.load(directory / 'flare-operators.pt', map_location='cpu', weights_only=True)
    right = torch.load(directory / 'vllm-operators.pt', map_location='cpu', weights_only=True)
    records = [file_record(operands / name) for name in ('fixture.pt', 'flare.pt', 'vllm.pt')]
    for name, payload in (('flare', left), ('vllm', right)):
        if (payload.get('schema_version') != 1 or payload.get('backend') != name
                or payload.get('completed') is not True or not payload.get('cases')):
            raise ValueError(f'{name}: incomplete operator probe')
        if payload['operand_sources'] != records:
            raise ValueError(f'{name}: recorded operator inputs changed')
    baseline = {name: torch.load(operands / f'{name}.pt', map_location='cpu', weights_only=True)
                for name in ('flare', 'vllm')}
    same_inputs, reproduction = [], []
    for actual in left['cases']:
        matches = [ref for ref in right['cases'] if ref['step'] == actual['step'] and ref['operand_source'] == actual['operand_source']]
        if not matches:
            raise ValueError('missing matching operator reference case')
        for reference in matches:
            if actual['inputs'] != reference['inputs']:
                raise ValueError('operator comparisons require exactly identical live input bytes')
            if actual['outputs'].keys() != reference['outputs'].keys():
                raise ValueError('operator output coverage differs')
            same_inputs.append({'step': actual['step'], 'operand_source': actual['operand_source'],
                'live_rows': actual['live_rows'], 'flare_capacity': actual['capacity'],
                'vllm_capacity': reference['capacity'], 'operators': {
                    name: _measure(value, reference['outputs'][name]) for name, value in actual['outputs'].items()}})
    for backend, payload in (('flare', left), ('vllm', right)):
        for case in payload['cases']:
            if case['operand_source'] != backend or (backend == 'vllm' and case['capacity'] != case['live_rows']):
                continue
            saved = baseline[backend][case['step']]['stages']
            reproduction.append({'backend': backend, 'step': case['step'], 'operators': {
                name: _measure(case['outputs'][name], saved[stage]) for name, stage in REPRODUCTION.items()}})
    return {'acceptance_threshold': None, 'identical_live_inputs': same_inputs,
            'recorded_own_input_reproduction': reproduction,
            'shared_weight_identity': {name: left['weights'][name] == right['weights'][name]
                for name in left['weights'].keys() & right['weights'].keys()},
            'sources': [file_record(directory / f'{name}-operators.pt') for name in ('flare', 'vllm')],
            'operator_script': file_record(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operands', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output / 'operator-comparison.json'
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, compare(args.output, args.operands))


if __name__ == '__main__':
    main()
