"""Diagnostic copies preserve inputs and observe delayed residual boundaries."""

from types import SimpleNamespace

import pytest
import torch

from qualify.vllm_layer_trace import LayerTrace, STAGE_WIDTHS


PROMPT = [760, 6511, 314, 9338, 369]
TOKENS = [11751, 13, 561, 6511]


def recorder():
    return LayerTrace(prompt_tokens=PROMPT, trace_tokens=TOKENS, device='cpu')


@pytest.mark.parametrize('id_dtype,position_dtype,axes', [
    (torch.int32, torch.int64, 1), (torch.int32, torch.int32, 3), (torch.int64, torch.int64, 1)])
def test_device_index_copies_compile_without_host_reads_and_preserve_operands(id_dtype, position_dtype, axes):
    trace = recorder()

    def observed(value, ids, positions):
        trace.begin(None, (), {'hidden_states': value, 'input_ids': ids, 'positions': positions,
            'prev_block_output': None, 'prev_injection': None,
            'query_start_loc': None, 'ngram_context': None})
        trace.record('layer.00', value)
        trace.counter.add_(1)
        return value

    compiled = torch.compile(observed, backend='eager', fullgraph=True)
    for step, rows in enumerate((8, 1, 1, 1)):
        value = (torch.arange(rows * 10240).reshape(rows, 10240) % 53).to(torch.bfloat16) + step
        saved = value.clone()
        ids = torch.tensor(PROMPT + [0] * 3 if step == 0 else [TOKENS[step - 1]], dtype=id_dtype)
        positions = torch.arange(rows, dtype=position_dtype) if step == 0 else torch.tensor([4 + step], dtype=position_dtype)
        if axes == 3:
            positions = positions.expand(3, -1)
        ids_before, positions_before = ids.clone(), positions.clone()
        assert compiled(value, ids, positions) is value
        assert torch.equal(value, saved)
        assert torch.equal(ids, ids_before) and torch.equal(positions, positions_before)
        count = min(rows, 5)
        assert torch.equal(trace.banks['layer.00'][step, :count], saved[:count].float())
        assert torch.equal(trace.input_ids[step, :count], ids[:count].long())
        assert torch.equal(trace.positions[step, :axes, :count], positions.reshape(axes, -1)[:, :count].long())
    assert trace.counter.tolist() == [4]
    assert trace.hit_views['layer.00'].tolist() == [1, 1, 1, 1]
    assert trace.capacity_views['layer.00'].tolist() == [8, 1, 1, 1]
    assert not torch.cuda.is_initialized()


class HC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_size, self.workspace, self.use_combine = 1, object(), True
        self.input_mix_weight_down_block_inject = torch.nn.Identity()
        self.input_mix_weight_up = torch.nn.Identity()

    def combine(self, result, *args):
        return result

    def combine_and_mix(self, result, *args):
        return result

    def mix(self, result):
        return result

    def _mix_normalized(self, result):
        return result


class Attention(torch.nn.Identity):
    def __init__(self):
        super().__init__()
        self.overlap_input_projections = False
        self.in_proj_qkvz = torch.nn.Identity()
        self.in_proj_ba = torch.nn.Identity()
        self.out_proj = torch.nn.Identity()


def test_delayed_boundary_copies_actual_return_before_ple_and_final_mixer():
    trace = recorder()
    layers = torch.nn.ModuleList()
    for i in range(48):
        layer = torch.nn.Module()
        layer.attn_hyper_connection, layer.mlp_hyper_connection = HC(), HC()
        layer.ple = torch.nn.Identity() if i == 1 else None
        layer.linear_attn, layer.mlp = Attention(), torch.nn.Identity()
        layer.mlp.experts = SimpleNamespace(
            router=SimpleNamespace(select_experts=lambda **kwargs: None),
            _apply_quant_method=lambda **kwargs: None)
        layers.append(layer)
    final = HC()
    language = SimpleNamespace(layers=layers, start_layer=0, end_layer=48, hyper_connection_mixer=final)
    trace.install(SimpleNamespace(language_model=SimpleNamespace(model=language)))
    source = torch.full((5, 10240), 19, dtype=torch.bfloat16)
    assert layers[1].attn_hyper_connection.combine(source, None, None) is source
    source.fill_(23)
    assert (trace.banks['layer.00'][0] == 19).all()
    assert (trace.banks['mlp_residual'][0] == 19).all()
    for i in range(2, 48):
        value = torch.full((5, 10240), i, dtype=torch.bfloat16)
        result = (value, None, None)
        assert layers[i].attn_hyper_connection.combine_and_mix(result, None, None) is result
        assert torch.equal(trace.banks[f'layer.{i - 1:02d}'][0], value.float())
    value = torch.full((5, 10240), 48, dtype=torch.bfloat16)
    result = (value, value[:, :2560], None)
    assert final.combine_and_mix(result, None, None) is result
    assert torch.equal(trace.banks['layer.47'][0], value.float())
    assert trace.counter.tolist() == [1]
    assert trace.hits[:48, 0].tolist() == [1] * 48


def test_snapshot_rejects_missing_stage_or_changed_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    trace = recorder()
    trace.counter.fill_(4)
    trace.hits.fill_(1)
    trace.capacities.fill_(512)
    trace.source_dtype_codes.zero_()
    trace.position_axes.fill_(1)
    for tensor in trace.banks.values():
        tensor.fill_(7)
    for step in range(4):
        ids = PROMPT if step == 0 else TOKENS[step - 1:step]
        positions = list(range(5)) if step == 0 else [4 + step]
        trace.input_ids[step, :len(ids)] = torch.tensor(ids)
        trace.positions[step, 0, :len(ids)] = torch.tensor(positions)
    result = trace.snapshot()
    assert result['passed']
    assert [tuple(x.shape) for x in result['layer_outputs']] == [(48, 5, 10240)] + [(48, 1, 10240)] * 3
    assert all(set(stages) == set(STAGE_WIDTHS) for stages in result['stage_outputs'])
    trace.hit_views['layer.00'][1] = 0
    trace.input_ids[2, 0] += 1
    result = trace.snapshot()
    assert not result['passed']
    assert not result['checks']['each_stage_once_per_step']
    assert not result['checks']['step2_input_ids']


@pytest.mark.parametrize('id_dtype', [torch.int32, torch.int64])
def test_moe_copies_actual_router_and_finalized_outputs_without_replacing_them(id_dtype, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    trace = recorder()
    rows = 8
    logits = torch.arange(rows * 512, dtype=torch.bfloat16).reshape(rows, 512)
    ids = torch.arange(rows * 10, dtype=id_dtype).reshape(rows, 10) % 512
    weights = torch.arange(rows * 10, dtype=torch.float32).reshape(rows, 10) / 100
    shared = torch.full((rows, 2560), 3, dtype=torch.bfloat16)
    routed = torch.full_like(shared, 7)
    inputs = torch.full_like(shared, 11)
    route_result, output_result = (weights, ids), (shared, routed)
    runner = SimpleNamespace(router=SimpleNamespace(select_experts=lambda **kwargs: route_result),
        _apply_quant_method=lambda **kwargs: output_result)
    trace.install_moe(runner)

    def observed():
        selected = runner.router.select_experts(hidden_states=inputs, router_logits=logits,
            topk_indices_dtype=id_dtype, input_ids=None)
        result = runner._apply_quant_method(hidden_states=inputs, router_logits=logits,
            shared_experts_input=inputs, input_ids=None, shared_experts_overlapping=True)
        return selected, result

    selected, result = observed()
    assert selected is route_result and result is output_result
    originals = [x.clone() for x in (logits, weights, ids, shared, routed, inputs)]
    compiled = torch.compile(observed, backend='eager', fullgraph=True)
    compiled()
    for name, source in [('moe_router_logits', logits), ('moe_weights', weights),
                         ('moe_ids', ids), ('moe_shared', shared), ('moe_routed', routed)]:
        bank = trace.banks[name]
        assert bank.dtype == (torch.int64 if name == 'moe_ids' else torch.float32)
        assert torch.equal(bank[0], source[:5].to(bank.dtype))
        assert trace.hit_views[name].tolist() == [2, 0, 0, 0]
        assert trace.capacity_views[name][0].item() == rows
    assert all(torch.equal(a, b) for a, b in zip(originals,
        (logits, weights, ids, shared, routed, inputs), strict=True))
    snapshot = trace.snapshot()
    assert snapshot['stage_outputs'][0]['moe_ids'].dtype == torch.int64
    assert snapshot['stage_source_dtypes'][0]['moe_ids'] == str(id_dtype)
    assert snapshot['stage_storage_dtype']['moe_ids'] == 'torch.int64'
    trace.reset()
    assert (trace.banks['moe_ids'] == -1).all()
    assert torch.isnan(trace.banks['moe_routed']).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bad', [None, object(), (torch.ones(1, 2560), object()),
    (torch.ones(1, 2560), torch.ones(2, 2560))])
def test_moe_observer_rejects_missing_or_deferred_output(bad):
    trace = recorder()
    runner = SimpleNamespace(router=SimpleNamespace(select_experts=lambda **kwargs: None),
        _apply_quant_method=lambda **kwargs: bad)
    trace.install_moe(runner)
    with pytest.raises(TypeError, match='MoE'):
        runner._apply_quant_method(hidden_states=torch.ones(1, 2560))


def test_moe_id_bank_rejects_floating_ids_and_preserves_integer_values():
    trace = recorder()
    with pytest.raises(TypeError, match='moe_ids'):
        trace.record('moe_ids', torch.ones(1, 10))
    values = torch.full((1, 10), 2**53 + 1, dtype=torch.int64)
    trace.record('moe_ids', values)
    assert torch.equal(trace.banks['moe_ids'][0, :1], values)
