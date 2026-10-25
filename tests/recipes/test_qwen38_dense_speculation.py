"""Dense MTP admission, memory geometry and native semantic contract."""
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.execution import GraphSignature
from ignition.preparation import runtime_package_pb2 as p
from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig
from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.contract import prepare
from ignition.recipes.qwen38_dense.pool import continuation_layout, page_bytes, page_specs
from ignition.recipes.qwen38_dense.recipe import RECIPE


def config():
    return ModelConfig(dict(hidden_size=5120, vocab_size=248320, num_key_value_heads=4,
        head_dim=256, layer_types=["full_attention" if i % 4 == 3 else "linear_attention"
                                  for i in range(64)]), {})


@pytest.mark.parametrize('depth,row_bytes', [(0, 165302528), (2, 170283264), (3, 172249344)])
def test_checkpoint_columns_and_memory(depth, row_bytes):
    model = config()
    layout = continuation_layout(model, dtype=torch.float32, element_bytes=2, draft_tokens=depth)
    assert layout.row_bytes == row_bytes
    recurrent = [r for r in layout.regions if r.name.endswith('.recurrent')]
    histories = [r for r in layout.regions if r.name.endswith('.convolution')]
    assert len(recurrent) == len(histories) == 48
    assert sum(r.nbytes for r in recurrent) == 144 << 20
    assert all(r.shape == (10240, 3 + depth) and r.dtype == torch.float32 for r in histories)
    assert page_bytes(model, element_bytes=2, mtp=depth > 0) == ((8 << 20) + (512 << 10 if depth else 0))
    assert {name for name, _ in page_specs(model, element_bytes=2, mtp=depth > 0)
            if name.startswith('draft')} == ({'draft.key', 'draft.value'} if depth else set())
    assert {r.name for r in layout.regions if r.name.startswith('draft')} == (
        {'draft.key', 'draft.value', 'draft.key.extra', 'draft.value.extra'} if depth else set())


@pytest.mark.parametrize('widths', [(), (2, 3), (1, 1), (1, 3, 2), (1, 6)])
def test_reject_invalid_catalog(widths):
    with pytest.raises(ValueError):
        PreparationConfig(mtp=True, speculative_widths=widths)


def test_maximum_verification_width_admits_four_draft_tokens():
    prepared = PreparationConfig(mtp=True, speculative_widths=(1, 5))
    assert prepared.draft_tokens == 4


def test_target_row_invariant_arithmetic_can_be_prepared_without_mtp():
    assert PreparationConfig().target_row_invariant is False
    assert PreparationConfig(mtp=True).target_row_invariant is True
    ordinary = PreparationConfig(max_active=1, mtp=False, row_invariant=True,
                                 retain_graphs=True, capture_activations=True, steering_max_sites=64)
    assert ordinary.target_row_invariant is True
    assert ordinary.mtp is False and ordinary.draft_tokens == 0
    with pytest.raises(ValueError, match="MTP requires"):
        PreparationConfig(mtp=True, row_invariant=False)
    for value in (0, 1, "true"):
        with pytest.raises(ValueError, match="bool or None"):
            PreparationConfig(row_invariant=value)


def test_native_contract_requires_all_state_columns_and_trailing_draft_pages():
    model = config()
    layout = continuation_layout(model, dtype=torch.float32, element_bytes=2, draft_tokens=3)
    pool = SimpleNamespace(layout=layout, pages={name: torch.empty((1, *shape), dtype=torch.uint8)
        for name, shape in page_specs(model, element_bytes=2, mtp=True)})
    buffers = {'draft_page_tables': torch.zeros((4, 64), dtype=torch.int32)}
    fields = ('request_state_rows', 'row_slot_ids', 'row_generations', 'live_rows',
              'page_tables', 'draft_page_tables', 'query_lengths', 'sampling_grammar_mask')
    signatures = {f'speculative4x{w}': GraphSignature('speculative', 4, w, fields,
                                                   speculation_enabled=True) for w in range(1, 5)}
    signatures['decode4'] = GraphSignature('decode', 4, 1, fields, speculation_enabled=False)
    contract = prepare(signatures, buffers, pool, vocabulary=248320)
    assert contract['cache']['page_tables'] == [dict(buffer='page_tables', page_offset=0),
                                               dict(buffer='draft_page_tables', page_offset=1)]
    for graph in contract['graphs']:
        if graph['kind'] == p.SPECULATIVE:
            # Replay-commit: one committed continuation row per request whatever the verification width.
            assert graph['state_rows_per_request'] == 1
            assert graph['draft_cursor']['lag_tokens'] == 1
            assert graph['pending_token']['buffer'] == 'request_pending_token'
        else:
            assert graph['state_rows_per_request'] == 1
            assert graph['draft_cursor'] is None
    assert RECIPE.max_draft_tokens == 4
    assert RECIPE.verification_rows == 0 and RECIPE.live_rows(4, RECIPE.speculation) == 1 + 4


def test_steering_catalog_declares_admission_inputs_and_forward_state():
    from ignition.components.steering import SteeringBuffers
    model = config()
    layout = continuation_layout(model, dtype=torch.float32, element_bytes=2, draft_tokens=0)
    pool = SimpleNamespace(layout=layout, pages={})
    steering = SteeringBuffers.allocate(64, 4, 4, 2, "cpu", intermediate=7)
    fields = ('request_state_rows', 'row_slot_ids', 'row_generations', 'live_rows')
    signatures = {kind: GraphSignature(kind, 4 if kind != 'prefill' else 1,
        128 if kind == 'prefill' else 1,
        (*fields, *steering.buffers()) if kind in ('init', 'restore') else fields)
        for kind in ('init', 'restore', 'prefill', 'decode', 'resume', 'snapshot')}
    contract = prepare(signatures, steering.buffers(), pool, vocabulary=248320, steering=steering)
    assert contract['steering'] == steering.contract()
    for graph in contract['graphs']:
        sources = {binding['source'] for binding in graph['inputs']}
        if graph['kind'] in (p.INIT, p.RESTORE):
            assert {p.STEERING_ADDENDS, p.STEERING_PHASES, p.STEERING_PROMPT_TOKENS} <= sources
        else:
            assert not {p.STEERING_ADDENDS, p.STEERING_PHASES, p.STEERING_PROMPT_TOKENS} & sources
        reads = {state['view']['buffer'] for state in graph['state'] if state['access'] == p.READ}
        if graph['kind'] in (p.PREFILL, p.DECODE):
            assert set(steering.buffers()) <= reads
        else:
            assert not set(steering.buffers()) & reads
    from google.protobuf.json_format import ParseDict
    decoded = ParseDict(contract, p.ModelContract())
    restored = p.ModelContract.FromString(decoded.SerializeToString())
    assert restored.steering.max_sites == 2
    assert restored.steering.sites[0].name == 'layers.0.mlp_out'
    assert restored.steering.sites[2].name == 'layers.0.mlp_input'
    assert restored.steering.sites[2].width == 7
