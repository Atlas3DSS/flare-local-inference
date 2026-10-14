"""Joint catalog coverage and dense native-to-role packing for mixed Qwen graphs."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.capacity import batch_capacities, mixed_shapes, prefill_capacities
from ignition.recipes.qwen38.bootstrap import PreparationConfig
from ignition.recipes.qwen38.decoder_state import DecoderInputs
from ignition.recipes.qwen38.mixed_state import MixedPacking, role_inputs
from ignition.recipes.qwen38.recipe import RECIPE
from ignition.recipes.declaration import Speculation


@pytest.mark.parametrize("concurrency", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("widths", [(1,), (1, 3, 6), tuple(range(1, 9))])
def test_joint_catalog_covers_both_axes_without_cross_product(concurrency, widths):
    prompts = prefill_capacities(4096)
    shapes = mixed_shapes(concurrency, 4096, widths)
    batches = batch_capacities(concurrency)[1:]
    assert len(shapes) == len(batches) * (len(prompts) + len(widths) - 1)
    assert len(shapes) == len(set(shapes))
    for batch in batches:
        assert {(b, 1, p) for b in (batch,) for p in prompts} <= set(shapes)
        for width in widths:
            assert any(b == batch and w == width for b, w, _ in shapes)
    config = PreparationConfig(max_active=concurrency, mtp=True, speculative_widths=widths)
    assert all(p + (b - 1) * w <= config.token_capacity for _, b, w, p in config.mixed_shapes)
    assert all(w == 1 for enabled, _, w, _ in config.mixed_shapes if not enabled)


def test_plan_and_preparation_use_identical_catalogs():
    for speculation in (None, Speculation(0), Speculation(5), Speculation(3, adaptive=False)):
        config = PreparationConfig(mtp=speculation is not None,
            speculative_widths=(1,) if speculation is None else speculation.widths)
        assert config.mixed_shapes == RECIPE.mixed_shapes(4, 4096, speculation)
    assert len(PreparationConfig(mtp=True).mixed_shapes) == 50
    assert PreparationConfig(max_active=1, mtp=True).mixed_shapes == ()


@pytest.mark.parametrize("prompt_row", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("width", [1, 3, 6])
def test_packing_preserves_native_row_order_and_physical_slots(prompt_row, width):
    batch, prefill, length = 4, 128, 65
    tokens = prefill + (batch - 1) * width
    inputs = DecoderInputs.allocate("cpu", context=256, tokens=tokens, capacity=batch)
    p, d = role_inputs(inputs, prefill), role_inputs(inputs, batch * width)
    packing = MixedPacking(inputs, p, d, batch, width, prefill)
    inputs.live_rows.fill_(batch)
    inputs.prefill_row.fill_(prompt_row)
    inputs.row_slot_ids.copy_(torch.tensor([2, 0, 3, 1]))
    inputs.row_generations.copy_(torch.tensor([21, 32, 43, 54]))
    inputs.query_lengths.fill_(width)
    if prompt_row >= 0:
        inputs.query_lengths[prompt_row] = length
    inputs.token_ids.copy_(torch.arange(tokens) + 1000)
    packing.upload_roles()
    assert d.row_slot_ids.tolist() == [(-1 if r == prompt_row else s) for r, s in enumerate([2, 0, 3, 1])]
    prompt_ids = torch.full((prefill,), -1)
    if prompt_row >= 0:
        prompt_ids[:length] = 0
        start = int(inputs.query_lengths[:prompt_row].sum())
        assert p.token_ids[:length].tolist() == list(range(1000 + start, 1000 + start + length))
    else:
        assert p.live_rows.item() == 0
    decode_ids = torch.arange(batch).repeat_interleave(width)
    decode_ids[decode_ids == prompt_row] = -1
    prompt = SimpleNamespace(qsa=SimpleNamespace(request_ids=prompt_ids))
    decode = SimpleNamespace(qsa=SimpleNamespace(request_ids=decode_ids),
        gdn=SimpleNamespace(query_start_loc=torch.arange(batch + 1) * width))
    packing.bind_metadata(prompt, decode)
    packed = torch.zeros(tokens, dtype=torch.int64)
    packing.combine((p.token_ids, inputs.token_ids[packing.decode_indices]), packed)
    live = int(inputs.query_lengths.sum())
    assert packed[:live].tolist() == inputs.token_ids[:live].tolist()
    assert not packing.prompt_valid.any() if prompt_row == -1 else packing.prompt_valid.sum() == length
