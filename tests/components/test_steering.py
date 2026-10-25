"""Additive-site semantics, slot ownership, and replay without recapture."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.steering import SteeringBuffers
from ignition.components.sm70_precision import round_bf16


def fixture(device="cpu"):
    buffers = SteeringBuffers.allocate(2, 4, 4, 2, device, intermediate=7)
    inputs = SimpleNamespace(row_slot_ids=torch.tensor([3, 1], device=device),
                             row_generations=torch.tensor([44, 22], device=device))
    requests = SimpleNamespace(generation=torch.tensor([11, 22, 33, 44], device=device))
    metadata = SimpleNamespace(requests=torch.tensor([0, 0, 1, 1, 1], device=device),
                               positions=torch.tensor([1, 2, 0, 3, -1], device=device))
    buffers.prompt_tokens.fill_(2)
    return buffers, inputs, requests, metadata


def test_disabled_and_zero_coordinates_preserve_every_original_bit():
    buffers, inputs, requests, metadata = fixture()
    bits = torch.tensor([0, -2147483648, 0x7FC01234, 0x7F800000], dtype=torch.int32)
    source = bits.view(torch.float32).repeat(5, 1)
    apply = buffers.bind(inputs, requests, metadata)
    assert torch.equal(apply(source, 0).view(torch.int32), source.view(torch.int32))
    buffers.phases.fill_(3)
    assert torch.equal(apply(source, 0).view(torch.int32), source.view(torch.int32))


def test_physical_slots_phase_boundary_padding_and_generation():
    buffers, inputs, requests, metadata = fixture()
    source = torch.arange(20, dtype=torch.float32).view(5, 4)
    buffers.site_addends(0)[3] = torch.tensor([1.25, 0, -2.5, 0])
    buffers.site_addends(0)[1] = 8
    buffers.phases[3, 0] = 1
    buffers.phases[1, 0] = 2
    apply = buffers.bind(inputs, requests, metadata)
    expected = source.clone()
    expected[0] = round_bf16(source[0] + buffers.site_addends(0)[3])
    expected[3] = round_bf16(source[3] + 8)
    assert torch.equal(apply(source, 0), expected)
    assert torch.equal(apply(source, 1), source)
    inputs.row_generations[0] = 43
    expected[0] = source[0]
    assert torch.equal(apply(source, 0), expected)
    inputs.row_slot_ids[1] = -1
    assert torch.equal(apply(source, 0), source)


def test_site_catalog_and_capacity_are_explicit():
    buffers, *_ = fixture()
    assert buffers.contract() == {"max_sites": 2, "sites": [
        {"name": f"layers.{layer}.{kind}", "width": width}
        for layer in range(2) for kind, width in (("mlp_out", 4), ("residual", 4), ("mlp_input", 7))]}
    assert buffers.addends.shape == (4, 30)
    assert buffers.offsets == (0, 4, 8, 15, 19, 23, 30)
    buffers.site_addends(2)[3].fill_(1.5)
    assert torch.equal(buffers.addends[3, 8:15], torch.full((7,), 1.5))
    assert torch.count_nonzero(buffers.addends[:, 15:]) == 0
    for limit in (0, 7, True):
        with pytest.raises(ValueError):
            SteeringBuffers.allocate(2, 4, 4, limit, "cpu", intermediate=7)


@pytest.mark.parametrize("gate_up", [False, True])
def test_neuron_site_changes_rounded_product_before_down_projection(gate_up):
    from ignition.recipes.qwen38_dense.decoder import SM70DecoderLayer
    from ignition.components.activation_capture import ActivationCapture
    buffers, inputs, requests, metadata = fixture()
    source = torch.arange(20, dtype=torch.float32).view(5, 4) / 8
    raw_gate = torch.arange(35, dtype=torch.float32).view(5, 7) / 19 - 1
    raw_up = torch.arange(35, dtype=torch.float32).view(5, 7) / 11 + 0.3
    matrix = torch.arange(28, dtype=torch.float32).view(4, 7) / 31 - 0.25
    observed = []

    def down(value):
        observed.append(value.clone())
        return value @ matrix.T

    mlp = SimpleNamespace(act_fn=torch.nn.functional.silu, down_proj=down)
    if gate_up:
        mlp.gate_up_proj = lambda hidden: torch.cat((raw_gate, raw_up), dim=-1)
    else:
        mlp.gate_proj, mlp.up_proj = lambda hidden: raw_gate, lambda hidden: raw_up
    layer = SimpleNamespace(block_type="linear_attention", mlp=mlp,
        linear_attn=lambda *, hidden_states: torch.zeros_like(hidden_states),
        input_layernorm=None, post_attention_layernorm=None)
    norms = (torch.nn.Identity(), torch.nn.Identity())
    baseline = SM70DecoderLayer(layer, norms)(source, None)
    product = (torch.nn.functional.silu(raw_gate.bfloat16().float()).bfloat16().float()
               * raw_up.bfloat16().float()).bfloat16().float()
    assert torch.equal(observed[-1], product)
    apply = buffers.bind(inputs, requests, metadata)
    capture = ActivationCapture.allocate(2, 5, 4, "cpu", intermediate=7)
    # Observe the final token of the first physical request while exercising all packed rows.
    capture.begin(SimpleNamespace(positions=metadata.positions, live_counts=torch.tensor([2])))
    operation = SM70DecoderLayer(layer, norms, steering=apply, layer_index=1, capture=capture)
    assert torch.equal(operation(source, None).view(torch.int32), baseline.view(torch.int32))
    delta = torch.tensor([0., 0.25, -0.5, -0., 4., -2., 0.])
    buffers.phases[3, 5] = 3
    for sign in (1, -1):
        buffers.site_addends(5)[3].copy_(sign * delta)
        expected = product.clone()
        expected[:2] = torch.where(delta != 0,
            (product[:2] + sign * delta).bfloat16().float(), product[:2])
        actual = operation(source, None)
        assert torch.equal(observed[-1].view(torch.int32), expected.view(torch.int32))
        assert torch.equal(capture.mlp_input_last[1], expected[1])
        expected_output = (source + (expected @ matrix.T).bfloat16().float()).bfloat16().float()
        assert torch.equal(actual, expected_output)
        assert not torch.equal(actual[:2], baseline[:2])
        assert torch.equal(actual[2:], baseline[2:])
    buffers.phases.zero_()
    assert torch.equal(operation(source, None).view(torch.int32), baseline.view(torch.int32))
    # The deployed unfused source has no fused-producer option to reject.
    from inspect import signature
    if "swiglu" in signature(SM70DecoderLayer).parameters:
        with pytest.raises(ValueError, match="explicit SwiGLU product"):
            SM70DecoderLayer(layer, norms, steering=apply, swiglu=True)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA opt-in required")
def test_compiled_captured_steering_reads_updated_request_inputs():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 arithmetic qualification")
    buffers, inputs, requests, metadata = fixture("cuda")
    original = torch.arange(20, dtype=torch.float32).view(5, 4) + 0.001
    source = original.cuda()
    output = torch.empty((2, *source.shape), device="cuda")
    product = torch.arange(35, dtype=torch.float32).view(5, 7) / 9
    product_input, product_output = product.cuda(), torch.empty_like(product, device="cuda")
    apply = buffers.bind(inputs, requests, metadata)

    def forward():
        output[0].copy_(apply(source, 0))
        output[1].copy_(apply(source, 1))
        product_output.copy_(apply(product_input, 2))

    compiled = torch.compile(forward, fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()

    def check(expected, expected_product=product):
        graph.replay()
        assert torch.equal(output.cpu().view(torch.int32), expected.view(torch.int32))
        assert torch.equal(product_output.cpu().view(torch.int32), expected_product.view(torch.int32))

    baseline = original.repeat(2, 1, 1)
    check(baseline)
    buffers.phases.fill_(3)
    check(baseline)  # Active sites with zero vectors preserve non-BF16 input bits.
    special = torch.tensor([0, -2147483648, 0x7FC01234, 0x7F800000], dtype=torch.int32)
    special = special.view(torch.float32).repeat(5, 1)
    source.copy_(special)
    check(special.repeat(2, 1, 1))
    source.copy_(original)

    positive = torch.tensor([1.25, 0., -2.5, -0.])
    negative = torch.tensor([-4., 1., 0., 0.])
    buffers.site_addends(0)[3].copy_(positive)
    buffers.site_addends(0)[1].fill_(-8)
    buffers.site_addends(1)[3].copy_(negative)
    buffers.site_addends(1)[1].fill_(0.5)
    buffers.phases[3, 0], buffers.phases[1, 0] = 1, 2
    buffers.phases[3, 1], buffers.phases[1, 1] = 3, 1

    def changed(row, delta):
        # Reference arithmetic runs on CPU; zero coordinates retain their source bits.
        return torch.where(delta != 0, round_bf16(original[row] + delta), original[row])

    expected = baseline.clone()
    expected[0, 0] = changed(0, positive)
    expected[0, 3] = round_bf16(original[3] - 8)
    expected[1, 0], expected[1, 1] = changed(0, negative), changed(1, negative)
    expected[1, 2] = round_bf16(original[2] + 0.5)
    check(expected)

    buffers.prompt_tokens[3] = 3
    expected[0, 1] = changed(1, positive)
    check(expected)

    # Reuse the same captured graph after packed rows exchange physical request slots.
    inputs.row_slot_ids.copy_(torch.tensor([1, 3]))
    inputs.row_generations.copy_(torch.tensor([22, 44]))
    expected = baseline.clone()
    expected[0, 1] = round_bf16(original[1] - 8)
    expected[0, 2] = changed(2, positive)
    expected[1, 0] = round_bf16(original[0] + 0.5)
    expected[1, 2], expected[1, 3] = changed(2, negative), changed(3, negative)
    check(expected)

    inputs.row_generations[1] = 43
    expected[:, 2:] = baseline[:, 2:]
    check(expected)
    inputs.row_slot_ids[0] = -1
    check(baseline)

    inputs.row_slot_ids.copy_(torch.tensor([3, 1]))
    inputs.row_generations.copy_(torch.tensor([44, 22]))
    buffers.phases.zero_()
    check(baseline)  # Disable nonzero programs without clearing or recapturing them.
    buffers.phases[3, 2] = 3
    delta = torch.tensor([0., 0.25, -0.5, -0., 4., -2., 0.])
    for sign in (1, -1, 0):
        buffers.site_addends(2)[3].copy_(sign * delta)
        expected_product = product.clone()
        expected_product[:2] = torch.where(sign * delta != 0,
            round_bf16(product[:2] + sign * delta), product[:2])
        check(baseline, expected_product)


def test_diagnostic_capture_tracks_last_live_row_and_reuses_capacity_storage():
    from ignition.components.activation_capture import ActivationCapture
    capture = ActivationCapture.allocate(2, 8, 4, "cpu", intermediate=7)
    short = capture.rows(4)
    assert short.residual_last.data_ptr() == capture.residual_last.data_ptr()
    assert short.residual_l2.data_ptr() == capture.residual_l2.data_ptr()
    assert short.mlp_input_last.data_ptr() == capture.mlp_input_last.data_ptr()
    metadata = SimpleNamespace(positions=torch.tensor([127, 128, 129, -1]),
                               live_counts=torch.tensor([3, 1]))
    source = torch.tensor([[3., 4., 0., 0.], [0., 0., 0., 2.], [1., 2., 2., 0.], [9., 9., 9., 9.]])
    original = source.clone()
    short.begin(metadata)
    short.record_pre_mlp(source, 1)
    product = torch.cat((source, source[:, :3]), dim=-1)
    short.record_mlp_input(product, 1)
    short.record_residual(source, 1)
    assert torch.equal(short.positions, metadata.positions)
    assert torch.equal(short.residual_l2[1], torch.tensor([5., 2., 3., 0.]))
    assert torch.equal(short.residual_last[1], source[2])
    assert torch.equal(short.pre_mlp_last[1], source[2])
    assert torch.equal(short.mlp_input_last[1], product[2])
    assert torch.equal(source, original)
    metadata.positions.fill_(-1)
    metadata.live_counts.zero_()
    short.begin(metadata)
    short.record_pre_mlp(source, 1)
    short.record_mlp_input(product, 1)
    short.record_residual(source, 1)
    assert torch.count_nonzero(short.residual_last[1]) == 0
    assert torch.count_nonzero(short.residual_l2[1]) == 0
    assert torch.count_nonzero(short.mlp_input_last[1]) == 0
