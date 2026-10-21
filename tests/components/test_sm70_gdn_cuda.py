"""Volta GDN sequence kernels against an independent CPU PyTorch recurrence."""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned V100")

KEYS, VALUES, DIM = 16, 48, 128
CHANNELS = (2 * KEYS + VALUES) * DIM


def _round(value):
    # CPU BF16 conversion is the reference; no software bit implementation is shared.
    return value.to(torch.bfloat16).float()


@pytest.fixture
def device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    result = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(result) != (7, 0):
        pytest.skip("requires a real sm_70 GPU")
    return result


def _case(device, activation, weight_bf16):
    generator = torch.Generator().manual_seed(48128)
    capacity, requests, slots = 17, 4, 6

    def random(shape, scale=.1, *, rounded=True):
        value = torch.randn(shape, generator=generator) * scale
        return _round(value) if rounded else value

    qkvz = random((capacity, CHANNELS + VALUES * DIM), .3)
    ba = random((capacity, 2 * VALUES), .4)
    conv = random((CHANNELS, 1, 4), .2)
    norm = 1 + random((DIM,), .1, rounded=False)
    if weight_bf16:
        norm = _round(norm)
    module = SimpleNamespace(num_k_heads=KEYS, num_v_heads=VALUES, head_k_dim=DIM,
        head_v_dim=DIM, conv_dim=CHANNELS, conv_kernel_size=4, activation="silu",
        conv1d=SimpleNamespace(weight=conv.to(device), bias=None),
        A_log=random((VALUES,), .2, rounded=False).to(device),
        dt_bias=random((VALUES,), .1, rounded=False).to(device),
        norm=SimpleNamespace(weight=norm.to(device), activation=activation, variance_epsilon=1e-6))
    packed, gates = qkvz.to(device), ba.to(device)
    scratch = SimpleNamespace(projected_qkv=packed[:, :CHANNELS], z=packed[:, CHANNELS:],
        a=gates[:, VALUES:], b=gates[:, :VALUES],
        convolved_qkv=torch.full((capacity, CHANNELS), 11.0, device=device),
        recurrence=torch.full((capacity, VALUES * DIM), 13.0, device=device),
        gated=torch.full((capacity, VALUES * DIM), 17.0, device=device))
    # Real state views include padding as in the hybrid cache pool.
    conv_backing = torch.full((slots, CHANNELS * 3 + 64), 19.0, device=device)
    state_backing = torch.full((slots, VALUES * DIM * DIM + 64), 23.0, device=device)
    state = SimpleNamespace(convolution=conv_backing[:, :CHANNELS * 3].view(slots, CHANNELS, 3),
                            recurrent=state_backing[:, :VALUES * DIM * DIM].view(slots, VALUES, DIM, DIM))
    state.convolution.copy_(random(state.convolution.shape))
    state.recurrent.copy_(random(state.recurrent.shape, .025, rounded=False))
    # The null recurrence row must never be read or overwritten.
    state.recurrent[0].fill_(31)
    def zeros(shape, dtype=torch.int32):
        return torch.zeros(shape, dtype=dtype, device=device)
    metadata = SimpleNamespace(query_start_loc=zeros((requests + 1,)),
        convolution_slots=torch.full((requests,), -1, dtype=torch.int64, device=device),
        fresh=zeros((requests,), torch.bool), accepted_tokens=torch.ones(requests, dtype=torch.int32, device=device),
        state_indices=zeros((requests, 1), torch.int64), initial_state_indices=zeros((requests,), torch.int64),
        final_state_indices=zeros((requests,), torch.int64), checkpoint_state_indices=zeros((requests,), torch.int64),
        checkpoint_offsets=zeros((requests,)), num_seqs=zeros((1,)), num_tokens=zeros((1,)))
    return module, scratch, state, metadata, (conv_backing, state_backing)


def _set_metadata(meta, lengths, slots, fresh, *, initial=None, checkpoint=None):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    assert len(lengths) == meta.convolution_slots.numel()
    device = meta.num_tokens.device
    meta.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
    meta.convolution_slots.copy_(torch.tensor(slots, dtype=torch.int64, device=device))
    meta.fresh.copy_(torch.tensor(fresh, dtype=torch.bool, device=device))
    final = torch.tensor([max(slot, 0) for slot in slots], dtype=torch.int64, device=device)
    meta.state_indices[:, 0].copy_(final)
    meta.final_state_indices.copy_(final)
    meta.initial_state_indices.copy_(final if initial is None else torch.tensor(initial, dtype=torch.int64, device=device))
    meta.checkpoint_state_indices.zero_()
    meta.checkpoint_offsets.zero_()
    if checkpoint is not None:
        request, slot, offset = checkpoint
        meta.checkpoint_state_indices[request] = slot
        meta.checkpoint_offsets[request] = offset
    meta.num_seqs.fill_(len(lengths))
    meta.num_tokens.fill_(starts[-1])


def _reference(module, scratch, recurrent, convolution, meta, mode, weight_bf16):
    """No Ignition kernel/reference helpers; FP32 PyTorch operations on CPU."""
    qkv, a, b, z = (getattr(scratch, name).cpu() for name in ("projected_qkv", "a", "b", "z"))
    convolved = scratch.convolved_qkv.cpu().clone()
    outputs = scratch.recurrence.cpu().clone()
    conv_weight = module.conv1d.weight[:, 0].cpu()
    starts = meta.query_start_loc.cpu().tolist()
    for sequence in range(meta.num_seqs.item()):
        start, end = starts[sequence:sequence + 2]
        if start == end:
            continue
        slot = int(meta.convolution_slots[sequence])
        fresh = bool(meta.fresh[sequence])
        if slot >= 0:
            history = torch.zeros_like(convolution[slot]) if fresh else convolution[slot].clone()
            for token in range(start, end):
                x = _round(qkv[token])
                summed = ((history[:, 0] * conv_weight[:, 0] + history[:, 1] * conv_weight[:, 1])
                          + history[:, 2] * conv_weight[:, 2]) + x * conv_weight[:, 3]
                convolved[token] = _round(F.silu(_round(summed)))
                history = torch.cat((history[:, 1:], x[:, None]), dim=1)
            convolution[slot] = history
        else:
            convolved[start:end].zero_()
        final = int(meta.final_state_indices[sequence] if mode == "prefill" else meta.state_indices[sequence, 0])
        initial = int(meta.initial_state_indices[sequence]) if mode == "prefill" else final
        if final == 0:
            outputs[start:end].zero_()
            continue
        current = torch.zeros_like(recurrent[0]) if fresh or initial == 0 else recurrent[initial].clone()
        for token in range(start, end):
            q, k, v = convolved[token].split((KEYS * DIM, KEYS * DIM, VALUES * DIM))
            q, k, v = q.view(KEYS, DIM), k.view(KEYS, DIM), v.view(VALUES, DIM)
            q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / math.sqrt(DIM)
            k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q, k = q.repeat_interleave(VALUES // KEYS, 0), k.repeat_interleave(VALUES // KEYS, 0)
            decay = torch.exp(-torch.exp(module.A_log.cpu()) * F.softplus(_round(a[token]) + module.dt_bias.cpu(), threshold=20))
            beta = _round(torch.sigmoid(_round(b[token])))
            current = current * decay[:, None, None]
            correction = (v - torch.einsum("hvk,hk->hv", current, k)) * beta[:, None]
            current = current + correction[:, :, None] * k[:, None, :]
            outputs[token] = _round(torch.einsum("hvk,hk->hv", current, q)).flatten()
            if mode == "prefill" and token - start + 1 == int(meta.checkpoint_offsets[sequence]):
                checkpoint = int(meta.checkpoint_state_indices[sequence])
                if checkpoint > 0:
                    recurrent[checkpoint] = current
        recurrent[final] = current
    live = int(meta.num_tokens)
    gated = torch.zeros_like(scratch.gated, device="cpu")
    hidden = outputs[:live].reshape(-1, VALUES, DIM)
    normalized = _round(hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + module.norm.variance_epsilon))
    weighted = normalized * module.norm.weight.cpu()
    if weight_bf16:
        weighted = _round(weighted)
    gate = _round(z[:live]).reshape(-1, VALUES, DIM)
    gate = F.silu(gate) if module.norm.activation == "silu" else torch.sigmoid(gate)
    gated[:live] = _round(weighted * gate).reshape(live, -1)
    return convolved, outputs, gated


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("weight_bf16", [False, True])
def test_prefill_continuation_decode_and_inactive_state(device, activation, weight_bf16):
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence

    module, scratch, state, meta, backing = _case(device, activation, weight_bf16)
    decode_scratch = SimpleNamespace(**{name: value[:4] for name, value in vars(scratch).items()})
    prefill = PreparedSM70GDNSequence(module, scratch, state, meta, mode="prefill",
                                     norm_weight_bf16=weight_bf16, checkpoint_export=True)
    decode = PreparedSM70GDNSequence(module, decode_scratch, state, meta, mode="decode",
                                    norm_weight_bf16=weight_bf16)
    graphs = {}
    for mode, operation in (("prefill", prefill), ("decode", decode)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        graphs[mode] = graph
    expected_state = state.recurrent.cpu().clone()
    expected_conv = state.convolution.cpu().clone()
    cases = [
        ("prefill", (4, 0, 7, 2), (1, 5, 2, -1), (True, True, False, False), (0, 0, 3, 0), (0, 4, 2)),
        ("prefill", (1, 3, 2, 0), (1, 4, 2, -1), (False, True, False, False), (1, 0, 2, 0), None),
        ("decode", (1, 0, 1, 1), (1, -1, 2, 4), (False, False, False, False), None, None),
        ("decode", (0, 1, 0, 0), (-1, 5, -1, -1), (False, True, False, False), None, None),
    ]
    for mode, lengths, slots, fresh, initial, checkpoint in cases:
        _set_metadata(meta, lengths, slots, fresh, initial=initial, checkpoint=checkpoint)
        target = scratch if mode == "prefill" else decode_scratch
        expected = _reference(module, target, expected_state, expected_conv, meta, mode, weight_bf16)
        graphs[mode].replay()
        torch.cuda.synchronize()
        live = sum(lengths)
        # Bounds follow existing GDN recovery/state and gated-norm component tests.
        torch.testing.assert_close(target.convolved_qkv[:live].cpu(), expected[0][:live], rtol=2**-7, atol=1e-7)
        torch.testing.assert_close(target.recurrence[:live].cpu(), expected[1][:live], rtol=2**-6, atol=1e-6)
        torch.testing.assert_close(target.gated.cpu(), expected[2], rtol=2**-6, atol=4e-3)
        torch.testing.assert_close(state.recurrent.cpu(), expected_state, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(state.convolution.cpu(), expected_conv, rtol=0, atol=0)
        assert not target.gated[live:].any()
        assert torch.all(backing[0][:, -64:] == 19)
        assert torch.all(backing[1][:, -64:] == 23)
    meta.num_seqs.zero_()
    meta.num_tokens.zero_()
    before = (state.convolution.clone(), state.recurrent.clone())
    graphs["prefill"].replay()
    torch.cuda.synchronize()
    assert torch.equal(state.convolution, before[0])
    assert torch.equal(state.recurrent, before[1])
    assert not scratch.gated.any()


def test_rejects_speculative_history_and_bf16_visible_operands(device):
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence

    module, scratch, state, meta, _ = _case(device, "silu", True)
    invalid = SimpleNamespace(convolution=torch.zeros((6, CHANNELS, 4), device=device), recurrent=state.recurrent)
    with pytest.raises(ValueError, match="three chronological"):
        PreparedSM70GDNSequence(module, scratch, invalid, meta, mode="prefill", norm_weight_bf16=True)
    invalid = SimpleNamespace(**vars(scratch))
    invalid.z = invalid.z.bfloat16()
    with pytest.raises(ValueError, match="FP32 token-major"):
        PreparedSM70GDNSequence(module, invalid, state, meta, mode="prefill", norm_weight_bf16=True)
    meta.state_indices = torch.zeros((4, 2), dtype=torch.int64, device=device)
    with pytest.raises(ValueError, match="state_indices"):
        PreparedSM70GDNSequence(module, scratch, state, meta, mode="prefill", norm_weight_bf16=True)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@torch.no_grad()
def test_recorded_preparation_compiled_capture_and_aliased_state(device, mode):
    from b12x._lib.compile_plan import observe_programs
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    module, scratch, original, meta, _ = _case(device, "silu", True)
    if mode == "decode":
        scratch = SimpleNamespace(**{name: value[:4] for name, value in vars(scratch).items()})
    # Production convolution and recurrence are disjoint views of one cache
    # allocation. Whole-forward compilation must preserve both mutations.
    conv_size, recurrent_size = CHANNELS * 3, VALUES * DIM * DIM
    backing = torch.full((6, conv_size + recurrent_size + 64), 29., device=device)
    state = SimpleNamespace(
        convolution=backing[:, :conv_size].view(6, CHANNELS, 3),
        recurrent=backing[:, conv_size:conv_size + recurrent_size].view(6, VALUES, DIM, DIM))
    state.convolution.copy_(original.convolution)
    state.recurrent.copy_(original.recurrent)
    expected_conv, expected_state = state.convolution.cpu(), state.recurrent.cpu()
    with RecordedPreparationSession(device=device, checkpoint="sm70-gdn-sequence",
            workload_schema="sm70-gdn-sequence", numerical_policy="bf16-rounded-fp32") as session:
        with observe_programs() as declarations:
            operation = PreparedSM70GDNSequence(module, scratch, state, meta, mode=mode,
                norm_weight_bf16=True, session=session, name="sequence")
        assert not declarations, "declaring the sequence must not compile its kernels"
        assert operation.plan.selection is None
        session.tune()
        torch.testing.assert_close(state.convolution.cpu(), expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(state.recurrent.cpu(), expected_state, rtol=0, atol=0)
        compiled = compile_forward(operation)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for continuation in (False, True):
            lengths = (4, 0, 7, 2) if mode == "prefill" else (1, 0, 1, 1)
            _set_metadata(meta, lengths, (1, 5, 2, -1),
                (not continuation, True, False, False),
                initial=(1 if continuation else 0, 0, 2 if continuation else 3, 0))
            expected = _reference(module, scratch, expected_state, expected_conv, meta, mode, True)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            live = sum(lengths)
            torch.testing.assert_close(scratch.convolved_qkv[:live].cpu(), expected[0][:live], rtol=2**-7, atol=1e-7)
            torch.testing.assert_close(scratch.recurrence[:live].cpu(), expected[1][:live], rtol=2**-6, atol=1e-6)
            torch.testing.assert_close(scratch.gated.cpu(), expected[2], rtol=2**-6, atol=4e-3)
            torch.testing.assert_close(state.recurrent.cpu(), expected_state, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(state.convolution.cpu(), expected_conv, rtol=0, atol=0)
            assert torch.all(backing[:, -64:] == 29)
        graph.reset()
