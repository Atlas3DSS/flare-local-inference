"""Captured-graph Sol-15 GDN prefill (conv1d + recurrence) is bit-identical to the Triton prefill sequence.

Outputs (convolved, recurrence, gated), the full recurrent-state pool and the full conv-history pool are compared as
int32 bit patterns after each replay of one captured graph per capacity (128 and 512) whose metadata and data change
between replays (Flare's situation). Chunking invariance: four chained 32-token chunks consuming their own state equal
one 128-token chunk, and an unchanged Triton decode step from either state matches bitwise.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned V100")

KEYS, VALUES, DIM, HISTORY = 16, 48, 128, 7
CHANNELS = (2 * KEYS + VALUES) * DIM


@pytest.fixture
def device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    result = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(result) != (7, 0):
        pytest.skip("requires a real sm_70 GPU")
    return result


def _bits(tensor):
    return tensor.view(torch.int32)


def _module(device, generator, activation="silu"):
    def random(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(device)
    return SimpleNamespace(num_k_heads=KEYS, num_v_heads=VALUES, head_k_dim=DIM, head_v_dim=DIM, conv_dim=CHANNELS,
        conv_kernel_size=4, activation="silu", conv1d=SimpleNamespace(weight=random((CHANNELS, 1, 4), .3), bias=None),
        A_log=random((VALUES,), .5), dt_bias=random((VALUES,), .3),
        norm=SimpleNamespace(weight=1 + random((DIM,), .1), activation=activation, variance_epsilon=1e-6),
        norm_weight_bf16=True)


def _case(device, capacity, requests, slots, pool_pad=64):
    """Production layout: qkvz [cap, 16384] with projected_qkv = [:, :10240]; ba [cap, 96]; padded pool rows."""
    qkvz = torch.zeros((capacity, CHANNELS + VALUES * DIM), device=device)
    ba = torch.zeros((capacity, 2 * VALUES), device=device)
    scratch = SimpleNamespace(projected_qkv=qkvz[:, :CHANNELS], z=qkvz[:, CHANNELS:], a=ba[:, VALUES:], b=ba[:, :VALUES],
        convolved_qkv=torch.zeros((capacity, CHANNELS), device=device),
        recurrence=torch.zeros((capacity, VALUES * DIM), device=device),
        gated=torch.zeros((capacity, VALUES * DIM), device=device), qkvz=qkvz, ba=ba)
    conv_size, rec_size = CHANNELS * HISTORY, VALUES * DIM * DIM
    backing = torch.zeros((slots, conv_size + rec_size + pool_pad), device=device)
    state = SimpleNamespace(convolution=backing[:, :conv_size].view(slots, CHANNELS, HISTORY),
                            recurrent=backing[:, conv_size:conv_size + rec_size].view(slots, VALUES, DIM, DIM), backing=backing)
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    metadata = SimpleNamespace(query_start_loc=torch.zeros(requests + 1, **i32),
        convolution_slots=torch.full((requests,), -1, **i64), fresh=torch.zeros(requests, dtype=torch.bool, device=device),
        accepted_tokens=torch.ones(requests, **i32), state_indices=torch.zeros((requests, HISTORY - 2), **i64),
        initial_state_indices=torch.zeros(requests, **i64), final_state_indices=torch.zeros(requests, **i64),
        checkpoint_state_indices=torch.zeros(requests, **i64), checkpoint_offsets=torch.zeros(requests, **i32),
        num_seqs=torch.zeros(1, **i32), num_tokens=torch.zeros(1, **i32))
    return scratch, state, metadata


def _meta(m, starts, slots, fresh, initial, final, checkpoints=None, offsets=None, num_seqs=None):
    device = m.num_seqs.device
    n = len(slots)
    m.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
    m.convolution_slots.copy_(torch.tensor(slots, dtype=torch.int64, device=device))
    m.fresh.copy_(torch.tensor(fresh, dtype=torch.bool, device=device))
    m.initial_state_indices.copy_(torch.tensor(initial, dtype=torch.int64, device=device))
    m.final_state_indices.copy_(torch.tensor(final, dtype=torch.int64, device=device))
    m.state_indices[:, 0].copy_(torch.tensor(final, dtype=torch.int64, device=device))
    m.checkpoint_state_indices.copy_(torch.tensor(checkpoints or [0] * n, dtype=torch.int64, device=device))
    m.checkpoint_offsets.copy_(torch.tensor(offsets or [0] * n, dtype=torch.int32, device=device))
    m.num_seqs.fill_(n if num_seqs is None else num_seqs)
    m.num_tokens.fill_(starts[-1] - starts[0])


def _fill(scratch, state, generator, *, adversarial=False):
    device = scratch.qkvz.device
    scratch.qkvz.copy_(torch.nn.functional.silu(torch.randn(scratch.qkvz.shape, generator=generator) * 1.5).to(device))
    scratch.ba.copy_((torch.randn(scratch.ba.shape, generator=generator) * 3.0).to(device))
    state.backing.copy_((torch.randn(state.backing.shape, generator=generator) * .05).to(device))
    for tensor, fill in ((scratch.convolved_qkv, 11.0), (scratch.recurrence, 13.0), (scratch.gated, 17.0)):
        tensor.fill_(fill)
    if adversarial:
        x = scratch.projected_qkv
        x[3, 5] = float("nan"); x[4, 2048 + 7] = float("inf"); x[5, 4096 + 9] = -float("inf")
        x[6, :2048] = 0.0; x[7, 2048:4096] = 1e-30; x[8, 4096:] = 3e38
        scratch.a[9, :] = -200.0; scratch.a[10, :] = 25.0; scratch.a[11, :] = float("nan"); scratch.b[12, :] = float("nan")
        scratch.b[13, :] = 1e30; scratch.a[17, :] = 19.99; scratch.a[18, :] = 20.0


def _snapshot(scratch, state):
    return tuple(t.clone() for t in (scratch.convolved_qkv, scratch.recurrence, scratch.gated, state.backing))


def _restore(scratch, state, saved):
    for tensor, value in zip((scratch.convolved_qkv, scratch.recurrence, scratch.gated, state.backing), saved, strict=True):
        tensor.copy_(value)


def _assert_identical(got, expected):
    for name, a, b in zip(("convolved", "recurrence", "gated", "pools"), got, expected, strict=True):
        assert torch.equal(_bits(a), _bits(b)), f"{name} differs from the Triton prefill sequence"


def _prepare_captured(module, scratch, state, metadata, device):
    """Recorded preparation (compile jobs, tuning, capture, freeze) of the Sol-15 prefill sequence on a scratch arena."""
    from ignition.components.sm70_gdn_prefill import PreparedSM70GDNPrefillSequence
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    arena = ScratchArena(device, overlap=False)
    with RecordedPreparationSession(device=device, checkpoint="sm70-gdn-prefill", workload_schema="sm70-gdn-prefill",
                                    numerical_policy="bf16-rounded-fp32") as session:
        session.register_scratch(arena)     # as the recipe bootstrap does: tune() sizes, finalizes (bind) and freezes it
        operation = PreparedSM70GDNPrefillSequence(module, scratch, state, metadata, norm_weight_bf16=True,
                                                   session=session, name="sequence", workspace=arena)
        assert operation.plan.selection is None
        session.tune()
        assert arena.frozen and operation.scratch is not None and arena.nbytes >= operation.nbytes
        compiled = compile_forward(operation)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
    return graph, operation


@pytest.mark.parametrize("capacity", [128, 512])
@torch.no_grad()
def test_captured_prefill_matches_triton_bits(device, capacity):
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence

    generator = torch.Generator().manual_seed(1500 + capacity)
    module = _module(device, generator)
    scratch, state, metadata = _case(device, capacity, requests=3, slots=12)
    reference = PreparedSM70GDNSequence(module, scratch, state, metadata, mode="prefill", norm_weight_bf16=True)
    graph, operation = _prepare_captured(module, scratch, state, metadata, device)
    full = capacity
    cases = [  # (starts, slots, fresh, initial, final, checkpoints, offsets, num_seqs, adversarial)
        ("full-loaded", [0, full, full, full], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], None, None, 1, False),
        ("M1-fresh-checkpoint", [0, 1, 1, 1], [3, -1, -1], [True, False, False], [3, 0, 0], [5, 0, 0], [7, 0, 0], [1, 0, 0], 1, False),
        ("M2-fresh", [0, 2, 2, 2], [4, -1, -1], [True, False, False], [0, 0, 0], [4, 0, 0], None, None, 1, False),
        ("M33-checkpoint", [0, 33, 33, 33], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], [7, 0, 0], [16, 0, 0], 1, False),
        ("M127-fresh-checkpoint", [0, 127, 127, 127], [3, -1, -1], [True, False, False], [3, 0, 0], [5, 0, 0], [7, 0, 0], [63, 0, 0], 1, False),
        ("initial0-not-fresh", [0, 100, 100, 100], [2, -1, -1], [False, False, False], [0, 0, 0], [2, 0, 0], [4, 0, 0], [100, 0, 0], 1, False),
        ("final0-zero-output", [0, 77, 77, 77], [3, -1, -1], [False, False, False], [3, 0, 0], [0, 0, 0], None, None, 1, False),
        ("slot-negative", [0, 90, 90, 90], [-1, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], None, None, 1, False),
        ("numseqs0", [0, full, full, full], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], None, None, 0, False),
        ("start-offset", [17, full, full, full], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], [6, 0, 0], [40, 0, 0], 1, False),
        ("empty-range", [40, 40, 40, 40], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], None, None, 1, False),
        ("three-requests", [0, 50, 51, full], [3, 7, 9], [False, True, False], [3, 0, 9], [5, 6, 10], [0, 0, 11], [0, 0, 30], 3, False),
        ("adversarial", [0, full, full, full], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], [6, 0, 0], [20, 0, 0], 1, True),
    ]
    for name, starts, slots, fresh, initial, final, checkpoints, offsets, num_seqs, adversarial in cases:
        _fill(scratch, state, generator, adversarial=adversarial)
        _meta(metadata, starts, slots, fresh, initial, final, checkpoints, offsets, num_seqs)
        before = _snapshot(scratch, state)
        reference()
        torch.cuda.synchronize()
        expected = _snapshot(scratch, state)
        _restore(scratch, state, before)
        graph.replay()
        torch.cuda.synchronize()
        got = _snapshot(scratch, state)
        _assert_identical(got, expected)
        _restore(scratch, state, before)
        graph.replay()
        torch.cuda.synchronize()
        _assert_identical(_snapshot(scratch, state), got)   # deterministic
    # Checkpoint sweep: a boundary at every position of one loaded chunk.
    _fill(scratch, state, generator)
    base = _snapshot(scratch, state)
    # Every boundary at capacity 128; at 512 every 7th plus the chunk and tile edges (Argon: 128, 129, 256, 257, full).
    boundaries = set(range(1, full + 1, 1 if full == 128 else 7)) | {1, 127, 128, 129, 255, 256, 257, full - 1, full}
    for boundary in sorted(b for b in boundaries if 1 <= b <= full):
        _meta(metadata, [0, full, full, full], [3, -1, -1], [False, False, False], [3, 0, 0], [5, 0, 0], [6, 0, 0], [boundary, 0, 0])
        _restore(scratch, state, base)
        reference()
        torch.cuda.synchronize()
        expected = _snapshot(scratch, state)
        _restore(scratch, state, base)
        graph.replay()
        torch.cuda.synchronize()
        _assert_identical(_snapshot(scratch, state), expected)
    graph.reset()


@torch.no_grad()
def test_chained_chunks_equal_one_chunk_and_hand_off_to_triton_decode(device):
    """4 x 32 chained Sol-15 chunks == 1 x 128 Triton chunk (state pools + outputs), then one unchanged Triton decode
    step from either state is bit-identical."""
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence
    from ignition.components.sm70_gdn_prefill import PreparedSM70GDNPrefillSequence

    generator = torch.Generator().manual_seed(2015)
    module = _module(device, generator)
    scratch, state, metadata = _case(device, 128, requests=1, slots=8)
    reference = PreparedSM70GDNSequence(module, scratch, state, metadata, mode="prefill", norm_weight_bf16=True)
    _fill(scratch, state, generator)
    _meta(metadata, [0, 128], [3], [True], [0], [3])
    before = _snapshot(scratch, state)
    reference()
    torch.cuda.synchronize()
    expected = _snapshot(scratch, state)
    # Candidate: four 32-token chunks through a capacity-32 Sol-15 sequence sharing the same pools.
    chunk, chunk_state, chunk_meta = _case(device, 32, requests=1, slots=8)
    chunk_state.backing.copy_(before[3])
    candidate = PreparedSM70GDNPrefillSequence(module, chunk, chunk_state, chunk_meta, norm_weight_bf16=True)
    outputs = []
    for n in range(4):
        rows = slice(32 * n, 32 * n + 32)
        chunk.qkvz.copy_(scratch.qkvz[rows])
        chunk.ba.copy_(scratch.ba[rows])
        _meta(chunk_meta, [0, 32], [3], [n == 0], [0 if n == 0 else 3], [3])
        candidate()
        torch.cuda.synchronize()
        outputs.append((chunk.convolved_qkv.clone(), chunk.recurrence.clone(), chunk.gated.clone()))
    for index, name in enumerate(("convolved", "recurrence", "gated")):
        assert torch.equal(_bits(torch.cat([o[index] for o in outputs])), _bits(expected[index])), name
    assert torch.equal(_bits(chunk_state.backing), _bits(expected[3])), "state pools differ after 4 x 32"
    # Hand-off: the unchanged Triton decode kernels step once from each state and must agree bitwise.
    results = []
    for pools in (expected[3], chunk_state.backing.clone()):
        dec_scratch, dec_state, dec_meta = _case(device, 1, requests=1, slots=8)
        dec_state.backing.copy_(pools)
        decode = PreparedSM70GDNSequence(module, dec_scratch, dec_state, dec_meta, mode="decode", norm_weight_bf16=True)
        dec_scratch.qkvz.copy_(torch.nn.functional.silu(torch.randn((1, CHANNELS + VALUES * DIM), generator=torch.Generator().manual_seed(7)) * 1.5).to(device))
        dec_scratch.ba.copy_((torch.randn((1, 2 * VALUES), generator=torch.Generator().manual_seed(8)) * 3.0).to(device))
        _meta(dec_meta, [0, 1], [3], [False], [3], [3])
        dec_meta.state_indices[0, 0] = 3
        decode()
        torch.cuda.synchronize()
        results.append((dec_scratch.gated.clone(), dec_state.backing.clone()))
    assert torch.equal(_bits(results[0][0]), _bits(results[1][0]))
    assert torch.equal(_bits(results[0][1]), _bits(results[1][1]))


def test_rejects_operands_outside_the_reviewed_contract(device):
    from ignition.components.sm70_gdn_prefill import PreparedSM70GDNPrefillSequence, admits

    generator = torch.Generator().manual_seed(3)
    module = _module(device, generator)
    scratch, state, metadata = _case(device, 128, requests=1, slots=4)
    assert admits(module, scratch, state)
    narrow = SimpleNamespace(convolution=torch.zeros((4, CHANNELS, 6), device=device), recurrent=state.recurrent)
    metadata.state_indices = torch.zeros((1, 4), dtype=torch.int64, device=device)
    assert not admits(module, scratch, narrow)
    with pytest.raises(ValueError, match="reviewed Sol-15 contract"):
        PreparedSM70GDNPrefillSequence(module, scratch, narrow, metadata, norm_weight_bf16=True)
    wide, wide_state, wide_meta = _case(device, 513, requests=1, slots=4)
    assert not admits(module, wide, wide_state)
    with pytest.raises(ValueError, match="reviewed Sol-15 contract"):
        PreparedSM70GDNPrefillSequence(module, wide, wide_state, wide_meta, norm_weight_bf16=True)
