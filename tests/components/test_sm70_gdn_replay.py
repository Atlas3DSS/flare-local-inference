"""Replay-commit for the Volta GDN decode recurrence: stored operands replayed from the committed row.

The CPU test models the two arithmetic paths with the same torch expressions in the same order: today's per-position
candidate states (the ``_reference`` of ``test_sm70_gdn_cuda``) and the replay of the operands the verification pass
stores, laid out exactly as the ``_recurrence`` REPLAY branch writes them and read back as ``_replay`` reads them. It
proves the layout, packing (query boundaries), fresh/null/inactive masks and count semantics bit for bit. The CUDA tests
(FLARE_RUN_CUDA_TESTS=1, V100) prove the kernels themselves: the replay-mode sequence writes no state and emits the same
outputs as the candidate-column sequence, and the replay commit reproduces every candidate column byte for byte.
"""

import math
import os
import re
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

KEYS, VALUES, DIM = 16, 48, 128
CHANNELS = (2 * KEYS + VALUES) * DIM
OPERAND = (KEYS + VALUES) * DIM + 2 * VALUES          # 8288 FP32 words per token and layer
OPERAND_KEY, OPERAND_VALUE = 0, KEYS * DIM
OPERAND_DECAY, OPERAND_BETA = (KEYS + VALUES) * DIM, (KEYS + VALUES) * DIM + VALUES


def _round(value):
    return value.to(torch.bfloat16).float()


def _same_bits(actual, expected):
    """Bit-for-bit equality. torch.equal on floats accepts +0.0 == -0.0 (and would hide a sign-of-zero difference
    between the candidate path and the replay), so every comparison goes through uint8 views of the same shape/dtype."""
    actual, expected = actual.detach().contiguous(), expected.detach().contiguous()
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))


def _step(state, k, value, decay, beta):
    """One recurrence update of one value head's [rows x keys] tile: the kernel's expressions in the kernel's order."""
    state = state * decay
    delta = (value - (state * k[None, :]).sum(1)) * beta
    return state + delta[:, None] * k[None, :]


def _operands(convolved, a, b, a_log, dt_bias, token):
    """The FP32 operands the decode recurrence consumes for one token (every value head)."""
    q, k, v = convolved[token].split((KEYS * DIM, KEYS * DIM, VALUES * DIM))
    k = k.view(KEYS, DIM)
    k = k * (1.0 / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6))
    decay = torch.exp(-torch.exp(a_log) * F.softplus(_round(a[token]) + dt_bias, threshold=20))
    beta = _round(torch.sigmoid(_round(b[token])))
    return k, v.view(VALUES, DIM), decay, beta


def test_operand_width_is_the_documented_33152_bytes():
    assert OPERAND == 8288 and OPERAND * 4 == 33_152


@pytest.mark.parametrize("width", [1, 4, 5])
def test_generic_gdn_replay_history_capacity_admission(monkeypatch, width):
    """Exercise PreparedGDN's real admission before any CUDA compilation or allocation.

    Meta tensors carry production shapes/strides with a CUDA device label. Only hardware detection and the downstream sequence
    constructor are replaced; the generic geometry, history, token and column checks run unchanged.
    """
    from ignition.components import sm70_gdn, sm70_projection
    from ignition.components.gated_delta import GDNScratch, PreparedGDN
    from ignition.components.recurrent import DeltaMetadata

    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("admission test initialized CUDA"))
    monkeypatch.setattr(sm70_projection, "is_sm70", lambda device: True)
    reached = []
    sentinel = object()

    def prepare_sequence(*args, **kwargs):
        reached.append(kwargs["replay"])
        return sentinel

    monkeypatch.setattr(sm70_gdn, "PreparedSM70GDNSequence", prepare_sequence)
    class CudaMetadata:
        # Admission reads metadata only. Avoid constructing FakeTensor(cuda), which can initialize CUDA on a
        # GPU-equipped test runner even though this test must run entirely on the CPU.
        device = torch.device("cuda:0")

        def __init__(self, tensor):
            self.tensor = tensor

        def __getattr__(self, name):
            return getattr(self.tensor, name)

    def fake(shape, dtype=torch.float32):
        return CudaMetadata(torch.empty(shape, dtype=dtype, device="meta"))

    module = SimpleNamespace(hidden_size=32, num_k_heads=KEYS, num_v_heads=VALUES, head_k_dim=DIM,
        head_v_dim=DIM, conv_dim=CHANNELS, conv_kernel_size=4, activation="silu", norm_weight_bf16=True,
        conv1d=SimpleNamespace(weight=fake((CHANNELS, 1, 4)), bias=None), A_log=fake((VALUES,)),
        dt_bias=fake((VALUES,)), norm=SimpleNamespace(weight=fake((DIM,)), activation="silu"))
    requests = 4
    state = SimpleNamespace(convolution=fake((requests + 1, CHANNELS, width + 2)),
                            recurrent=fake((requests + 1, VALUES, DIM, DIM)))

    def construct(tokens, columns, replay):
        scratch = GDNScratch.allocate(tokens, torch.device("meta"), hidden_size=32, dtype=torch.float32)
        scratch = replace(scratch, **{field.name: CudaMetadata(getattr(scratch, field.name)) for field in fields(scratch)})
        metadata = DeltaMetadata.allocate(requests, "cpu", verification_columns=columns)
        operands = fake((tokens, OPERAND)) if replay else None
        return PreparedGDN(module, dict.fromkeys(("in_proj_qkv", "in_proj_z", "in_proj_ba", "out_proj")),
            fake((tokens, 32)), scratch, state, metadata, mode="decode", session=None, name="admission",
            prepare_projections=False, replay=operands)

    accepted = construct(requests * width, 1, True)
    assert accepted.sm70_sequence is sentinel and reached[-1] is not None
    assert accepted.tokens == requests * width and accepted.speculative_tokens == width - 1
    ordinary = construct(requests * width, width, False)
    assert ordinary.sm70_sequence is sentinel and reached[-1] is None
    calls = len(reached)
    for tokens, columns, replay in ((requests * width + 1, 1, True), (requests, 2, True),
                                    (requests * width + 1, width, False), (requests + 1, 1, False),
                                    (requests, width + 1, False)):
        with pytest.raises(ValueError, match="decode token/verification capacity exceeds its convolution history"):
            construct(tokens, columns, replay)
    assert len(reached) == calls, "invalid capacities reached kernel preparation"


@pytest.mark.parametrize("fresh", [(False, False, True, False), (True, True, True, True)])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_cpu_replay_of_stored_operands_equals_every_candidate_column(width, fresh):
    torch.manual_seed(7035 + width)
    requests, slots, layers = 4, 6, 2
    lengths = (width, 0, width, width)                      # request 1 inactive (zero query length, count 0)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = (1, 2, 3, 4)                                      # one committed row per request; 0 = null, 5 = bystander
    convolved = _round(torch.randn(layers, tokens, CHANNELS) * .3)
    a = _round(torch.randn(layers, tokens, VALUES) * .4)
    b = _round(torch.randn(layers, tokens, VALUES) * .4)
    a_log = torch.randn(layers, VALUES) * .2
    dt_bias = torch.randn(layers, VALUES) * .1
    initial = torch.randn(layers, slots, VALUES, DIM, DIM) * .025
    # Signed-zero fixtures: a head tile of -0.0, a head tile of alternating +0.0/-0.0 (state*decay keeps the sign, the
    # products with k produce both zero signs, and sums of mixed zero signs round to +0.0).
    initial[:, rows[0], 3] = -0.0
    initial[:, rows[0], 4, :, ::2], initial[:, rows[0], 4, :, 1::2] = 0.0, -0.0
    initial[:, rows[3], 9] = 0.0
    initial[:, rows[3], 9, :, 0] = torch.randn(layers, DIM) * .05            # one-hot rows: sum = state[:, 0] * k[0]
    # Cancellation fixture (widths > 1): request 2 is fresh in both patterns, so after its first token the state is
    # delta[:, None] * k[None, :]. Its second token uses a one-hot k for key head 0 and a value equal, by the same
    # expression, to the dot product the update computes: value - sum is exactly 0.0 and delta is a signed zero.
    if width > 1:
        second = starts[2] + 1
        for layer in range(layers):
            convolved[layer, second, KEYS * DIM:KEYS * DIM + DIM] = 0.0
            convolved[layer, second, KEYS * DIM] = 1.0
            k1, v1, decay1, beta1 = _operands(convolved[layer], a[layer], b[layer], a_log[layer], dt_bias[layer], starts[2])
            k2, _, _, _ = _operands(convolved[layer], a[layer], b[layer], a_log[layer], dt_bias[layer], second)
            for head in range(VALUES // KEYS):                              # the value heads of key head 0
                state1 = _step(torch.zeros(DIM, DIM), k1[0], v1[head], decay1[head], beta1[head])
                convolved[layer, second, 2 * KEYS * DIM + head * DIM:2 * KEYS * DIM + (head + 1) * DIM] = state1[:, 0] * k2[0, 0]
        convolved[:, second, 2 * KEYS * DIM + 5 * DIM:2 * KEYS * DIM + 5 * DIM + 8] = -0.0      # signed-zero values

    # Path 1: today's verification, candidate j of request r = sequential state after inputs 0..j from the row.
    candidates = {}
    for layer in range(layers):
        for request in range(requests):
            if lengths[request] == 0:
                continue
            state = torch.zeros(VALUES, DIM, DIM) if fresh[request] else initial[layer, rows[request]].clone()
            for position in range(lengths[request]):
                token = starts[request] + position
                k, value, decay, beta = _operands(convolved[layer], a[layer], b[layer], a_log[layer], dt_bias[layer], token)
                for head in range(VALUES):
                    state[head] = _step(state[head], k[head // (VALUES // KEYS)], value[head], decay[head], beta[head])
                candidates[layer, request, position] = state.clone()

    # Path 2: the REPLAY-mode recurrence stores the operands it consumed in the kernel's layout, nothing else.
    operands = torch.full((layers, tokens, OPERAND), float("nan"))
    for layer in range(layers):
        for request in range(requests):
            for position in range(lengths[request]):
                token = starts[request] + position
                k, value, decay, beta = _operands(convolved[layer], a[layer], b[layer], a_log[layer], dt_bias[layer], token)
                for head in range(VALUES):
                    key_head = head // (VALUES // KEYS)
                    if head % (VALUES // KEYS) == 0:
                        operands[layer, token, OPERAND_KEY + key_head * DIM:OPERAND_KEY + (key_head + 1) * DIM] = k[key_head]
                    operands[layer, token, OPERAND_VALUE + head * DIM:OPERAND_VALUE + (head + 1) * DIM] = value[head]
                    operands[layer, token, OPERAND_DECAY + head] = decay[head]
                    operands[layer, token, OPERAND_BETA + head] = beta[head]
    assert not operands[:, :tokens].isnan().any(), "every operand word of a live token is stored"

    # Path 2 continued: the replay commit reads back exactly those words, `count` steps from the committed row.
    def replay(pool, counts):
        for layer in range(layers):
            for request in range(requests):
                row, count = rows[request], counts[request]
                if count <= 0 or row <= 0:
                    continue
                for head in range(VALUES):
                    key_head = head // (VALUES // KEYS)
                    state = torch.zeros(DIM, DIM) if fresh[request] else pool[layer, row, head].clone()
                    for step in range(count):
                        word = operands[layer, starts[request] + step]
                        state = _step(state, word[OPERAND_KEY + key_head * DIM:OPERAND_KEY + (key_head + 1) * DIM],
                                      word[OPERAND_VALUE + head * DIM:OPERAND_VALUE + (head + 1) * DIM],
                                      word[OPERAND_DECAY + head], word[OPERAND_BETA + head])
                    pool[layer, row, head] = state

    for accepted in range(1, width + 1):
        pool = initial.clone()
        counts = [accepted if length else 0 for length in lengths]
        replay(pool, counts)
        for layer in range(layers):
            for request in range(requests):
                if lengths[request]:
                    assert _same_bits(pool[layer, rows[request]], candidates[layer, request, accepted - 1])
                else:
                    assert _same_bits(pool[layer, rows[request]], initial[layer, rows[request]])
            assert _same_bits(pool[layer, 0], initial[layer, 0]) and _same_bits(pool[layer, 5], initial[layer, 5])
    pool = initial.clone()
    replay(pool, [0] * requests)
    assert _same_bits(pool, initial), "zero counts leave every row untouched"


# --------------------------------------------------------------------------------------------------------------------
# CUDA (V100) tests
# --------------------------------------------------------------------------------------------------------------------

_LAYOUT = re.compile(r"#[a-z_]*[0-9]* = #ttg\.[a-z_]*<\{[^>]*\}>")
# A floating-point fused multiply-add in PTX, with or without a guard predicate, scalar or packed, any rounding /
# ftz / sat modifiers; integer mad (mad.lo / mad.wide / mad.hi on s/u types, address arithmetic) does not match.
_FP_FUSED = re.compile(r"^\s*(?:@!?%p\d+\s+)?(?:fma|mad)(?:\.[a-z0-9]+)*\.(?:f16|f16x2|bf16|bf16x2|f32|f64)\b.*$", re.M)


def test_fp_fusion_pattern_cases():
    caught = ["\tfma.rn.f32 \t%r1, %r2, %r3, %r4;", "\t@%p1 fma.rn.f32 \t%r1, %r2, %r3, %r4;",
              "\t@!%p12 fma.rn.ftz.f32 \t%r1, %r2, %r3, %r4;", "\tfma.rn.f16x2 \t%r1, %r2, %r3, %r4;",
              "\tfma.rn.bf16 \t%rs1, %rs2, %rs3, %rs4;", "\tmad.f32 \t%r1, %r2, %r3, %r4;", "\tmad.rn.f64 \t%fd1, %fd2, %fd3, %fd4;",
              "\tfma.rn.sat.f32 \t%r1, %r2, %r3, %r4;"]
    ignored = ["\tmad.lo.s64 \t%rd1, %rd2, 16, %rd3;", "\tmad.wide.u32 \t%rd4, %r1, 4, %rd5;",
               "\t@%p3 mad.wide.s32 \t%rd4, %r1, 4, %rd5;", "\tmad.hi.u16 \t%rs1, %rs2, %rs3, %rs4;",
               "\tmul.rn.f32 \t%r1, %r2, %r3;", "\tadd.rn.f32 \t%r1, %r2, %r3;", "\t// fma.rn.f32 in a comment is still a line: see below"]
    for line in caught:
        assert _FP_FUSED.search(line), line
    for line in ignored[:-1]:
        assert not _FP_FUSED.search(line), line
    assert len(_FP_FUSED.findall("\n".join(caught + ignored[:-1]))) == len(caught)

cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                          reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned V100")


@pytest.fixture
def device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    result = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(result) != (7, 0):
        pytest.skip("requires a real sm_70 GPU")
    return result


def _module(generator, device, activation="silu"):
    def random(shape, scale, *, rounded=True):
        value = torch.randn(shape, generator=generator) * scale
        return _round(value) if rounded else value

    norm = _round(1 + random((DIM,), .1, rounded=False))
    return SimpleNamespace(num_k_heads=KEYS, num_v_heads=VALUES, head_k_dim=DIM, head_v_dim=DIM, conv_dim=CHANNELS,
        conv_kernel_size=4, activation="silu", conv1d=SimpleNamespace(weight=random((CHANNELS, 1, 4), .2).to(device), bias=None),
        A_log=random((VALUES,), .2, rounded=False).to(device), dt_bias=random((VALUES,), .1, rounded=False).to(device),
        norm=SimpleNamespace(weight=norm.to(device), activation=activation, variance_epsilon=1e-6))


def _scratch(generator, device, capacity):
    def random(shape, scale):
        return _round(torch.randn(shape, generator=generator) * scale)

    packed = random((capacity, CHANNELS + VALUES * DIM), .3).to(device)
    gates = random((capacity, 2 * VALUES), .4).to(device)
    return SimpleNamespace(projected_qkv=packed[:, :CHANNELS], z=packed[:, CHANNELS:], a=gates[:, VALUES:], b=gates[:, :VALUES],
        convolved_qkv=torch.full((capacity, CHANNELS), 11.0, device=device),
        recurrence=torch.full((capacity, VALUES * DIM), 13.0, device=device),
        gated=torch.full((capacity, VALUES * DIM), 17.0, device=device))


class _Pool:
    """Two linear-attention layers in one byte-pool row: [conv L0 | recurrent L0 | conv L1 | recurrent L1 | pad]."""

    def __init__(self, generator, device, slots, history, layers=2, pad=64):
        self.layers, self.history = layers, history
        conv, rec = CHANNELS * history, VALUES * DIM * DIM
        self.floats = torch.full((slots, layers * (conv + rec) + pad), 23.0, device=device)
        self.backing = self.floats.view(torch.uint8)
        self.states = []
        for layer in range(layers):
            base = layer * (conv + rec)
            convolution = self.floats[:, base:base + conv].view(slots, CHANNELS, history)
            recurrent = self.floats[:, base + conv:base + conv + rec].view(slots, VALUES, DIM, DIM)
            convolution.copy_(_round(torch.randn(convolution.shape, generator=generator) * .1).to(device))
            recurrent.copy_((torch.randn(recurrent.shape, generator=generator) * .025).to(device))
            recurrent[0].fill_(31)                       # the null row is never read nor written
            self.states.append(SimpleNamespace(convolution=convolution, recurrent=recurrent))
        self.pad = self.floats[:, -pad:]


def _metadata(device, requests, columns):
    def zeros(shape, dtype=torch.int32):
        return torch.zeros(shape, dtype=dtype, device=device)

    return SimpleNamespace(query_start_loc=zeros((requests + 1,)),
        convolution_slots=torch.full((requests,), -1, dtype=torch.int64, device=device),
        fresh=zeros((requests,), torch.bool), accepted_tokens=torch.ones(requests, dtype=torch.int32, device=device),
        state_indices=zeros((requests, columns), torch.int64), initial_state_indices=zeros((requests,), torch.int64),
        final_state_indices=zeros((requests,), torch.int64), checkpoint_state_indices=zeros((requests,), torch.int64),
        checkpoint_offsets=zeros((requests,)), num_seqs=zeros((1,)), num_tokens=zeros((1,)))


def _set(meta, lengths, slots, fresh, columns):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    device = meta.num_tokens.device
    meta.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
    meta.convolution_slots.copy_(torch.tensor(slots, dtype=torch.int64, device=device))
    meta.fresh.copy_(torch.tensor(fresh, dtype=torch.bool, device=device))
    meta.state_indices.copy_(torch.tensor(columns, dtype=torch.int64, device=device))
    meta.accepted_tokens.fill_(1)
    meta.num_seqs.fill_(len(lengths))
    meta.num_tokens.fill_(starts[-1])


@cuda
@pytest.mark.parametrize("width", [1, 4, 5])
@torch.no_grad()
def test_replay_commit_matches_candidate_columns_byte_for_byte(device, width):
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence, PreparedSM70ReplayCommit, operand_floats

    assert operand_floats(KEYS, VALUES) == OPERAND
    generator = torch.Generator().manual_seed(48128 + width)
    requests, layers = 4, 2
    modules = [_module(generator, device) for _ in range(layers)]
    scratch = _scratch(generator, device, requests * width)
    # Candidate-column pool: w private rows per request. Replay pool: one committed row per request (+ a bystander).
    old = _Pool(generator, device, 1 + requests * width, width + 2)
    new = _Pool(generator, device, 2 + requests, width + 2)
    canonical_old = [1 + request * width for request in range(requests)]
    canonical_new = [1 + request for request in range(requests)]
    # Signed-zero and cancellation fixtures alongside the random inputs (the comparisons below are bit-for-bit, so a
    # +0.0/-0.0 difference between the candidate path and the replay would fail them):
    #  - request 0: head 3 state tile all -0.0, head 4 alternating +0.0/-0.0, head 9 one-hot rows;
    #  - request 0, key head 2 (value heads 6..8): the k inputs (conv window and this round's channels) are exact zeros
    #    with a -0.0 sprinkled in, so k normalises to signed zeros, state*k products carry both zero signs and the
    #    reduction cancels to a signed zero before the subtraction;
    #  - request 3: a value-head tile of exact zeros (state*decay = 0, delta*k is the whole update).
    key2 = slice(KEYS * DIM + 2 * DIM, KEYS * DIM + 3 * DIM)
    for layer in range(layers):
        tile = old.states[layer].recurrent[canonical_old[0]]
        tile[3] = -0.0
        tile[4, :, ::2], tile[4, :, 1::2] = 0.0, -0.0
        tile[9] = 0.0
        tile[9, :, 0] = torch.randn(DIM, generator=generator).to(device) * .05
        old.states[layer].recurrent[canonical_old[3]][11] = 0.0
        old.states[layer].convolution[canonical_old[0], key2] = 0.0
        old.states[layer].convolution[canonical_old[0], key2.start, :] = -0.0
    scratch.projected_qkv[:width, key2] = 0.0                          # request 0 packs first: tokens 0..width-1
    scratch.projected_qkv[:width, key2.start + 1] = -0.0
    for layer in range(layers):
        for request in range(requests):
            new.states[layer].recurrent[canonical_new[request]].copy_(old.states[layer].recurrent[canonical_old[request]])
            new.states[layer].convolution[canonical_new[request]].copy_(old.states[layer].convolution[canonical_old[request]])
    operands = torch.full((layers, requests * width, OPERAND), float("nan"), device=device)
    meta_old, meta_new = _metadata(device, requests, width), _metadata(device, requests, 1)
    sequences = []
    for layer in range(layers):
        before = PreparedSM70GDNSequence(modules[layer], scratch, old.states[layer], meta_old, mode="decode", norm_weight_bf16=True)
        after = PreparedSM70GDNSequence(modules[layer], scratch, new.states[layer], meta_new, mode="decode",
                                        norm_weight_bf16=True, replay=operands[layer])
        for operation in (before, after):
            for program in operation.prepared.programs:
                assert program.asm["cubin"] and ".target sm_70" in program.asm["ptx"]
        sequences.append((before, after))
    counts = torch.zeros(requests, dtype=torch.int32, device=device)
    commit = PreparedSM70ReplayCommit(new.backing, {layer: new.states[layer].recurrent for layer in range(layers)},
                                      operands, torch.tensor(canonical_new, dtype=torch.int64, device=device), counts,
                                      meta_new.query_start_loc, meta_new.fresh, key_heads=KEYS, value_heads=VALUES)
    assert commit.operation.compiled.asm["cubin"] and ".target sm_70" in commit.operation.compiled.asm["ptx"]
    # The programs that actually run (Triton's launch specialization on alignment/divisibility, not a synthetic
    # compile) must share one tile layout, so the per-row key sum lowers to the same association in the verification
    # and in the replay: four consecutive keys per thread, 32 threads per row, one row per warp. A runtime-loaded,
    # unaligned replay base once gave _replay a one-key-per-thread layout and a different sum tree (1-2 ulp).
    expected = "sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]"
    layouts = {}
    for name, program in (("candidate", sequences[0][0].prepared.programs[1]),
                          ("replay_mode", sequences[0][1].prepared.programs[1]), ("replay", commit.operation.compiled)):
        assert program.metadata.name in ("_recurrence", "_replay"), (name, program.metadata.name)
        layouts[name] = sorted(set(_LAYOUT.findall(program.asm["ttgir"])))
    assert layouts["candidate"] == layouts["replay_mode"] == layouts["replay"], layouts
    assert any(expected in layout for layout in layouts["replay"]), layouts["replay"]
    # No floating-point fused multiply-add in the replay commit, predicated or not, scalar or packed (integer
    # mad.lo/mad.wide are address arithmetic; see _FP_FUSED and test_fp_fusion_pattern_cases).
    fused = _FP_FUSED.findall(commit.operation.compiled.asm["ptx"])
    assert not fused, ("replay commit must not fuse floating-point multiply-adds", fused[:4])

    lengths, fresh = (width, 0, width, width), (False, False, True, False)      # request 1 inactive this round
    columns = [[canonical_old[request] + column for column in range(width)] for request in range(requests)]
    _set(meta_old, lengths, canonical_old, fresh, columns)
    _set(meta_new, lengths, canonical_new, fresh, [[row] for row in canonical_new])
    initial = [state.recurrent.clone() for state in new.states]
    outputs = {}
    for layer, (before, after) in enumerate(sequences):
        before()
        torch.cuda.synchronize()
        outputs[layer] = tuple(getattr(scratch, name).clone() for name in ("convolved_qkv", "recurrence", "gated"))
        after()
        torch.cuda.synchronize()
        # Same convolution, recurrence outputs and gated outputs; no recurrent state written; conv windows identical.
        for name, expected in zip(("convolved_qkv", "recurrence", "gated"), outputs[layer], strict=True):
            assert _same_bits(getattr(scratch, name), expected), (layer, name)
        assert _same_bits(new.states[layer].recurrent, initial[layer]), "verification wrote recurrent state"
        for request in range(requests):
            assert _same_bits(new.states[layer].convolution[canonical_new[request]],
                              old.states[layer].convolution[canonical_old[request]])
    packed = sum(lengths)
    assert not operands[:, :packed].isnan().any(), "every packed live token stored all of its operands"
    assert operands[:, packed:].isnan().all(), "no token beyond the packed count was touched"

    for accepted in range(1, width + 1):
        for layer in range(layers):
            new.states[layer].recurrent.copy_(initial[layer])
        counts.copy_(torch.tensor([accepted if length else 0 for length in lengths], dtype=torch.int32, device=device))
        commit()
        torch.cuda.synchronize()
        for layer in range(layers):
            for request in range(requests):
                row = canonical_new[request]
                if lengths[request]:
                    assert _same_bits(new.states[layer].recurrent[row],
                                      old.states[layer].recurrent[columns[request][accepted - 1]]), (accepted, layer, request)
                else:
                    assert _same_bits(new.states[layer].recurrent[row], initial[layer][row])
            assert _same_bits(new.states[layer].recurrent[0], initial[layer][0])
            assert _same_bits(new.states[layer].recurrent[-1], initial[layer][-1])
    assert torch.all(new.pad == 23) and torch.all(old.pad == 23)


@cuda
@torch.no_grad()
def test_replay_commit_captures_without_allocation_and_skips_zero_counts(device):
    from ignition.components.sm70_gdn import PreparedSM70ReplayCommit

    generator = torch.Generator().manual_seed(2026)
    requests, width, layers = 4, 5, 2
    pool = _Pool(generator, device, 1 + requests, width + 2)
    operands = (torch.randn((layers, requests * width, OPERAND), generator=generator) * .1).to(device)
    rows = torch.arange(1, 1 + requests, dtype=torch.int64, device=device)
    counts = torch.zeros(requests, dtype=torch.int32, device=device)
    starts = torch.arange(0, requests * width + 1, width, dtype=torch.int32, device=device)
    fresh = torch.zeros(requests, dtype=torch.bool, device=device)
    commit = PreparedSM70ReplayCommit(pool.backing, {layer: pool.states[layer].recurrent for layer in range(layers)},
                                      operands, rows, counts, starts, fresh, key_heads=KEYS, value_heads=VALUES)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit()
    before = pool.floats.clone()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    assert _same_bits(pool.floats, before), "zero counts leave the pool untouched"
    counts.copy_(torch.tensor([1, 0, 5, 3], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    changed = pool.floats != before
    for request, count in enumerate((1, 0, 5, 3)):
        assert bool(changed[1 + request].any()) == (count > 0), request
    # Only the recurrent regions of committed rows change; convolution regions and the null row never do.
    conv = CHANNELS * (width + 2)
    for layer in range(layers):
        base = layer * (conv + VALUES * DIM * DIM)
        assert not changed[:, base:base + conv].any(), "convolution regions untouched"
    assert not changed[0].any() and torch.all(pool.pad == 23)
    graph.reset()


@cuda
def test_replay_rejections(device):
    from ignition.components.sm70_gdn import PreparedSM70GDNSequence, PreparedSM70ReplayCommit

    generator = torch.Generator().manual_seed(11)
    module, scratch = _module(generator, device), _scratch(generator, device, 8)
    pool = _Pool(generator, device, 6, 4, layers=1)
    operands = torch.zeros((1, 8, OPERAND), device=device)
    with pytest.raises(ValueError, match="decode"):
        PreparedSM70GDNSequence(module, scratch, pool.states[0], _metadata(device, 4, 1), mode="prefill",
                                norm_weight_bf16=True, replay=operands[0])
    with pytest.raises(ValueError, match="one committed state column"):
        PreparedSM70GDNSequence(module, scratch, pool.states[0], _metadata(device, 4, 2), mode="decode",
                                norm_weight_bf16=True, replay=operands[0])
    with pytest.raises(ValueError, match="one committed state column"):
        PreparedSM70GDNSequence(module, scratch, pool.states[0], _metadata(device, 4, 1), mode="decode",
                                norm_weight_bf16=True, replay=operands[0, :4])
    with pytest.raises(ValueError, match="state_indices columns"):
        PreparedSM70GDNSequence(module, scratch, pool.states[0], _metadata(device, 4, 1), mode="decode",
                                norm_weight_bf16=True)
    rows = torch.arange(1, 5, dtype=torch.int64, device=device)
    counts = torch.zeros(4, dtype=torch.int32, device=device)
    starts = torch.arange(0, 9, 2, dtype=torch.int32, device=device)
    fresh = torch.zeros(4, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="operand row"):
        PreparedSM70ReplayCommit(pool.backing, {0: pool.states[0].recurrent}, operands[:, :, :100], rows, counts, starts,
                                 fresh, key_heads=KEYS, value_heads=VALUES)
    with pytest.raises(ValueError, match="views of the supplied pool"):
        PreparedSM70ReplayCommit(pool.backing, {0: pool.states[0].recurrent.clone()}, operands, rows, counts, starts,
                                 fresh, key_heads=KEYS, value_heads=VALUES)
    # The replay base must be provably 16-float aligned (that alignment is what gives _replay the verification's
    # tile layout): a recurrent view at an unaligned offset, a pool whose row stride is not a multiple of 16 floats,
    # and a pool whose base address is not 64-byte aligned are all refused.
    rec = VALUES * DIM * DIM
    floats = torch.zeros((6, rec + 32), device=device)
    with pytest.raises(ValueError, match="aligned"):
        PreparedSM70ReplayCommit(floats.view(torch.uint8), {0: floats[:, 1:1 + rec].view(6, VALUES, DIM, DIM)}, operands,
                                 rows, counts, starts, fresh, key_heads=KEYS, value_heads=VALUES)
    odd = torch.zeros((6, rec + 4), device=device)
    with pytest.raises(ValueError, match="aligned"):
        PreparedSM70ReplayCommit(odd.view(torch.uint8), {0: odd[:, :rec].view(6, VALUES, DIM, DIM)}, operands,
                                 rows, counts, starts, fresh, key_heads=KEYS, value_heads=VALUES)
    shifted = torch.zeros(6 * (rec + 16) * 4 + 16, dtype=torch.uint8, device=device)[16:].view(6, (rec + 16) * 4)
    assert shifted.data_ptr() % 64 == 16
    with pytest.raises(ValueError, match="aligned"):
        PreparedSM70ReplayCommit(shifted, {0: shifted.view(torch.float32)[:, :rec].view(6, VALUES, DIM, DIM)}, operands,
                                 rows, counts, starts, fresh, key_heads=KEYS, value_heads=VALUES)
