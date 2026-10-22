"""Qwen GDN sequence arithmetic on Volta with FP32 storage and BF16 rounding.

The FP32 tensors hold BF16-rounded activations and chronological convolution
history. Recurrence remains FP32, value by key. Decode writes one state column
per candidate; packed prefill supports distinct initial, final and checkpoint slots.
Metadata retains DeltaMetadata's scheduler-owned bounds and unique-slot rules.
No projection or model scheduling is implemented here.

Replay-commit (speculative verification): a decode sequence prepared with a
``replay`` operand buffer stores, per token and layer, the FP32 operands its
recurrence step consumed (normalized k, value, decay, beta) and writes no
candidate state at all; the committed row stays the untouched initial state.
``PreparedSM70ReplayCommit`` then replays each request's accepted prefix from
that row with the same sequential arithmetic (same tile, warps, expression
order, no FP contraction), for all layers in one launch, so the committed row
equals the state of ``count`` ordinary single-token decodes byte for byte.
"""

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from b12x._lib.program_cache import program_cache

from ignition.preparation.triton import launch_triton, warmup_triton


@triton.jit
def _bf16(value):
    """Round FP32 to BF16, retaining an FP32 value without BF16 instructions."""
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = tl.where(nan, (bits & 0xFFFF0000) | 0x00400000, rounded)
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _convolution(X, Weight, History, Starts, Slots, Fresh, Accepted, NumSeqs, NumTokens, Output,
                 x_stride, weight_stride, history_stride, channel_stride,
                 CHANNELS: tl.constexpr, HISTORY: tl.constexpr, PREFILL: tl.constexpr,
                 BLOCK: tl.constexpr):
    sequence = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = channels < CHANNELS
    if sequence < tl.load(NumSeqs):
        start = tl.load(Starts + sequence)
        end = tl.load(Starts + sequence + 1)
        slot = tl.load(Slots + sequence).to(tl.int64)
        if end > start:
            if slot >= 0:
                offsets = slot * history_stride + channels * channel_stride
                fresh = tl.load(Fresh + sequence)
                selected = 0
                if not PREFILL:
                    selected = tl.load(Accepted + sequence) - 1
                h0 = tl.load(History + offsets + selected, mask & ~fresh, other=0.0)
                h1 = tl.load(History + offsets + selected + 1, mask & ~fresh, other=0.0)
                h2 = tl.load(History + offsets + selected + 2, mask & ~fresh, other=0.0)
                w0 = tl.load(Weight + channels * weight_stride, mask, other=0.0)
                w1 = tl.load(Weight + channels * weight_stride + 1, mask, other=0.0)
                w2 = tl.load(Weight + channels * weight_stride + 2, mask, other=0.0)
                w3 = tl.load(Weight + channels * weight_stride + 3, mask, other=0.0)
                for token in range(start, end):
                    value = _bf16(tl.load(X + token.to(tl.int64) * x_stride + channels, mask, other=0.0))
                    convolved = _bf16(((h0 * w0 + h1 * w1) + h2 * w2) + value * w3)
                    activated = _bf16(convolved / (1.0 + tl.exp(-convolved)))
                    tl.store(Output + token.to(tl.int64) * CHANNELS + channels, activated, mask)
                    h0, h1, h2 = h1, h2, value
                    if not PREFILL:
                        # Window i:i+3 is the history after candidate i. Load
                        # the selected old window before overwriting any column.
                        if token == start:
                            tl.store(History + offsets, h0, mask)
                            tl.store(History + offsets + 1, h1, mask)
                        tl.store(History + offsets + token - start + 2, h2, mask)
                if PREFILL:
                    tl.store(History + offsets, h0, mask)
                    tl.store(History + offsets + 1, h1, mask)
                    tl.store(History + offsets + 2, h2, mask)
                for column in tl.static_range(3, HISTORY):
                    unused = True
                    if not PREFILL:
                        unused = column >= end - start + 2
                    tl.store(History + offsets + column, 0.0, mask & unused)
            else:
                for token in range(start, end):
                    tl.store(Output + token.to(tl.int64) * CHANNELS + channels, 0.0, mask)


def operand_floats(key_heads, value_heads):
    """FP32 words one token's replay operands occupy for one layer: k per key head, value per value head, decay, beta."""
    return (key_heads + value_heads) * 128 + 2 * value_heads


@triton.jit
def _recurrence(Mixed, A, B, ALog, DtBias, State, Starts, Indices, Accepted, Initial, Final,
                Checkpoints, CheckpointOffsets, Fresh, NumSeqs, NumTokens, Output, Operands,
                mixed_stride, a_stride, b_stride, state_stride, index_stride,
                KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, PREFILL: tl.constexpr,
                REPLAY: tl.constexpr, ROWS: tl.constexpr):
    tile, request_head = tl.program_id(0), tl.program_id(1)
    request, head = request_head // VALUE_HEADS, request_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    if request < tl.load(NumSeqs):
        start, end = tl.load(Starts + request), tl.load(Starts + request + 1)
        rows = tile * ROWS + tl.arange(0, ROWS)
        keys = tl.arange(0, 128)
        if PREFILL:
            initial = tl.load(Initial + request).to(tl.int64)
            final = tl.load(Final + request).to(tl.int64)
            checkpoint = tl.load(Checkpoints + request).to(tl.int64)
            boundary = tl.load(CheckpointOffsets + request)
        else:
            accepted = tl.load(Accepted + request)
            initial = tl.load(Indices + request * index_stride + accepted - 1).to(tl.int64)
            final = tl.load(Indices + request * index_stride).to(tl.int64)
        if end > start:
            if final == 0:
                for token in range(start, end):
                    tl.store(Output + token.to(tl.int64) * (VALUE_HEADS * 128) + head * 128 + rows, 0.0)
            else:
                elements = head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
                fresh = tl.load(Fresh + request)
                state = tl.load(State + initial * state_stride + elements,
                                mask=(initial != 0) & ~fresh, other=0.0)
                a_scale = tl.exp(tl.load(ALog + head))
                bias = tl.load(DtBias + head)
                for position in range(start, end):
                    token = position.to(tl.int64)
                    q = tl.load(Mixed + token * mixed_stride + key_head * 128 + keys)
                    k = tl.load(Mixed + token * mixed_stride + KEY_HEADS * 128 + key_head * 128 + keys)
                    q = q * (1.0 / tl.sqrt_rn(tl.sum(q * q, 0) + 1.0e-6)) * 0.08838834764831845
                    k = k * (1.0 / tl.sqrt_rn(tl.sum(k * k, 0) + 1.0e-6))
                    a = _bf16(tl.load(A + token * a_stride + head)) + bias
                    softplus = tl.where(a <= 20.0, libdevice.log1p(tl.exp(a)), a)
                    decay = tl.exp(-a_scale * softplus)
                    beta = _bf16(1.0 / (1.0 + tl.exp(-_bf16(tl.load(B + token * b_stride + head)))))
                    value = tl.load(Mixed + token * mixed_stride + 2 * KEY_HEADS * 128 + head * 128 + rows)
                    state = state * decay
                    delta = (value - tl.sum(state * k[None, :], 1)) * beta
                    state = state + delta[:, None] * k[None, :]
                    output = _bf16(tl.sum(state * q[None, :], 1))
                    tl.store(Output + token * (VALUE_HEADS * 128) + head * 128 + rows, output)
                    if PREFILL:
                        if (checkpoint > 0) & (token - start + 1 == boundary):
                            tl.store(State + checkpoint * state_stride + elements, state)
                    elif REPLAY:
                        # Replay-commit: keep exactly the operands this step consumed (one k per key
                        # head, this tile's value rows, the scalar gates); write no candidate state.
                        operand = Operands + token * ((KEY_HEADS + VALUE_HEADS) * 128 + 2 * VALUE_HEADS)
                        tl.store(operand + key_head * 128 + keys, k,
                                 (tile == 0) & (head % (VALUE_HEADS // KEY_HEADS) == 0))
                        tl.store(operand + KEY_HEADS * 128 + head * 128 + rows, value)
                        tl.store(operand + (KEY_HEADS + VALUE_HEADS) * 128 + head, decay, tile == 0)
                        tl.store(operand + (KEY_HEADS + VALUE_HEADS) * 128 + VALUE_HEADS + head, beta, tile == 0)
                    else:
                        destination = tl.load(Indices + request * index_stride + position - start).to(tl.int64)
                        tl.store(State + destination * state_stride + elements, state, destination != 0)
                if PREFILL:
                    tl.store(State + final * state_stride + elements, state)


@triton.jit
def _norm(Hidden, Gate, Weight, Output, NumTokens, gate_stride,
          HEADS: tl.constexpr, EPS: tl.constexpr, SILU: tl.constexpr,
          BF16_WEIGHT: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    columns = tl.arange(0, 128)
    live = row < tl.load(NumTokens)
    offsets = row * (HEADS * 128) + head * 128 + columns
    hidden = tl.load(Hidden + offsets, live, other=0.0)
    z = _bf16(tl.load(Gate + row.to(tl.int64) * gate_stride + head * 128 + columns, live, other=0.0))
    gate = 1.0 / (1.0 + tl.exp(-z))
    if SILU:
        gate = z * gate
    weight = tl.load(Weight + columns)
    normalized = _bf16(hidden * tl.rsqrt(tl.sum(hidden * hidden, 0) / 128.0 + EPS))
    weighted = normalized * weight
    if BF16_WEIGHT:
        weighted = _bf16(weighted)
    value = _bf16(weighted * gate)
    tl.store(Output + offsets, tl.where(live, value, 0.0))


@triton.jit
def _replay(Operands, Backing, Offsets, Rows, Counts, Starts, Fresh, operand_layer_stride,
            STRIDE: tl.constexpr, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, ROWS: tl.constexpr):
    """Replay ``Counts[request]`` stored verification steps from the committed row ``Rows[request]`` in place.

    One program per (key tile, request x value head, layer) carries the same [ROWS x 128] state tile as the decode
    recurrence and applies the same update expressions in the same order to the stored FP32 operands, so the result
    equals the decode recurrence's state after those steps bit for bit. A zero count or null row leaves the row alone.
    """
    tile, request_head, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    request, head = request_head // VALUE_HEADS, request_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    count = tl.load(Counts + request)
    row = tl.load(Rows + request).to(tl.int64)
    if (count > 0) & (row > 0):
        start = tl.load(Starts + request)
        fresh = tl.load(Fresh + request)
        rows = tile * ROWS + tl.arange(0, ROWS)
        keys = tl.arange(0, 128)
        elements = head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
        base = Backing + row * STRIDE + tl.load(Offsets + layer)
        state = tl.load(base + elements, mask=~fresh, other=0.0)
        layer_operands = Operands + layer.to(tl.int64) * operand_layer_stride
        for step in range(0, count):
            operand = layer_operands + (start + step).to(tl.int64) * ((KEY_HEADS + VALUE_HEADS) * 128 + 2 * VALUE_HEADS)
            k = tl.load(operand + key_head * 128 + keys)
            value = tl.load(operand + KEY_HEADS * 128 + head * 128 + rows)
            decay = tl.load(operand + (KEY_HEADS + VALUE_HEADS) * 128 + head)
            beta = tl.load(operand + (KEY_HEADS + VALUE_HEADS) * 128 + VALUE_HEADS + head)
            state = state * decay
            delta = (value - tl.sum(state * k[None, :], 1)) * beta
            state = state + delta[:, None] * k[None, :]
        tl.store(base + elements, state)


@dataclass(frozen=True)
class Query:
    operands: tuple
    key_heads: int
    value_heads: int
    mode: str
    epsilon: float
    activation: str
    norm_weight_bf16: bool
    replay: bool = False


def _launches(tensors, query):
    (x, a, b, z, weight, a_log, bias, norm_weight, starts, slots, fresh, accepted, indices,
     initial, final, checkpoints, offsets, sequences, tokens, history, state,
     convolved, recurrence, gated, *rest) = tensors
    # Without a replay operand buffer the recurrence never dereferences the pointer (REPLAY is false).
    operands = rest[0] if query.replay else state
    requests = starts.numel() - 1
    channels = (2 * query.key_heads + query.value_heads) * 128
    common = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
    yield (_convolution, (x, weight, history, starts, slots, fresh, accepted, sequences, tokens, convolved,
                         x.stride(0), weight.stride(0), history.stride(0), history.stride(1)),
           (requests, triton.cdiv(channels, 128), 1), {**common, "CHANNELS": channels,
            "HISTORY": history.shape[2], "PREFILL": query.mode == "prefill", "BLOCK": 128})
    yield (_recurrence, (convolved, a, b, a_log, bias, state, starts, indices, accepted, initial, final,
                        checkpoints, offsets, fresh, sequences, tokens, recurrence, operands,
                        convolved.stride(0), a.stride(0), b.stride(0), state.stride(0), indices.stride(0)),
           (16, requests * query.value_heads, 1), {**common, "KEY_HEADS": query.key_heads,
            "VALUE_HEADS": query.value_heads, "PREFILL": query.mode == "prefill", "REPLAY": query.replay,
            "ROWS": 8})
    yield (_norm, (recurrence, z, norm_weight, gated, tokens, z.stride(0)),
           (x.shape[0], query.value_heads, 1), {**common, "HEADS": query.value_heads,
            "EPS": query.epsilon, "SILU": query.activation == "silu", "BF16_WEIGHT": query.norm_weight_bf16})


@dataclass(frozen=True)
class _Prepared:
    query: Query
    programs: tuple

    def run(self, tensors):
        for program, (kernel, args, grid, options) in zip(self.programs, _launches(tensors, self.query), strict=True):
            launch_triton(kernel, program, args, grid, **options)


@program_cache(scope="preparation")
def compile_sequence(payload, ordinal):
    from b12x._lib.compile_plan import compile_only_launches, launch_triton as compile_launch

    query = Query(**dict(payload))
    tensors = tuple(torch.empty_strided(shape, stride, dtype=getattr(torch, dtype), device="meta")
                    for shape, stride, dtype in query.operands)
    with torch.cuda.device(ordinal), compile_only_launches():
        return tuple(compile_launch(kernel, grid, *args, **options)
                     for kernel, args, grid, options in _launches(tensors, query))


@torch.library.custom_op("flare::sm70_gdn_sequence", mutates_args=("state", "outputs"))
def _execute(inputs: list[torch.Tensor], state: list[torch.Tensor], outputs: list[torch.Tensor], handle: int) -> None:
    from b12x.preparation.types import plan_from_handle, require_prepared

    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, inputs[0].device).run((*inputs, *state, *outputs))


@_execute.register_fake
def _fake(inputs, state, outputs, handle):
    pass


@dataclass(frozen=True)
class ValidatedSequence:
    """Operand tuples of one validated GDN sequence, in ``_launches`` order (inputs, state, outputs, replay)."""
    inputs: tuple
    state: tuple
    outputs: tuple
    keys: int
    values: int
    epsilon: float
    history: int
    replay: tuple = ()

    @property
    def tensors(self):
        return (*self.inputs, *self.state, *self.outputs, *self.replay)


def validate_sequence(module, scratch, state, metadata, *, mode, norm_weight_bf16, checkpoint_export, replay=None):
    """Admit the FP32 token-major operands, state pools and metadata of one prefill or decode sequence.

    ``replay`` (decode only) is this layer's replay operand buffer, one row of ``operand_floats`` FP32 words per
    token of the scratch capacity; the sequence then stores its operands there and writes no candidate state.
    """
    if (mode not in ("decode", "prefill") or type(norm_weight_bf16) is not bool
            or type(checkpoint_export) is not bool):
        raise ValueError("SM70 GDN requires an ordinary decode/prefill mode and explicit norm precision")
    if checkpoint_export and mode != "prefill":
        raise ValueError("SM70 GDN checkpoint export is a prefill operation")
    if replay is not None and mode != "decode":
        raise ValueError("SM70 GDN replay operands are stored by decode (verification) sequences only")
    keys, values = module.num_k_heads, module.num_v_heads
    channels = (2 * keys + values) * 128
    if (keys <= 0 or values <= 0 or values % keys or module.head_k_dim != 128 or module.head_v_dim != 128
            or module.conv_dim != channels or module.conv_kernel_size != 4
            or module.activation != "silu" or module.norm.activation not in ("silu", "sigmoid")
            or module.conv1d.bias is not None):
        raise ValueError("SM70 GDN requires grouped 128-element heads and a bias-free SiLU conv4")
    x = scratch.projected_qkv
    if x.ndim != 2 or x.shape[1] != channels or x.shape[0] <= 0:
        raise ValueError("SM70 GDN requires nonempty packed QKV token rows")
    capacity, device = x.shape[0], x.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError("SM70 GDN requires a Volta CUDA device")
    for field, width in (("projected_qkv", channels), ("convolved_qkv", channels),
                         ("a", values), ("b", values), ("z", values * 128),
                         ("recurrence", values * 128), ("gated", values * 128)):
        tensor = getattr(scratch, field)
        if (tensor.shape != (capacity, width) or tensor.dtype != torch.float32
                or tensor.device != device or tensor.stride(1) != 1 or tensor.stride(0) < width
                or (field in ("convolved_qkv", "recurrence", "gated") and not tensor.is_contiguous())):
            raise ValueError(f"SM70 GDN scratch {field} requires its FP32 token-major layout")
    if state.recurrent.ndim != 4 or state.convolution.ndim != 3:
        raise ValueError("SM70 GDN requires rank-four recurrence and rank-three convolution history")
    slots = state.recurrent.shape[0]
    history = state.convolution.shape[2]
    if (state.recurrent.shape != (slots, values, 128, 128) or slots < 2
            or state.recurrent.stride()[1:] != (16384, 128, 1)
            or state.recurrent.stride(0) < values * 16384
            or not 3 <= history <= 10 or state.convolution.shape != (slots, channels, history)
            or state.convolution.stride()[1:] != (history, 1)
            or state.convolution.stride(0) < channels * history
            or any(t.device != device or t.dtype != torch.float32 for t in (state.convolution, state.recurrent))):
        raise ValueError("SM70 GDN requires FP32 recurrence and three chronological FP32 history columns plus zero through seven drafts")
    conv_weight = module.conv1d.weight.squeeze(1)
    for tensor, shape in ((conv_weight, (channels, 4)), (module.A_log, (values,)),
                          (module.dt_bias, (values,)), (module.norm.weight, (128,))):
        if tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != device or not tensor.is_contiguous():
            raise ValueError("SM70 GDN canonical parameters must retain their values in contiguous FP32 storage")
    requests = metadata.query_start_loc.numel() - 1
    columns = metadata.state_indices.shape[1] if metadata.state_indices.ndim == 2 else 0
    if not 1 <= columns <= history - 2:
        raise ValueError("SM70 GDN state_indices columns exceed convolution history capacity")
    expected = {"query_start_loc": ((requests + 1,), torch.int32),
                "convolution_slots": ((requests,), torch.int64), "fresh": ((requests,), torch.bool),
                "accepted_tokens": ((requests,), torch.int32), "state_indices": ((requests, columns), torch.int64),
                "initial_state_indices": ((requests,), torch.int64), "final_state_indices": ((requests,), torch.int64),
                "checkpoint_state_indices": ((requests,), torch.int64), "checkpoint_offsets": ((requests,), torch.int32),
                "num_seqs": ((1,), torch.int32), "num_tokens": ((1,), torch.int32)}
    if requests <= 0 or (mode == "decode" and replay is None and capacity > requests * columns):
        raise ValueError("SM70 GDN decode token capacity exceeds its state_indices columns")
    if replay is not None and (columns != 1 or replay.shape != (capacity, operand_floats(keys, values))
                               or replay.dtype != torch.float32 or replay.device != device
                               or not replay.is_contiguous()):
        raise ValueError("SM70 GDN replay requires one committed state column and a contiguous FP32 operand row "
                         "per scratch token")
    for field, (shape, dtype) in expected.items():
        tensor = getattr(metadata, field)
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"SM70 GDN metadata {field} has an unsupported shape, dtype or stride")
    if metadata.num_tokens.item() or metadata.num_seqs.item():
        raise ValueError("SM70 GDN preparation requires zero live counts")
    epsilon = float(module.norm.variance_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("SM70 GDN norm epsilon must be finite and positive")
    inputs = (x, scratch.a, scratch.b, scratch.z, conv_weight, module.A_log, module.dt_bias,
              module.norm.weight, metadata.query_start_loc, metadata.convolution_slots, metadata.fresh, metadata.accepted_tokens,
              metadata.state_indices, metadata.initial_state_indices, metadata.final_state_indices,
              metadata.checkpoint_state_indices, metadata.checkpoint_offsets, metadata.num_seqs, metadata.num_tokens)
    return ValidatedSequence(inputs, (state.convolution, state.recurrent),
                             (scratch.convolved_qkv, scratch.recurrence, scratch.gated), keys, values, epsilon, history,
                             () if replay is None else (replay,))


class PreparedSM70GDNSequence:
    """Convolve, update recurrence and normalize already projected Qwen operands.

    All floating operands use FP32 storage. ``norm_weight_bf16`` records whether
    the original norm multiplication rounded to BF16; converting the weight's
    storage to FP32 must not erase that arithmetic distinction. Runtime metadata
    must obey DeltaMetadata's bounds/unique-owner rules. Decode resumes from
    ``state_indices[:, accepted_tokens - 1]`` and writes every candidate's FP32
    checkpoint, or, given ``replay`` operand storage, stores each step's operands
    for ``PreparedSM70ReplayCommit`` and writes no state (replay-commit).
    A candidate's convolution window starts at its zero-based index.
    A supplied preparation session declares compilation jobs and warms only
    after its compile phase. Without a session, kernels compile for direct tests.
    """

    def __init__(self, module, scratch, state, metadata, *, mode, norm_weight_bf16,
                 session=None, name=None, checkpoint_export=False, replay=None):
        validated = validate_sequence(module, scratch, state, metadata, mode=mode, norm_weight_bf16=norm_weight_bf16,
                                      checkpoint_export=checkpoint_export, replay=replay)
        device = validated.inputs[0].device
        self.inputs, self.state, self.outputs = validated.inputs, validated.state, validated.outputs
        self.replay = validated.replay
        self.tensors = validated.tensors
        query = Query(tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype).removeprefix("torch.")) for t in self.tensors),
                      validated.keys, validated.values, mode, validated.epsilon, module.norm.activation, norm_weight_bf16,
                      replay is not None)
        self.plan = None
        if session is None:
            programs = tuple(warmup_triton(kernel, args, grid, **options)
                             for kernel, args, grid, options in _launches(self.tensors, query))
            self.prepared = _Prepared(query, programs)
        else:
            if not name:
                raise ValueError("prepared SM70 GDN requires an operation name")
            from b12x._lib.compile_plan import attach_programs, load_programs
            from b12x._lib.compile_pool import CompileJob
            from b12x.preparation import PreparedCall
            from ignition.preparation.tuning import declare
            from .native_operator import native_contract, native_plan

            contract = native_contract(component_id="flare.sm70_gdn_sequence", query_type=Query,
                                       tactics=lambda query: ("sequential",), semantic_version=3)
            payload = contract.encode_query(query)

            def materialize(tactic):
                programs = compile_sequence(payload, device.index)
                load_programs(programs)
                return attach_programs(_Prepared(query, programs), programs)

            self.plan = native_plan(contract=contract, query=query, device=device,
                scratch_bytes=lambda tactic: 0, materialize=materialize,
                compile_jobs=lambda tactic, detected: (CompileJob.create(
                    "ignition.components.sm70_gdn:compile_sequence", payload, detected.ordinal),))
            declare(session, (self.plan.request(name=name,
                prepare_call=lambda prepared: PreparedCall(run=lambda: prepared.run(self.tensors), owners=(self.tensors,))),))

    def __call__(self):
        if self.plan is None:
            self.prepared.run(self.tensors)
        else:
            _execute(list(self.inputs), list(self.state), [*self.outputs, *self.replay], self.plan._handle)
        return self.outputs[2]

    run_sequence = __call__


@triton.jit
def _commit_convolution(Backing, Offsets, Rows, Counts, STRIDE: tl.constexpr,
                        ROW_STRIDE: tl.constexpr, CHANNELS: tl.constexpr,
                        COLUMNS: tl.constexpr, BLOCK: tl.constexpr):
    layer, row, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + row)
    slot = tl.load(Rows + row * ROW_STRIDE).to(tl.int64)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    active = (count > 0) & (slot > 0) & (channel < CHANNELS)
    base = Backing + slot * STRIDE + tl.load(Offsets + layer) + channel * COLUMNS
    # All three source values must be read before the overlapping destination
    # is written. The suffix is private candidate state and is normalized away.
    first = tl.load(base + count - 1, active, other=0.0)
    second = tl.load(base + count, active, other=0.0)
    third = tl.load(base + count + 1, active, other=0.0)
    tl.store(base, first, active)
    tl.store(base + 1, second, active)
    tl.store(base + 2, third, active)
    for column in tl.static_range(3, COLUMNS):
        tl.store(base + column, 0.0, active)


class PreparedSM70ConvolutionCommit:
    """Normalize accepted FP32 history windows for all layers in one pool.

    ``histories`` maps layer IDs to convolution views of ``backing``. ``rows``
    names canonical pool rows, and ``counts`` contains accepted input counts;
    zero counts leave their rows untouched. Counts and unique positive rows
    follow the same scheduler bounds as DeltaMetadata. Recurrent column copies
    must be limited to recurrent regions so they preserve these histories.
    """

    def __init__(self, backing, histories, rows, counts):
        from ignition.preparation.triton import prepare_triton

        if (backing.ndim != 2 or backing.dtype != torch.uint8 or not backing.is_contiguous()
                or backing.shape[1] % 4 or backing.device.type != "cuda"
                or torch.cuda.get_device_capability(backing.device) != (7, 0)):
            raise ValueError("SM70 convolution commit requires contiguous CUDA byte-pool rows")
        if (rows.ndim != 1 or rows.numel() == 0 or rows.dtype != torch.int64
                or rows.device != backing.device or rows.stride(0) <= 0
                or counts.shape != rows.shape or counts.dtype != torch.int32
                or counts.device != backing.device or not counts.is_contiguous() or not histories):
            raise ValueError("SM70 convolution commit requires state rows, accepted counts and layer histories")
        floats = backing.view(torch.float32)
        first = next(iter(histories.values()))
        if first.ndim != 3:
            raise ValueError("SM70 convolution commit requires rank-three histories")
        slots, channels, columns = first.shape
        if slots != backing.shape[0] or channels <= 0 or not 3 <= columns <= 10:
            raise ValueError("SM70 convolution commit history geometry exceeds its pool or draft capacity")
        offsets = []
        for history in histories.values():
            if (history.shape != first.shape or history.dtype != torch.float32
                    or history.device != backing.device
                    or history.stride() != (floats.stride(0), columns, 1)
                    or history.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()):
                raise ValueError("SM70 convolution histories must be FP32 views of the supplied pool")
            offset = history.storage_offset() - floats.storage_offset()
            if offset < 0 or offset + channels * columns > floats.shape[1]:
                raise ValueError("SM70 convolution history crosses a pool row boundary")
            offsets.append(offset)
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=backing.device)
        self.operation = prepare_triton(_commit_convolution,
            (floats, self.offsets, rows, counts),
            (len(offsets), rows.numel(), triton.cdiv(channels, 128)),
            STRIDE=floats.stride(0), ROW_STRIDE=rows.stride(0), CHANNELS=channels,
            COLUMNS=columns, BLOCK=128, num_warps=4, num_stages=1)

    def __call__(self):
        self.operation()


class PreparedSM70ReplayCommit:
    """Replay accepted verification prefixes into the committed recurrent rows of every layer, in one launch.

    ``recurrents`` maps layer IDs to the FP32 recurrent views ``[slots, value_heads, 128, 128]`` of ``backing``;
    ``operands`` is the ``[layers, tokens, operand_floats]`` buffer the verification sequences of those layers
    filled (layer order = mapping order); ``rows`` names each request's committed pool row, ``counts`` its accepted
    input count, ``starts`` the packed query boundaries and ``fresh`` whether the request began with no history
    (zero initial state), exactly as the verification decode read them. Zero counts and null rows leave their
    rows untouched. Callers validate rows against ``memory_capacity[0]`` before execution, as the dense round's
    batch validator does; this kernel has no bound of its own, like the convolution commit.
    """

    def __init__(self, backing, recurrents, operands, rows, counts, starts, fresh, *, key_heads, value_heads):
        from ignition.preparation.triton import prepare_triton

        if (backing.ndim != 2 or backing.dtype != torch.uint8 or not backing.is_contiguous()
                or backing.shape[1] % 4 or backing.device.type != "cuda"
                or torch.cuda.get_device_capability(backing.device) != (7, 0)):
            raise ValueError("SM70 replay commit requires contiguous CUDA byte-pool rows")
        device = backing.device
        if (rows.ndim != 1 or rows.numel() == 0 or rows.dtype != torch.int64 or counts.shape != rows.shape
                or counts.dtype != torch.int32 or starts.shape != (rows.numel() + 1,) or starts.dtype != torch.int32
                or fresh.shape != rows.shape or fresh.dtype != torch.bool
                or any(t.device != device or not t.is_contiguous() for t in (rows, counts, starts, fresh))):
            raise ValueError("SM70 replay commit requires contiguous committed rows, accepted counts, query boundaries "
                             "and fresh flags")
        if (type(key_heads) is not int or type(value_heads) is not int or key_heads <= 0
                or value_heads <= 0 or value_heads % key_heads):
            raise ValueError("SM70 replay commit requires grouped key/value heads")
        width = operand_floats(key_heads, value_heads)
        if (operands.ndim != 3 or operands.dtype != torch.float32 or operands.device != device
                or operands.shape[0] != len(recurrents) or operands.shape[1] <= 0 or operands.shape[2] != width
                or operands.stride(2) != 1 or operands.stride(1) != width
                or operands.stride(0) < operands.shape[1] * width):
            raise ValueError("SM70 replay commit requires one contiguous FP32 operand row per token for every layer")
        floats = backing.view(torch.float32)
        offsets = []
        for recurrent in recurrents.values():
            if (recurrent.shape != (backing.shape[0], value_heads, 128, 128) or recurrent.dtype != torch.float32
                    or recurrent.device != device or recurrent.stride() != (floats.stride(0), 16384, 128, 1)
                    or recurrent.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()):
                raise ValueError("SM70 replay commit recurrents must be FP32 views of the supplied pool")
            offset = recurrent.storage_offset() - floats.storage_offset()
            if offset < 0 or offset + value_heads * 16384 > floats.shape[1]:
                raise ValueError("SM70 replay commit recurrent crosses a pool row boundary")
            offsets.append(offset)
        if not offsets:
            raise ValueError("SM70 replay commit requires at least one layer")
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        self.operands, self.rows, self.counts, self.starts, self.fresh = operands, rows, counts, starts, fresh
        self.operation = prepare_triton(_replay,
            (operands, floats, self.offsets, rows, counts, starts, fresh, operands.stride(0)),
            (16, rows.numel() * value_heads, len(offsets)),
            STRIDE=floats.stride(0), KEY_HEADS=key_heads, VALUE_HEADS=value_heads, ROWS=8,
            num_warps=4, num_stages=1, enable_fp_fusion=False)

    def __call__(self):
        self.operation()
