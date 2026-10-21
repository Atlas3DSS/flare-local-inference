"""Scalar CUDA paged attention for Volta with FP32 queries and BF16 cache bits.

The device loop visits only the causal prefix of each packed query. Cache state
and inactive rows follow the existing paged-attention component contract. UInt16
pages retain BF16's complete range without BF16 tensors or native instructions.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import benchmark_resource, declare
from .native_operator import native_contract, native_plan


@triton.jit
def _pack_bf16(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = tl.where(nan, bits | 0x00400000, rounded)
    return (rounded >> 16).to(tl.uint16)


@triton.jit
def _unpack_bf16(value):
    return (value.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _write(Key, Value, KCache, VCache, Slots, KS: tl.constexpr, VS: tl.constexpr,
           WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row).to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    valid = (slot >= 0) & (cols < WIDTH)
    k = tl.load(Key + row * KS + cols, valid, 0)
    v = tl.load(Value + row * VS + cols, valid, 0)
    tl.store(KCache + slot * WIDTH + cols, _pack_bf16(k), valid)
    tl.store(VCache + slot * WIDTH + cols, _pack_bf16(v), valid)


@triton.jit
def _attention(Q, K, V, Pages, Lengths, Cu, Out, QS: tl.constexpr, OS: tl.constexpr,
               HEADS: tl.constexpr, KVHEADS: tl.constexpr, DIM: tl.constexpr,
               PAGE: tl.constexpr, PAGECOLS: tl.constexpr, ROWS: tl.constexpr,
               BLOCK: tl.constexpr, WINDOW: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    request, begin, end = -1, 0, 0
    for row in range(ROWS):
        left = tl.load(Cu + row)
        right = tl.load(Cu + row + 1)
        if (token >= left) & (token < right):
            request, begin, end = row, left, right
    dims = tl.arange(0, DIM)
    result = tl.full((DIM,), 0, tl.float32)
    if request >= 0:
        length = tl.load(Lengths + request)
        visible = length - (end - begin) + token - begin + 1
        first = tl.maximum(0, visible - WINDOW - 1) if WINDOW >= 0 else 0
        q = tl.load(Q + token * QS + head * DIM + dims).to(tl.float32)
        kvhead = head // (HEADS // KVHEADS)
        maximum, denominator = float("-inf"), 0.0
        for start in range(first, visible, BLOCK):
            positions = start + tl.arange(0, BLOCK)
            valid = positions < visible
            page = tl.load(Pages + request * PAGECOLS + positions // PAGE, valid, 0).to(tl.int64)
            base = ((page * PAGE + positions % PAGE) * KVHEADS + kvhead) * DIM
            keys = _unpack_bf16(tl.load(K + base[:, None] + dims[None, :], valid[:, None], 0))
            scores = tl.sum(keys * q[None, :], 1) * (1.0 / tl.sqrt(float(DIM)))
            scores = tl.where(valid, scores, float("-inf"))
            updated = tl.maximum(maximum, tl.max(scores, 0))
            alpha = tl.exp(maximum - updated)
            probability = tl.exp(scores - updated)
            values = _unpack_bf16(tl.load(V + base[:, None] + dims[None, :], valid[:, None], 0))
            result = result * alpha + tl.sum(values * probability[:, None], 0)
            denominator = denominator * alpha + tl.sum(probability, 0)
            maximum = updated
        result = result / tl.maximum(denominator, 1e-30)
    tl.store(Out + token * OS + head * DIM + dims, result)


@dataclass(frozen=True)
class Query:
    kind: str
    operands: tuple
    window: int = -1
    row_invariant: bool = False


CONTRACT = native_contract(component_id="flare.sm70.paged_attention", query_type=Query,
    tactics=lambda q: ((256, 4),) if q.kind == "write" else (((32, 4),) if q.row_invariant else ((16, 4), (32, 4))),
    semantic_version=2)


def _invocation(q, tensors, tactic):
    block, warps = tactic
    if q.kind == "write":
        key, value, kc, vc, slots = tensors
        width = key.shape[1] * key.shape[2]
        return _write, (key.shape[0], triton.cdiv(width, block), 1), (*tensors,), dict(
            KS=key.stride(0), VS=value.stride(0), WIDTH=width, BLOCK=block, num_warps=warps)
    query, key, value, pages, lengths, cu, output = tensors
    return _attention, (query.shape[0], query.shape[1], 1), tensors, dict(
        QS=query.stride(0), OS=output.stride(0), HEADS=query.shape[1], KVHEADS=key.shape[2],
        DIM=query.shape[2], PAGE=key.shape[1], PAGECOLS=pages.shape[1], ROWS=pages.shape[0],
        BLOCK=block, WINDOW=q.window, num_warps=warps)


@program_cache(scope="preparation")
def compile_attention(payload, tactic, ordinal):
    query = Query(**dict(payload))
    tensors = tuple(torch.empty_strided(shape, stride, dtype=getattr(torch, dtype), device="meta")
                    for shape, stride, dtype in query.operands)
    kernel, grid, args, kwargs = _invocation(query, tensors, tactic)
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(kernel, grid, *args, **kwargs)


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    program: object

    def run(self, tensors):
        _, grid, args, kwargs = _invocation(self.query, tensors, self.tactic)
        self.program[grid](*args, *(v for k, v in kwargs.items() if k != "num_warps"))


@torch.library.custom_op("flare::sm70_paged_attention", mutates_args=("output",))
def attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pages: torch.Tensor,
           lengths: torch.Tensor, cu: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, q.device).run((q, key, value, pages, lengths, cu, output))


@attend.register_fake
def _fake_attend(q, key, value, pages, lengths, cu, output, handle):
    pass


@torch.library.custom_op("flare::sm70_paged_kv_write", mutates_args=("key_cache", "value_cache"))
def write(key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor,
          value_cache: torch.Tensor, slots: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, key.device).run((key, value, key_cache, value_cache, slots))


@write.register_fake
def _fake_write(key, value, key_cache, value_cache, slots, handle):
    pass


def _declare(kind, tensors, *, session, name, window=-1, row_invariant=False):
    query = Query(kind, tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype).removeprefix("torch."))
                             for t in tensors), window, row_invariant)
    payload = CONTRACT.encode_query(query)
    device = tensors[0].device

    def materialize(tactic):
        program = compile_attention(payload, tactic, device.index)
        load_programs(program)
        return attach_programs(State(query, tactic, program), program)

    plan = native_plan(contract=CONTRACT, query=query, device=device,
        scratch_bytes=lambda _: 0, materialize=materialize,
        compile_jobs=lambda t, d: (CompileJob.create(
            "ignition.components.sm70_attention:compile_attention", payload, t, d.ordinal),))

    def benchmark(state):
        def fixture():
            if kind == "write":
                # Use a private cache so tuning cannot change live model state.
                key, value, kc, vc, slots = tensors
                pages = min(kc.shape[0], max(1, triton.cdiv(slots.numel(), kc.shape[1])))
                shape = (pages, *kc.shape[1:])
                result = [torch.randn_like(key), torch.randn_like(value),
                          torch.zeros(shape, dtype=kc.dtype, device=device),
                          torch.zeros(shape, dtype=vc.dtype, device=device),
                          torch.arange(slots.numel(), device=device, dtype=slots.dtype)]
                if slots.numel() > pages * kc.shape[1]:
                    # Duplicate destinations are not a valid write fixture.
                    result[-1][pages * kc.shape[1]:] = -1
            else:
                q, key, value, pages, lengths, cu, output = tensors
                result = [torch.randn_like(q), key, value, torch.zeros_like(pages),
                          torch.zeros_like(lengths), torch.zeros_like(cu), torch.zeros_like(output)]
                # One live row safely visits the first backed page repeatedly.
                result[4][0] = max(q.shape[0], min(4096, pages.shape[1] * key.shape[1]))
                result[5][1:] = q.shape[0]
            return tuple(result)
        args = benchmark_resource(session, ("sm70_attention", name), fixture)
        seed = args[0].clone()
        return PreparedCall(run=lambda: state.run(args), produce=lambda: args[0].copy_(seed), owners=(args, seed),
                            output=None if kind == "write" else args[-1])

    declare(session, (plan.request(name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(tensors)), benchmark_call=benchmark),))
    return plan


def _device(tensors):
    device = tensors[0].device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError("SM70 attention requires a V100")
    if any(t.device != device for t in tensors):
        raise ValueError("SM70 attention operands must share one CUDA device")


class PreparedAttention:
    def __init__(self, q, key, value, pages, lengths, cu, output, *, session, name, window=-1, row_invariant=False):
        tensors = (q, key, value, pages, lengths, cu, output)
        _device(tensors)
        if (q.dtype != torch.float32 or output.dtype != torch.float32 or key.dtype != torch.uint16
                or value.dtype != torch.uint16 or q.ndim != 3 or output.shape != q.shape
                or key.ndim != 4 or value.shape != key.shape or q.shape[2] != key.shape[3]
                or q.shape[1] % key.shape[2] or q.shape[2] & (q.shape[2] - 1)
                or q.stride()[1:] != (q.shape[2], 1) or output.stride()[1:] != (q.shape[2], 1)
                or not all(t.is_contiguous() for t in (key, value, pages, lengths, cu))
                or pages.ndim != 2 or lengths.shape != (pages.shape[0],) or cu.shape != (pages.shape[0] + 1,)
                or pages.dtype not in (torch.int32, torch.int64) or lengths.dtype != torch.int32
                or cu.dtype != torch.int32 or window < -1):
            raise ValueError("SM70 paged attention requires FP32 queries/output and UInt16 BF16-bit pages")
        self.tensors = tensors
        self.plan = _declare("attention", tensors, session=session, name=name, window=window, row_invariant=row_invariant)

    def __call__(self):
        attend(*self.tensors, self.plan.handle)
        return self.tensors[-1]


class PreparedWriter:
    def __init__(self, key, value, key_cache, value_cache, slots, *, session, name):
        tensors = (key, value, key_cache, value_cache, slots)
        _device(tensors)
        if (key.dtype != torch.float32 or value.dtype != torch.float32
                or key_cache.dtype != torch.uint16 or value_cache.dtype != torch.uint16
                or key.ndim != 3 or value.shape != key.shape or key_cache.ndim != 4
                or value_cache.shape != key_cache.shape or key.shape[1:] != key_cache.shape[2:]
                or key.stride()[1:] != (key.shape[2], 1) or value.stride()[1:] != (value.shape[2], 1)
                or not all(t.is_contiguous() for t in (key_cache, value_cache, slots))
                or slots.shape != (key.shape[0],) or slots.dtype != torch.int64):
            raise ValueError("SM70 KV writes require FP32 source heads, UInt16 BF16-bit pages and int64 slots")
        self.tensors = tensors
        self.plan = _declare("write", tensors, session=session, name=name)

    def __call__(self):
        write(*self.tensors, self.plan.handle)
