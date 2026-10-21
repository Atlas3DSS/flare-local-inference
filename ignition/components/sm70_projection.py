"""SM70 packed projections with fused GEMV and bounded mixed-precision cuBLAS tiles.

V100 cannot multiply BF16 or FP8 operands. Conversion to FP32 preserves every
BF16/FP16 value, including BF16 values outside FP16's range. NVFP4 uses the
existing weight-only arithmetic: E2M1 times its E4M3 block scale, an FP32 dot,
then the checkpoint global scale. Tensor-FP8 retains calibrated activation
rounding to E4M3 and applies the combined scale after the dot. Weights remain
packed; serialized projections reuse caller-owned expansion scratch. Tensor-FP8
prefill uses exact FP16 operands with FP32 accumulation and result storage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import MemoryRequirements, PersistentMemory, Plan, PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.scratch import ScratchArena, prime_owners, scratch_storage
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer

from .native_operator import native_contract, native_plan


CAPABILITY = (7, 0)
COMPONENT = "flare.sm70.projection"
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def is_sm70(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == CAPABILITY


@dataclass(frozen=True)
class ProjectionQuery:
    kind: str
    rows: int
    in_features: int
    padded_in_features: int
    out_features: int
    global_scale_kind: str = "none"
    weight_dtype: str = "uint8"
    input_dtype: str = "float32"
    output_dtype: str = "float32"
    has_bias: bool = False
    row_invariant: bool = False
    repacked: bool = False
    column_gains: bool = False
    member_widths: tuple = ()
    member_gains: tuple = ()
    gain_value: float = 1.0
    input_scale_value: float = 1.0


def _tiles(query):
    from .sm70_dense import tactic as dense_tactic
    dense = dense_tactic(query)
    if dense is not None:
        return (dense,)
    if query.repacked:
        from .sm70_projection_tc import tactic
        return (tactic(query),)
    # Exact speculative commits require the same reduction for a row in every
    # verification bucket. Shape-dependent GEMV/GEMM tuning is not equivalent
    # at BF16 midpoints, even when both dots satisfy their error bounds.
    if query.row_invariant:
        return (("row_fp32", min(query.out_features, 256)),) if query.kind == "dense" else (("row_gemv", 4, 512),)
    if query.kind != "dense" and query.rows <= 4:
        return tuple(("gemv", bn, bk) for bn, bk in
                     ((4, 512), (2, 256), (2, 512), (4, 256), (8, 256), (8, 512)))
    fp32 = tuple(("fp32", width) for width in
                 dict.fromkeys(min(query.out_features, width) for width in (256, 1024)))
    if query.kind == "tensor_fp8":
        return tuple(("hgemm", width) for width in
                     dict.fromkeys(min(query.out_features, width) for width in (1024, 2048, 4096))) + fp32
    return fp32


CONTRACT = native_contract(component_id=COMPONENT, query_type=ProjectionQuery, tactics=_tiles,
                           semantic_version=8)


def _scratch_shapes(query, tactic):
    if tactic[0] in ("gemv", "row_gemv"):
        return ((query.rows, query.in_features),) if query.kind == "tensor_fp8" else ((1, 1),)
    tile = tactic[1]
    return ((query.rows, query.in_features), (tile, query.in_features), (query.rows, tile))


def workspace_bytes(query, tactic):
    if tactic[0] == "row_dense":
        return 0
    if tactic[0] in ("row_tc", "prefill_tc"):
        from .sm70_projection_tc import workspace_bytes as tc_workspace_bytes
        return tc_workspace_bytes(query, tactic)
    return sum((rows * columns * (2 if tactic[0] == "hgemm" and index < 2 else 4) + 255) // 256 * 256
               for index, (rows, columns) in enumerate(_scratch_shapes(query, tactic)))


def _scratch_views(query, tactic, workspace):
    views, offset = [], 0
    for rows, columns in _scratch_shapes(query, tactic):
        dtype = torch.float16 if tactic[0] == "hgemm" and len(views) < 2 else torch.float32
        size = rows * columns * dtype.itemsize
        views.append(workspace[offset:offset + size].view(dtype).view(rows, columns))
        offset += (size + 255) // 256 * 256
    return views


@triton.jit
def _e4m3_to_float(bits):
    """Decode storage bits without Volta-incompatible FP8 instructions."""
    bits = bits.to(tl.uint32)
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * (1.0 / 512.0), normal)
    magnitude = tl.where((exponent == 15) & (mantissa == 7), float("nan"), magnitude)
    signed = magnitude.to(tl.uint32, bitcast=True) | ((bits & 128) << 24)
    return signed.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["start"])
def _expand_nvfp4_kernel(Values, Scales, Output, start, K: tl.constexpr,
                         PADDED_K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    """One flat launch expands a tile using b12x's F8_128x4 scale storage."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = start + index // K, index % K
    valid = row < N
    packed = tl.load(Values + row * (PADDED_K // 2) + column // 2, valid, 0).to(tl.uint32)
    code = (packed >> ((column % 2) * 4)) & 15
    magnitude = code & 7
    decoded = tl.where(magnitude < 4, magnitude.to(tl.float32) * 0.5,
                       tl.where(magnitude == 4, 2.0, tl.where(magnitude == 5, 3.0,
                                tl.where(magnitude == 6, 4.0, 6.0))))
    decoded = (decoded.to(tl.uint32, bitcast=True) | ((code & 8) << 28)).to(tl.float32, bitcast=True)
    group = column // 16
    offset = ((row // 128 * ((PADDED_K // 16 + 3) // 4) + group // 4) * 512
              + row % 32 * 16 + row // 32 % 4 * 4 + group % 4)
    scale = _e4m3_to_float(tl.load(Scales + offset, valid, 0))
    tl.store(Output + index, decoded * scale, valid)


def _expansion_options(query):
    return dict(K=query.in_features, PADDED_K=query.padded_in_features,
                N=query.out_features, BLOCK=256, num_warps=4, enable_fp_fusion=False)


@program_cache(scope="preparation")
def compile_expansion(payload, tile, ordinal):
    query = ProjectionQuery(**dict(payload))
    values = torch.empty((query.out_features, query.padded_in_features // 2), dtype=torch.uint8, device="meta")
    scale_size = triton.cdiv(query.out_features, 128) * triton.cdiv(query.padded_in_features // 16, 4) * 512
    scales = torch.empty((scale_size,), dtype=torch.uint8, device="meta")
    output = torch.empty((tile, query.in_features), dtype=torch.float32, device="meta")
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_expand_nvfp4_kernel, (triton.cdiv(tile * query.in_features, 256), 1, 1),
                             values, scales, output, 0, **_expansion_options(query))


def _expand_nvfp4(program, values, scales, output, start, stop, query):
    program[(triton.cdiv((stop - start) * query.in_features, 256), 1, 1)](values, scales, output, start)


def _hgemm(inputs, weight, result):
    # These flags affect cuBLAS dispatch, while spawned complete-forward
    # compilers guard their own process state. Scope them to the opaque call.
    matmul = torch.backends.cuda.matmul
    previous = (matmul.allow_fp16_reduced_precision_reduction, matmul.allow_fp16_accumulation)
    try:
        matmul.allow_fp16_reduced_precision_reduction = False
        matmul.allow_fp16_accumulation = False
        torch.mm(inputs, weight.T, out_dtype=torch.float32, out=result)
    finally:
        matmul.allow_fp16_reduced_precision_reduction, matmul.allow_fp16_accumulation = previous


def _project(query, tactic, expansion, source, values, scales, gain, input_scale, output, workspace, bias):
    """All temporary allocations are bounded by a tile and captured before serving."""
    tile = tactic[1]
    inputs, expanded, product = _scratch_views(query, tactic, workspace)
    if query.kind == "tensor_fp8":
        inputs.copy_((source.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn))
    else:
        inputs.copy_(source)
    for start in range(0, query.out_features, tile):
        stop = min(query.out_features, start + tile)
        width = stop - start
        weight, result = expanded[:width], product[:, :width]
        if query.kind == "nvfp4":
            _expand_nvfp4(expansion, values, scales, weight, start, stop, query)
        elif query.kind == "tensor_fp8":
            weight.copy_(values[start:stop, :query.in_features].view(torch.float8_e4m3fn))
        else:
            weight.copy_(values.view(getattr(torch, query.weight_dtype))[start:stop, :query.in_features])
        if tactic[0] == "hgemm":
            # CUDA_R_16F operands, CUDA_R_32F C, CUBLAS_COMPUTE_32F.
            # Warmup owns cuBLAS's per-stream workspace through capture/export.
            _hgemm(inputs, weight, result)
        elif tactic[0] == "row_fp32":
            for row in range(query.rows):
                torch.mm(inputs[row:row + 1], weight.T, out=result[row:row + 1])
        else:
            torch.mm(inputs, weight.T, out=result)
        if gain is not None:
            if query.global_scale_kind == "reciprocal":
                result.div_(gain)
            else:
                result.mul_(gain)
        # The unquantized operator adds its bias before the final output cast.
        if bias is not None and query.kind == "dense":
            result.add_(bias[start:stop])
        output[:, start:stop].copy_(result)
    # QuantizedProjection's existing bias contract rounds before adding bias.
    if bias is not None and query.kind != "dense":
        output.add_(bias)
    return output


@triton.jit
def _e2m1_to_float(code):
    magnitude = code & 7
    value = tl.where(magnitude < 4, magnitude.to(tl.float32) * 0.5,
                     tl.where(magnitude == 4, 2., tl.where(magnitude == 5, 3.,
                              tl.where(magnitude == 6, 4., 6.))))
    return (value.to(tl.uint32, bitcast=True) | ((code & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _load_float(pointer, offsets, mask, BF16: tl.constexpr):
    value = tl.load(pointer + offsets, mask, other=0)
    if BF16:
        return (value.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return value.to(tl.float32)


@triton.jit
def _bf16_bits(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    return tl.where((bits & 0x7fffffff) > 0x7f800000,
                    (bits >> 16) | 0x40, rounded).to(tl.uint16)


@triton.jit
def _output_round(value, DTYPE: tl.constexpr):
    if DTYPE == "bfloat16":
        return (_bf16_bits(value).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    elif DTYPE == "float16":
        return value.to(tl.float16).to(tl.float32)
    return value


@triton.jit(do_not_specialize=["x_row_stride", "y_row_stride"])
def _gemv_kernel(X, Values, Scales, Gain, Bias, Y, x_row_stride, y_row_stride,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PADDED_K: tl.constexpr,
                 KIND: tl.constexpr, GAIN_KIND: tl.constexpr, INPUT_DTYPE: tl.constexpr,
                 OUTPUT_DTYPE: tl.constexpr, HAS_BIAS: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, ROW_INVARIANT: tl.constexpr):
    # Each packed byte, scale, and FP8 weight is fetched once per CTA slice,
    # then reused across all prepared rows. No inter-CTA reduction or atomics.
    n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    pair = tl.arange(0, BK // 2).to(tl.int64)
    rows = tl.arange(0, triton.next_power_of_2(M)).to(tl.int64)
    if ROW_INVARIANT:
        X += tl.program_id(1).to(tl.int64) * x_row_stride
        Y += tl.program_id(1).to(tl.int64) * y_row_stride
    acc = tl.full((triton.next_power_of_2(M), BN), 0., tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + 2 * pair
        if KIND == "nvfp4":
            packed = tl.load(Values + n[:, None] * (PADDED_K // 2) + k[None, :] // 2,
                             (n[:, None] < N) & (k[None, :] < K), 0).to(tl.uint32)
            group = start * (BK // 16) + tl.arange(0, BK // 16).to(tl.int64)
            offset = ((n[:, None] // 128 * tl.cdiv(PADDED_K // 16, 4) + group[None, :] // 4) * 512
                      + n[:, None] % 32 * 16 + n[:, None] // 32 % 4 * 4 + group[None, :] % 4)
            scale = _e4m3_to_float(tl.load(Scales + offset,
                                  (n[:, None] < N) & (group[None, :] * 16 < K), 0))
            lo = (_e2m1_to_float(packed & 15).reshape(BN, BK // 16, 8) *
                  scale[:, :, None]).reshape(BN, BK // 2)
            hi = (_e2m1_to_float(packed >> 4).reshape(BN, BK // 16, 8) *
                  scale[:, :, None]).reshape(BN, BK // 2)
        else:
            lo = _e4m3_to_float(tl.load(Values + n[:, None] * PADDED_K + k[None, :],
                                (n[:, None] < N) & (k[None, :] < K), 0))
            hi = _e4m3_to_float(tl.load(Values + n[:, None] * PADDED_K + k[None, :] + 1,
                                (n[:, None] < N) & (k[None, :] + 1 < K), 0))
        xlo = _load_float(X, rows[:, None] * x_row_stride + k[None, :],
                         (rows[:, None] < M) & (k[None, :] < K), INPUT_DTYPE == "bfloat16")
        xhi = _load_float(X, rows[:, None] * x_row_stride + k[None, :] + 1,
                         (rows[:, None] < M) & (k[None, :] + 1 < K), INPUT_DTYPE == "bfloat16")
        products = xlo[:, None, :] * lo[None, :, :] + xhi[:, None, :] * hi[None, :, :]
        acc = acc + tl.sum(products, 2)
    gain = tl.load(Gain)
    if GAIN_KIND == "reciprocal":
        acc = tl.div_rn(acc, gain)
    else:
        acc = acc * gain
    acc = _output_round(acc, OUTPUT_DTYPE)
    if HAS_BIAS:
        bias = _load_float(Bias, n, n < N, OUTPUT_DTYPE == "bfloat16")
        acc = _output_round(acc + bias[None, :], OUTPUT_DTYPE)
    if OUTPUT_DTYPE == "bfloat16":
        result = _bf16_bits(acc)
    else:
        result = acc
    tl.store(Y + rows[:, None] * y_row_stride + n[None, :], result,
             (rows[:, None] < M) & (n[None, :] < N))


def _gemv_options(query, tactic):
    return dict(M=1 if query.row_invariant else query.rows, N=query.out_features, K=query.in_features,
                PADDED_K=query.padded_in_features, KIND=query.kind,
                GAIN_KIND=query.global_scale_kind,
                INPUT_DTYPE="float32" if query.kind == "tensor_fp8" else query.input_dtype,
                OUTPUT_DTYPE=query.output_dtype, HAS_BIAS=query.has_bias,
                BN=tactic[1], BK=tactic[2], ROW_INVARIANT=query.row_invariant,
                num_warps=4, enable_fp_fusion=False)


@program_cache(scope="preparation")
def compile_gemv(payload, tactic, ordinal):
    query = ProjectionQuery(**dict(payload))
    input_dtype = torch.float32 if query.kind == "tensor_fp8" else getattr(torch, query.input_dtype)
    if input_dtype == torch.bfloat16:
        input_dtype = torch.uint16
    output_dtype = torch.uint16 if query.output_dtype == "bfloat16" else getattr(torch, query.output_dtype)
    source = torch.empty((query.rows, query.in_features), dtype=input_dtype, device="meta")
    values = torch.empty((query.out_features, query.padded_in_features), dtype=torch.uint8, device="meta")
    scales = torch.empty((1,), dtype=torch.uint8, device="meta") if query.kind == "nvfp4" else None
    gain = torch.empty((1,), dtype=torch.float32, device="meta")
    bias = torch.empty((query.out_features,), dtype=output_dtype, device="meta") if query.has_bias else None
    output = torch.empty((query.rows, query.out_features), dtype=output_dtype, device="meta")
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_gemv_kernel, (triton.cdiv(query.out_features, tactic[1]),
                             query.rows if query.row_invariant else 1, 1),
                             source, values, scales, gain, bias, output,
                             query.in_features, query.out_features, **_gemv_options(query, tactic))


@dataclass(frozen=True)
class PreparedProjection:
    query: ProjectionQuery
    tactic: tuple
    expansion: object = None
    gemv: object = None
    kernel_cache: str | None = None
    cutlass_include: str | None = None

    @property
    def tile(self):
        return min(self.query.out_features, 1024) if self.tactic[0] in ("gemv", "row_gemv") else self.tactic[1]

    @property
    def required_workspace(self):
        return workspace_bytes(self.query, self.tactic)

    def run(self, source, values, scales, gain, input_scale, output, workspace, bias=None):
        if self.tactic[0] == 'row_dense':
            from .sm70_dense import run
            return run(self.query, self.tactic, source, values, output, bias, kernel_cache=self.kernel_cache)
        if self.tactic[0] in ("row_tc", "prefill_tc"):
            from .sm70_projection_tc import run
            return run(self.query, self.tactic, source, values, scales, output, workspace, bias, gain,
                       kernel_cache=self.kernel_cache, cutlass_include=self.cutlass_include)
        if self.tactic[0] in ("gemv", "row_gemv"):
            if self.query.kind == "tensor_fp8":
                inputs, = _scratch_views(self.query, self.tactic, workspace)
                inputs.copy_((source.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn))
                source = inputs
            if source.dtype == torch.bfloat16:
                source = source.view(torch.uint16)
            if bias is not None and bias.dtype == torch.bfloat16:
                bias = bias.view(torch.uint16)
            destination = output.view(torch.uint16) if output.dtype == torch.bfloat16 else output
            self.gemv[(triton.cdiv(self.query.out_features, self.tactic[1]),
                       self.query.rows if self.query.row_invariant else 1, 1)](
                source, values, scales, gain, bias, destination, source.stride(0), output.stride(0))
            return output
        return _project(self.query, self.tactic, self.expansion, source, values, scales, gain, input_scale,
                        output, workspace, bias)


@torch.library.custom_op("flare::sm70_projection_out", mutates_args=("output", "workspace"))
def _projection_out(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor | None,
                    gain: torch.Tensor | None, input_scale: torch.Tensor | None,
                    output: torch.Tensor, workspace: torch.Tensor, bias: torch.Tensor | None,
                    handle: int) -> None:
    state = require_prepared(plan_from_handle(handle), COMPONENT, source.device)
    state.run(source, values, scales, gain, input_scale, output, workspace, bias)


@_projection_out.register_fake
def _fake_projection_out(source, values, scales, gain, input_scale, output, workspace, bias, handle):
    pass


def _check_operands(source, values, output, bias, k, n):
    if not is_sm70(source.device):
        raise ValueError("SM70 projections require a V100 CUDA device")
    if (source.ndim != 2 or source.shape[1] != k or source.dtype not in _FLOAT_DTYPES
            or output.shape != (source.shape[0], n) or output.dtype not in _FLOAT_DTYPES
            or source.shape[0] <= 0 or source.stride(-1) != 1 or output.stride(-1) != 1
            or min(k, n) <= 0 or values.ndim != 2):
        raise ValueError("SM70 projections require positive matching floating matrices with contiguous columns")
    if values.device != source.device or output.device != source.device or not values.is_contiguous():
        raise ValueError("SM70 projection weights and output must share the source device")
    if bias is not None and (bias.shape != (n,) or bias.dtype != output.dtype
                             or bias.device != source.device or not bias.is_contiguous()):
        raise ValueError("SM70 projection bias must match the output width, dtype and device")
    from torch._C import _overlaps

    if any(_overlaps(output, tensor) for tensor in (source, values, bias) if tensor is not None):
        raise ValueError("SM70 projection output must not overlap an input")


class _Projection:
    def _initialize(self, query, source, values, output, *, scales=None, gain=None, input_scale=None,
                    bias=None, session, name, workspace):
        query = replace(query, input_dtype=str(source.dtype).removeprefix("torch."),
                        output_dtype=str(output.dtype).removeprefix("torch."), has_bias=bias is not None)
        _check_operands(source, values, output, bias, query.in_features, query.out_features)
        for tensor in (scales, gain, input_scale):
            if tensor is not None and tensor.device != source.device:
                raise ValueError("SM70 projection scales must share the source device")
        for scalar in (gain, input_scale):
            expected = query.out_features if scalar is gain and query.column_gains else 1
            if scalar is not None and (scalar.dtype != torch.float32 or scalar.numel() != expected
                    or not bool((torch.isfinite(scalar) & (scalar > 0)).all())):
                raise ValueError("SM70 projection scales must be positive finite FP32 scalars")
        if query.repacked:
            if source.dtype != torch.float32 or output.dtype != torch.float32:
                raise ValueError("reviewed SM70 tensor-core preparation requires FP32 physical input/output")
            query = replace(query, gain_value=1.0 if gain is None or query.column_gains else float(gain),
                            input_scale_value=1.0 if input_scale is None else float(input_scale))
        for tensor in (values, scales, gain, input_scale, bias):
            if tensor is not None:
                record_numeric_tensor(tensor)
        self.output, self.values, self.scales = output, values, scales
        self.gain, self.input_scale, self.bias = gain, input_scale, bias
        payload = CONTRACT.encode_query(query)
        kernel_cache = getattr(session, 'kernel_cache', None)
        cutlass_include = getattr(session, 'cutlass_include', None)

        def compile_jobs(tactic, device):
            if tactic[0] in ("row_tc", "prefill_tc", "row_dense"):
                return ()  # Native sm_70 library was built before checkpoint repacking.
            jobs = []
            if query.kind == "nvfp4":
                tile = min(query.out_features, 1024) if tactic[0] in ("gemv", "row_gemv") else tactic[1]
                jobs.append(CompileJob.create("ignition.components.sm70_projection:compile_expansion",
                                              payload, tile, device.ordinal))
            if tactic[0] in ("gemv", "row_gemv"):
                jobs.append(CompileJob.create("ignition.components.sm70_projection:compile_gemv",
                                              payload, tactic, device.ordinal))
            return tuple(jobs)

        def materialize(tactic):
            if tactic[0] == 'row_dense':
                from .sm70_dense import library as dense_library
                dense_library(kernel_cache)
                return PreparedProjection(query, tactic, kernel_cache=kernel_cache)
            if tactic[0] in ("row_tc", "prefill_tc"):
                from .sm70_projection_tc import library
                library(kernel_cache)  # Retained native modules; no compilation during capture/replay.
                if tactic[0] == 'prefill_tc':
                    from .sm70_prefill import library as prefill_library
                    prefill_library(kernel_cache, cutlass_include)
                return PreparedProjection(query, tactic, kernel_cache=kernel_cache, cutlass_include=cutlass_include)
            expansion, gemv = None, None
            programs = []
            if query.kind == "nvfp4":
                tile = min(query.out_features, 1024) if tactic[0] in ("gemv", "row_gemv") else tactic[1]
                expansion = compile_expansion(payload, tile, source.device.index)
                programs.append(expansion)
            if tactic[0] in ("gemv", "row_gemv"):
                gemv = compile_gemv(payload, tactic, source.device.index)
                programs.append(gemv)
            for program in programs:
                load_programs(program)
            return attach_programs(PreparedProjection(query, tactic, expansion, gemv), *programs)

        # Without an external arena, keep one owned arena across preparation calls.
        arena = ScratchArena(source.device) if workspace is None else workspace
        if workspace is None:
            def memory(config, device):
                from b12x.preparation import current_plan

                needed = max(arena.nbytes, workspace_bytes(query, config.tactic))
                return MemoryRequirements((), (PersistentMemory(
                    (COMPONENT, current_plan()), needed, arena.nbytes),))

            self.plan = Plan(contract=CONTRACT, query=query,
                _compile_jobs=lambda config, device: compile_jobs(config.tactic, device),
                _memory_requirements=memory,
                _materialize=lambda selection, device: materialize(selection.config.tactic),
                _device=source.device)
        else:
            self.plan = native_plan(contract=CONTRACT, query=query, device=source.device,
                scratch_bytes=lambda tile: workspace_bytes(query, tile),
                materialize=materialize, compile_jobs=compile_jobs)

        def storage(nbytes=None):
            if nbytes is None:
                nbytes = require_prepared(self.plan, COMPONENT, source.device).required_workspace
            result = scratch_storage(arena, nbytes=nbytes, device=source.device)
            from torch._C import _overlaps

            if any(_overlaps(result, tensor) for tensor in (source, values, scales, gain, input_scale, output, bias)
                   if tensor is not None):
                raise ValueError("SM70 projection scratch must not overlap operands")
            return result

        def prepare(state):
            scratch = storage(state.required_workspace)
            return PreparedCall(run=lambda: state.run(source, values, scales, gain, input_scale,
                output, scratch, bias), owners=prime_owners(arena, scratch))

        acquire = input_factory(source, output, storage=getattr(session, "benchmark_storage", None))

        def benchmark(state):
            from b12x.preparation._measurement import capture_benchmark_call

            operands, scratch = acquire(), storage(state.required_workspace)
            call = PreparedCall(run=lambda: state.run(operands.source, values, scales, gain, input_scale,
                operands.output, scratch, bias), produce=operands.produce, output=operands.output,
                owners=(operands, scratch, self.weight))
            # ATen's bounded temporaries may request memory while a candidate is
            # timed. CUDA allocation can synchronize, which deadlocks behind the
            # tuning stream gate. Serving replays a captured graph; measure that
            # same allocation-free operation while keeping live input producers.
            captured = capture_benchmark_call(call)
            # The first launch may upload a large graph's executable buffers.
            # Complete that work before b12x holds the timing stream gate.
            captured.produce()
            captured.invoke()
            torch.cuda.current_stream(source.device).synchronize()
            return captured

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=benchmark),))
        register_finalizer(session, lambda: setattr(self, "workspace", storage()), workspace=workspace)
        self.arena = arena

    def __call__(self, source):
        _projection_out(source, self.values, self.scales, self.gain, self.input_scale,
                        self.output, self.workspace, self.bias, self.plan.handle)
        return self.output


class PackedProjection(_Projection):
    """Checkpoint NVFP4 with legal W4A16 arithmetic; no activation requantization."""

    def __init__(self, source, weight, output, *, session, name, activation_mode="auto",
                 bias=None, workspace=None, row_invariant=False):
        from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight, scale_storage

        if not isinstance(weight, NVFP4LinearWeight):
            raise TypeError("SM70 packed projections currently require NVFP4 weights")
        if activation_mode not in ("auto", "a16"):
            raise ValueError("SM70 NVFP4 supports weight-only a16 arithmetic; quantized activations are unsupported")
        if weight.global_scale_kind not in ("multiplier", "reciprocal"):
            raise ValueError("NVFP4 requires a multiplier or reciprocal global scale")
        k, n = weight.in_features, weight.out_features
        if (k % 16 or k > weight.padded_in_features or weight.values.dtype != torch.uint8
                or weight.values.shape != (n, weight.padded_in_features // 2)):
            raise ValueError("NVFP4 projection has invalid packed dimensions")
        self.weight, self.activation_mode = weight, "a16"
        from .sm70_projection_tc import RepackedNVFP4
        repacked = isinstance(weight, RepackedNVFP4)
        scales = weight.scale_mma if repacked else scale_storage(weight.scale_mma, n, weight.padded_in_features, 16)
        query = ProjectionQuery("nvfp4", source.shape[0], k, weight.padded_in_features, n,
                                weight.global_scale_kind, row_invariant=row_invariant, repacked=repacked,
                                column_gains=repacked and weight.global_scale.numel() > 1,
                                member_widths=weight.member_widths if repacked else (),
                                member_gains=weight.member_gains if repacked else ())
        self._initialize(query, source, weight.values, output, scales=scales, gain=weight.global_scale,
                         bias=bias, session=session, name=name, workspace=workspace)


class TensorFP8Projection(_Projection):
    """E4M3 activation rounding and the checkpoint's combined output scale."""

    def __init__(self, source, weight, output, *, session, name, workspace=None, row_invariant=False):
        from b12x.gemm.blockscaled._linear import TensorFP8LinearWeight

        packed = weight.packed
        if (not isinstance(packed, TensorFP8LinearWeight) or packed.values.dtype != torch.float8_e4m3fn
                or packed.values.shape != (packed.out_features, packed.padded_in_features)
                or not 0 < packed.in_features <= packed.padded_in_features):
            raise ValueError("SM70 tensor-FP8 requires a packed E4M3 matrix with logical K")
        # pack_tensor_fp8_linear_weight defines these as unit scale planes. Reject
        # altered packed objects instead of discarding meaningful block scales.
        if not bool((packed.block_scale.float() == 1).all()):
            raise ValueError("tensor-FP8 block scales must retain their packed unit values")
        self.weight = weight
        from .sm70_projection_tc import RepackedFP8
        query = ProjectionQuery("tensor_fp8", source.shape[0], packed.in_features,
                                packed.padded_in_features, packed.out_features, "multiplier", row_invariant=row_invariant,
                                repacked=isinstance(packed, RepackedFP8),
                                column_gains=isinstance(packed, RepackedFP8) and packed.output_scale.numel() > 1,
                                member_widths=packed.member_widths if isinstance(packed, RepackedFP8) else (),
                                member_gains=packed.member_gains if isinstance(packed, RepackedFP8) else ())
        # SM70 Inductor cannot type FP8 operands even for an opaque call: its
        # functionalization pattern pass skips such nodes. The checkpoint's
        # bits cross the boundary as bytes; only the implementation views E4M3.
        self._initialize(query, source, packed.values.view(torch.uint8), output, gain=packed.output_scale,
                         input_scale=weight.input_scale, session=session, name=name, workspace=workspace)


class UnquantizedProjection(_Projection):
    """Exact FP32 expansion of BF16/FP16 storage, with FP32 GEMM accumulation."""

    def __init__(self, source, weight, output, *, session, name, bias=None, workspace=None, row_invariant=False):
        if weight.ndim != 2 or weight.dtype not in _FLOAT_DTYPES:
            raise ValueError("SM70 unquantized projections require a floating matrix")
        self.weight = weight
        n, k = weight.shape
        query = ProjectionQuery("dense", source.shape[0], k, k, n,
                                weight_dtype=str(weight.dtype).removeprefix("torch."), row_invariant=row_invariant)
        self._initialize(query, source, weight.view(torch.uint8), output, bias=bias, session=session,
                         name=name, workspace=workspace)
