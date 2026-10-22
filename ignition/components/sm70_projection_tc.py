"""Reviewed Sol-14 projections (M <= 24, one pinned tuple per shape) with fixed reductions and prepared, bounded storage."""
from dataclasses import dataclass
from functools import lru_cache
import ctypes as C
from contextlib import nullcontext
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess

import torch
from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight
from b12x.gemm.blockscaled._linear import TensorFP8LinearWeight

from ignition.preparation.tensor_metadata import record_numeric_tensor


@dataclass(frozen=True)
class RepackedNVFP4(NVFP4LinearWeight):
    """Same byte count; values/scales use the reviewed N-partitioned layout."""
    member_widths: tuple = ()
    member_gains: tuple = ()


@dataclass(frozen=True)
class RepackedFP8(TensorFP8LinearWeight):
    """Same byte count; values use the reviewed N-partitioned layout."""
    member_widths: tuple = ()
    member_gains: tuple = ()


# Reviewed Sol-14 tuple (BK, warps, split, path) per shape, fixed across every row count 1..24 (row invariance):
# sm70_projection_plan14.json (sol-14/plan14.json). Fused members carry the fused shape's tuple. Shapes whose split
# changed against Sol-10 (tensor_fp8 5120->14336 and its q/k/v members, 4 -> 2) produce new FP32 bits; every other
# served projection is bit-identical to the Sol-10 build (sol-14/final-bit-reference.json, astra-8/bit-identity.json).
PINNED = {
    ('nvfp4', 5120, 34816): (256, 4, 2, 60),
    ('tensor_fp8', 5120, 16384): (256, 4, 2, 60),
    ('tensor_fp8', 5120, 14336): (128, 2, 2, 60),
    ('tensor_fp8', 6144, 5120): (64, 2, 8, 60),
    ('tensor_fp8', 5120, 10240): (256, 4, 2, 60),
    ('tensor_fp8', 5120, 6144): (256, 4, 2, 60),
    ('nvfp4', 5120, 17408): (256, 4, 2, 60),
    ('nvfp4', 17408, 5120): (128, 4, 8, 51),
    ('tensor_fp8', 5120, 12288): (128, 2, 2, 60),
    ('tensor_fp8', 5120, 1024): (128, 2, 2, 60),
    ('nvfp4', 5120, 248320): (128, 4, 1, 47),
    # Draft-only NVFP4 overlay shapes (not in plan14.json): the compiled path-51 instance at the Sol-10 split, so
    # their bits are unchanged; the former 47/128/4/4 q shape moves to path 51 at the same split (same per-row
    # HMMA sequence and fixed-order finish; sol-14 admits path 47 only at split 1).
    ('nvfp4', 10240, 5120): (128, 4, 8, 51),
    ('nvfp4', 6144, 5120): (128, 4, 4, 51),
    ('nvfp4', 5120, 12288): (128, 4, 4, 51),
    ('nvfp4', 5120, 1024): (128, 4, 8, 51),
}
# Rows per launch: the R=20 instances serve every decode/speculative row count (4 agents x depth 4 = 20) in one
# call with the real M; longer row counts (draft prefill) run in 20-row chunks. R=24 is admitted but not used.
MAX_ROWS = 20
ADMISSIONS = []


def tactic(query):
    if query.rows > 20:
        from .sm70_prefill import PINNED as prefill_shapes
        if query.member_widths or (query.kind, query.in_features, query.out_features) in prefill_shapes:
            return ('prefill_tc', 512)
    bk, warps, split, path = PINNED[(query.kind, query.in_features, query.out_features)]
    return ('row_tc', bk, warps, split, path)


@lru_cache(None)
def compile_library(kernel_cache=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if CUDA_HOME is None:
        raise RuntimeError('SM70 projection preparation requires the configured CUDA toolkit')
    directory = Path(__file__).parent
    sources = [directory / name for name in ('sm70_projection_tc.cu', 'kernels14.cuh', 'local14_kernel.cuh',
        'deep14_kernel.cuh', 'fast14_kernel.cuh', 'step14.cuh', 'stage14.cuh', 'sm70_projection_tuples.inc')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources) + Path(__file__).read_bytes()).hexdigest()
    cache = Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + '.so')
    with (cache / (digest + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            nvcc = Path(CUDA_HOME) / 'bin/nvcc'
            command = [str(nvcc), '--threads', str(len(os.sched_getaffinity(0))),
                '--split-compile=' + str(len(os.sched_getaffinity(0))), '-std=c++17', '-O3',
                '-arch=compute_70', '-code=sm_70', '-Xcompiler=-fPIC', '--shared',
                '--fmad=false', '-DSOL14_PIPE2=0', '-DSOL14_STAGER=1', '-DSOL14_LDS128=1',
                str(sources[0]), '-o', str(output)]
            subprocess.run(command, check=True)
    return str(output)


@lru_cache(None)
def library(kernel_cache=None):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('SM70 projection library must be loaded before capture')
    lib = C.CDLL(compile_library(kernel_cache))
    lib.atlas_init.argtypes = []
    lib.atlas_tuple_status.argtypes = [C.c_int] * 6
    lib.atlas_produce.argtypes = [C.c_void_p] * 3 + [C.c_int] * 3 + [C.c_float, C.c_void_p]
    lib.atlas_project.argtypes = [C.c_void_p] * 4 + [C.c_float, C.c_int] + [C.c_void_p] * 3 + [C.c_int] * 9 + [C.c_void_p] * 2
    lib.atlas_repack.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.c_void_p]
    lib.atlas_position_exact.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    lib.atlas_fp8_exact.argtypes = [C.c_void_p] * 3
    # Once per process, before any capture: dynamic shared-memory opt-in for the >48 KiB R=24 instances.
    checked(lib.atlas_init())
    return lib


def ptr(tensor):
    return None if tensor is None else C.c_void_p(tensor.data_ptr())


def checked(status):
    if status:
        raise RuntimeError(f'SM70 tensor-core CUDA launch failed: {status}')


def load_weight(loader, prefix, *, storage=None, kernel_cache=None):
    # Keep temporary native/reverse-repacked weights out of the retained pool.
    # Otherwise their interleaved large allocations strand device segments.
    with storage.transient() if storage is not None else nullcontext():
        return _load_weight(loader, prefix, storage=storage, kernel_cache=kernel_cache)


def _weight_pool(storage):
    if not hasattr(storage, '_sm70_repacked_weights'):
        pool = torch.cuda.MemPool(storage._allocator.allocator())
        storage._preparation_pools.add(pool.id)
        storage._sm70_repacked_weights = pool
    return storage._sm70_repacked_weights


def _load_weight(loader, prefix, *, storage, kernel_cache):
    """Read native checkpoint bytes; prove both repack round trips before use."""
    from .tensor_fp8 import TensorFP8Weight

    spec = loader.quantization[prefix]
    fp4 = spec['quant_algo'] == 'NVFP4'
    if spec['quant_algo'] not in ('NVFP4', 'FP8'):
        raise ValueError(f'{prefix}: unsupported reviewed projection format')
    values = loader.load_tensor(prefix + '.weight')
    n, stored = values.shape
    k = stored * (2 if fp4 else 1)
    kind = 'nvfp4' if fp4 else 'tensor_fp8'
    if n % 128 or k % 64 or (kind, k, n) not in PINNED:
        raise ValueError(f'{prefix}: shape is outside reviewed SM70 projection admission: {kind} {k}->{n}')
    bk, warps, split, path = PINNED[(kind, k, n)]
    if n % (warps * 32) or k % (split * bk) or n * (k // 16) > 2**31 - 1:
        raise ValueError(f'{prefix}: aligned reviewed projection admission failed')
    lib = library(kernel_cache)
    # The pinned tuple must be a compiled, feasible instance at every padded row count (sol-10 C3, sol-14 step 3).
    if any(lib.atlas_tuple_status(fp4, path, bk, warps, split, m) for m in (1, 16, 20, 24)):
        raise ValueError(f'{prefix}: pinned tuple {(bk, warps, split, path)} is not compiled or infeasible for M <= 24')
    scales = loader.load_tensor(prefix + '.weight_scale')
    input_scale = loader.load_tensor(prefix + '.input_scale').float()
    if fp4:
        if values.dtype != torch.uint8 or tuple(scales.shape) != (n, k // 16):
            raise ValueError(f'{prefix}: native NVFP4 storage geometry differs')
        gain = loader.load_tensor(prefix + '.weight_scale_2').float()
        codes = scales.view(torch.uint8)
    else:
        if values.dtype != torch.float8_e4m3fn or scales.numel() != 1:
            raise ValueError(f'{prefix}: native tensor-FP8 storage differs')
        gain = scales.float() * input_scale
        codes = values.view(torch.uint8)
    # Every tensor is admitted, including every layer and the vocabulary head.
    if bool(((codes & 127) == 127).any()):
        raise ValueError(f'{prefix}: finite-only reviewed kernel rejects E4M3 NaN codes')
    if any(t.numel() != 1 or not bool((torch.isfinite(t) & (t > 0)).all()) for t in (gain, input_scale)):
        raise ValueError(f'{prefix}: invalid positive finite calibration scalar')
    raw = values.view(torch.uint8)
    native_scales = codes.contiguous().view(-1) if fp4 else None
    with torch.cuda.use_mem_pool(_weight_pool(storage), device=values.device) if storage is not None else nullcontext():
        packed = torch.empty_like(raw)
        packed_scales = torch.empty_like(native_scales) if fp4 else None
        gain = gain.clone()
        input_scale = input_scale.clone()
        activation_scale = input_scale.reciprocal() if fp4 else None
        unit = torch.ones(1, device=values.device) if not fp4 else None
    inverse = torch.empty_like(raw)
    inverse_scales = torch.empty_like(native_scales) if fp4 else None
    stream = C.c_void_p(torch.cuda.current_stream(values.device).cuda_stream)
    checked(lib.atlas_repack(ptr(raw), ptr(native_scales), ptr(packed), ptr(packed_scales), n, k, fp4, 0, stream))
    checked(lib.atlas_repack(ptr(packed), ptr(packed_scales), ptr(inverse), ptr(inverse_scales), n, k, fp4, 1, stream))
    if not torch.equal(inverse, raw):
        raise ValueError(f'{prefix}: weight repack round trip failed')
    if fp4:
        restored = inverse_scales.view(n, k // 16)
        if not torch.equal(restored, codes):
            raise ValueError(f'{prefix}: native scale repack round trip failed')
        result = RepackedNVFP4(packed, packed_scales, gain, 'multiplier', k, n, activation_scale)
    else:
        result = TensorFP8Weight(RepackedFP8(packed.view(torch.float8_e4m3fn), unit, unit, gain, k, k, n), input_scale)
    for tensor in (packed, packed_scales, gain, input_scale):
        if tensor is not None:
            record_numeric_tensor(tensor)
    ADMISSIONS.append({'prefix': prefix, 'kind': kind, 'K': k, 'N': n,
        'finite_codes_checked': codes.numel(), 'weight_bytes_roundtripped': raw.numel(),
        'native_scale_bytes_roundtripped': codes.numel() if fp4 else 0})
    return result


def workspace_layout(query, selected):
    if selected[0] == 'prefill_tc':
        from .sm70_prefill import workspace_layout
        return workspace_layout(query)
    m = min(MAX_ROWS, query.rows)
    k, n, split = query.in_features, query.out_features, selected[3]
    return ((torch.float32, (m, k)), (torch.float16, (m, k)), (torch.float32, (m,)),
            (torch.float32, (split if selected[4] != 60 and split > 1 else 0, m, n)), (torch.float32, (m, n)))


def workspace_bytes(query, selected):
    import math
    return sum((math.prod(shape) * dtype.itemsize + 255) // 256 * 256
               for dtype, shape in workspace_layout(query, selected))


def run(query, selected, source, values, scales, output, workspace, bias, gain=None, *, live_count=None,
        kernel_cache=None, cutlass_include=None):
    if selected[0] == 'prefill_tc':
        from .sm70_prefill import run
        return run(query, source, values, scales, output, workspace, bias, live_count=live_count,
                   kernel_cache=kernel_cache, cutlass_include=cutlass_include)
    # Row tactics compute every row (their bits never depend on other rows); a device live count only lets the
    # prefill tactic skip padded rows, so it is not forwarded here.
    import math
    views, offset = [], 0
    for dtype, shape in workspace_layout(query, selected):
        size = math.prod(shape) * dtype.itemsize
        views.append(workspace[offset:offset + size].view(dtype).view(shape))
        offset += (size + 255) // 256 * 256
    staging, half, inverse, partial, result = views
    lib = library(kernel_cache)
    stream = C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream)
    bk, warps, split, path = selected[1:]
    for begin in range(0, query.rows, MAX_ROWS):
        # The real row count is passed as M (never host-padded to R): row invariance makes every row's bits
        # independent of M, and produce/stores touch only rows [0, M).
        live = min(MAX_ROWS, query.rows - begin)
        x = source[begin:begin + live]
        if not x.is_contiguous():
            staging[:live].copy_(x)
            x = staging[:live]
        destination = output[begin:begin + live]
        direct = destination.is_contiguous()
        y = destination if direct else result[:live]
        checked(lib.atlas_produce(ptr(x), ptr(half), ptr(inverse), live, query.in_features,
            query.kind == 'tensor_fp8', query.input_scale_value, stream))
        checked(lib.atlas_project(ptr(half), ptr(values), ptr(scales), ptr(inverse), query.gain_value,
            query.global_scale_kind == 'reciprocal', ptr(bias), ptr(y), ptr(partial) if partial.numel() else None, live,
            query.out_features, query.in_features, query.padded_in_features, split,
            query.kind == 'nvfp4', bk, warps, path, ptr(gain) if query.column_gains else None, stream))
        if not direct:
            destination.copy_(y)
    return output


def load_fused(loader, prefixes, *, storage=None, kernel_cache=None):
    """Concatenate admitted 32-column blocks without changing checkpoint codes."""
    from .tensor_fp8 import TensorFP8Weight
    roles = tuple(prefix.rsplit('.', 1)[-1] for prefix in prefixes)
    if roles not in (('gate_proj', 'up_proj'), ('in_proj_qkv', 'in_proj_z'), ('q_proj', 'k_proj', 'v_proj')):
        raise ValueError('unreviewed fusion member order')
    if any(prefix + '.bias' in loader.manifest.tensors for prefix in prefixes):
        raise ValueError('reviewed Qwen fusion requires bias-free checkpoint members')
    with storage.transient() if storage is not None else nullcontext():
        members = [load_weight(loader, prefix, kernel_cache=kernel_cache) for prefix in prefixes]
        fp8 = isinstance(members[0], TensorFP8Weight)
        if any(isinstance(member, TensorFP8Weight) != fp8 for member in members):
            raise ValueError('fused projections must share their format')
        packed = [member.packed if fp8 else member for member in members]
        first = packed[0]
        if any(type(part) is not type(first) or part.values.dtype != first.values.dtype
               or part.in_features != first.in_features or part.padded_in_features != first.padded_in_features
               or part.out_features % 32 for part in packed):
            raise ValueError('fused projections require equal input geometry and 32-column blocks')
        if fp8:
            if any(not torch.equal(member.input_scale, members[0].input_scale) for member in members[1:]):
                raise ValueError('FP8 fusion requires bit-identical calibrated input scales')
            gains = [part.output_scale for part in packed]
        else:
            if any(part.global_scale_kind != first.global_scale_kind for part in packed):
                raise ValueError('fused projections must share the reciprocal convention')
            gains = [part.global_scale for part in packed]
        widths = tuple(part.out_features for part in packed)
        scalars = tuple(float(gain) for gain in gains)
        kind = 'tensor_fp8' if fp8 else 'nvfp4'
        if (kind, first.in_features, sum(widths)) not in PINNED:
            raise ValueError('unreviewed fused projection shape')
        with torch.cuda.use_mem_pool(_weight_pool(storage), device=first.values.device) if storage is not None else nullcontext():
            # Packed N blocks are consecutive. Byte views avoid FP8 cat kernels on SM70.
            values = torch.cat([part.values.view(torch.uint8) for part in packed]).view(first.values.dtype)
            column_gains = torch.cat([gain.expand(width) for gain, width in zip(gains, widths, strict=True)])
            if fp8:
                unit = first.block_scale.clone()
                result = TensorFP8Weight(RepackedFP8(values, unit, unit, column_gains,
                    first.in_features, first.padded_in_features, sum(widths), widths, scalars), members[0].input_scale.clone())
            else:
                scales = torch.cat([part.scale_mma for part in packed])
                result = RepackedNVFP4(values, scales, column_gains, first.global_scale_kind,
                    first.in_features, sum(widths), first.activation_global_scale.clone(), widths, scalars)
        for tensor in (values, column_gains, result.input_scale if fp8 else scales):
            record_numeric_tensor(tensor)
        return result
