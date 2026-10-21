"""Reviewed Sol-7 projections with fixed reductions and prepared, bounded storage."""
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


@dataclass(frozen=True)
class RepackedFP8(TensorFP8LinearWeight):
    """Same byte count; values use the reviewed N-partitioned layout."""


# Reviewed M=16 tuple (BK, warps, split, path), fixed across all row counts.
PINNED = {
    ('tensor_fp8', 6144, 5120): (128, 4, 4, 51),
    ('tensor_fp8', 5120, 10240): (128, 4, 4, 51),
    ('tensor_fp8', 5120, 6144): (128, 4, 4, 51),
    ('nvfp4', 5120, 17408): (256, 8, 2, 51),
    ('nvfp4', 17408, 5120): (128, 4, 8, 51),
    ('tensor_fp8', 5120, 12288): (128, 4, 4, 47),
    ('tensor_fp8', 5120, 1024): (128, 4, 8, 51),
    ('nvfp4', 5120, 248320): (256, 4, 1, 47),
    # Draft-only shapes use the same admitted body and fixed reductions.
    ('nvfp4', 10240, 5120): (128, 4, 8, 51),
    ('nvfp4', 6144, 5120): (128, 4, 4, 51),
    ('nvfp4', 5120, 12288): (128, 4, 4, 47),
    ('nvfp4', 5120, 1024): (128, 4, 8, 51),
}
ROWS = (1, 2, 4, 8, 12, 16)
ADMISSIONS = []


def tactic(query):
    bk, warps, split, path = PINNED[(query.kind, query.in_features, query.out_features)]
    return ('row_tc', bk, warps, split, path)


@lru_cache(None)
def compile_library():
    directory = Path(__file__).parent
    sources = [directory / name for name in ('sm70_projection_tc.cu', 'repacked.cu', 'repacked_kernel.cuh', 'fast_kernel.cuh', 'deep_kernel.cuh')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    cache = Path(os.environ.get('FLARE_SM70_KERNEL_CACHE', '/mnt/storage16tb/flare/kernel-cache'))
    cache.mkdir(exist_ok=True)
    output = cache / (digest + '.so')
    with (cache / (digest + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            nvcc = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-12.9')) / 'bin/nvcc'
            command = [str(nvcc), '--threads', str(len(os.sched_getaffinity(0))),
                '--split-compile=' + str(len(os.sched_getaffinity(0))), '-std=c++17', '-O3',
                '-arch=compute_70', '-code=sm_70', '-Xcompiler=-fPIC', '--shared',
                '--fmad=false', '-DSOL_STAGE=1', str(sources[0]), '-o', str(output)]
            subprocess.run(command, check=True)
    return str(output)


@lru_cache(None)
def library():
    lib = C.CDLL(compile_library())
    lib.atlas_produce.argtypes = [C.c_void_p] * 3 + [C.c_int] * 3 + [C.c_float, C.c_void_p]
    lib.atlas_project.argtypes = [C.c_void_p] * 4 + [C.c_float, C.c_int] + [C.c_void_p] * 3 + [C.c_int] * 9 + [C.c_void_p]
    lib.atlas_repack.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.c_void_p]
    lib.atlas_position_exact.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    lib.atlas_fp8_exact.argtypes = [C.c_void_p] * 3
    return lib


def ptr(tensor):
    return None if tensor is None else C.c_void_p(tensor.data_ptr())


def checked(status):
    if status:
        raise RuntimeError(f'SM70 tensor-core CUDA launch failed: {status}')


def load_weight(loader, prefix, *, storage=None):
    # Keep temporary native/reverse-repacked weights out of the retained pool.
    # Otherwise their interleaved large allocations strand device segments.
    with storage.transient() if storage is not None else nullcontext():
        return _load_weight(loader, prefix, storage=storage)


def _weight_pool(storage):
    if not hasattr(storage, '_sm70_repacked_weights'):
        pool = torch.cuda.MemPool(storage._allocator.allocator())
        storage._preparation_pools.add(pool.id)
        storage._sm70_repacked_weights = pool
    return storage._sm70_repacked_weights


def _load_weight(loader, prefix, *, storage):
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
    lib = library()
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
    m = min(16, next((m for m in ROWS if m >= query.rows), 16))
    k, n, split = query.in_features, query.out_features, selected[3]
    return ((torch.float32, (m, k)), (torch.float16, (m, k)), (torch.float32, (m,)),
            (torch.float32, (split, m, n)), (torch.float32, (m, n)))


def workspace_bytes(query, selected):
    import math
    return sum((math.prod(shape) * dtype.itemsize + 255) // 256 * 256
               for dtype, shape in workspace_layout(query, selected))


def run(query, selected, source, values, scales, output, workspace, bias):
    import math
    views, offset = [], 0
    for dtype, shape in workspace_layout(query, selected):
        size = math.prod(shape) * dtype.itemsize
        views.append(workspace[offset:offset + size].view(dtype).view(shape))
        offset += (size + 255) // 256 * 256
    staging, half, inverse, partial, result = views
    lib = library()
    stream = C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream)
    bk, warps, split, path = selected[1:]
    for begin in range(0, query.rows, 16):
        live = min(16, query.rows - begin)
        m = next(m for m in ROWS if m >= live)
        if m != live:
            staging[:m].zero_()
        x = source[begin:begin + live]
        if m != live or not x.is_contiguous():
            staging[:live].copy_(x)
            x = staging[:m]
        destination = output[begin:begin + live]
        direct = live == m and destination.is_contiguous()
        y = destination if direct else result[:m]
        checked(lib.atlas_produce(ptr(x), ptr(half), ptr(inverse), m, query.in_features,
            query.kind == 'tensor_fp8', query.input_scale_value, stream))
        checked(lib.atlas_project(ptr(half), ptr(values), ptr(scales), ptr(inverse), query.gain_value,
            query.global_scale_kind == 'reciprocal', ptr(bias), ptr(y), ptr(partial), m,
            query.out_features, query.in_features, query.padded_in_features, split,
            query.kind == 'nvfp4', bk, warps, path, stream))
        if not direct:
            destination.copy_(y[:live])
    return output
