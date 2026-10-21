"""Sol-8 prefill with one reviewed split/tile per parent matrix and static scratch."""
import ctypes as C
from functools import lru_cache
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess

import torch

from .sm70_projection_tc import checked, ptr, library as projection_library


# (tile, split), independent of row count and position. No cuBLAS routes.
PINNED = {
    ('tensor_fp8', 6144, 5120): (23, 1),
    ('tensor_fp8', 5120, 10240): (15, 1),
    ('tensor_fp8', 5120, 6144): (9, 1),
    ('tensor_fp8', 5120, 12288): (9, 1),
    ('tensor_fp8', 5120, 1024): (23, 4),
    ('nvfp4', 5120, 17408): (14, 1),
    ('nvfp4', 17408, 5120): (23, 2),
    ('nvfp4', 5120, 248320): (14, 1),
}


def _cutlass_headers(include):
    if include is None:
        raise ValueError('SM70 prefill requires a configured cutlass_include directory')
    root = Path(include).expanduser().resolve()
    pending = [root / 'cutlass/gemm/warp' / name for name in
               ('mma_tensor_op_sm70.h', 'mma_tensor_op_policy.h')]
    for path in pending:
        if not path.is_file():
            raise ValueError(f'SM70 prefill CUTLASS header is missing: {path}')
    headers = {}
    while pending:
        path = pending.pop()
        if path in headers:
            continue
        headers[path] = path.read_bytes()
        for name in re.findall(rb'^\s*#\s*include\s*[<"]([^>"]+)[>"]', headers[path], re.MULTILINE):
            for candidate in (path.parent / name.decode(), root / name.decode()):
                candidate = candidate.resolve()
                if candidate.is_relative_to(root) and candidate.is_file():
                    pending.append(candidate)
                    break
    return root, tuple((str(path.relative_to(root)), headers[path]) for path in sorted(headers))


@lru_cache(None)
def library(kernel_cache=None, cutlass_include=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('SM70 prefill library must be loaded before capture')
    if CUDA_HOME is None:
        raise RuntimeError('SM70 prefill preparation requires the configured CUDA toolkit')
    directory = Path(__file__).parent
    sources = [directory / name for name in ('sm70_prefill.cu', 'repacked.cu',
        'repacked_kernel.cuh', 'fast_kernel.cuh', 'deep_kernel.cuh')]
    includes, headers = _cutlass_headers(cutlass_include)
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in sources) + Path(__file__).read_bytes()
        + str(includes).encode() + b''.join(name.encode() + data for name, data in headers)).hexdigest()
    cache = Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + '.so')
    with (cache / (digest + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            nvcc = Path(CUDA_HOME) / 'bin/nvcc'
            workers = str(len(os.sched_getaffinity(0)))
            subprocess.run([str(nvcc), '--threads', workers, '--split-compile=' + workers,
                '-std=c++17', '-O3', '-arch=compute_70', '-code=sm_70', '-Xcompiler=-fPIC',
                '--shared', '--fmad=false', '-DSOL_STAGE=1', '-I' + str(includes),
                str(sources[0]), '-o', str(output)], check=True)
    lib = C.CDLL(str(output))
    lib.atlas_prefill.argtypes = [C.c_void_p] * 4 + [C.c_float, C.c_int] + [C.c_void_p] * 3 + [C.c_int] * 7 + [C.c_void_p]
    checked(lib.atlas_prefill_init())
    return lib


def members(query):
    widths = query.member_widths or (query.out_features,)
    gains = query.member_gains or (query.gain_value,)
    return tuple(zip(widths, gains, strict=True))


def workspace_layout(query):
    m, k = min(query.rows, 512), query.in_features
    shapes = [(n, PINNED[query.kind, k, n][1]) for n, _ in members(query)]
    partial = max((n * split for n, split in shapes if split > 1), default=0)
    return ((torch.float32, (m, k)), (torch.float16, (m, k)), (torch.float32, (m,)),
            (torch.float32, (m * partial,)), (torch.float32, (m, max(n for n, _ in shapes))))


def run(query, source, values, scales, output, workspace, bias, *, kernel_cache=None, cutlass_include=None):
    views, offset = [], 0
    for dtype, shape in workspace_layout(query):
        size = math.prod(shape) * dtype.itemsize
        views.append(workspace[offset:offset + size].view(dtype).view(shape))
        offset += (size + 255) // 256 * 256
    staging, half, inverse, partial, result = views
    lib, producer = library(kernel_cache, cutlass_include), projection_library(kernel_cache)
    stream = C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream)
    k, pk = query.in_features, query.padded_in_features
    for begin in range(0, query.rows, 512):
        m = min(512, query.rows - begin)
        x = source[begin:begin + m]
        if not x.is_contiguous():
            staging[:m].copy_(x)
            x = staging[:m]
        checked(producer.atlas_produce(ptr(x), ptr(half), ptr(inverse), m, k,
            query.kind == 'tensor_fp8', query.input_scale_value, stream))
        column = 0
        for n, gain in members(query):
            tile, split = PINNED[query.kind, k, n]
            w = values[column:column + n]
            scale = None if scales is None else scales[column * (pk // 16):(column + n) * (pk // 16)]
            destination = output[begin:begin + m, column:column + n]
            direct = destination.is_contiguous()
            y = destination if direct else result.view(-1)[:m * n].view(m, n)
            member_bias = None if bias is None else bias[column:column + n]
            checked(lib.atlas_prefill(ptr(half), ptr(w), ptr(scale), ptr(inverse), gain,
                query.global_scale_kind == 'reciprocal', ptr(member_bias), ptr(y), ptr(partial),
                m, n, k, pk, split, query.kind == 'nvfp4', tile, stream))
            if not direct:
                destination.copy_(y)
            column += n
    return output
