"""Sol-20 r4 prefill projections (Sol-8 bits) with one reviewed split per parent matrix and static scratch."""
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


# (Sol-8 reference tile, split), independent of row count and position. No cuBLAS routes. The split fixes every
# element's reduction order; serving runs the Sol-20 tile of TILES below with the same split (same bits).
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


def _sol20(mt, ng, w, bk, pf, minb):
    """pgemm4 configuration id (sm70_pgemm.cuh pgemm4_cfg)."""
    return mt | ng << 4 | w << 8 | (bk // 32) << 12 | pf << 16 | minb << 20 | 1 << 28


# Sol-20 r4 instantiation per (graph capacity, shape) (astra-18). Every id produces the PINNED split's Sol-8 bits for
# live rows (sol-20 identity4b: 0 mismatches at capacities 128/256/512); the capacity only selects the fastest measured
# tile: capacity 128 = the Sol-20 live-choice result, capacity 256 = fastest full-M=256 tile of the Sol-20 live sweep.
# Graphs with more than 128 rows use the 256 choices. sm70_pgemm_configs.inc compiles exactly these instantiations.
TILES = {
    128: {
        ('tensor_fp8', 6144, 5120): _sol20(4, 1, 4, 64, 1, 4),
        ('tensor_fp8', 5120, 10240): _sol20(6, 1, 4, 64, 1, 3),
        ('tensor_fp8', 5120, 6144): _sol20(4, 1, 4, 64, 1, 4),
        ('tensor_fp8', 5120, 12288): _sol20(4, 2, 4, 32, 1, 1),
        ('tensor_fp8', 5120, 1024): _sol20(3, 1, 4, 64, 1, 5),
        ('nvfp4', 5120, 17408): _sol20(8, 2, 8, 64, 1, 1),
        ('nvfp4', 17408, 5120): _sol20(4, 2, 4, 32, 1, 1),
        ('nvfp4', 5120, 248320): _sol20(8, 2, 8, 64, 2, 1),
    },
    256: {
        ('tensor_fp8', 6144, 5120): _sol20(4, 2, 4, 32, 1, 1),
        ('tensor_fp8', 5120, 10240): _sol20(8, 2, 8, 64, 1, 1),
        ('tensor_fp8', 5120, 6144): _sol20(4, 2, 4, 32, 1, 1),
        ('tensor_fp8', 5120, 12288): _sol20(4, 2, 4, 32, 1, 1),
        ('tensor_fp8', 5120, 1024): _sol20(4, 1, 4, 64, 1, 4),
        ('nvfp4', 5120, 17408): _sol20(8, 2, 8, 64, 2, 1),
        ('nvfp4', 17408, 5120): _sol20(8, 2, 8, 64, 1, 1),
        ('nvfp4', 5120, 248320): _sol20(8, 2, 8, 64, 2, 1),
    },
}


def tile(kind, k, n, capacity):
    return TILES[128 if capacity <= 128 else 256][kind, k, n]


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
        'repacked_kernel.cuh', 'fast_kernel.cuh', 'deep_kernel.cuh', 'sm70_pgemm.cuh', 'sm70_pgemm_configs.inc')]
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
    lib.atlas_pgemm4.argtypes = ([C.c_void_p] * 4 + [C.c_float, C.c_int] + [C.c_void_p] * 3 + [C.c_int] * 7
                                 + [C.c_void_p] * 2)
    lib.atlas_pgemm4_describe.argtypes = [C.c_int, C.c_int, C.c_void_p]
    checked(lib.atlas_prefill_init())
    checked(lib.atlas_pgemm_init())
    # Fail closed: every served (capacity, shape) id must resolve to a compiled instantiation of its format.
    described = (C.c_int * 6)()
    for choices in TILES.values():
        for (kind, _, _), config in choices.items():
            if lib.atlas_pgemm4_describe(kind == 'nvfp4', config, described):
                raise RuntimeError(f'SM70 prefill library lacks Sol-20 instantiation {config:#x} for {kind}')
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


def run(query, source, values, scales, output, workspace, bias, *, live_count=None, kernel_cache=None,
        cutlass_include=None):
    views, offset = [], 0
    for dtype, shape in workspace_layout(query):
        size = math.prod(shape) * dtype.itemsize
        views.append(workspace[offset:offset + size].view(dtype).view(shape))
        offset += (size + 255) // 256 * 256
    staging, half, inverse, partial, result = views
    lib, producer = library(kernel_cache, cutlass_include), projection_library(kernel_cache)
    stream = C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream)
    k, pk = query.in_features, query.padded_in_features
    # `live_count` (device Int32 scalar, e.g. metadata.live_counts[:1]): rows < *live_count keep Sol-8's bits, rows in
    # [*live_count, M) are written +0.0 without reading their activations or inverse scales. One launch per graph.
    if live_count is not None and query.rows > 512:
        raise ValueError('a device live count requires a prefill graph of at most 512 rows')
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
            split = PINNED[query.kind, k, n][1]
            w = values[column:column + n]
            scale = None if scales is None else scales[column * (pk // 16):(column + n) * (pk // 16)]
            destination = output[begin:begin + m, column:column + n]
            direct = destination.is_contiguous()
            y = destination if direct else result.view(-1)[:m * n].view(m, n)
            member_bias = None if bias is None else bias[column:column + n]
            checked(lib.atlas_pgemm4(ptr(half), ptr(w), ptr(scale), ptr(inverse), gain,
                query.global_scale_kind == 'reciprocal', ptr(member_bias), ptr(y), ptr(partial),
                m, n, k, pk, split, query.kind == 'nvfp4', tile(query.kind, k, n, query.rows), ptr(live_count),
                stream))
            if not direct:
                destination.copy_(y)
            column += n
    return output
