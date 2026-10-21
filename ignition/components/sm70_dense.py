"""Reviewed BF16 weight-only SM70 projections with one reduction per matrix."""

import ctypes as C
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess


# Independent Sol-9 plan: K/V use the same reduction in draft and absorption.
# These tuples never depend on the row count or the projection's call site.
PINNED = {
    (5120, 96): 22,
    (10240, 5120): 42,
    (5120, 12288): 36,
    (5120, 1024): 54,
    (6144, 5120): 43,
    (5120, 17408): 36,
    (17408, 5120): 42,
}
MAX_ROWS = 20


def tactic(query):
    if (query.kind != "dense" or query.weight_dtype != "bfloat16"
            or query.input_dtype != "float32" or query.output_dtype != "float32"
            or query.has_bias or query.padded_in_features != query.in_features):
        return None
    selected = PINNED.get((query.in_features, query.out_features))
    return None if selected is None else ("row_dense", selected)


@lru_cache(None)
def compile_library(kernel_cache=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if CUDA_HOME is None:
        raise RuntimeError("SM70 dense preparation requires the configured CUDA toolkit")
    source = Path(__file__).with_suffix(".cu")
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    cache = Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + ".so")
    with (cache / (digest + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            workers = len(os.sched_getaffinity(0))
            subprocess.run([str(Path(CUDA_HOME) / "bin/nvcc"), "--threads", str(workers),
                "--split-compile=" + str(workers), "-std=c++17", "-O3",
                "-arch=compute_70", "-code=sm_70", "-Xcompiler=-fPIC", "--shared",
                str(source), "-o", str(output)], check=True)
    return str(output)


@lru_cache(None)
def library(kernel_cache=None):
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM70 dense library must be loaded during preparation, before capture")
    lib = C.CDLL(compile_library(kernel_cache))
    lib.sol9_dense.argtypes = [C.c_void_p] * 5 + [C.c_int] * 8 + [C.c_void_p]
    lib.sol9_dense.restype = C.c_int
    return lib


def run(query, selected, source, values, output, bias=None, *, kernel_cache=None):
    import torch

    if selected != tactic(query) or selected is None or bias is not None:
        raise ValueError("SM70 dense requires an admitted BF16, FP32-in/out, bias-free plan")
    k, n = query.in_features, query.out_features
    if (source.dtype != torch.float32 or output.dtype != torch.float32
            or source.shape != (query.rows, k) or output.shape != (query.rows, n)
            or query.rows < 1 or source.device.type != "cuda"
            or output.device != source.device or values.device != source.device
            or source.stride(1) != 1 or output.stride(1) != 1
            or source.stride(0) < k or source.stride(0) % 4 or output.stride(0) < n
            or source.data_ptr() % 16 or values.data_ptr() % 16
            or not values.is_contiguous()
            or not ((values.dtype == torch.uint8 and values.shape == (n, 2 * k))
                    or (values.dtype == torch.bfloat16 and values.shape == (n, k)))):
        raise ValueError("SM70 dense operands must retain their prepared shapes, strides and storage")
    lib = library(kernel_cache)
    stream = C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream)
    # Prefill and absorption retain the same row-local reduction as decode and
    # verification; no scratch, padding, conversion or output-copy launches.
    for begin in range(0, query.rows, MAX_ROWS):
        live = min(MAX_ROWS, query.rows - begin)
        status = lib.sol9_dense(
            C.c_void_p(source.data_ptr() + begin * source.stride(0) * 4),
            C.c_void_p(values.data_ptr()), None,
            C.c_void_p(output.data_ptr() + begin * output.stride(0) * 4), None,
            live, n, k, source.stride(0), output.stride(0), 0, 0, selected[1], stream)
        if status:
            raise RuntimeError(f"SM70 dense CUDA launch failed: {status}")
    return output
