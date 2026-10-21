"""Exact alias-safe copies of admitted recurrent rows on SM70."""

import ctypes as C
import fcntl
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import subprocess

import torch

from .state import gather_rows, scatter_rows


@lru_cache(None)
def compile_library(kernel_cache=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if CUDA_HOME is None:
        raise RuntimeError("SM70 commit preparation requires the configured CUDA toolkit")
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
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM70 commit library must be loaded before capture")
    lib = C.CDLL(compile_library(kernel_cache))
    lib.sol12_commit.argtypes = [C.c_void_p, C.c_int, C.c_int] + [C.c_void_p] * 3 + [C.c_int, C.c_void_p]
    lib.sol12_commit.restype = C.c_int
    return lib


@torch.library.custom_op("flare::sm70_recurrent_commit", mutates_args=("backing",))
def _execute(backing: torch.Tensor, descriptors: torch.Tensor, source: torch.Tensor,
             destination: torch.Tensor, active: torch.Tensor, width: int, kernel_cache: str) -> None:
    status = library(kernel_cache).sol12_commit(C.c_void_p(descriptors.data_ptr()), descriptors.shape[0], width,
        C.c_void_p(source.data_ptr()), C.c_void_p(destination.data_ptr()), C.c_void_p(active.data_ptr()),
        source.numel(), C.c_void_p(torch.cuda.current_stream(backing.device).cuda_stream))
    if status:
        raise RuntimeError(f"SM70 recurrent commit CUDA launch failed: {status}")


@_execute.register_fake
def _fake(backing, descriptors, source, destination, active, width, kernel_cache):
    pass


class PreparedSM70RecurrentCommit:
    """Copy unique admitted destinations from persistent source/active metadata.

    Layer views share a byte backing, have uniform widths, and occupy disjoint
    regions of each row. Sources may alias destinations, including cycles.
    Callers validate active source rows against ``memory_capacity[0]`` before
    execution, as dense MTP's batch validator does. The descriptor destination
    bound is refreshed from that device binding on every replay, so VMM growth
    never relies on the preparation-time backed row count. Fixed-address package
    storage preserves the pointers embedded in the descriptor byte tensor.
    """

    def __init__(self, backing, layers, source, destination, active, memory_capacity, *, kernel_cache=None):
        from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

        self.kernel_cache = str(Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve())
        if (backing.ndim != 2 or backing.dtype != torch.uint8 or not backing.is_contiguous()
                or backing.shape[1] % 4 or not 2 <= backing.shape[0] <= 2**31 - 1):
            raise ValueError("SM70 recurrent commit requires contiguous byte-pool rows")
        if backing.is_cuda and torch.cuda.get_device_capability(backing.device) != (7, 0):
            raise ValueError("SM70 recurrent commit requires a V100 CUDA device")
        if (source.ndim != 1 or not 1 <= source.numel() <= 16
                or source.dtype != torch.int64 or destination.dtype != torch.int64
                or active.dtype != torch.uint8 or destination.shape != source.shape
                or active.shape != source.shape or memory_capacity.shape != (2,)
                or memory_capacity.dtype != torch.int64
                or any(t.device != backing.device or not t.is_contiguous()
                       for t in (source, destination, active, memory_capacity))):
            raise ValueError("SM70 recurrent commit requires persistent contiguous row and capacity metadata")
        self.layers = tuple(layers)
        if not 1 <= len(self.layers) <= 65535:
            raise ValueError("SM70 recurrent commit requires one through 65535 layers")
        first = self.layers[0]
        if first.ndim != 2 or not 1 <= first.shape[1] <= 2**31 - 1 - 1023:
            raise ValueError("SM70 recurrent commit requires positive bounded layer widths")
        self.width = first.shape[1]
        intervals, descriptors = [], []
        for layer in self.layers:
            if (layer.ndim != 2 or layer.shape != (backing.shape[0], self.width)
                    or layer.dtype != torch.float32 or layer.device != backing.device
                    or layer.stride(1) != 1 or layer.stride(0) < self.width
                    or layer.stride(0) * 4 != backing.stride(0)
                    or layer.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()):
                raise ValueError("SM70 recurrent layers require uniform widths and nonoverlapping pool-row strides")
            offset = layer.data_ptr() - backing.data_ptr()
            end = offset + self.width * 4
            if offset < 0 or end > backing.shape[1] or any(offset < stop and start < end for start, stop in intervals):
                raise ValueError("SM70 recurrent layer regions must be disjoint within each pool row")
            if layer.data_ptr() % 16 or layer.stride(0) % 4 or self.width % 4:
                raise ValueError("SM70 recurrent layer vectors require 16-byte aligned bases, rows and widths")
            intervals.append((offset, end))
            descriptors.append((layer.data_ptr(), layer.stride(0), self.width))
        self.backing, self.source, self.destination = backing, source, destination
        self.active, self.memory_capacity = active, memory_capacity
        # Three int64 words exactly encode {pointer, int64 stride, int32 width,
        # int32 slots}. Keep this pointer-bearing storage unmarked as numeric.
        self.descriptors = torch.tensor(descriptors, dtype=torch.int64, device=backing.device).view(torch.uint8)
        if self.descriptors.device != backing.device or self.descriptors.data_ptr() % 8:
            raise ValueError("SM70 commit descriptors require device-resident 8-byte aligned storage")
        self.copy = None if backing.is_cuda else torch.empty(
            (source.numel(), self.width), dtype=torch.float32, device=backing.device)
        logging.getLogger(__name__).info(
            "SM70 recurrent commit: %d layers, width=%d, row_stride=%d, 16-byte vectors aligned",
            len(self.layers), self.width, first.stride(0))

    @property
    def descriptor_slots(self):
        return self.descriptors.view(torch.int32).view(-1, 6)[:, 5]

    def prepare(self):
        if self.backing.is_cuda:
            library(self.kernel_cache)

    def __call__(self):
        self.descriptor_slots.copy_(self.memory_capacity[:1])
        if self.backing.is_cuda:
            _execute(self.backing, self.descriptors, self.source, self.destination, self.active, self.width,
                     self.kernel_cache)
        else:
            for layer in self.layers:
                gather_rows(layer, self.source, self.copy)
                scatter_rows(self.copy, self.destination, layer,
                             self.active.bool() & (self.destination < self.memory_capacity[0]))
