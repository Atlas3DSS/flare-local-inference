"""Reviewed Sol-11s tensor-core prefill attention for the 16 full-attention layers on SM70.

One prefill request per captured graph. Captured graphs cannot carry host integers, so the per-chunk plan (M, context,
split plan) is derived on the device from the request metadata the prefill graph already owns (``lengths``,
``query_start_loc``); page metadata is recomputed for every page of the request at every chunk (never incremental).
A non-zero status word (non-finite K/V, a V range above 2^80 across pages, or an inadmissible (M, context)) makes the
combine write NaN rows, which the server turns into an explicit request error: there is no in-graph fallback.
Workspace, plan and metadata live in the shared operator scratch arena: the status word is reset inside the graph, so
no scratch state survives a chunk. Decode, resume and speculative verification keep the scalar paged attention.
"""

import ctypes as C
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import subprocess

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import scratch_storage
from ignition.preparation.tuning import declare, register_finalizer
from .native_operator import native_contract, native_plan

PAGE_TOKENS = 128
MAX_TOKENS = 512          # graph token capacity admitted by the kernel
MAX_CONTEXT = 34816       # cached tokens admitted by plan_kernel (the served context of the real-world package)
MAX_PAGES = 276           # page-table width admitted: ceil((MAX_CONTEXT + MAX_TOKENS) / PAGE_TOKENS)
HEADS, KV_HEADS, DIM = 24, 4, 256
PLAN_WORDS = 16        # prefill-lab EXP-10 (G128): 8 header words + (span, splits) for 4 groups of 128 rows
STATUS_BYTES = 16         # workspace header holding the sticky int32 status word
ALIGN = 256


@lru_cache(None)
def compile_library(kernel_cache=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if CUDA_HOME is None:
        raise RuntimeError("SM70 prefill attention preparation requires the configured CUDA toolkit")
    source = Path(__file__).with_suffix(".cu")
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    cache = Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + ".so")
    with (cache / (digest + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            workers = len(os.sched_getaffinity(0))
            # Same flags as the Sol-11s build.sh: the FP64-oracle and admission gates ran on this exact flag set.
            subprocess.run([str(Path(CUDA_HOME) / "bin/nvcc"), "--threads", str(workers), "-std=c++17", "-O3",
                "-arch=compute_70", "-code=sm_70", "--fmad=false", "-Xcompiler=-fPIC", "--shared",
                str(source), "-o", str(output)], check=True)
    return str(output)


@lru_cache(None)
def library(kernel_cache=None):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM70 prefill attention library must be loaded before capture")
    lib = C.CDLL(compile_library(kernel_cache))
    lib.sol11s_attention_workspace.restype = C.c_size_t
    lib.sol11s_attention_init.argtypes = [C.c_void_p, C.c_void_p]
    lib.sol11s_plan.argtypes = [C.c_void_p] * 3 + [C.c_int, C.c_int, C.c_void_p]
    lib.sol11s_meta_dev.argtypes = [C.c_void_p] * 6 + [C.c_int, C.c_void_p]
    lib.sol11s_attention_dev.argtypes = [C.c_void_p] * 8 + [C.c_int, C.c_int, C.c_void_p]
    for name in ("sol11s_attention_init", "sol11s_plan", "sol11s_meta_dev", "sol11s_attention_dev"):
        getattr(lib, name).restype = C.c_int
    return lib


def workspace_bytes(kernel_cache=None):
    return int(library(kernel_cache).sol11s_attention_workspace())


def _aligned(nbytes):
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def scratch_layout(physical_pages, kernel_cache=None):
    """Byte offsets of (workspace, plan, meta) inside one contiguous scratch buffer, and its total size."""
    workspace = _aligned(workspace_bytes(kernel_cache))
    plan = _aligned(PLAN_WORDS * 4)
    meta = _aligned(physical_pages * KV_HEADS * 2 * 4)
    return dict(workspace=(0, workspace), plan=(workspace, plan), meta=(workspace + plan, meta),
                total=workspace + plan + meta)


def _ptr(tensor):
    return C.c_void_p(tensor.data_ptr())


def _checked(status, what):
    if status:
        raise RuntimeError(f"SM70 prefill attention {what} failed: CUDA error {status}")


@dataclass(frozen=True)
class Query:
    capacity: int
    physical_pages: int
    max_pages: int


CONTRACT = native_contract(component_id="flare.sm70.prefill_attention", query_type=Query, tactics=lambda q: ((1,),))


@dataclass(frozen=True)
class State:
    lib: object
    capacity: int
    max_pages: int

    def run(self, q, key_cache, value_cache, pages, lengths, cu, output, workspace, plan, meta):
        stream = C.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
        _checked(self.lib.sol11s_plan(_ptr(lengths), _ptr(cu), _ptr(plan), self.capacity, self.max_pages, stream), "plan")
        _checked(self.lib.sol11s_meta_dev(_ptr(key_cache), _ptr(value_cache), _ptr(pages), _ptr(meta), _ptr(workspace),
                                          _ptr(plan), self.max_pages, stream), "metadata")
        _checked(self.lib.sol11s_attention_dev(_ptr(q), _ptr(key_cache), _ptr(value_cache), _ptr(pages), _ptr(meta),
                                               _ptr(output), _ptr(workspace), _ptr(plan), self.capacity, PAGE_TOKENS,
                                               stream), "attention")


@torch.library.custom_op("flare::sm70_prefill_attention", mutates_args=("output", "workspace", "plan", "meta"))
def _execute(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, pages: torch.Tensor,
             lengths: torch.Tensor, cu: torch.Tensor, output: torch.Tensor, workspace: torch.Tensor,
             plan: torch.Tensor, meta: torch.Tensor, handle: int) -> None:
    prepared = plan_from_handle(handle)
    require_prepared(prepared, prepared.component_id, q.device).run(
        q, key_cache, value_cache, pages, lengths, cu, output, workspace, plan, meta)


@_execute.register_fake
def _fake(q, key_cache, value_cache, pages, lengths, cu, output, workspace, plan, meta, handle):
    pass


def admits(q, key_cache, pages):
    """Static shape admission of the reviewed Sol-11s contract (one request per graph, Qwen3.8-27B geometry)."""
    return (q.ndim == 3 and tuple(q.shape[1:]) == (HEADS, DIM) and 1 <= q.shape[0] <= MAX_TOKENS
            and key_cache.ndim == 4 and tuple(key_cache.shape[1:]) == (PAGE_TOKENS, KV_HEADS, DIM)
            and pages.ndim == 2 and pages.shape[0] == 1 and 1 <= pages.shape[1] <= MAX_PAGES)


class PreparedPrefillAttention:
    """One prefill request per graph: Q/out FP32 [capacity, 24, 256], BF16-bit pages [physical, 128, 4, 256]."""

    def __init__(self, q, key_cache, value_cache, pages, lengths, cu, output, *, session, name, workspace=None):
        device = q.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
            raise ValueError("SM70 prefill attention requires a V100")
        if not admits(q, key_cache, pages):
            raise ValueError("SM70 prefill attention operands are outside the reviewed Sol-11s contract")
        capacity = q.shape[0]
        physical = key_cache.shape[0]
        if (q.dtype != torch.float32 or output.dtype != torch.float32 or output.shape != q.shape
                or key_cache.dtype != torch.uint16 or value_cache.shape != key_cache.shape
                or value_cache.dtype != torch.uint16 or pages.dtype != torch.int32
                or lengths.shape != (1,) or lengths.dtype != torch.int32 or cu.shape != (2,) or cu.dtype != torch.int32
                or not all(t.is_contiguous() for t in (q, output, key_cache, value_cache, pages, lengths, cu))
                or any(t.device != device for t in (key_cache, value_cache, pages, lengths, cu, output))
                or torch._C._overlaps(output, q)):
            raise ValueError("SM70 prefill attention requires contiguous FP32 queries/output, UInt16 BF16-bit pages "
                             "and int32 request metadata on one V100")
        self.kernel_cache = getattr(session, "kernel_cache", None)
        self.tensors = (q, key_cache, value_cache, pages, lengths, cu, output)
        self.workspace, self.scratch = workspace, None
        self.layout = scratch_layout(physical, self.kernel_cache)
        query = Query(capacity, physical, pages.shape[1])
        nbytes = self.layout["total"]

        def materialize(tactic):
            # Outside capture (tuning phase): load the library and set the dynamic shared-memory attribute of both
            # tc_attention instantiations before the first launch; the status header is zeroed again in-graph.
            lib = library(self.kernel_cache)
            stream = C.c_void_p(torch.cuda.current_stream(device).cuda_stream)
            _checked(lib.sol11s_attention_init(_ptr(self.views()[0]), stream), "init")
            return State(lib, capacity, pages.shape[1])

        self.plan = native_plan(contract=CONTRACT, query=query, device=device,
                                scratch_bytes=lambda _: nbytes, materialize=materialize)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.tensors, *self.views()))),))
        register_finalizer(session, self.bind, workspace=workspace)
        logging.getLogger(__name__).info(
            "SM70 prefill attention %s: capacity %d tokens, page table %d pages, %d physical pages, scratch %d bytes",
            name, capacity, pages.shape[1], physical, nbytes)

    def views(self):
        """Carve (workspace, plan, meta) from the operator scratch; re-read the arena address on every call before
        it is frozen (growth moves it), and keep the finalized binding afterwards."""
        if self.scratch is not None:
            return self.scratch
        storage = scratch_storage(self.workspace, nbytes=self.layout["total"], device=self.tensors[0].device)
        physical = self.tensors[1].shape[0]
        offset, size = self.layout["workspace"]
        workspace = storage[offset:offset + size]
        offset, size = self.layout["plan"]
        plan = storage[offset:offset + PLAN_WORDS * 4].view(torch.int32)
        offset, size = self.layout["meta"]
        meta = storage[offset:offset + physical * KV_HEADS * 2 * 4].view(torch.int32).view(physical, KV_HEADS, 2)
        if workspace.data_ptr() % 16:
            raise ValueError("SM70 prefill attention workspace requires 16-byte alignment")
        return workspace, plan, meta

    def bind(self):
        """After tuning, before capture: fix the scratch views and set the kernels' dynamic shared-memory attribute."""
        self.scratch = None
        self.scratch = self.views()
        stream = C.c_void_p(torch.cuda.current_stream(self.tensors[0].device).cuda_stream)
        _checked(library(self.kernel_cache).sol11s_attention_init(_ptr(self.scratch[0]), stream), "init")

    @property
    def status_word(self):
        """int32[1] view of the sticky status word for in-graph accumulation (valid right after this chunk ran)."""
        return self.views()[0][:4].view(torch.int32)

    @property
    def status(self):
        """Sticky admission status word of the last chunk (0 admitted; 1 non-finite K/V; 2 V range > 2^80; 4 inadmissible
        (M, context)). Reads the shared scratch arena: only meaningful right after this layer's chunk ran."""
        return self.views()[0][:4].view(torch.int32)[0]

    def __call__(self):
        _execute(*self.tensors, *self.views(), self.plan.handle)
        return self.tensors[6]
