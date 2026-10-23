"""Reviewed Sol-21 r2 S1L tensor-core decode/verification attention for the 16 full-attention layers on SM70.

Sol-21 r2 = Sol-13c with predicated K/V loads, bit-identical to Sol-13c on every gate. Sol-13c keeps Sol-13b's ABI
(sol13b_* entry points, variants, workspace) and adds the Flare row contract: verification row j of an agent with R rows
is bit-identical to the single-row launch at visible length ctx0 + j + 1 (per-row boundary-tile conversion, per-row
ladder rung), so accepting k rows leaves the state of k ordinary greedy decodes (forced_prefix gate).
The kernel sizes its own grid (gsplits grows at a ladder straddle) and workspace from max_length; the wrapper only asks
sol13b_workspace_bytes and never derives a launch size itself.

Raw-BF16 mode (variant = R_cap | 128: partition ladder, no S1 writer, meta = NULL): K/V pages stay canonical BF16 bits
for the prefill kernel, the KV writer, prefix-cache offload and snapshot/restore; bits equal the encoded mode. The kernel
is fixed at four agents (``cu[5]``, ``lengths[4]``, ``pages[4, pagecols]``): the decoder packs Flare's b-request metadata
into that layout once per forward (``pack_decode_agents``). ``merge_kernel`` leaves rows at or beyond ``cu[4]`` untouched,
so the wrapper zeroes them against the DEVICE row count after every call (capacity-sized ``o_proj`` must see defined
rows). Admission (Argon, REVIEW-ARGON-S1L.md option (a)): max_length <= 32,768 at partition 512; the package must be
prepared with ``--context 32768``. Non-finite visible K/V propagate to non-finite outputs, which the samplers' existing
non-finite check turns into error code 1 on every decode and speculative step.
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
AGENTS = 4                 # the kernel's fixed agent count (cu[5], lengths[4], grid 4 agents x 4 KV heads)
MAX_ROWS = 20              # 4 agents x 5 rows
MAX_ROWS_PER_AGENT = 5
MAX_LENGTH = 32768         # valid_cfg: n <= 32768 and splits(n, 512) <= 64
PARTITION = 512
LADDER = 128               # variant flag: 128/256/512-token partitions per ROW from its own committed context ctx0 + j
HEADS, KV_HEADS, DIM = 24, 4, 256


@lru_cache(None)
def compile_library(kernel_cache=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if CUDA_HOME is None:
        raise RuntimeError("SM70 decode attention preparation requires the configured CUDA toolkit")
    source = Path(__file__).with_suffix(".cu")
    header = Path(__file__).with_suffix(".h")
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    cache = Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + ".so")
    with (cache / (digest + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            workers = len(os.sched_getaffinity(0))
            # Exactly sol-21/build.sh (= sol-13c/sol-13b flags; the qualified attention.so): the SASS must stay identical.
            subprocess.run([str(Path(CUDA_HOME) / "bin/nvcc"), "--threads", str(workers), "-std=c++17", "-O3",
                "-arch=compute_70", "-code=sm_70", "-Xcompiler=-fPIC", "--shared", "-lineinfo", "--fmad=false",
                str(source), "-o", str(output)], check=True)
    return str(output)


@lru_cache(None)
def library(kernel_cache=None):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM70 decode attention library must be loaded before capture")
    lib = C.CDLL(compile_library(kernel_cache))
    lib.sol13b_init.argtypes = []
    lib.sol13b_init.restype = C.c_int
    lib.sol13b_workspace_bytes.argtypes = [C.c_int, C.c_int]
    lib.sol13b_workspace_bytes.restype = C.c_size_t
    lib.sol13b_attention.argtypes = [C.c_void_p] * 11 + [C.c_int] * 4 + [C.c_void_p]
    lib.sol13b_attention.restype = C.c_int
    return lib


def workspace_bytes(max_length, kernel_cache=None):
    nbytes = int(library(kernel_cache).sol13b_workspace_bytes(max_length, PARTITION))
    if nbytes <= 0:
        raise ValueError(f"SM70 decode attention admits no configuration for max_length {max_length}")
    return nbytes


def _ptr(tensor):
    return C.c_void_p(tensor.data_ptr())


def admits(q, key_cache, pages):
    """Static shape admission (Qwen3.8-27B geometry, b <= 4 agents, w <= 5 rows each, served context <= 32,768)."""
    return (q.ndim == 3 and tuple(q.shape[1:]) == (HEADS, DIM) and 1 <= q.shape[0] <= MAX_ROWS
            and key_cache.ndim == 4 and tuple(key_cache.shape[1:]) == (PAGE_TOKENS, KV_HEADS, DIM)
            and pages.ndim == 2 and 1 <= pages.shape[0] <= AGENTS and q.shape[0] % pages.shape[0] == 0
            and 1 <= q.shape[0] // pages.shape[0] <= MAX_ROWS_PER_AGENT
            and 1 <= pages.shape[1] * PAGE_TOKENS <= MAX_LENGTH)


@dataclass(frozen=True)
class Query:
    rows: int
    rows_per_agent: int
    pagecols: int
    max_length: int


CONTRACT = native_contract(component_id="flare.sm70.decode_attention", query_type=Query, tactics=lambda q: ((1,),))


@dataclass(frozen=True)
class DecodeAgents:
    """The kernel's four-agent view of a decode graph's metadata, refreshed once per forward (captured copies)."""
    lengths: torch.Tensor      # int32[4]: committed context + rows for active agents, 0 for empty or padded agents
    cu: torch.Tensor           # int32[5]: monotone prefix sums; padded agents repeat cu[b]
    pages: torch.Tensor        # int32[4, pagecols]: padded rows -1
    row_index: torch.Tensor    # int64[M, 1, 1]: row ordinals for the padded-row mask

    @classmethod
    def allocate(cls, rows, pagecols, device):
        return cls(torch.zeros(AGENTS, dtype=torch.int32, device=device),
                   torch.zeros(AGENTS + 1, dtype=torch.int32, device=device),
                   torch.full((AGENTS, pagecols), -1, dtype=torch.int32, device=device),
                   torch.arange(rows, device=device).view(rows, 1, 1))

    def pack(self, lengths, starts, cu, page_table):
        """lengths/starts/cu/page_table: the graph's b-request metadata (b = page_table.shape[0] <= 4).

        Argon (REVIEW-ARGON-S1L.md, required change 2): lengths[a] = 0 for every agent with no rows, both padded
        (a >= b) and inactive (cu[a+1] == cu[a]); padded page rows stay -1.
        """
        b = page_table.shape[0]
        rows = cu[1:b + 1] - cu[:b]
        active = (lengths > 0) & (rows > 0)
        self.lengths[:b].copy_(torch.where(active, starts + lengths, 0))
        self.lengths[b:].zero_()
        self.cu[:b + 1].copy_(cu)
        self.cu[b + 1:].copy_(cu[b:b + 1].expand(AGENTS - b))
        self.pages[:b].copy_(page_table)
        self.pages[b:].fill_(-1)


@dataclass(frozen=True)
class State:
    lib: object
    pagecols: int
    max_length: int
    variant: int

    def run(self, q, key_cache, value_cache, agents, output, workspace):
        stream = C.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
        status = self.lib.sol13b_attention(_ptr(q), _ptr(key_cache), _ptr(value_cache), None, None, None,
            _ptr(agents.pages), _ptr(agents.lengths), _ptr(agents.cu), _ptr(output), _ptr(workspace),
            self.pagecols, self.max_length, PARTITION, self.variant, stream)
        if status:
            raise RuntimeError(f"SM70 decode attention launch failed: CUDA error {status}")
        # merge_kernel returns before writing rows >= cu[4]: define them as +0.0 against the DEVICE row count
        # (Argon required change 1; the active row count varies between replays of one captured graph).
        output.masked_fill_(agents.row_index >= agents.cu[AGENTS:AGENTS + 1], 0.0)


@torch.library.custom_op("flare::sm70_decode_attention", mutates_args=("output", "workspace"))
def _execute(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, lengths: torch.Tensor,
             cu: torch.Tensor, pages: torch.Tensor, row_index: torch.Tensor, output: torch.Tensor,
             workspace: torch.Tensor, handle: int) -> None:
    prepared = plan_from_handle(handle)
    require_prepared(prepared, prepared.component_id, q.device).run(
        q, key_cache, value_cache, DecodeAgents(lengths, cu, pages, row_index), output, workspace)


@_execute.register_fake
def _fake(q, key_cache, value_cache, lengths, cu, pages, row_index, output, workspace, handle):
    pass


class PreparedDecodeAttention:
    """Q/out FP32 [M <= 20, 24, 256] packed rows, BF16-bit pages [physical, 128, 4, 256], four-agent metadata."""

    def __init__(self, q, key_cache, value_cache, agents, output, *, session, name, workspace=None, rows_per_agent):
        device = q.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
            raise ValueError("SM70 decode attention requires a V100")
        pagecols = agents.pages.shape[1]
        max_length = pagecols * PAGE_TOKENS
        if (q.dtype != torch.float32 or output.dtype != torch.float32 or output.shape != q.shape
                or q.ndim != 3 or tuple(q.shape[1:]) != (HEADS, DIM) or not 1 <= q.shape[0] <= MAX_ROWS
                or key_cache.dtype != torch.uint16 or value_cache.dtype != torch.uint16
                or value_cache.shape != key_cache.shape or tuple(key_cache.shape[1:]) != (PAGE_TOKENS, KV_HEADS, DIM)
                or not 1 <= rows_per_agent <= MAX_ROWS_PER_AGENT
                or agents.lengths.shape != (AGENTS,) or agents.cu.shape != (AGENTS + 1,)
                or tuple(agents.pages.shape) != (AGENTS, pagecols) or agents.row_index.shape != (q.shape[0], 1, 1)
                or any(t.dtype != torch.int32 for t in (agents.lengths, agents.cu, agents.pages))
                or agents.row_index.dtype != torch.int64
                or not all(t.is_contiguous() for t in (q, output, key_cache, value_cache, agents.lengths, agents.cu,
                                                       agents.pages, agents.row_index))
                or any(t.device != device for t in (key_cache, value_cache, output, agents.lengths, agents.cu,
                                                    agents.pages, agents.row_index))
                or torch._C._overlaps(output, q)):
            raise ValueError("SM70 decode attention operands are outside the reviewed Sol-13c contract")
        if not 1 <= max_length <= MAX_LENGTH:
            # Argon required change 4 (option (a)): the kernel admits 32,768 tokens; prepare with --context 32768.
            raise ValueError(f"SM70 decode attention admits a served context of at most {MAX_LENGTH} tokens "
                             f"(page table {pagecols} pages = {max_length}); prepare with --context 32768")
        self.kernel_cache = getattr(session, "kernel_cache", None)
        self.tensors = (q, key_cache, value_cache, agents.lengths, agents.cu, agents.pages, agents.row_index, output)
        self.workspace, self.scratch = workspace, None
        self.nbytes = workspace_bytes(max_length, self.kernel_cache)
        variant = (5 if rows_per_agent == 5 else 4) | LADDER
        query = Query(q.shape[0], rows_per_agent, pagecols, max_length)

        def materialize(tactic):
            lib = library(self.kernel_cache)
            status = lib.sol13b_init()                     # dynamic shared-memory attribute, outside capture
            if status:
                raise RuntimeError(f"SM70 decode attention init failed: CUDA error {status}")
            return State(lib, pagecols, max_length, variant)

        self.plan = native_plan(contract=CONTRACT, query=query, device=device,
                                scratch_bytes=lambda _: self.nbytes, materialize=materialize)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(q, key_cache, value_cache, agents, output, self.views()))),))
        register_finalizer(session, self.bind, workspace=workspace)
        logging.getLogger(__name__).info(
            "SM70 decode attention %s: %d rows (%d per agent), page table %d pages, max_length %d, variant %d, scratch %d bytes",
            name, q.shape[0], rows_per_agent, pagecols, max_length, variant, self.nbytes)

    def views(self):
        if self.scratch is not None:
            return self.scratch
        storage = scratch_storage(self.workspace, nbytes=self.nbytes, device=self.tensors[0].device)
        if storage.data_ptr() % 16:
            raise ValueError("SM70 decode attention workspace requires 16-byte alignment")
        return storage[:self.nbytes]

    def bind(self):
        self.scratch = None
        self.scratch = self.views()

    def __call__(self):
        _execute(*self.tensors, self.views(), self.plan.handle)
        return self.tensors[-1]
