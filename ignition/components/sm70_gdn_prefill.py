"""Reviewed Sol-15 GDN prefill kernels (conv1d + recurrence) for the 48 linear-attention layers on SM70.

Prefill mode only: the token-parallel conv1d (``sm70_gdn_prefill_convolution.cu``) and the warp-per-row recurrence
scan (``sm70_gdn_prefill_recurrence.cu``, variant 3) replace the two sequential Triton kernels of
``PreparedSM70GDNSequence``; the gated norm stays the same Triton ``_norm`` launch. Both kernels reproduce the
reference PTX operation by operation, so outputs, the recurrent-state pool and the conv-history pool are bit-identical
(the captured-graph gates prove it); the prefill -> decode handoff, decode and speculative verification are untouched.
The recurrence pre-pass workspace lives in the shared operator scratch arena. The kernels have no device-side admission
status: their only refusals are host argument checks, which fail preparation loudly.
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
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import scratch_storage
from ignition.preparation.triton import launch_triton, warmup_triton
from ignition.preparation.tuning import declare, register_finalizer
from .native_operator import native_contract, native_plan
from .sm70_gdn import _norm, validate_sequence

KEY_HEADS, VALUE_HEADS, DIM = 16, 48, 128     # kernel constants of the reviewed Qwen3.8-27B geometry
CHANNELS = (2 * KEY_HEADS + VALUE_HEADS) * DIM
HISTORY = 7                                   # the only instantiated conv history width (3 + draft depth 4)
MAX_TOKENS = 512                              # graph token capacities the gates covered (128 and 512)
VARIANT = 3                                   # 4 warps x 4 rows, reduce-scatter row sums (REVIEW-ARGON-RS.md)
SOURCES = ("sm70_gdn_prefill_convolution.cu", "sm70_gdn_prefill_recurrence.cu")
NORM_OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


@lru_cache(None)
def compile_library(source_name, kernel_cache=None):
    from torch.utils.cpp_extension import CUDA_HOME
    from ignition.preparation.compilation import DEFAULT_CACHE_ROOT

    if CUDA_HOME is None:
        raise RuntimeError("SM70 GDN prefill preparation requires the configured CUDA toolkit")
    source = Path(__file__).with_name(source_name)
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    cache = Path(DEFAULT_CACHE_ROOT if kernel_cache is None else kernel_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + ".so")
    with (cache / (digest + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            # Exactly Sol-15's flags (sol-15/HANDOFF.md): the bit-identity, sanitizer and speed gates ran on this build.
            # nvcc writes a temporary file that is renamed into place, so an interrupted build never leaves a partial .so.
            partial = cache / (digest + f".{os.getpid()}.partial.so")
            subprocess.run([str(Path(CUDA_HOME) / "bin/nvcc"), "-O3", "-arch=compute_70", "-code=sm_70", "--fmad=false",
                            "-Wno-deprecated-gpu-targets", "-Xcompiler", "-fPIC", "-shared", "-o", str(partial),
                            str(source)], check=True)
            os.replace(partial, output)
    return str(output)


@dataclass(frozen=True)
class Library:
    convolution: object
    recurrence: object


@lru_cache(None)
def library(kernel_cache=None):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM70 GDN prefill libraries must be loaded before capture")
    p, i = C.c_void_p, C.c_int
    conv = C.CDLL(compile_library(SOURCES[0], kernel_cache))
    conv.sol15_conv_prefill.restype = i
    conv.sol15_conv_prefill.argtypes = [p] * 8 + [i] * 7 + [p]
    rec = C.CDLL(compile_library(SOURCES[1], kernel_cache))
    rec.sol15_gdn_prefill_workspace.restype = C.c_size_t
    rec.sol15_gdn_prefill_workspace.argtypes = [i]
    rec.sol15_gdn_prefill.restype = i
    rec.sol15_gdn_prefill.argtypes = [p] * 14 + [i] * 6 + [p, i, p]
    return Library(conv, rec)


def workspace_bytes(capacity, kernel_cache=None):
    return int(library(kernel_cache).recurrence.sol15_gdn_prefill_workspace(capacity))


def _ptr(tensor):
    return C.c_void_p(tensor.data_ptr())


def _checked(status, what):
    if status:
        raise RuntimeError(f"SM70 GDN prefill {what} failed: CUDA error {status}")


def admits(module, scratch, state):
    """Static admission of the reviewed Sol-15 contract: Qwen3.8-27B GDN geometry, history 7, capacity <= 512."""
    x = scratch.projected_qkv
    return (module.num_k_heads == KEY_HEADS and module.num_v_heads == VALUE_HEADS and x.ndim == 2
            and 1 <= x.shape[0] <= MAX_TOKENS and x.shape[1] == CHANNELS
            and state.convolution.ndim == 3 and state.convolution.shape[2] == HISTORY
            and state.recurrent.ndim == 4 and state.recurrent.stride(0) % 4 == 0)


@dataclass(frozen=True)
class Query:
    operands: tuple
    value_heads: int
    epsilon: float
    activation: str
    norm_weight_bf16: bool


CONTRACT = native_contract(component_id="flare.sm70.gdn_prefill", query_type=Query, tactics=lambda q: ("sol15-v3",))


def _norm_launch(tensors, query):
    """The unchanged ``sm70_gdn._launches`` norm launch over this sequence's operand tuple."""
    (x, a, b, z, weight, a_log, bias, norm_weight, starts, slots, fresh, accepted, indices,
     initial, final, checkpoints, offsets, sequences, tokens, history, state, convolved, recurrence, gated) = tensors
    return (_norm, (recurrence, z, norm_weight, gated, tokens, z.stride(0)), (x.shape[0], query.value_heads, 1),
            {**NORM_OPTIONS, "HEADS": query.value_heads, "EPS": query.epsilon, "SILU": query.activation == "silu",
             "BF16_WEIGHT": query.norm_weight_bf16})


@program_cache(scope="preparation")
def compile_norm(payload, ordinal):
    from b12x._lib.compile_plan import compile_only_launches, launch_triton as compile_launch

    query = Query(**dict(payload))
    tensors = tuple(torch.empty_strided(shape, stride, dtype=getattr(torch, dtype), device="meta")
                    for shape, stride, dtype in query.operands)
    kernel, args, grid, options = _norm_launch(tensors, query)
    with torch.cuda.device(ordinal), compile_only_launches():
        return (compile_launch(kernel, grid, *args, **options),)


@dataclass(frozen=True)
class State:
    lib: Library
    query: Query
    programs: tuple

    def run(self, tensors, workspace):
        (x, a, b, z, weight, a_log, bias, norm_weight, starts, slots, fresh, accepted, indices,
         initial, final, checkpoints, offsets, sequences, tokens, history, state, convolved, recurrence, gated) = tensors
        capacity, requests = x.shape[0], starts.numel() - 1
        stream = C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        _checked(self.lib.convolution.sol15_conv_prefill(
            _ptr(x), _ptr(weight), _ptr(history), _ptr(starts), _ptr(slots), _ptr(fresh), _ptr(sequences), _ptr(convolved),
            x.stride(0), weight.stride(0), history.stride(0), history.stride(1), history.shape[2], capacity, requests,
            stream), "convolution")
        _checked(self.lib.recurrence.sol15_gdn_prefill(
            _ptr(convolved), _ptr(a), _ptr(b), _ptr(a_log), _ptr(bias), _ptr(state), _ptr(starts), _ptr(initial),
            _ptr(final), _ptr(checkpoints), _ptr(offsets), _ptr(fresh), _ptr(sequences), _ptr(recurrence),
            convolved.stride(0), a.stride(0), b.stride(0), state.stride(0), capacity, requests, _ptr(workspace), VARIANT,
            stream), "recurrence")
        kernel, args, grid, options = _norm_launch(tensors, self.query)
        launch_triton(kernel, self.programs[0], args, grid, **options)


@torch.library.custom_op("flare::sm70_gdn_prefill_sequence", mutates_args=("state", "outputs", "workspace"))
def _execute(inputs: list[torch.Tensor], state: list[torch.Tensor], outputs: list[torch.Tensor],
             workspace: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, inputs[0].device).run((*inputs, *state, *outputs), workspace)


@_execute.register_fake
def _fake(inputs, state, outputs, workspace, handle):
    pass


class PreparedSM70GDNPrefillSequence:
    """Prefill-mode drop-in for ``PreparedSM70GDNSequence``: same operands, same validation, Sol-15 kernels.

    ``workspace`` is the shared operator scratch arena (or a uint8 buffer / None for direct tests). With a recorded
    session the norm program is a declared compile job and the libraries load at materialization, outside capture;
    without a session everything prepares immediately for direct tests.
    """

    def __init__(self, module, scratch, state, metadata, *, norm_weight_bf16, session=None, name=None,
                 workspace=None, checkpoint_export=False):
        validated = validate_sequence(module, scratch, state, metadata, mode="prefill",
                                      norm_weight_bf16=norm_weight_bf16, checkpoint_export=checkpoint_export)
        if not admits(module, scratch, state):
            raise ValueError("SM70 GDN prefill operands are outside the reviewed Sol-15 contract "
                             "(16/48 heads, 7 history columns, capacity <= 512, float4 state rows)")
        x, convolved = validated.inputs[0], validated.outputs[0]
        if torch._C._overlaps(x, convolved) or any(t.data_ptr() % 16 for t in (x, convolved, validated.outputs[1],
                                                                                validated.state[1])):
            raise ValueError("SM70 GDN prefill requires distinct, 16-byte aligned QKV, convolved, recurrence and state buffers")
        device, capacity = x.device, x.shape[0]
        self.inputs, self.state, self.outputs = validated.inputs, validated.state, validated.outputs
        self.tensors = validated.tensors
        self.kernel_cache = getattr(session, "kernel_cache", None)
        self.workspace, self.scratch = workspace, None
        self.nbytes = workspace_bytes(capacity, self.kernel_cache)
        query = Query(tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype).removeprefix("torch.")) for t in self.tensors),
                      validated.values, validated.epsilon, module.norm.activation, norm_weight_bf16)
        self.plan = None
        if session is None:
            kernel, args, grid, options = _norm_launch(self.tensors, query)
            self.prepared = State(library(None), query, (warmup_triton(kernel, args, grid, **options),))
            self.bind()
        else:
            if not name:
                raise ValueError("prepared SM70 GDN prefill requires an operation name")
            from b12x._lib.compile_plan import attach_programs, load_programs
            from b12x._lib.compile_pool import CompileJob

            payload = CONTRACT.encode_query(query)
            kernel_cache = self.kernel_cache

            def materialize(tactic):
                programs = compile_norm(payload, device.index)
                load_programs(programs)
                return attach_programs(State(library(kernel_cache), query, programs), programs)

            self.plan = native_plan(contract=CONTRACT, query=query, device=device,
                                    scratch_bytes=lambda tactic: self.nbytes, materialize=materialize,
                                    compile_jobs=lambda tactic, detected: (CompileJob.create(
                                        "ignition.components.sm70_gdn_prefill:compile_norm", payload, detected.ordinal),))
            declare(session, (self.plan.request(name=name, prepare_call=lambda prepared: PreparedCall(
                run=lambda: prepared.run(self.tensors, self.views()), owners=(self.tensors,))),))
            register_finalizer(session, self.bind, workspace=workspace)
        logging.getLogger(__name__).info("SM70 GDN prefill %s: capacity %d tokens, %d requests, workspace %d bytes",
                                         name, capacity, self.inputs[8].numel() - 1, self.nbytes)

    def views(self):
        """The recurrence workspace carved from the operator scratch; re-read the arena address on every call before
        it is frozen (growth moves it), and keep the finalized binding afterwards."""
        if self.scratch is not None:
            return self.scratch
        storage = scratch_storage(self.workspace, nbytes=self.nbytes, device=self.inputs[0].device)
        return storage[:self.nbytes]

    def bind(self):
        """After tuning, before capture: fix the scratch view (the arena's finalizer; also the session-less path)."""
        self.scratch = None
        self.scratch = self.views()
        if self.scratch.data_ptr() % 16:
            raise ValueError("SM70 GDN prefill workspace requires 16-byte alignment")

    def __call__(self):
        if self.plan is None:
            self.prepared.run(self.tensors, self.views())
        else:
            _execute(list(self.inputs), list(self.state), list(self.outputs), self.views(), self.plan._handle)
        return self.outputs[2]

    run_sequence = __call__
