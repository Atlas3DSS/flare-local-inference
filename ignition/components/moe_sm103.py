"""Routed experts on SM103 (GB300/B300) through FlashInfer's SM100-family MoE kernels.

b12x executes routed experts on SM120/SM121 only. On compute capability 10.3
the same checkpoint storage runs through FlashInfer:

* NVFP4 experts with NVFP4 activations, including CSF-encoded scales, use
  trtllm-gen's routed FP4 MoE. Compressed scales are expanded losslessly once
  during loading and stay resident per layer. FC1 inputs are quantized once
  with the checkpoint's shared FC1 input scale; FC2 inputs are quantized per
  expert in trtllm-gen's FC1 epilogue with each expert's FC2 input scale.
* W4A16 NVFP4 experts use FlashInfer's CuTe-DSL NVFP4xBF16 grouped GEMMs.

Weights are re-laid out once, in place. Routing remains b12x's portable
``route_topk``. Native kernels allocate their graph-local intermediates from
PyTorch's caching allocator, which the capture's graph pool owns; caller-owned
scratch holds the quantized activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .sm103 import native_contract, native_plan

_CHUNK = 64


@cache
def _trtllm():
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    return get_trtllm_moe_sm100_module().moe_op


@cache
def _fp4_quantizer():
    from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm103_module

    return gen_fp4_quantization_sm103_module().build_and_load()


def _vector(values, count, device, name):
    values = torch.as_tensor(values).reshape(-1).to("cpu", torch.float64)
    if values.shape != (count,) or not bool((torch.isfinite(values) & (values > 0)).all()):
        raise ValueError(f"{name} must be {count} positive finite values")
    return values


class Sm103Experts:
    """Routed experts prepared for a FlashInfer-native SM103 kernel."""

    has_csf_blocks = False

    @property
    def num_experts(self) -> int:
        return self.w13.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w2.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.w2.shape[2] * 2

    @property
    def device(self) -> torch.device:
        return self.w13.device


@dataclass(frozen=True, eq=False)
class TrtllmNvfp4Experts(Sm103Experts):
    """NVFP4 x NVFP4 SwiGLU experts in trtllm-gen's shuffled MajorK layout.

    FC1 rows are ``[up; gate]`` reordered for the gated epilogue; block scales
    follow the same rows and the 128x4 interleave. Scalars follow trtllm-gen:
    ``output1_gate_scale`` dequantizes FC1, ``output1_scale`` additionally
    applies each expert's FC2 input global scale, and ``output2_scale``
    dequantizes FC2.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    input_global_scale: torch.Tensor
    output1_scale: torch.Tensor
    output1_gate_scale: torch.Tensor
    output2_scale: torch.Tensor
    gate_alpha: torch.Tensor
    kind = "trtllm_nvfp4"


@dataclass(frozen=True, eq=False)
class CuteW4A16Experts(Sm103Experts):
    """NVFP4 weights for BF16 activations in FlashInfer's CuTe-DSL grouped GEMMs.

    FC1 rows interleave 64-row ``up`` and ``gate`` groups. Scale storage is the
    128x4-interleaved buffer; the ``*_scale`` fields are its MMA-layout views.
    ``*_alpha`` are the per-expert weight global scales.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_alpha: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_alpha: torch.Tensor
    scale_storage: tuple[torch.Tensor, torch.Tensor]
    kind = "cute_w4a16"


def _check_packed(w13, s13, w2, s2):
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("packed NVFP4 experts must be rank-three uint8 tensors")
    experts, rows, half = w13.shape
    intermediate, hidden = rows // 2, half * 2
    if rows % 2 or w2.shape != (experts, hidden, intermediate // 2):
        raise ValueError("FC1 must hold [up; gate] rows matching FC2 geometry")
    for scales, shape in ((s13, (experts, rows, hidden // 16)), (s2, (experts, hidden, intermediate // 16))):
        if scales.dtype not in (torch.uint8, torch.float8_e4m3fn) or tuple(scales.shape) != shape:
            raise ValueError("NVFP4 block scales must be row-major E4M3 with one scale per 16 values")
    tensors = (w13, s13, w2, s2)
    if any(t.device != w13.device or not t.is_contiguous() for t in tensors):
        raise ValueError("expert tensors must be contiguous on one CUDA device")
    return experts, hidden, intermediate


def _permute_rows(tensor, permutation):
    """Reorder dimension one of every expert in bounded chunks, in place."""
    for start in range(0, tensor.shape[0], _CHUNK):
        part = tensor[start:start + _CHUNK]
        part.copy_(part.index_select(1, permutation))


def _interleaved_scales(scales, permutation):
    """Row-permuted, 128x4-interleaved E4M3 scales with the source's logical shape."""
    from flashinfer.fp4_quantization import block_scale_interleave

    experts, rows, columns = scales.shape
    output = torch.empty((experts, rows, columns), dtype=torch.uint8, device=scales.device)
    flat = output.view(experts, -1)
    for start in range(0, experts, _CHUNK):
        part = scales[start:start + _CHUNK].view(torch.uint8)
        if permutation is not None:
            part = part.index_select(1, permutation)
        flat[start:start + part.shape[0]].copy_(
            block_scale_interleave(part.contiguous()).view(part.shape[0], -1))
    return output


def prepare_trtllm_nvfp4_experts(w13, s13, w2, s2, scale_vectors) -> TrtllmNvfp4Experts:
    """Re-lay out packed NVFP4 experts for trtllm-gen; ``w13`` and ``w2`` change in place.

    ``s13``/``s2`` are expanded row-major E4M3 scales. ``scale_vectors`` holds
    the FC1 and FC2 weight global scales followed by the FC1 and FC2
    activation global scales (reciprocal checkpoint input scales). FC1 inputs
    are quantized once, so every expert must share one FC1 input scale.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    experts, hidden, intermediate = _check_packed(w13, s13, w2, s2)
    if (2 * intermediate) % 128 or hidden % 128 or (hidden // 16) % 4 or (intermediate // 16) % 4:
        raise ValueError("trtllm-gen NVFP4 experts require 128-row tiles and 4-aligned scale columns")
    if len(scale_vectors) != 4:
        raise ValueError("NVFP4 A4 experts require weight and activation global scales")
    w13_global, w2_global, a1, a2 = (
        _vector(v, experts, w13.device, name) for v, name in zip(scale_vectors, (
            "FC1 weight global scales", "FC2 weight global scales",
            "FC1 activation global scales", "FC2 activation global scales"), strict=True))
    if not bool((a1 == a1[0]).all()):
        raise ValueError("SM103 NVFP4 experts require one FC1 input scale shared by all experts")
    cache = {}
    w13_rows = _maybe_get_cached_w3_w1_permute_indices(cache, w13[0], 128, is_gated_act_gemm=True)
    s13_rows = _maybe_get_cached_w3_w1_permute_indices(
        cache, s13[0].view(torch.uint8), 128, num_elts_per_sf=16, is_gated_act_gemm=True)
    w2_rows = get_w2_permute_indices_with_cache(cache, w2[0], 128, is_gated_act_gemm=True)
    s2_rows = get_w2_permute_indices_with_cache(
        cache, s2[0].view(torch.uint8), 128, num_elts_per_sf=16, is_gated_act_gemm=True)
    _permute_rows(w13, w13_rows.to(w13.device))
    _permute_rows(w2, w2_rows.to(w2.device))
    scales13 = _interleaved_scales(s13, s13_rows.to(w13.device)).view(torch.float8_e4m3fn)
    scales2 = _interleaved_scales(s2, s2_rows.to(w13.device)).view(torch.float8_e4m3fn)
    gate = w13_global / a1
    device = w13.device
    as_device = partial(torch.as_tensor, dtype=torch.float32, device=device)
    return TrtllmNvfp4Experts(
        w13=w13, w13_scale=scales13, w2=w2, w2_scale=scales2,
        input_global_scale=as_device(a1[:1]),
        output1_scale=as_device(a2 * gate),
        output1_gate_scale=as_device(gate),
        output2_scale=as_device(w2_global / a2),
        gate_alpha=torch.ones(experts, dtype=torch.float32, device=device),
    )


def prepare_cute_w4a16_experts(w13, s13, w2, s2, w13_global, w2_global) -> CuteW4A16Experts:
    """Re-lay out packed NVFP4 experts for BF16-activation CuTe-DSL GEMMs, in place."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    experts, hidden, intermediate = _check_packed(w13, s13, w2, s2)
    if intermediate % 64 or hidden % 16:
        raise ValueError("CuTe-DSL W4A16 experts require 64-row gate/up groups")
    rows = torch.arange(2 * intermediate, device=w13.device).view(2, intermediate // 64, 64)
    rows = rows.transpose(0, 1).reshape(-1)
    _permute_rows(w13, rows)
    storage13 = _interleaved_scales(s13, rows)
    storage2 = _interleaved_scales(s2, None)
    device = w13.device
    alphas = tuple(torch.as_tensor(_vector(v, experts, device, name), dtype=torch.float32, device=device)
                   for v, name in ((w13_global, "FC1 weight global scales"),
                                   (w2_global, "FC2 weight global scales")))
    return CuteW4A16Experts(
        w13=w13, w13_scale=convert_sf_to_mma_layout(
            storage13.view(-1), m=2 * intermediate, k=hidden, num_groups=experts, sf_vec_size=16),
        w13_alpha=alphas[0], w2=w2,
        w2_scale=convert_sf_to_mma_layout(
            storage2.view(-1), m=hidden, k=intermediate, num_groups=experts, sf_vec_size=16),
        w2_alpha=alphas[1], scale_storage=(storage13, storage2))


@dataclass(frozen=True)
class RoutedExpertsQuery:
    """Exact routed-expert invocation; tactics depend only on these values."""

    kind: str
    num_tokens: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int


@cache
def _trtllm_tactics(query: RoutedExpertsQuery):
    from flashinfer.tllm_enums import ActivationType, DtypeTrtllmGen, Fp8QuantizationType, WeightLayout

    configs = _trtllm().trtllm_get_valid_moe_configs(
        int(DtypeTrtllmGen.E2m1), int(DtypeTrtllmGen.E2m1), int(Fp8QuantizationType.NoneFp8),
        query.top_k, query.hidden_size, query.hidden_size, query.intermediate_size,
        query.num_experts, int(ActivationType.Swiglu), True, int(WeightLayout.MajorK), False,
        query.num_tokens, False)
    return tuple((int(tile), int(config)) for tile, config in configs)


@cache
def _w4a16_tactics(query: RoutedExpertsQuery):
    import cutlass
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel import Sm100W4A16GroupedGemmKernel
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens
    from flashinfer.fused_moe.cute_dsl.tuner import W4A16_MOE_TACTICS

    def implementable(m, k, slots, tactic):
        tiler, cluster, _ = tactic
        return Sm100W4A16GroupedGemmKernel.can_implement(
            mnkl=(m, slots, k, query.num_experts), a_dtype=cutlass.Float4E2M1FN,
            b_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16, a_major="k", b_major="k",
            c_major="m", mma_tiler=tiler, cluster_shape_mn=cluster, use_2cta_instrs=tiler[0] == 256)

    hidden, intermediate = query.hidden_size, query.intermediate_size
    tactics = []
    for fc1, fc2 in W4A16_MOE_TACTICS:
        slots = get_max_num_permuted_tokens(query.num_tokens, query.top_k, query.num_experts, fc1[0][1])
        if implementable(2 * intermediate, hidden, slots, fc1) and implementable(hidden, intermediate, slots, fc2):
            tactics.append((fc1, fc2))
    return tuple(tactics)


TRTLLM_CONTRACT = native_contract(component_id="flare.moe.sm103_trtllm_nvfp4",
                                  query_type=RoutedExpertsQuery, tactics=_trtllm_tactics)
# Bump a contract's semantic version whenever its tactic space or launch semantics change,
# so cached selections from earlier spaces are never looked up.
W4A16_CONTRACT = native_contract(component_id="flare.moe.sm103_cute_w4a16",
                                 query_type=RoutedExpertsQuery, tactics=_w4a16_tactics,
                                 semantic_version=2)


def _aligned(nbytes):
    return (nbytes + 255) // 256 * 256


def _scratch_bytes(query):
    """NVFP4 activations and their row-major scales, padded to 16 rows as FlashInfer writes them."""
    if query.kind != "trtllm_nvfp4":
        return 0
    tokens, hidden = query.num_tokens, query.hidden_size
    return _aligned(tokens * hidden // 2) + _aligned((tokens + 15) // 16 * 16 * hidden // 16)


_REDUCE_BLOCK = 512


@triton.jit
def _reduce_routes(Rows, Permuted, Weights, Shared, Out, HIDDEN: tl.constexpr, TOP_K: tl.constexpr,
                   BLOCK: tl.constexpr):
    """Out = BF16(BF16(sum_k weight * FC2 row) + shared), routes summed in trtllm-gen's order."""
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for route in tl.static_range(TOP_K):
        row = tl.load(Permuted + token * TOP_K + route).to(tl.int64)
        weight = tl.load(Weights + token * TOP_K + route)
        total += weight * tl.load(Rows + row * HIDDEN + columns).to(tl.float32)
    routed = total.to(tl.bfloat16).to(tl.float32)
    shared = tl.load(Shared + token * HIDDEN + columns).to(tl.float32)
    tl.store(Out + token * HIDDEN + columns, (routed + shared).to(tl.bfloat16))


# b12x evicts compile-planning placeholders only from the Triton functions it tracks, which by
# default are its own; tracking this one lets in-process compilation (compile_workers=0) lower it.
_NATIVE_JITS.add(_reduce_routes)


@program_cache(scope="preparation")
def compile_route_reduction(tokens, hidden, top_k, ordinal):
    """Warm the routed-row reduction that folds trtllm-gen's finalize into the shared-expert sum."""
    with torch.cuda.device(ordinal):
        return _reduce_routes.warmup(torch.bfloat16, torch.int32, torch.float32, torch.bfloat16,
                                     torch.bfloat16, HIDDEN=hidden, TOP_K=top_k, BLOCK=_REDUCE_BLOCK,
                                     grid=(tokens, hidden // _REDUCE_BLOCK), num_warps=4)


def _route_reduction_job(query, ordinal):
    return CompileJob.create("ignition.components.moe_sm103:compile_route_reduction",
                             query.num_tokens, query.hidden_size, query.top_k, ordinal)


class _TrtllmNvfp4State:
    """One trtllm-gen tactic over NVFP4 activations quantized into caller scratch.

    ``run`` finalizes inside trtllm-gen. ``stage_rows`` leaves FC2's
    permuted rows unreduced, in trtllm-gen's graph-local pool, until the next
    ``reduce`` folds the routed sum into the shared expert's output with one
    kernel, in trtllm-gen's summation order. Flare's compiled-forward storage
    calls return no values, so the rows pass through the state; both custom
    ops mutate the routed scratch, which orders them.
    """

    implementation = "trtllm_nvfp4"

    def __init__(self, experts: TrtllmNvfp4Experts, query: RoutedExpertsQuery, tactic):
        from flashinfer.tllm_enums import ActivationType, RoutingInputMode, RoutingMethodType

        self.experts, self.query, self.tactic = experts, query, [int(tactic[0]), int(tactic[1])]
        self.quantize = _fp4_quantizer().fp4_quantize
        self.moe = _trtllm().trtllm_fp4_block_scale_moe
        self.enums = (int(RoutingInputMode.UnpackedPrecomputed), int(RoutingMethodType.Renormalize),
                      int(ActivationType.Swiglu))
        ordinal = experts.device.index if experts.device.index is not None else torch.cuda.current_device()
        # Load the compiled reduction while preparing; serving freezes kernel resolution.
        self.reduction = load_programs(compile_route_reduction(query.num_tokens, query.hidden_size,
                                                               query.top_k, ordinal))
        # trtllm-gen takes a zero-width output when it leaves FC2 rows unfinalized.
        self.unfinalized = torch.empty((query.num_tokens, 0), dtype=torch.bfloat16, device=experts.device)
        self.staged = None

    def scratch_bytes(self):
        return _scratch_bytes(self.query)

    def _launch(self, a, topk_ids, topk_weights, output, scratch, finalize):
        e, q = self.experts, self.query
        tokens, hidden = a.shape
        values = scratch[:tokens * hidden // 2].view(tokens, hidden // 2)
        offset = _aligned(tokens * hidden // 2)
        scales = scratch[offset:offset + (tokens + 15) // 16 * 16 * hidden // 16]
        self.quantize(a, e.input_global_scale, values, scales, 16, False, False, False, False, True, 0)
        scales = scales[:tokens * hidden // 16].view(torch.float8_e4m3fn).view(tokens, hidden // 16)
        routing, method, activation = self.enums
        return self.moe(routing, None, topk_ids, topk_weights, None, values, scales,
                        e.w13, e.w13_scale, None, None, e.gate_alpha, None, None,
                        e.w2, e.w2_scale, None, e.output1_scale, e.output1_gate_scale, e.output2_scale,
                        None, q.num_experts, q.top_k, 0, None, None, q.intermediate_size, 0,
                        q.num_experts, None, method, finalize, True, activation, output, self.tactic,
                        True, None, [], [], False, None, None)

    def run(self, a, topk_ids, topk_weights, output, scratch):
        self._launch(a, topk_ids, topk_weights, output, scratch, True)

    def stage_rows(self, a, topk_ids, topk_weights, scratch):
        """Run quantization, routing, FC1 and FC2, keeping the unreduced FC2 rows for ``reduce``."""
        if self.staged is not None:
            raise RuntimeError("SM103 routed rows were staged twice without a reduction")
        result = self._launch(a, topk_ids, topk_weights, self.unfinalized, scratch, False)
        rows, permuted = (t if isinstance(t, torch.Tensor) else torch.from_dlpack(t)
                          for t in (result[0], result[2]))
        if (rows.ndim != 2 or rows.shape[1] != self.query.hidden_size or not rows.is_contiguous()
                or permuted.numel() != topk_ids.numel()):
            raise RuntimeError("trtllm-gen FC2 rows differ from their planned geometry")
        self.staged = rows, permuted.view(-1)

    def reduce(self, topk_weights, shared, output):
        if self.staged is None:
            raise RuntimeError("SM103 routed reduction requires staged FC2 rows")
        (rows, permuted), self.staged = self.staged, None
        tokens, hidden = output.shape
        self.reduction[(tokens, hidden // _REDUCE_BLOCK, 1)](
            rows, permuted, topk_weights, shared, output, hidden, self.query.top_k, _REDUCE_BLOCK)


class _CuteW4A16State:
    """One CuTe-DSL W4A16 tactic over BF16 activations.

    FC2 writes permuted rows that ``moe_unpermute`` reduces deterministically;
    the fused finalize's BF16 atomic reduction is not used.
    """

    implementation = "cute_w4a16"

    def __init__(self, experts: CuteW4A16Experts, query: RoutedExpertsQuery, tactic):
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import launch_w4a16_moe
        from flashinfer.tllm_enums import ActivationType

        self.experts, self.query, self.tactic = experts, query, tuple(tactic)
        self.launch = partial(launch_w4a16_moe, activation_type=ActivationType.Swiglu)

    def scratch_bytes(self):
        return _scratch_bytes(self.query)

    def run(self, a, topk_ids, topk_weights, output, scratch):
        e, q = self.experts, self.query
        self.launch(a, topk_ids, topk_weights, e.w13, e.w13_scale, e.w13_alpha,
                    e.w2, e.w2_scale, e.w2_alpha, q.num_experts, q.num_experts, 0, output,
                    use_fused_finalize=False, enable_pdl=True, tactic=self.tactic)


def _state_type(experts):
    if isinstance(experts, TrtllmNvfp4Experts):
        return TRTLLM_CONTRACT, _TrtllmNvfp4State
    if isinstance(experts, CuteW4A16Experts):
        return W4A16_CONTRACT, _CuteW4A16State
    raise TypeError("SM103 routed experts require FlashInfer-prepared expert storage")


def plan_routed_experts(experts: Sm103Experts, *, tokens: int, top_k: int):
    """Declare one exact-shape native routed-expert operator for ``experts``."""
    contract, state_type = _state_type(experts)
    query = RoutedExpertsQuery(experts.kind, int(tokens), experts.num_experts, experts.hidden_size,
                               experts.intermediate_size, int(top_k))
    jobs = ((lambda tactic, detected: (_route_reduction_job(query, detected.ordinal),))
            if state_type is _TrtllmNvfp4State else (lambda tactic, detected: ()))
    return native_plan(contract=contract, query=query, device=experts.device,
                       scratch_bytes=lambda tactic: _scratch_bytes(query),
                       materialize=lambda tactic: state_type(experts, query, tactic), compile_jobs=jobs)


@torch.library.custom_op("flare::sm103_routed_experts", mutates_args=("output", "scratch"))
def _sm103_routed_experts(a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                          output: torch.Tensor, scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, a.device).run(a, topk_ids, topk_weights, output, scratch)


@_sm103_routed_experts.register_fake
def _fake_sm103_routed_experts(a, topk_ids, topk_weights, output, scratch, plan_handle) -> None:
    pass


@torch.library.custom_op("flare::sm103_stage_routed_rows", mutates_args=("scratch",))
def _sm103_stage_routed_rows(a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                             scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, a.device).stage_rows(a, topk_ids, topk_weights, scratch)


@_sm103_stage_routed_rows.register_fake
def _fake_sm103_stage_routed_rows(a, topk_ids, topk_weights, scratch, plan_handle) -> None:
    pass


@torch.library.custom_op("flare::sm103_reduce_routes", mutates_args=("output", "scratch"))
def _sm103_reduce_routes(topk_weights: torch.Tensor, shared: torch.Tensor, output: torch.Tensor,
                         scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, output.device).reduce(topk_weights, shared, output)


@_sm103_reduce_routes.register_fake
def _fake_sm103_reduce_routes(topk_weights, shared, output, scratch, plan_handle) -> None:
    pass


@dataclass(frozen=True, eq=False)
class Sm103Binding:
    """Exact tensors of one prepared native routed-expert launch.

    Eager calls launch the bound state directly. Traced calls record the
    compiler-visible custom op, which resolves the prepared plan by handle.
    """

    plan: object
    state: object
    a: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    scratch: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm103_routed_experts(self.a, self.topk_ids, self.topk_weights,
                                                 self.output, self.scratch, self.plan._handle)
        else:
            self.state.run(self.a, self.topk_ids, self.topk_weights, self.output, self.scratch)
        return self.output

    def stage_rows(self):
        """Launch the routed experts without their reduction; ``reduce`` must follow."""
        if _tracing():
            torch.ops.flare.sm103_stage_routed_rows(self.a, self.topk_ids, self.topk_weights,
                                                    self.scratch, self.plan._handle)
        else:
            self.state.stage_rows(self.a, self.topk_ids, self.topk_weights, self.scratch)
        return self.scratch

    def reduce(self, shared, output):
        """output = routed sum + shared, rounding the routed sum to BF16 first as finalize does."""
        if shared.shape != output.shape or shared.dtype != torch.bfloat16 or not shared.is_contiguous():
            raise ValueError("the shared-expert output must be contiguous BF16 matching the routed output")
        if _tracing():
            torch.ops.flare.sm103_reduce_routes(self.topk_weights, shared, output, self.scratch,
                                                self.plan._handle)
        else:
            self.state.reduce(self.topk_weights, shared, output)
        return output


def _tracing():
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def _bind(plan, state, a, topk_ids, topk_weights, output, scratch):
    tokens, hidden = a.shape
    query = plan.query
    if (a.dtype != torch.bfloat16 or (tokens, hidden) != (query.num_tokens, query.hidden_size)
            or not a.is_contiguous()):
        raise ValueError("SM103 routed experts require the prepared contiguous BF16 source")
    for tensor, dtype in ((topk_ids, torch.int32), (topk_weights, torch.float32)):
        if tensor.dtype != dtype or tensor.shape != (tokens, query.top_k) or not tensor.is_contiguous():
            raise ValueError("SM103 routes require contiguous int32 IDs and FP32 weights per token")
    if output.dtype != torch.bfloat16 or output.shape != a.shape or not output.is_contiguous():
        raise ValueError("SM103 routed output must be contiguous BF16 matching the source")
    if scratch.dtype != torch.uint8 or scratch.ndim != 1 or scratch.numel() < state.scratch_bytes():
        raise ValueError("SM103 routed-expert scratch is smaller than its plan")
    if any(t.device != a.device for t in (topk_ids, topk_weights, output, scratch)):
        raise ValueError("SM103 routed-expert tensors must share one device")
    return Sm103Binding(plan, state, a, topk_ids, topk_weights, output, scratch, state.implementation)


def _benchmark_call(state, *, plan, source, experts, top_k, session, name, workspace,
                    allocate_router, router_weight, router_plan, correction_bias=None):
    from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs

    operands = benchmark_resource(session, ("moe_inputs", name),
        lambda: ExpertBenchmarkInputs(source, experts=experts.num_experts, top_k=top_k,
            allocate_router=allocate_router, router_weight=router_weight, router_plan=router_plan,
            correction_bias=correction_bias))
    scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
    binding = _bind(plan, state, operands.source, operands.router.ids, operands.router.weights,
                    operands.output, scratch)
    return PreparedCall(run=binding.run, output=operands.output, produce=operands.producers[0],
                        benchmark_producers=operands.producers,
                        owners=(operands, scratch, binding, experts))


class Sm103RoutedExperts:
    """Prepared native expert execution with fixed source and route storage.

    Mirrors ``PreparedRoutedExperts``: one exact-shape plan declared to the
    preparation session, tuned over the native kernel's tactics, then bound
    to caller-owned scratch after tuning.
    """

    def __init__(self, source, experts: Sm103Experts, ids, weights, output, *, session, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None, router_weight=None,
                 router_plan=None, correction_bias=None, a4_prefill=False):
        from ignition.components.moe import RouterScratch

        if a4_prefill:
            raise ValueError("SM103 routed experts do not provide A4 prefill for A16 experts")
        if source.ndim != 2 or source.shape[1] != experts.hidden_size or ids.ndim != 2:
            raise ValueError("SM103 routed experts require [tokens, hidden] source and [tokens, top_k] routes")
        self.source, self.output, self.scratch = source, output, None
        self.plan = plan_routed_experts(experts, tokens=source.shape[0], top_k=ids.shape[1])
        plan = self.plan

        def prepare_call(state):
            scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
            self.scratch = scratch
            binding = _bind(plan, state, source, ids, weights, output, scratch)
            return PreparedCall(run=binding.run, owners=prime_owners(workspace, scratch, binding))

        declare(session, (plan.request(name=name, prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_call, plan=plan, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate, router_weight=router_weight,
                router_plan=router_plan, correction_bias=correction_bias)),))

        def finalize():
            self.scratch = planned_scratch(plan.scratch_specs(), workspace)
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind(plan, state, source, ids, weights, output, self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    @property
    def reduces_into_shared(self) -> bool:
        """Whether ``stage_rows``/``reduce`` can replace the finalized output plus a shared add."""
        return self.plan.contract is TRTLLM_CONTRACT

    def __call__(self) -> torch.Tensor:
        return self.binding.run()

    def stage_rows(self) -> torch.Tensor:
        """Run the routed experts up to unreduced FC2 rows; returns the scratch that orders ``reduce``."""
        return self.binding.stage_rows()

    def reduce(self, shared: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return self.binding.reduce(shared, output)
