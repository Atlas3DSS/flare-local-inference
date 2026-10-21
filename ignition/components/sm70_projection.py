"""SM70 projections with packed checkpoint weights and bounded FP32 cuBLAS operands.

V100 cannot multiply BF16 or FP8 operands. Conversion to FP32 preserves every
BF16/FP16 value, including BF16 values outside FP16's range. NVFP4 uses the
existing weight-only arithmetic: E2M1 times its E4M3 block scale, an FP32 dot,
then the checkpoint global scale. Tensor-FP8 retains calibrated activation
rounding to E4M3 and applies the combined scale after the dot. Weights remain
packed; serialized projections reuse caller-owned expansion scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.preparation import MemoryRequirements, PersistentMemory, Plan, PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.scratch import ScratchArena, prime_owners, scratch_storage
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer

from .native_operator import native_contract, native_plan


CAPABILITY = (7, 0)
COMPONENT = "flare.sm70.projection"
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def is_sm70(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == CAPABILITY


@dataclass(frozen=True)
class ProjectionQuery:
    kind: str
    rows: int
    in_features: int
    padded_in_features: int
    out_features: int
    global_scale_kind: str = "none"
    weight_dtype: str = "uint8"


def _tiles(query):
    return tuple(dict.fromkeys(min(query.out_features, width) for width in (256, 1024)))


CONTRACT = native_contract(component_id=COMPONENT, query_type=ProjectionQuery, tactics=_tiles)


def _scratch_shapes(query, tile):
    return ((query.rows, query.in_features), (tile, query.in_features), (query.rows, tile))


def workspace_bytes(query, tile):
    return sum((rows * columns * 4 + 255) // 256 * 256
               for rows, columns in _scratch_shapes(query, tile))


def _scratch_views(query, tile, workspace):
    views, offset = [], 0
    for rows, columns in _scratch_shapes(query, tile):
        size = rows * columns * 4
        views.append(workspace[offset:offset + size].view(torch.float32).view(rows, columns))
        offset += (size + 255) // 256 * 256
    return views


def _expand_nvfp4(values, scales, output, start, stop, query):
    """Expand logical K using b12x's F8_128x4 scale storage; ignore padded K."""
    k = query.in_features
    packed = values[start:stop, :k // 2]
    code = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(stop - start, k)
    magnitude = code & 7
    decoded = torch.where(magnitude < 4, magnitude.float() * .5,
        torch.where(magnitude == 4, 2., torch.where(magnitude == 5, 3.,
                                                  torch.where(magnitude == 6, 4., 6.))))
    decoded = torch.where(code >= 8, -decoded, decoded)
    row = torch.arange(start, stop, device=values.device)[:, None]
    group = torch.arange(k // 16, device=values.device)[None, :]
    groups = query.padded_in_features // 16
    offset = ((row // 128 * ((groups + 3) // 4) + group // 4) * 512
              + row % 32 * 16 + row // 32 % 4 * 4 + group % 4)
    block_scale = scales[offset].view(torch.float8_e4m3fn).float()
    output.copy_((decoded.view(stop - start, k // 16, 16) * block_scale[:, :, None]).view_as(output))


def _project(query, tile, source, values, scales, gain, input_scale, output, workspace, bias):
    """All temporary allocations are bounded by a tile and captured before serving."""
    inputs, expanded, product = _scratch_views(query, tile, workspace)
    if query.kind == "tensor_fp8":
        inputs.copy_((source.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn))
    else:
        inputs.copy_(source)
    for start in range(0, query.out_features, tile):
        stop = min(query.out_features, start + tile)
        width = stop - start
        weight, result = expanded[:width], product[:, :width]
        if query.kind == "nvfp4":
            _expand_nvfp4(values, scales, weight, start, stop, query)
        elif query.kind == "tensor_fp8":
            weight.copy_(values[start:stop, :query.in_features].view(torch.float8_e4m3fn))
        else:
            weight.copy_(values.view(getattr(torch, query.weight_dtype))[start:stop, :query.in_features])
        torch.mm(inputs, weight.T, out=result)
        if gain is not None:
            if query.global_scale_kind == "reciprocal":
                result.div_(gain)
            else:
                result.mul_(gain)
        # The unquantized operator adds its bias before the final output cast.
        if bias is not None and query.kind == "dense":
            result.add_(bias[start:stop])
        output[:, start:stop].copy_(result)
    # QuantizedProjection's existing bias contract rounds before adding bias.
    if bias is not None and query.kind != "dense":
        output.add_(bias)
    return output


@dataclass(frozen=True)
class PreparedProjection:
    query: ProjectionQuery
    tile: int

    @property
    def required_workspace(self):
        return workspace_bytes(self.query, self.tile)

    def run(self, source, values, scales, gain, input_scale, output, workspace, bias=None):
        return _project(self.query, self.tile, source, values, scales, gain, input_scale,
                        output, workspace, bias)


@torch.library.custom_op("flare::sm70_projection_out", mutates_args=("output", "workspace"))
def _projection_out(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor | None,
                    gain: torch.Tensor | None, input_scale: torch.Tensor | None,
                    output: torch.Tensor, workspace: torch.Tensor, bias: torch.Tensor | None,
                    handle: int) -> None:
    state = require_prepared(plan_from_handle(handle), COMPONENT, source.device)
    state.run(source, values, scales, gain, input_scale, output, workspace, bias)


@_projection_out.register_fake
def _fake_projection_out(source, values, scales, gain, input_scale, output, workspace, bias, handle):
    pass


def _check_operands(source, values, output, bias, k, n):
    if not is_sm70(source.device):
        raise ValueError("SM70 projections require a V100 CUDA device")
    if (source.ndim != 2 or source.shape[1] != k or source.dtype not in _FLOAT_DTYPES
            or output.shape != (source.shape[0], n) or output.dtype not in _FLOAT_DTYPES
            or source.shape[0] <= 0 or source.stride(-1) != 1 or output.stride(-1) != 1
            or min(k, n) <= 0 or values.ndim != 2):
        raise ValueError("SM70 projections require positive matching floating matrices with contiguous columns")
    if values.device != source.device or output.device != source.device or not values.is_contiguous():
        raise ValueError("SM70 projection weights and output must share the source device")
    if bias is not None and (bias.shape != (n,) or bias.dtype != output.dtype
                             or bias.device != source.device or not bias.is_contiguous()):
        raise ValueError("SM70 projection bias must match the output width, dtype and device")
    from torch._C import _overlaps

    if any(_overlaps(output, tensor) for tensor in (source, values, bias) if tensor is not None):
        raise ValueError("SM70 projection output must not overlap an input")


class _Projection:
    def _initialize(self, query, source, values, output, *, scales=None, gain=None, input_scale=None,
                    bias=None, session, name, workspace):
        _check_operands(source, values, output, bias, query.in_features, query.out_features)
        for tensor in (scales, gain, input_scale):
            if tensor is not None and tensor.device != source.device:
                raise ValueError("SM70 projection scales must share the source device")
        for scalar in (gain, input_scale):
            if scalar is not None and (scalar.dtype != torch.float32 or scalar.numel() != 1
                    or not bool((torch.isfinite(scalar) & (scalar > 0)).all())):
                raise ValueError("SM70 projection scales must be positive finite FP32 scalars")
        for tensor in (values, scales, gain, input_scale, bias):
            if tensor is not None:
                record_numeric_tensor(tensor)
        self.output, self.values, self.scales = output, values, scales
        self.gain, self.input_scale, self.bias = gain, input_scale, bias
        # Without an external arena, keep one owned arena across preparation calls.
        arena = ScratchArena(source.device) if workspace is None else workspace
        if workspace is None:
            def memory(config, device):
                from b12x.preparation import current_plan

                needed = max(arena.nbytes, workspace_bytes(query, config.tactic))
                return MemoryRequirements((), (PersistentMemory(
                    (COMPONENT, current_plan()), needed, arena.nbytes),))

            self.plan = Plan(contract=CONTRACT, query=query,
                _compile_jobs=lambda config, device: (),
                _memory_requirements=memory,
                _materialize=lambda selection, device: PreparedProjection(query, selection.config.tactic),
                _device=source.device)
        else:
            self.plan = native_plan(contract=CONTRACT, query=query, device=source.device,
                scratch_bytes=lambda tile: workspace_bytes(query, tile),
                materialize=lambda tile: PreparedProjection(query, tile))

        def storage(nbytes=None):
            if nbytes is None:
                nbytes = require_prepared(self.plan, COMPONENT, source.device).required_workspace
            result = scratch_storage(arena, nbytes=nbytes, device=source.device)
            from torch._C import _overlaps

            if any(_overlaps(result, tensor) for tensor in (source, values, scales, gain, input_scale, output, bias)
                   if tensor is not None):
                raise ValueError("SM70 projection scratch must not overlap operands")
            return result

        def prepare(state):
            scratch = storage(state.required_workspace)
            return PreparedCall(run=lambda: state.run(source, values, scales, gain, input_scale,
                output, scratch, bias), owners=prime_owners(arena, scratch))

        acquire = input_factory(source, output, storage=getattr(session, "benchmark_storage", None))

        def benchmark(state):
            operands, scratch = acquire(), storage(state.required_workspace)
            return PreparedCall(run=lambda: state.run(operands.source, values, scales, gain, input_scale,
                operands.output, scratch, bias), produce=operands.produce, output=operands.output,
                owners=(operands, scratch, self.weight))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=benchmark),))
        register_finalizer(session, lambda: setattr(self, "workspace", storage()), workspace=workspace)
        self.arena = arena

    def __call__(self, source):
        _projection_out(source, self.values, self.scales, self.gain, self.input_scale,
                        self.output, self.workspace, self.bias, self.plan.handle)
        return self.output


class PackedProjection(_Projection):
    """Checkpoint NVFP4 with legal W4A16 arithmetic; no activation requantization."""

    def __init__(self, source, weight, output, *, session, name, activation_mode="auto",
                 bias=None, workspace=None):
        from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight, scale_storage

        if not isinstance(weight, NVFP4LinearWeight):
            raise TypeError("SM70 packed projections currently require NVFP4 weights")
        if activation_mode not in ("auto", "a16"):
            raise ValueError("SM70 NVFP4 supports weight-only a16 arithmetic; quantized activations are unsupported")
        if weight.global_scale_kind not in ("multiplier", "reciprocal"):
            raise ValueError("NVFP4 requires a multiplier or reciprocal global scale")
        k, n = weight.in_features, weight.out_features
        if (k % 16 or k > weight.padded_in_features or weight.values.dtype != torch.uint8
                or weight.values.shape != (n, weight.padded_in_features // 2)):
            raise ValueError("NVFP4 projection has invalid packed dimensions")
        self.weight, self.activation_mode = weight, "a16"
        scales = scale_storage(weight.scale_mma, n, weight.padded_in_features, 16)
        query = ProjectionQuery("nvfp4", source.shape[0], k, weight.padded_in_features, n,
                                weight.global_scale_kind)
        self._initialize(query, source, weight.values, output, scales=scales, gain=weight.global_scale,
                         bias=bias, session=session, name=name, workspace=workspace)


class TensorFP8Projection(_Projection):
    """E4M3 activation rounding and the checkpoint's combined output scale."""

    def __init__(self, source, weight, output, *, session, name, workspace=None):
        from b12x.gemm.blockscaled._linear import TensorFP8LinearWeight

        packed = weight.packed
        if (not isinstance(packed, TensorFP8LinearWeight) or packed.values.dtype != torch.float8_e4m3fn
                or packed.values.shape != (packed.out_features, packed.padded_in_features)
                or not 0 < packed.in_features <= packed.padded_in_features):
            raise ValueError("SM70 tensor-FP8 requires a packed E4M3 matrix with logical K")
        # pack_tensor_fp8_linear_weight defines these as unit scale planes. Reject
        # altered packed objects instead of discarding meaningful block scales.
        if not bool((packed.block_scale.float() == 1).all()):
            raise ValueError("tensor-FP8 block scales must retain their packed unit values")
        self.weight = weight
        query = ProjectionQuery("tensor_fp8", source.shape[0], packed.in_features,
                                packed.padded_in_features, packed.out_features, "multiplier")
        # SM70 Inductor cannot type FP8 operands even for an opaque call: its
        # functionalization pattern pass skips such nodes. The checkpoint's
        # bits cross the boundary as bytes; only the implementation views E4M3.
        self._initialize(query, source, packed.values.view(torch.uint8), output, gain=packed.output_scale,
                         input_scale=weight.input_scale, session=session, name=name, workspace=workspace)


class UnquantizedProjection(_Projection):
    """Exact FP32 expansion of BF16/FP16 storage, with FP32 GEMM accumulation."""

    def __init__(self, source, weight, output, *, session, name, bias=None, workspace=None):
        if weight.ndim != 2 or weight.dtype not in _FLOAT_DTYPES:
            raise ValueError("SM70 unquantized projections require a floating matrix")
        self.weight = weight
        n, k = weight.shape
        query = ProjectionQuery("dense", source.shape[0], k, k, n,
                                weight_dtype=str(weight.dtype).removeprefix("torch."))
        self._initialize(query, source, weight.view(torch.uint8), output, bias=bias, session=session,
                         name=name, workspace=workspace)
