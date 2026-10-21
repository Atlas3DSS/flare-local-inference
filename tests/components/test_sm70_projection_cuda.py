"""V100 packed arithmetic, graph replay, scratch rebinding and compiled dispatch."""

from dataclasses import replace
import os

import pytest
import torch

from ignition.components import sm70_projection as sm70
from ignition.components.linear import UnquantizedProjection
from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from ignition.preparation.scratch import ScratchArena


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100 device")


def _require_v100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")


def _weight(kind, n, k):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight

    generator = torch.Generator().manual_seed(715)
    if kind == "nvfp4":
        stored_k = (k + 127) // 128 * 128
        codes = torch.randint(0, 16, (n, stored_k), generator=generator, dtype=torch.uint8)
        values = codes[:, ::2] | (codes[:, 1::2] << 4)
        scales = torch.exp2(torch.randint(-3, 2, (n, stored_k // 16), generator=generator).float())
        gain = torch.tensor(.03125)
        packed = pack_weight(values.cuda(), swizzle_block_scale(scales.to(torch.float8_e4m3fn).cuda()),
                             recipe="nvfp4", global_scale=gain.cuda(), global_scale_kind="multiplier")
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        reference = table[codes.long()][:, :k] * scales.repeat_interleave(16, dim=1)[:, :k]
        return replace(packed, in_features=k), reference, gain, None
    values = (torch.randint(-32, 33, (n, k), generator=generator).float() / 4).to(torch.float8_e4m3fn)
    gain, input_scale = torch.tensor(.017 * .031), torch.tensor(.031)
    packed = pack_weight(values.cuda(), gain.cuda(), recipe="tensor_fp8")
    return TensorFP8Weight(packed, input_scale.cuda()), values.float(), gain, input_scale


@pytest.mark.parametrize("kind", ["nvfp4", "tensor_fp8"])
@torch.no_grad()
def test_recorded_tuning_measures_captured_projection(kind):
    from ignition.preparation.tuning import RecordedPreparationSession

    _require_v100()
    weight, matrix, gain, input_scale = _weight(kind, 384, 160)
    source = torch.full((17, 160), .25, device="cuda")
    output = torch.empty((17, 384), device="cuda")
    with RecordedPreparationSession(device=source.device, checkpoint="sm70-packed",
            workload_schema="sm70-packed", numerical_policy="fp32") as session:
        constructor = sm70.PackedProjection if kind == "nvfp4" else sm70.TensorFP8Projection
        operation = constructor(source, weight, output, session=session, name="tuned")
        assert operation.plan.selection is None
        session.tune()
        operation(source)
        reference = source.cpu()
        if kind == "tensor_fp8":
            reference = (reference / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(output.cpu(), (reference @ matrix.T) * gain, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["nvfp4", "tensor_fp8"])
@pytest.mark.parametrize("rows", [1, 17, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_packed_projection_retains_calibration_and_replays(kind, rows, dtype):
    from b12x.preparation import PreparationSession

    _require_v100()
    n, k = 384, 144 if kind == "nvfp4" else 160
    weight, matrix, gain, input_scale = _weight(kind, n, k)
    source_storage = torch.zeros((rows, k + 16), dtype=dtype, device="cuda")
    source = source_storage[:, :k]
    seed = torch.arange(rows * k).reshape(rows, k).remainder(17).sub(8).to(dtype)
    output_storage = torch.full((rows, n + 16), 19., dtype=dtype, device="cuda")
    output = output_storage[:, :n]
    arena = ScratchArena(source.device)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        constructor = sm70.PackedProjection if kind == "nvfp4" else sm70.TensorFP8Projection
        operation = constructor(source, weight, output, session=session, name="sm70-projection", workspace=arena)
        previous = operation.workspace
        arena.reserve(arena.nbytes + 256)
        arena.finalize()
        assert operation.workspace is arena.tensor and operation.workspace is not previous
        arena.freeze()
        source.copy_(seed)
        operation(source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation(source)
        for factor in (1., -.5, 2.):
            current = seed * factor
            source.copy_(current)
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            reference_source = current.float()
            if kind == "tensor_fp8":
                reference_source = (reference_source / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            # Dyadic operand values make these small reference dots exact in FP32.
            expected = ((reference_source @ matrix.T) * gain).to(dtype)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            assert torch.all(output_storage[:, n:] == 19)


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_dense_preserves_bfloat_range_and_bias(source_dtype, output_dtype):
    from b12x.preparation import PreparationSession

    _require_v100()
    source = torch.ones((17, 32), dtype=source_dtype, device="cuda")
    weight = torch.full((48, 32), 131072., dtype=torch.bfloat16, device="cuda")
    output = torch.empty((17, 48), dtype=output_dtype, device="cuda")
    bias = torch.full((48,), 32768., dtype=output_dtype, device="cuda")
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = UnquantizedProjection(source, weight, output, bias=bias, session=session, name="dense-range")
        operation(source)
        expected = (source.float() @ weight.float().T + bias.float()).to(output_dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("kind", ["tensor_fp8", "dense"])
@torch.no_grad()
def test_fullgraph_float32_buffers_preserve_opaque_weight_arithmetic(kind):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward

    _require_v100()
    weight, matrix, gain, scale = _weight("tensor_fp8", 384, 160)
    if kind == "dense":
        weight, gain = matrix.bfloat16().cuda(), torch.tensor(1.)
    source = torch.full((17, 160), .25, dtype=torch.float32, device="cuda")
    output = torch.empty((17, 384), dtype=torch.float32, device="cuda")
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        constructor = TensorFP8Projection if kind == "tensor_fp8" else UnquantizedProjection
        operation = constructor(source, weight, output, session=session, name="compiled-weight")
        compiled = compile_forward(lambda: operation(source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for value in (.125, -1., 32.):
            source.fill_(value)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            calibrated = source.cpu()
            if kind == "tensor_fp8":
                calibrated = (calibrated / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(output.cpu(), (calibrated @ matrix.T) * gain, rtol=0, atol=0)


@torch.no_grad()
def test_nvfp4_does_not_silently_drop_explicit_activation_quantization():
    _require_v100()
    weight, _, _, _ = _weight("nvfp4", 128, 128)
    source = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="quantized activations are unsupported"):
        sm70.PackedProjection(source, weight, source.clone(), session=None, name="unsupported",
                              activation_mode="quantized")
