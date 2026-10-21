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
    stored_k = (k + 31) // 32 * 32
    values = (torch.randint(-32, 33, (n, stored_k), generator=generator).float() / 4).to(torch.float8_e4m3fn)
    gain, input_scale = torch.tensor(.017 * .031), torch.tensor(.031)
    packed = pack_weight(values.cuda(), gain.cuda(), recipe="tensor_fp8")
    return TensorFP8Weight(replace(packed, in_features=k), input_scale.cuda()), values[:, :k].float(), gain, input_scale


@pytest.mark.parametrize("kind,n,k", [("nvfp4", 384, 160), ("tensor_fp8", 384, 160),
                                    ("nvfp4", 248320, 16)])
@torch.no_grad()
def test_recorded_tuning_measures_captured_projection(kind, n, k):
    from ignition.preparation.tuning import RecordedPreparationSession

    _require_v100()
    weight, matrix, gain, input_scale = _weight(kind, n, k)
    source = torch.full((17, k), .25, device="cuda")
    output = torch.empty((17, n), device="cuda")
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
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 17, 128, 129])
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


@pytest.mark.parametrize("kind", ["nvfp4", "tensor_fp8", "dense"])
@pytest.mark.parametrize("rows", [1, 4, 17])
@torch.no_grad()
def test_fullgraph_float32_buffers_preserve_opaque_weight_arithmetic(kind, rows):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward

    _require_v100()
    weight, matrix, gain, scale = _weight("nvfp4" if kind == "nvfp4" else "tensor_fp8", 384, 160)
    if kind == "dense":
        weight, gain = matrix.bfloat16().cuda(), torch.tensor(1.)
    source = torch.full((rows, 160), .25, dtype=torch.float32, device="cuda")
    output = torch.empty((rows, 384), dtype=torch.float32, device="cuda")
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        constructor = {"nvfp4": sm70.PackedProjection, "tensor_fp8": TensorFP8Projection,
                       "dense": UnquantizedProjection}[kind]
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


@pytest.mark.parametrize("k", [144, 528])
@torch.no_grad()
def test_fused_nvfp4_expansion_exact_bits_and_swizzled_padding(k):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.preparation import PreparationSession
    from b12x.preparation.types import require_prepared

    _require_v100()
    n = 384
    weight, _, _, _ = _weight("nvfp4", n, k)
    padded_k = weight.padded_in_features
    # Every finite E4M3 code, including both signs of zero and subnormals.
    finite = torch.tensor([i for i in range(256) if i & 127 != 127], dtype=torch.uint8)
    indices = torch.arange(n * padded_k // 16).remainder(finite.numel())
    scale_values = finite[indices].reshape(n, padded_k // 16).view(torch.float8_e4m3fn)
    scales = swizzle_block_scale(scale_values.cuda()).view(torch.uint8).reshape(-1)
    packed = weight.values.cpu()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, padded_k)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = table[codes.long()][:, :k] * scale_values.float().repeat_interleave(16, dim=1)[:, :k]
    source = torch.ones((1, k), device="cuda")
    output = torch.empty((1, n), device="cuda")
    expanded = torch.empty((n, k), device="cuda")
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = sm70.PackedProjection(source, weight, output, session=session, name="expansion-bits")
        state = require_prepared(operation.plan, sm70.COMPONENT, source.device)

        def expand():
            for start in range(0, n, state.tile):
                stop = min(n, start + state.tile)
                sm70._expand_nvfp4(state.expansion, weight.values, scales, expanded[start:stop],
                                  start, stop, state.query)

        expand()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            expand()
        expanded.fill_(float("nan"))
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(expanded.cpu().view(torch.int32), expected.contiguous().view(torch.int32),
                                   rtol=0, atol=0)


@torch.no_grad()
def test_nvfp4_does_not_silently_drop_explicit_activation_quantization():
    _require_v100()
    weight, _, _, _ = _weight("nvfp4", 128, 128)
    source = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="quantized activations are unsupported"):
        sm70.PackedProjection(source, weight, source.clone(), session=None, name="unsupported",
                              activation_mode="quantized")


@pytest.mark.parametrize("kind", ["nvfp4", "tensor_fp8"])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 17, 128, 129])
@torch.no_grad()
def test_alternating_users_poison_padding_and_rebind(kind, rows):
    from b12x.preparation import PreparationSession

    _require_v100()
    n, k = 384, 528
    weight, matrix, gain, scale = _weight(kind, n, k)
    packed = weight if kind == "nvfp4" else weight.packed
    # All padded bytes are poison; every logical input and every prepared row,
    # including zero rows, must still be computed. No active-row metadata exists.
    packed.values.view(torch.uint8)[:, k // 2 if kind == "nvfp4" else k:].fill_(255)
    source = torch.empty((rows, k + 32), device="cuda")[:, :k]
    destination = torch.full((rows, n + 32), 19., device="cuda")
    output = destination[:, :n]
    arena = ScratchArena(source.device)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        constructor = sm70.PackedProjection if kind == "nvfp4" else sm70.TensorFP8Projection
        first = constructor(source, weight, output, session=session, name="first", workspace=arena)
        second = constructor(source, weight, output, session=session, name="second", workspace=arena)
        old = first.workspace
        arena.reserve(arena.nbytes + 256)
        arena.finalize()
        assert first.workspace is second.workspace is arena.tensor
        assert first.workspace is not old
        arena.freeze()
        source.fill_(.25)
        first(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            first(source)
            second(source)
        for value in (0., -.5, .25):
            source.fill_(value)
            # This poisons both users' shared scratch before the first producer.
            arena.tensor.fill_(255)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            calibrated = source.cpu()
            if kind == "tensor_fp8":
                calibrated = (calibrated / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(output.cpu(), (calibrated @ matrix.T) * gain, rtol=0, atol=0)
            assert torch.all(destination[:, n:] == 19)


@pytest.mark.parametrize("kind,k", [("nvfp4", 5120), ("tensor_fp8", 6144), ("nvfp4", 17408)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@torch.no_grad()
def test_gemv_fp64_cancellation_bound(kind, k, rows):
    from b12x.preparation import PreparationSession
    from b12x.preparation.types import require_prepared
    import math

    _require_v100()
    weight, matrix, gain, scale = _weight(kind, 384, k)
    generator = torch.Generator().manual_seed(724)
    source_cpu = torch.randn((rows, k), generator=generator)
    source = torch.empty((rows, k + 16), device="cuda")[:, :k]
    source.copy_(source_cpu)
    output = torch.empty((rows, 400), device="cuda")[:, :384]
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        constructor = sm70.PackedProjection if kind == "nvfp4" else sm70.TensorFP8Projection
        operation = constructor(source, weight, output, session=session, name="fp64-bound")
        operation(source)
        state = require_prepared(operation.plan, sm70.COMPONENT, source.device)
        assert state.tactic[0] == "gemv"
        if kind == "tensor_fp8":
            source_cpu = (source_cpu / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        reference = (source_cpu.double() @ matrix.double().T) * gain.double()
        magnitude = (source_cpu.double().abs() @ matrix.double().abs().T) * gain.double().abs()
        bk = state.tactic[2]
        depth = 2 * math.ceil(k / bk) + math.ceil(math.log2(bk // 2)) + 2
        unit = 2 ** -24
        gamma = depth * unit / (1 - depth * unit)
        error = (output.cpu().double() - reference).abs()
        assert torch.isfinite(output).all()
        assert bool((error <= (gamma + 2 * unit) * magnitude).all()), float((error / magnitude).max())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gain_kind", ["multiplier", "reciprocal"])
@torch.no_grad()
def test_gemv_quantized_cast_bias_order_and_bf16_range(dtype, gain_kind):
    from b12x.preparation import PreparationSession

    _require_v100()
    weight, matrix, gain, _ = _weight("nvfp4", 384, 144)
    weight = replace(weight, global_scale_kind=gain_kind)
    seed = torch.ones((3, 144), dtype=dtype)
    if dtype == torch.bfloat16:
        seed[0].fill_(131072.)
        seed[1].fill_(2. ** -100)
    source = seed.cuda()
    output = torch.empty((3, 384), dtype=dtype, device="cuda")
    bias = torch.linspace(-1., 1., 384).to(dtype).cuda()
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = sm70.PackedProjection(source, weight, output, bias=bias, session=session, name="bias-order")
        operation(source)
        reference = seed.float() @ matrix.T
        reference = reference / gain if gain_kind == "reciprocal" else reference * gain
        expected = (reference.to(dtype).float() + bias.cpu().float()).to(dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@torch.no_grad()
def test_gemv_nonfinite_classification_and_signed_zero():
    from b12x.preparation import PreparationSession

    _require_v100()
    weight, matrix, gain, _ = _weight("nvfp4", 384, 144)
    source = torch.zeros((4, 144), device="cuda")
    output = torch.empty((4, 384), dtype=torch.bfloat16, device="cuda")
    source[1].fill_(-0.)
    source[2].fill_(float("inf"))
    source[3].fill_(float("nan"))
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = sm70.PackedProjection(source, weight, output, session=session, name="classification")
        operation(source)
        expected = ((source.cpu() @ matrix.T) * gain).bfloat16()
        actual = output.cpu()
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isinf(actual), torch.isinf(expected))
        finite = torch.isfinite(expected)
        assert torch.equal(actual[finite].view(torch.int16), expected[finite].view(torch.int16))


@torch.no_grad()
def test_full_vocabulary_real_k_streamed_reference():
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight
    from b12x.preparation import PreparationSession

    _require_v100()
    n, k = 248320, 5120
    generator = torch.Generator().manual_seed(726)
    # No full decoded vocabulary matrix is materialized on either host or GPU.
    values = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.uint8).cuda()
    scales = swizzle_block_scale(torch.ones((n, k // 16), device="cuda").to(torch.float8_e4m3fn))
    gain = torch.tensor(.03125, device="cuda")
    weight = pack_weight(values, scales, recipe="nvfp4", global_scale=gain,
                         global_scale_kind="multiplier")
    source = torch.full((1, k), .25, device="cuda")
    output = torch.empty((1, n), device="cuda")
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = sm70.PackedProjection(source, weight, output, session=session, name="vocabulary")
        operation(source)
        for start in range(0, n, 1024):
            packed = values[start:start+1024].cpu()
            codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], k)
            matrix = table[codes.long()].double()
            expected = matrix.sum(-1) * (.25 * .03125)
            torch.testing.assert_close(output[0, start:start+1024].cpu().double(), expected, rtol=0, atol=0)


@torch.no_grad()
def test_gemv_bf16_nearest_even_bits_and_nan_handling():
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight
    from b12x.preparation import PreparationSession

    _require_v100()
    n, k = 128, 144
    values = torch.zeros((n, 128), dtype=torch.uint8, device="cuda")
    values[:, 0] = 2  # E2M1 +1 at logical k=0; all other weights are zero.
    scales = swizzle_block_scale(torch.ones((n, 16), device="cuda").to(torch.float8_e4m3fn))
    weight = replace(pack_weight(values, scales, recipe="nvfp4",
        global_scale=torch.ones((), device="cuda"), global_scale_kind="multiplier"), in_features=k)
    source = torch.zeros((4, k), device="cuda")
    output = torch.empty((4, n), dtype=torch.bfloat16, device="cuda")
    # Ties on both parities, subnormals, FP16-exceeding range, infinities and NaNs.
    patterns = [0x00000000, 0x80000000, 0x00008000, 0x00018000,
                0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
                0x47800000, 0x0d800000, 0x7f7f0000, 0xff7f0000,
                0x7f800000, 0xff800000, 0x7f800001, 0xffc01234]
    signed = [value if value < 2**31 else value - 2**32 for value in patterns]
    seeds = torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = sm70.PackedProjection(source, weight, output, session=session, name="round-bits")
        for start in range(0, len(patterns), 4):
            source[:, 0].copy_(seeds[start:start+4])
            operation(source)
            actual = output[:, 0].cpu()
            # Dot products add positive zero, so the all-zero dot is +0.
            expected = (seeds[start:start+4] + 0.).bfloat16()
            assert torch.equal(torch.isnan(actual), torch.isnan(expected))
            valid = ~torch.isnan(expected)
            assert torch.equal(actual[valid].view(torch.int16), expected[valid].view(torch.int16))


@pytest.mark.parametrize("kind", ["nvfp4", "tensor_fp8"])
@torch.no_grad()
def test_gemv_output_tail_and_odd_fp8_logical_k(kind):
    from b12x.preparation import PreparationSession

    _require_v100()
    k = 144 if kind == "nvfp4" else 145
    weight, matrix, gain, scale = _weight(kind, 384, k)
    n = 383
    if kind == "nvfp4":
        weight = replace(weight, out_features=n, values=weight.values[:n])
    else:
        weight = replace(weight, packed=replace(weight.packed, out_features=n, values=weight.packed.values[:n]))
    source = torch.full((3, k), .25, device="cuda")
    storage = torch.full((3, n + 16), 19., device="cuda")
    output = storage[:, :n]
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        constructor = sm70.PackedProjection if kind == "nvfp4" else sm70.TensorFP8Projection
        operation = constructor(source, weight, output, session=session, name="tail")
        operation(source)
        calibrated = source.cpu()
        if kind == "tensor_fp8":
            calibrated = (calibrated / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(output.cpu(), (calibrated @ matrix[:n].T) * gain, rtol=0, atol=0)
        assert torch.all(storage[:, n:] == 19)


@torch.no_grad()
def test_gemv_every_e2m1_and_finite_signed_e4m3_scale_code():
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight
    from b12x.preparation import PreparationSession

    _require_v100()
    n, k = 4096, 144
    finite = torch.tensor([i for i in range(256) if i & 127 != 127], dtype=torch.uint8)
    codes = torch.arange(n).remainder(16)
    selected = finite[(torch.arange(n) // 16).remainder(finite.numel())]
    values = torch.zeros((n, 128), dtype=torch.uint8, device="cuda")
    values[:, 0].copy_(codes.to(torch.uint8))
    scale_bits = torch.full((n, 16), 0x38, dtype=torch.uint8)
    scale_bits[:, 0] = selected
    scales = swizzle_block_scale(scale_bits.view(torch.float8_e4m3fn).cuda())
    weight = replace(pack_weight(values, scales, recipe="nvfp4", global_scale=torch.ones((), device="cuda"),
                                 global_scale_kind="multiplier"), in_features=k)
    source = torch.zeros((4, k), device="cuda")
    source[:, 0].copy_(torch.tensor([1., .5, -1., 0.]))
    output = torch.empty((4, n), device="cuda")
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        operation = sm70.PackedProjection(source, weight, output, session=session, name="all-codes")
        operation(source)
        expected = source[:, :1].cpu() * (table[codes] * selected.view(torch.float8_e4m3fn).float())[None, :]
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
