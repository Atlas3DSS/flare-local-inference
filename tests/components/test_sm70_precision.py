"""Software BF16 boundaries against independent CPU BF16 arithmetic."""

import copy
import os

import pytest
import torch
import torch.nn.functional as F

from ignition.components.sm70_precision import apply_rotary, rms_norm, round_bf16


def _values():
    # Ties with even/odd high bits, signed zeros, subnormals, BF16 overflow,
    # infinities and small/large NaN payloads exercise the integer rounding.
    special = [0, 0x80000000, 1, 0x80000001, 0x00008000, 0x00018000,
               0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000,
               0x47800000, 0xC7800000, 0x7F7F8000, 0xFF7F8000,
               0x7F800000, 0xFF800000, 0x7F800001, 0xFFFFFFFF]
    generator = torch.Generator().manual_seed(7016)
    random = torch.randint(-(2**31), 2**31, (8192,), generator=generator, dtype=torch.int32)
    return torch.cat((torch.tensor(special, dtype=torch.int64).to(torch.int32), random)).view(torch.float32)


def _assert_rounding(result, expected):
    assert result.dtype == torch.float32
    nan = expected.isnan()
    assert torch.equal(result.isnan(), nan)
    # Bit comparison also protects signed zero, infinity and subnormal values.
    assert torch.equal(result[~nan].view(torch.int32), expected[~nan].view(torch.int32))


def test_integer_rounding_matches_cpu_bf16_conversion():
    values = _values()
    _assert_rounding(round_bf16(values), values.bfloat16().float())
    assert round_bf16(torch.tensor([2.0**80])).item() == 2.0**80
    with pytest.raises(TypeError, match="FP32"):
        round_bf16(torch.ones(1, dtype=torch.float16))


def _rms_reference(value, weight, epsilon):
    result = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon)
    return (result * (1.0 + weight.float())).bfloat16()


def _rotary_reference(value, cos, sin):
    width = cos.shape[-1]
    active = value[..., :width]
    half = width // 2
    rotated = torch.cat((-active[..., half:], active[..., :half]), -1)
    return torch.cat((active * cos[:, None] + rotated * sin[:, None], value[..., width:]), -1)


def test_norm_and_partial_rotary_preserve_canonical_bf16_boundaries():
    generator = torch.Generator().manual_seed(7035)
    q = torch.randn((7, 6, 128), generator=generator).bfloat16()
    k = torch.randn((7, 2, 128), generator=generator).bfloat16()
    weight = (torch.randn(128, generator=generator) * .2).bfloat16()
    angle = torch.randn((7, 32), generator=generator)
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    expected_q, expected_k = (_rms_reference(value, weight, 1e-6) for value in (q, k))
    actual_q, actual_k = (rms_norm(value.float(), weight.float(), 1e-6) for value in (q, k))
    torch.testing.assert_close(actual_q, expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k.float(), rtol=0, atol=0)
    result = apply_rotary(actual_q, actual_k, cos.float(), sin.float())
    for actual, expected in zip(result, (expected_q, expected_k), strict=True):
        torch.testing.assert_close(actual, _rotary_reference(expected, cos, sin).float(), rtol=0, atol=0)


_cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                            reason="requires an explicitly assigned V100")


def _device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (7, 0):
        pytest.skip("requires a real sm_70 GPU")
    return device


@_cuda
@torch.no_grad()
def test_rounding_compiles_and_replays_without_bf16_storage():
    from ignition.preparation.compilation import compile_forward

    device = _device()
    values = _values().to(device)
    output = torch.empty_like(values)

    def forward():
        output.copy_(round_bf16(values))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for offset in (0, 13):
        current = _values().roll(offset)
        values.copy_(current)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        _assert_rounding(output.cpu(), current.bfloat16().float())
    graph.reset()


class _LinearMixer(torch.nn.Module):
    """A real dot product isolates layer arithmetic from attention scheduling."""

    def __init__(self, width, full):
        super().__init__()
        self.projection = torch.nn.Linear(width, width, bias=False)
        self.full = full

    def forward(self, hidden_states, **kwargs):
        result = self.projection(hidden_states)
        return (result, None) if self.full else result


def _layer_fixture(full):
    layer = torch.nn.Module()
    layer.block_type = "full_attention" if full else "linear_attention"
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = torch.nn.Module()
        norm.weight = torch.nn.Parameter(torch.randn(32) * .1, requires_grad=False)
        norm.eps = 1e-6
        setattr(layer, name, norm)
    setattr(layer, "self_attn" if full else "linear_attn", _LinearMixer(32, full))
    layer.mlp = torch.nn.Module()
    layer.mlp.gate_proj = torch.nn.Linear(32, 64, bias=False)
    layer.mlp.up_proj = torch.nn.Linear(32, 64, bias=False)
    layer.mlp.down_proj = torch.nn.Linear(64, 32, bias=False)
    layer.mlp.act_fn = F.silu
    return layer.bfloat16().requires_grad_(False)


def _layer_reference(layer, hidden):
    residual = hidden
    norm = layer.input_layernorm
    hidden = _rms_reference(hidden, norm.weight, norm.eps)
    if layer.block_type == "full_attention":
        hidden, _ = layer.self_attn(hidden)
    else:
        hidden = layer.linear_attn(hidden)
    hidden = residual + hidden
    residual = hidden
    norm = layer.post_attention_layernorm
    hidden = _rms_reference(hidden, norm.weight, norm.eps)
    hidden = layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(hidden)) * layer.mlp.up_proj(hidden))
    return (residual + hidden).float()


@_cuda
@pytest.mark.parametrize("full", [False, True])
@torch.no_grad()
def test_decoder_layer_boundaries_compile_and_replay(full):
    from b12x.preparation import PreparationSession
    from ignition.components.sm70_projection import UnquantizedProjection
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.qwen38_dense.decoder import Projection, SM70DecoderLayer

    device = _device()
    torch.manual_seed(7035)
    reference = _layer_fixture(full)
    source = torch.randn((11, 32)).bfloat16().float().to(device)
    output = torch.empty_like(source)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        layer = copy.deepcopy(reference).float().to(device)
        # Exercise the production projection boundary. A plain torch.nn.Linear
        # instead enters Inductor's unrelated GEMM-template path, which loses
        # the dtype of a preceding integer/FP32 reinterpretation on this build.
        paths = ("self_attn.projection" if full else "linear_attn.projection",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        for path in paths:
            linear = reference.get_submodule(path)
            weight = linear.weight.detach().to(device)
            inputs = torch.zeros((11, weight.shape[1]), device=device)
            result = torch.zeros((11, weight.shape[0]), device=device)
            projection = UnquantizedProjection(inputs, weight, result, session=session, name=path)
            parent, _, field = path.rpartition(".")
            setattr(layer.get_submodule(parent), field, Projection(projection))
        operation = SM70DecoderLayer(layer)

        def forward():
            output.copy_(operation(source, position_embeddings=None))

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for scale in (1., -.5):
            source.mul_(scale)
            expected = _layer_reference(reference, source.cpu().bfloat16())
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            actual = output.cpu()
            assert not ((actual.view(torch.int32) & 65535) != 0).any()
            torch.testing.assert_close(actual, expected, rtol=2**-7, atol=1e-5)
        graph.reset()
