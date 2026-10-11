"""Direct packed-weight loading and graph replay for ordinary FP4 storage."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def checkpoint(path, tensors, layers):
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}}))
    return inspect_indexed_checkpoint(path)


def packed(rows, columns, fmt, generator):
    group = 32 if fmt == "mxfp4" else 16
    w = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator)
    if fmt == "mxfp4":
        s = torch.randint(121, 125, (rows, columns // group), dtype=torch.uint8, generator=generator)
    else:
        s = (torch.rand((rows, columns // group), generator=generator) * .02 + .01).to(torch.float8_e4m3fn)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    values = lut[torch.stack((w & 15, w >> 4), dim=-1).long()].flatten(-2)
    scales = torch.exp2(s.float() - 127) if fmt == "mxfp4" else s.float()
    tensors = {"weight": w, "weight_scale": s}
    if fmt != "mxfp4":
        tensors["weight_scale_2"] = torch.tensor(.5)
        scales = scales * .5
    return tensors, values * scales.repeat_interleave(group, dim=-1)


def mxfp4_activation(x):
    from ignition.reference.qwen38_quant import _FP4_VALUES

    groups = x.float().reshape(x.shape[0], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.where(amax > 0, amax / 6, 1).log2().ceil().clamp(-127, 127))
    v = groups / scale
    z = v.abs()
    code = ((z > .25).int() + (z >= .75).int() + (z > 1.25).int() + (z >= 1.75).int()
            + (z > 2.5).int() + (z >= 3.5).int() + (z > 5).int())
    code |= torch.where((v < 0) & (code != 0), 8, 0)
    return (torch.tensor(_FP4_VALUES)[code] * scale).reshape(x.shape)


@pytest.mark.parametrize("rows", [1, 17, 128])
@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_a4", "mxfp4"])
def test_ordinary_fp4_linear_graph(tmp_path, rows, fmt):
    from b12x.preparation import PreparationSession
    from ignition.components.linear import QuantizedProjection
    from ignition.preparation.scratch import ScratchArena

    gen = torch.Generator().manual_seed(871)
    tensors, weight = packed(256, 512, fmt, gen)
    algo, group = ("MXFP4", 32) if fmt == "mxfp4" else ("W4A16_NVFP4", 16)
    if fmt == "nvfp4_a4":
        algo = "NVFP4"
        tensors["input_scale"] = torch.tensor(.03125)
    manifest = checkpoint(tmp_path, {"proj." + name: t for name, t in tensors.items()},
                          {"proj": {"quant_algo": algo, "group_size": group}})
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        w = loader.load_linear("proj", rows=(128, 256), columns=(32, 320))
        assert loader._session.torch_copy_bytes == 0
    cpu = torch.randn(rows, 288, generator=gen).to(torch.bfloat16)
    x, out = cpu.cuda(), torch.empty((rows, 128), device="cuda", dtype=torch.bfloat16)
    with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
        projection = QuantizedProjection(x, w, out, session=session, name="linear", activation_mode="auto",
                                         workspace=ScratchArena(x.device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection(x)
        for factor in (1, -.5):
            x.copy_(cpu * factor)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            source = mxfp4_activation(cpu * factor) if fmt == "mxfp4" else (cpu * factor).float()
            expected = source @ weight[128:256, 32:320].T
            if fmt == "nvfp4_a4":
                from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_linear_reference

                stored = Nvfp4Projection(tensors["weight"][128:256, 16:160],
                    tensors["weight_scale"][128:256, 2:20], tensors["weight_scale_2"], tensors["input_scale"])
                expected = nvfp4_linear_reference(cpu * factor, stored)
            torch.testing.assert_close(out.float().cpu(), expected, rtol=.025, atol=.015)


@pytest.mark.parametrize("fmt,mode", [("nvfp4", "a16"), ("nvfp4", "a4"), ("mxfp4", "a8")])
def test_ordinary_fp4_experts_graph(tmp_path, fmt, mode):
    from b12x.preparation import PreparationSession
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_expert_reference, quantize_mxfp8_activation

    gen = torch.Generator().manual_seed(192)
    tensors, decoded, serialized = {}, [], []
    for expert in range(2):
        weights, stored = [], []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            data, logical = packed(256, 256, fmt, gen)
            tensors.update({f"experts.{expert}.{projection}.{name}": t for name, t in data.items()})
            if mode == "a4":
                data["input_scale"] = torch.tensor(.03125)
                tensors[f"experts.{expert}.{projection}.input_scale"] = data["input_scale"]
            weights.append(logical)
            stored.append(data)
        decoded.append(weights)
        serialized.append(stored)
    manifest = checkpoint(tmp_path, tensors, {})
    scale_vectors = (torch.full((2,), .5), torch.full((2,), .5), torch.full((2,), 32.), torch.full((2,), 32.))
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        experts = loader.load_fp4_experts(("experts.0", "experts.1"), hidden_size=256, intermediate_size=256,
            intermediate_range=(128, 256), format=fmt, activation_mode=mode,
            scale_vectors=scale_vectors if mode == "a4" else None)
        assert loader._session.torch_copy_bytes == 0
    cpu = (torch.randn(2, 256, generator=gen) * .1).to(torch.bfloat16)
    x, out = cpu.cuda(), torch.empty_like(cpu, device="cuda")
    ids = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    probabilities = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
        operation = PreparedRoutedExperts(x, experts, ids, probabilities, out, session=session, name="moe")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        for order in ((0, 1), (1, 0)):
            ids.copy_(torch.tensor(order, dtype=torch.int32)[:, None])
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = []
            for row, expert in enumerate(order):
                gate, up, down = decoded[expert]
                source = cpu[row:row + 1].float()
                if mode == "a4":
                    projections = []
                    for index, data in enumerate(serialized[expert]):
                        w, sf = data["weight"], data["weight_scale"]
                        w, sf = (w[:, 64:128], sf[:, 8:16]) if index == 2 else (w[128:256], sf[128:256])
                        projections.append(Nvfp4Projection(w, sf, data["weight_scale_2"], data["input_scale"]))
                    result = nvfp4_expert_reference(source, *projections).down
                else:
                    if mode == "a8":
                        source = quantize_mxfp8_activation(source)
                    hidden = (torch.nn.functional.silu(source @ gate[128:256].T) * (source @ up[128:256].T)).to(torch.bfloat16).float()
                    if mode == "a8":
                        hidden = quantize_mxfp8_activation(hidden)
                    result = hidden @ down[:, 128:256].T
                expected.append(result)
            expected = torch.cat(expected)
            relative = torch.linalg.vector_norm(out.float().cpu() - expected) / torch.linalg.vector_norm(expected)
            assert relative < .035, float(relative)
