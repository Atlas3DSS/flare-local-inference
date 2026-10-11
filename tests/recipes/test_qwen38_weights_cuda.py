"""Actual checkpoint components on an explicitly assigned SM120/SM121 GPU."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.qwen38 import MODEL_ID, REVISION, TEXT_PREFIX
from flare.recipes.qwen38.weights import QwenWeightLoader, allocate_expert_scale_scratch
from flare.reference.qwen38_quant import (
    CheckpointTensors,
    mxfp8_linear_reference,
    nvfp4_expert_reference,
    nvfp4_linear_reference,
    read_nvfp4_projection,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(scope="module")
def manifest():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    assert torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    default = Path.home() / ".cache/huggingface/hub" / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / REVISION
    return inspect_qwen38_checkpoint(Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default)))


def _assert_numerical(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    assert float(relative_l2) < .008, f"relative L2 error {float(relative_l2)}"
    torch.testing.assert_close(actual, expected, rtol=.015, atol=float(expected.abs().max()) * .01 + 1e-6)


@pytest.mark.parametrize("prefix,mode,rows,columns", [
    (f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_a", "quantized", None, None),
    ("model.visual.blocks.0.mlp.linear_fc1", "a16", None, None),
    ("model.visual.blocks.0.mlp.linear_fc2", "a16", None, None),
    ("model.visual.blocks.0.mlp.linear_fc1", "a16", (2152, 4304), None),
    ("model.visual.blocks.0.mlp.linear_fc2", "a16", None, (16, 144)),
    ("model.visual.blocks.0.mlp.linear_fc2", "a16", (3, 131), (16, 160)),
])
def test_actual_dense_weight_prepared_graph_replay(manifest, prefix, mode, rows, columns):
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._a16 import _weight_parts
    from b12x.preparation import PreparationSession, PreparedCall

    with QwenWeightLoader(manifest, "cuda:0", prefixes=(prefix + ".",)) as loader:
        packed = loader.load_linear(prefix, rows=rows, columns=columns)
        assert loader._session.torch_copy_bytes == 0
    if prefix.endswith("linear_fc2") and columns is None:
        assert packed.in_features == 4304 and packed.padded_in_features == 4352
        assert torch.count_nonzero(packed.values[:, 2152:]) == 0
    cpu_input = torch.randn(3, packed.in_features, generator=torch.Generator().manual_seed(39)).to(torch.bfloat16)
    source = cpu_input.cuda()
    output = torch.empty((3, packed.out_features), dtype=torch.bfloat16, device="cuda")
    query = blockscaled.query_from_call(source, packed, activation_mode=mode, out=output)
    declaration = blockscaled.plan(query)
    values, scales, gain, _ = _weight_parts(packed)
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        session.prepare((declaration.request(
            name=f"checkpoint:{prefix}",
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, values, scales, gain, out=output)),
        ),))
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blockscaled.mm(source, packed, out=output, plan=declaration)
        for factor in (1., -0.5):
            source.copy_(cpu_input * factor)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            with CheckpointTensors(manifest, byte_budget=32 << 20) as reader:
                if mode == "quantized":
                    expected = mxfp8_linear_reference(cpu_input * factor, reader.read(prefix + ".weight"), reader.read(prefix + ".weight_scale"))
                else:
                    projection = read_nvfp4_projection(reader, prefix)
                    r0, r1 = rows or (0, projection.packed.shape[0])
                    c0, c1 = columns or (0, projection.packed.shape[1] * 2)
                    projection = replace(projection,
                        packed=projection.packed[r0:r1, c0 // 2:c1 // 2],
                        scales=projection.scales[r0:r1, c0 // 16:c1 // 16])
                    expected = nvfp4_linear_reference(cpu_input * factor, projection)
            _assert_numerical(output, expected)


def test_actual_bf16_direct_load_and_expansion(manifest):
    name = f"{TEXT_PREFIX}.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(name,)) as loader:
        bf16 = loader.load_tensor(name)
        fp32 = loader.load_tensor(name, dtype=torch.float32)
        assert loader._session.torch_copy_bytes == 0
    with CheckpointTensors(manifest) as reader:
        original = reader.read(name)
    torch.testing.assert_close(bf16.cpu(), original, rtol=0, atol=0)
    torch.testing.assert_close(fp32.cpu(), original.float(), rtol=0, atol=0)


def test_batched_tensors_are_ready_before_loader_exit(manifest):
    names = (
        f"{TEXT_PREFIX}.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        f"{TEXT_PREFIX}.layers.0.linear_attn.A_log",
        f"{TEXT_PREFIX}.layers.0.linear_attn.dt_bias",
    )
    with QwenWeightLoader(manifest, "cuda:0", prefixes=names) as loader:
        values = loader.load_tensors(names)
        assert loader._session.stats(flush=False)["batches"] == 1
        with CheckpointTensors(manifest) as reader:
            for name, value in zip(names, values, strict=True):
                torch.testing.assert_close(value.cpu(), reader.read(name), rtol=0, atol=0)


@pytest.mark.parametrize("rank,world", [(0, 1), (1, 2), (3, 4)])
def test_actual_csf_experts_prepared_graph_replay(manifest, rank, world):
    from dataclasses import replace
    from flare.components.parallel import TensorParallelShard

    shard = TensorParallelShard(rank, world)
    first, last = shard.partition(640)
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall

    ids = (0, 511)
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts"
    shared = allocate_expert_scale_scratch("cuda:0", len(ids), shard=shard)
    with QwenWeightLoader(manifest, "cuda:0", prefixes=tuple(f"{prefix}.{expert}." for expert in ids), shard=shard) as loader:
        experts = loader.load_experts(0, expert_ids=ids, scale_scratch=shared)
        assert loader._session.torch_copy_bytes == 0
    assert experts._impl.w1_blockscale.data_ptr() == shared[0].data_ptr()
    assert experts._impl.w2_blockscale.data_ptr() == shared[1].data_ptr()
    assert experts._impl.a2_gscale.unique().numel() == 2
    moe.expand_scales(experts)
    cpu_scales = []
    with CheckpointTensors(manifest, byte_budget=16 << 20) as reader:
        projections = []
        for expert in ids:
            projections.append(tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}") for name in ("gate_proj", "up_proj", "down_proj")))
    projections = [(replace(gate, packed=gate.packed[first:last], scales=gate.scales[first:last]),
                    replace(up, packed=up.packed[first:last], scales=up.scales[first:last]),
                    replace(down, packed=down.packed[:, first // 2:last // 2], scales=down.scales[:, first // 16:last // 16]))
                   for gate, up, down in projections]
    for storage in shared:
        _, rows, columns = storage.shape
        logical = storage.view(torch.uint8).reshape(len(ids), rows // 128, columns // 4, 32, 4, 4)
        cpu_scales.append(logical.permute(0, 1, 4, 3, 2, 5).reshape(len(ids), rows, columns).cpu())
    for local, (gate, up, down) in enumerate(projections):
        assert torch.equal(cpu_scales[0][local, :2 * (last - first)], torch.cat((up.scales, gate.scales)).view(torch.uint8))
        assert torch.equal(cpu_scales[1][local, :, :(last - first) // 16], down.scales.view(torch.uint8))
    cpu_input = (torch.randn(2, 2560, generator=torch.Generator().manual_seed(63)) * .1).to(torch.bfloat16)
    source = cpu_input.cuda()
    routes = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    probabilities = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    declaration = moe.plan_execution(
        experts=experts, capacity=moe.ExecutionCapacity(max_tokens=2, top_k=1),
        invocation={"fast_math": False}, routing=moe.RoutingSpec(deterministic_output=True),
    )

    def prepared_call(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device="cuda") for spec in state.scratch.scratch_specs())
        binding = state.bind(a=source, topk_ids=routes, topk_weights=probabilities, output=output, scratch=scratch, input_scales_static=True)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        session.prepare((declaration.request(name="checkpoint:layer0:experts0,511", prepare_call=prepared_call),))
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device="cuda") for spec in declaration.scratch_specs())
        binding = moe.bind(declaration, a=source, topk_ids=routes, topk_weights=probabilities, output=output, scratch=scratch, input_scales_static=True)
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.run(binding=binding)
        for order in ((0, 1), (1, 0)):
            routes.copy_(torch.tensor(order, dtype=torch.int32)[:, None])
            for buffer in shared:
                buffer.view(torch.uint8).fill_(0x7F)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = torch.cat([nvfp4_expert_reference(cpu_input[row:row + 1], *projections[expert]).down for row, expert in enumerate(order)])
            _assert_numerical(output, expected)


@pytest.mark.parametrize("layer,region,names", [
    (0, "linear_attn", ("in_proj_qkv", "in_proj_z")),
    (0, "linear_attn", ("in_proj_b", "in_proj_a")),
    (3, "self_attn", ("q_proj", "k_proj", "v_proj")),
])
def test_packed_mxfp8_projections_preserve_serialized_rows(manifest, layer, region, names):
    prefixes = tuple(f"{TEXT_PREFIX}.layers.{layer}.{region}.{name}" for name in names)
    with QwenWeightLoader(manifest, "cuda:0", prefixes=prefixes) as loader:
        packed = loader.load_linear_group(prefixes)
        assert loader._session.torch_copy_bytes == 0
    values = packed.weight.values.view(torch.uint8).cpu().reshape(packed.out_features, packed.padded_in_features)
    scales = packed.weight.scale_rows.view(torch.uint8).cpu().reshape(packed.out_features, -1)
    offset = 0
    with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
        for prefix in prefixes:
            original = reader.read(prefix + ".weight").view(torch.uint8)
            original_scales = reader.read(prefix + ".weight_scale").view(torch.uint8)
            rows, columns = original.shape
            assert torch.equal(values[offset:offset + rows, :columns], original)
            assert torch.equal(scales[offset:offset + rows, :columns // 32], original_scales[:rows, :columns // 32])
            offset += rows
    assert offset == packed.out_features


def test_complete_layer_csf_load_and_captured_expansion(manifest):
    from flare.reference.qwen38_quant import decode_csf_scales

    scratch = allocate_expert_scale_scratch('cuda:0')
    prefix = f'{TEXT_PREFIX}.layers.0.mlp.experts.'
    with QwenWeightLoader(manifest, 'cuda:0', prefixes=(prefix,)) as loader:
        prepared = loader.load_experts(0, scale_scratch=scratch)
        assert loader._session.torch_copy_bytes == 0
    routes = torch.tensor([0, 511, 7], dtype=torch.int32, device='cuda')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared._impl.nvfp4_csf.decode(routes, *scratch)
    for selected in ((0, 511, 7), (99, 3, 501)):
        routes.copy_(torch.tensor(selected, dtype=torch.int32, device='cuda'))
        for tensor in scratch:
            tensor.view(torch.uint8).fill_(0xD6)
        allocated = torch.cuda.memory_stats()['allocation.all.allocated']
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
        with CheckpointTensors(manifest, byte_budget=2 << 20) as reader:
            for expert in selected:
                scales = []
                for projection in ('up_proj', 'gate_proj', 'down_proj'):
                    name = f'{prefix}{expert}.{projection}.weight_scale.nvfp4_csf_'
                    rows, columns = (2560, 40) if projection == 'down_proj' else (640, 160)
                    scales.append(decode_csf_scales(reader.read(name + 'fixed'), reader.read(name + 'exceptions'),
                                                   rows=rows, columns=columns))
                for storage, rows, columns, expected in (
                    (scratch[0], 1280, 160, torch.cat(scales[:2])), (scratch[1], 2560, 40, scales[2]),
                ):
                    actual = storage[expert].view(torch.uint8).reshape(rows // 128, columns // 4, 32, 4, 4)
                    actual = actual.permute(0, 3, 2, 1, 4).reshape(rows, columns).cpu()
                    assert torch.equal(actual, expected.view(torch.uint8))


@pytest.mark.parametrize("rank,world", [(0, 1), (1, 2), (3, 4)])
def test_mtp_expert_shards_match_checkpoint_arithmetic(manifest, rank, world):
    from dataclasses import replace
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall
    from flare.components.parallel import TensorParallelShard
    from flare.recipes.qwen38.mtp_weights import MTPWeightLoader

    shard = TensorParallelShard(rank, world)
    first, last = shard.partition(640)
    ids = (0, 511)
    prefix = "mtp.layers.0.mlp.experts"
    with MTPWeightLoader(manifest, "cuda:0", shard=shard,
            prefixes=tuple(f"{prefix}.{expert}." for expert in ids)) as loader:
        experts = loader.load_experts(expert_ids=ids)
    with CheckpointTensors(manifest, byte_budget=16 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                       for name in ("gate_proj", "up_proj", "down_proj")) for expert in ids]
    projections = [(replace(gate, packed=gate.packed[first:last], scales=gate.scales[first:last]),
                    replace(up, packed=up.packed[first:last], scales=up.scales[first:last]),
                    replace(down, packed=down.packed[:, first // 2:last // 2], scales=down.scales[:, first // 16:last // 16]))
                   for gate, up, down in projections]
    inputs = (torch.randn(2, 2560, generator=torch.Generator().manual_seed(63)) * .1).bfloat16()
    source = inputs.cuda()
    routes = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    probabilities = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    output = torch.empty_like(source)
    plan = moe.plan_execution(experts=experts, capacity=moe.ExecutionCapacity(max_tokens=2, top_k=1),
                              routing=moe.RoutingSpec(deterministic_output=True))

    def prepare(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device="cuda") for spec in state.scratch.scratch_specs())
        binding = state.bind(a=source, topk_ids=routes, topk_weights=probabilities, output=output, scratch=scratch)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name=f"mtp.experts.tp{world}", prepare_call=prepare),))
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device="cuda") for spec in plan.scratch_specs())
        binding = moe.bind(plan, a=source, topk_ids=routes, topk_weights=probabilities, output=output, scratch=scratch)
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.run(binding=binding)
        for order in ((0, 1), (1, 0)):
            routes.copy_(torch.tensor(order, dtype=torch.int32)[:, None])
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = []
            for row, expert in enumerate(order):
                gate, up, down = projections[expert]
                gate_output = nvfp4_linear_reference(inputs[row:row + 1], gate)
                up_output = nvfp4_linear_reference(inputs[row:row + 1], up)
                intermediate = (torch.nn.functional.silu(gate_output) * up_output).bfloat16()
                expected.append(nvfp4_linear_reference(intermediate, down))
            expected = torch.cat(expected)
            _assert_numerical(output, expected)
