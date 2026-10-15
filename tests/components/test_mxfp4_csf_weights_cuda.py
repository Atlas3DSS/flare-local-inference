"""Lossless DS4.1 expert loading and compressed-scale replay with shared scratch."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.deepseek41.recipe import RECIPE

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def expanded_scale(read, name, rows, columns):
    fixed = read(name + ".mxfp4_csf_fixed").numpy().reshape(rows // 16, -1)
    base = fixed[:, :16].reshape(rows, 1)
    bits = np.unpackbits(fixed[:, 16:].reshape(rows, -1), axis=1, bitorder="little")[:, :columns]
    result = (base + bits).copy()
    words = read(name + ".mxfp4_csf_exceptions").numpy()
    result.reshape(-1)[words & 0xFFFFFF] = words >> 24
    return torch.from_numpy(result)


@pytest.fixture(scope="module")
def experts():
    from b12x.moe import fused_moe as moe

    checkpoint = RECIPE.checkpoints[0]
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + checkpoint.repository.replace("/", "--")) / "snapshots" / checkpoint.revision
    if not root.is_dir():
        pytest.skip("lossless DS4.1 CSF checkpoint is unavailable")
    manifest = inspect_indexed_checkpoint(root)
    e, h, n = 2, 5120, 576
    buffers = (torch.empty((e, h // 32, 2 * n), dtype=torch.uint8, device="cuda:0"),
               torch.empty((e, n // 32, h), dtype=torch.uint8, device="cuda:0"))
    owners = []
    for layer in (0, 1):
        prefixes = [f"layers.{layer}.ffn.experts.{i}" for i in range(e)]
        with DirectWeightLoader(manifest, "cuda:0", prefixes=tuple(p + "." for p in prefixes)) as loader:
            compressed = loader.load_mxfp4_csf_experts(prefixes, hidden_size=h, intermediate_size=2304,
                intermediate_range=(576, 1152), scale_scratch=buffers,
                projection_names=("w1", "w3", "w2"), scale_name="scale", swiglu_limit=10.)
            assert loader._session.torch_copy_bytes == 0
        assert compressed.has_csf_blocks
        assert compressed._impl.w1_blockscale.data_ptr() == buffers[0].data_ptr()
        assert compressed._impl.w2_blockscale.data_ptr() == buffers[1].data_ptr()
        w13, w2, s13, s2 = [], [], [], []
        for prefix in prefixes:
            path = root / manifest.tensors[prefix + ".w1.weight"].shard
            with safe_open(path, framework="pt", device="cpu") as reader:
                read = reader.get_tensor
                w13.append(torch.cat([read(prefix + f".{p}.weight")[576:1152].view(torch.uint8) for p in ("w1", "w3")]))
                w2.append(read(prefix + ".w2.weight")[:, 288:576].view(torch.uint8))
                s13.append(torch.cat([expanded_scale(read, prefix + f".{p}.scale", 2304, h // 32)[576:1152] for p in ("w1", "w3")]))
                s2.append(expanded_scale(read, prefix + ".w2.scale", h, 72)[:, 18:36])
        one = torch.ones(e, device="cuda:0")
        ordinary_plan = moe.plan_weights(source=compressed.plan.source, activation=compressed.plan.activation,
                                         geometry=compressed.plan.geometry)
        ordinary = moe.prepare_weights(plan=ordinary_plan, weights=moe.PackedWeights(
            w13=torch.stack(w13).cuda(), w2=torch.stack(w2).cuda(),
            w13_block_scales=torch.stack(s13).cuda(), w2_block_scales=torch.stack(s2).cuda(),
            w13_global_scales=one, w2_global_scales=one))
        for name in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale"):
            assert torch.equal(getattr(compressed._impl, name), getattr(ordinary._impl, name)), name
        owners.append((compressed, ordinary))
    return owners, buffers


@pytest.mark.parametrize("tokens", [1, 32, 256, 4096])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_csf_experts_match_expanded_scales_with_poisoned_shared_scratch(experts, tokens, compiled):
    from b12x.preparation import PreparationSession
    from ignition.components.moe import PreparedRoutedExperts

    owners, buffers = experts
    torch.manual_seed(754 + tokens)
    x = torch.randn((tokens, 5120), dtype=torch.bfloat16, device="cuda:0") * .1
    ids = (torch.arange(tokens, device="cuda:0") % 2).to(torch.int32)[:, None]
    weights = torch.ones((tokens, 1), dtype=torch.float32, device="cuda:0")
    graphs = []
    with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
        for layer in owners:
            pair = []
            for owner in layer:
                out = torch.empty_like(x)
                operation = PreparedRoutedExperts(x, owner, ids, weights, out, session=session,
                                                  name=f"expert-{len(graphs)}-{len(pair)}")
                call = operation
                if compiled:
                    from ignition.preparation.compilation import compile_forward
                    call = compile_forward(operation)
                call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                pair.append((graph, out, call))
            graphs.append(pair)
        for index in (0, 1, 0, 1):
            for buffer in buffers:
                buffer.fill_(255)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            for graph, _, _ in graphs[index]:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(graphs[index][0][1], graphs[index][1][1], rtol=0, atol=0)
