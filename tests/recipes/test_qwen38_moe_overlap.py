"""Concurrent expert branches retain graph dependencies and scratch lifetimes."""

import os
from pathlib import Path

import pytest
import torch
from b12x.preparation import PreparationSession
from cuda.bindings import driver as cuda

from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.compilation import compile_forward
from flare.preparation.qwen38 import TEXT_PREFIX
from flare.preparation.scratch import ScratchArena
from flare.recipes.qwen38.moe import PreparedQwenMoe, QwenMoeWeights
from flare.recipes.qwen38.weights import QwenWeightLoader, allocate_expert_scale_scratch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.fixture(scope="module")
def weights():
    snapshot = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_QWEN38_SNAPSHOT must identify the checkpoint")
    manifest = inspect_qwen38_checkpoint(Path(snapshot))
    scales = allocate_expert_scale_scratch("cuda:0")
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(f"{TEXT_PREFIX}.layers.0.mlp.",)) as loader:
        return QwenMoeWeights.load(loader, 0, scale_scratch=scales), scales


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_shared_routed_fork_join_and_arena_reuse(weights, rows, compiled):
    weights, _scale_owners = weights
    arena = ScratchArena("cuda:0")
    source = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda") * .1
    input_copy = torch.empty_like(source)
    output_copy = torch.empty_like(source)
    shared_copy = torch.empty_like(source)
    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        moe = PreparedQwenMoe(input_copy, weights, session=session, name="overlap",
                              workspace=arena, projection_workspace=arena)
        arena.freeze()
        arena.finalize()
        session.freeze()
        assert moe.concurrent_shared is not None
        routed_bytes = moe.experts.plan.memory_requirements().scratch_nbytes
        assert moe.experts.scratch.data_ptr() + routed_bytes <= moe.concurrent_shared.workspace.data_ptr()

        def forward():
            input_copy.copy_(source)
            moe(input_copy)
            output_copy.copy_(moe.output)
            shared_copy.copy_(moe.shared.output)
            arena.tensor.fill_(255)
            input_copy.zero_()

        call = compile_forward(forward) if compiled else forward
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            call()
        graph.instantiate()
        raw = cuda.CUgraph(graph.raw_cuda_graph())
        status, _, _, _, count = cuda.cuGraphGetEdges(raw)
        assert status == cuda.CUresult.CUDA_SUCCESS
        status, sources, destinations, _, _ = cuda.cuGraphGetEdges(raw, count)
        assert status == cuda.CUresult.CUDA_SUCCESS
        edges = [(int(a), int(b)) for a, b in zip(sources, destinations, strict=True)]
        assert any(sum(parent == node for parent, _ in edges) >= 2 for node, _ in edges)
        assert any(sum(child == node for _, child in edges) >= 2 for _, node in edges)
        for _ in range(3):
            source.normal_(std=.1)
            input_copy.copy_(source)
            moe.shared(input_copy)
            expected_shared = moe.shared.output.clone()
            moe.router_projection(input_copy)
            moe.routing(moe.router_logits)
            moe.experts()
            expected = moe.routed_output + expected_shared
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(shared_copy, expected_shared, rtol=0, atol=0)
            # Native routed experts permit atomic accumulation in either order.
            torch.testing.assert_close(output_copy, expected, rtol=.01, atol=.001)
        graph.reset()
