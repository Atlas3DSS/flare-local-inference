"""GLM shared and routed experts preserve compiled graph and scratch dependencies."""

import os
from pathlib import Path

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.moe import MoeScratch, PreparedGlmMoe
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.fixture(scope="module", params=[2, 4])
def expert_partition(request):
    snapshot = os.environ.get("FLARE_GLM53_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_GLM53_SNAPSHOT must identify the checkpoint")
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    shard = TensorParallelShard(0, request.param)
    tree = create_model_tree(manifest.snapshot)
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
                         prefixes=(f"{TEXT_PREFIX}.layers.3.mlp.",)) as loader:
        scales = loader.allocate_expert_scale_scratch()
        weights = loader.load_mlp(3, scale_scratch=scales)
    return tree.model.language_model.layers[3].mlp, weights, scales, shard.world_size


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("defer_launch", [False, True])
@torch.no_grad()
def test_autotuned_shared_routed_graph(expert_partition, rows, compiled, defer_launch):
    module, weights, _scale_owners, world_size = expert_partition
    arena = ScratchArena("cuda:0")
    source = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda") * .1
    input_copy = source.clone()
    output_copy, shared_copy = torch.empty_like(source), torch.empty_like(source)
    scratch = MoeScratch.allocate(rows, world_size, "cuda:0")
    with PreparationSession(device="cuda:0", autotune=True, compile_workers=0) as session:
        moe = PreparedGlmMoe(module, input_copy, weights, scratch, session=session,
                             name="overlap", workspace=arena)
        arena.freeze()
        arena.finalize()
        session.freeze()
        overlap = moe.concurrent_shared
        assert overlap is not None
        overlap.defer_launch = defer_launch
        routed_bytes = moe.experts.plan.memory_requirements().scratch_nbytes
        assert moe.experts.scratch.data_ptr() + routed_bytes <= overlap.workspace.data_ptr()

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
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(16):
            source.normal_(std=.1)
            input_copy.copy_(source)
            expected_shared = moe.shared(input_copy).clone()
            moe.router_projection(input_copy)
            moe.routing(scratch.router_logits)
            expected = moe.experts() + expected_shared
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(shared_copy, expected_shared, rtol=0, atol=0)
            # Native routed experts permit atomic accumulation in either order.
            torch.testing.assert_close(output_copy, expected, rtol=.01, atol=.001)
        graph.reset()
