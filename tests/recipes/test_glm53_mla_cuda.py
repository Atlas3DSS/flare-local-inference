"""Checkpoint GLM latent attention composes into allocation-stable native graphs."""

import os
from pathlib import Path

import pytest
import torch

from flare.components.parallel import TensorParallelShard
from flare.preparation.checkpoint import inspect_indexed_checkpoint
from flare.preparation.compilation import compile_forward
from flare.preparation.graphs import PreparedGraphBundle
from flare.preparation.scratch import ScratchArena
from flare.preparation.tuning import RecordedPreparationSession
from flare.recipes.glm53.indexer import IndexMetadata
from flare.recipes.glm53.mla import MLAScratch, MLAState, PreparedGlmMLA
from flare.recipes.glm53.model import create_model_tree
from flare.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module", params=[2, 4])
def model_weights(request):
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoints-masters/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    tree = create_model_tree(snapshot)
    manifest = inspect_indexed_checkpoint(snapshot)
    world = request.param
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(world - 1, world),
            prefixes=(f"{TEXT_PREFIX}.layers.3.self_attn.",)) as loader:
        weights = loader.load_attention(3)
    return tree.model.language_model.layers[3].self_attn, weights, world


@pytest.mark.parametrize("mode,rows", [("decode", 2), ("prefill", 65), ("prefill", 4096)])
@torch.no_grad()
def test_glm_mla_checkpoint_native_replay(model_weights, mode, rows):
    module, weights, world = model_weights
    device = torch.device("cuda:0")
    sequences = 2 if mode == "decode" else 1
    source = torch.zeros((rows, 4096), dtype=torch.bfloat16, device=device)
    scratch = MLAScratch.allocate(rows, 64 // world, 64, mode, device)
    state = MLAState.allocate(128, 4, device)
    metadata = IndexMetadata.allocate(rows, sequences, 64, device)
    arena = ScratchArena(device)
    graph, worker = torch.cuda.CUDAGraph(), None
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
            workload_schema="glm-mla-v1", numerical_policy="default") as session:
        try:
            session.register_scratch(arena)
            attention = PreparedGlmMLA(module, weights, source, scratch, state, metadata,
                mode=mode, session=session, name="mla", workspace=arena)
            session.tune()

            def forward():
                attention(source)
                arena.tensor.fill_(255)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={mode: graph},
                buffers={"source": source, "output": scratch.output, "cache": state.cache,
                         "index": state.index.cache, "tail": state.index.tail},
                resources=(compiled, attention, session, arena))
            worker = bundle.start_worker()
            state.cache.zero_()
            state.index.cache.zero_()
            state.index.tail.normal_(std=.1)
            metadata.block_table.copy_(torch.randperm(128, device=device, dtype=torch.int32)
                                       .reshape(2, 64)[:sequences])
            metadata.state_slots.copy_(torch.tensor([2, 1][:sequences], dtype=torch.int64, device=device))
            metadata.query_start_loc.copy_(torch.tensor([0, 1, 2] if mode == "decode" else [0, rows],
                                                         dtype=torch.int32, device=device))
            if mode == "decode":
                metadata.requests.copy_(torch.arange(2, dtype=torch.int32, device=device))
            stages = {key: getattr(scratch, key) for key in (
                "query_input", "query_residual", "latent_input", "latent", "query", "absorbed_query",
                "attended", "expanded", "output")}
            persistent = (state.cache, state.index.cache, state.index.tail)
            mismatches = []
            for step in range(3):
                source.normal_(std=.1)
                if mode == "decode":
                    metadata.positions.copy_(torch.tensor([255 + step, 3 + step], dtype=torch.int64, device=device))
                else:
                    metadata.positions.copy_(torch.arange(rows, device=device, dtype=torch.int64) + 3 + step * rows)
                metadata.token_slots.copy_(metadata.block_table[metadata.requests.long(), metadata.positions // 256].long()
                                          * 256 + metadata.positions % 256)
                if mode == "decode" and step == 2:
                    metadata.state_slots[-1] = -1
                    metadata.positions[-1] = -1
                    metadata.token_slots[-1] = -1
                initial = tuple(value.clone() for value in persistent)
                forward()
                expected = {key: value.clone() for key, value in stages.items()}
                expected_state = tuple(value.clone() for value in persistent)
                expected_indices = scratch.index.physical.sort().values
                expected_lengths = scratch.index.selected_lengths.clone()
                for value, saved in zip(persistent, initial, strict=True):
                    value.copy_(saved)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                for value, expected_value in zip(persistent, expected_state, strict=True):
                    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
                torch.testing.assert_close(scratch.index.physical.sort().values, expected_indices, rtol=0, atol=0)
                torch.testing.assert_close(scratch.index.selected_lengths, expected_lengths, rtol=0, atol=0)
                if mode == "decode" and step == 2:
                    assert not scratch.output[-1].count_nonzero()
                for key, value in stages.items():
                    assert torch.isfinite(value).all(), key
                    if not torch.equal(value, expected[key]):
                        difference = (value.float() - expected[key].float()).abs()
                        mismatches.append(f"step {step}, {key}: {difference.count_nonzero().item()} elements differ; "
                                          f"maximum absolute error {difference.max().item()}")
                        del difference
            assert not mismatches, "\n".join(mismatches)
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
