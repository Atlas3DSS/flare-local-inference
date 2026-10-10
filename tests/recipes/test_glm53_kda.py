"""GLM KDA state continuation through compiled graphs and native replay."""

import os
from pathlib import Path

import pytest
import torch

from flare.components.parallel import TensorParallelShard
from flare.components.recurrent import DeltaMetadata, DeltaState
from flare.preparation.checkpoint import inspect_indexed_checkpoint
from flare.preparation.compilation import compile_forward
from flare.preparation.graphs import PreparedGraphBundle
from flare.preparation.scratch import ScratchArena
from flare.preparation.tuning import RecordedPreparationSession
from flare.recipes.glm53.kda import KDAScratch, PreparedGlmKDA
from flare.recipes.glm53.model import create_model_tree
from flare.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader


@pytest.mark.parametrize("world", [2, 4])
def test_rank_state_and_scratch_geometry(world):
    heads = 64 // world
    state = DeltaState.allocate(6, "cpu", heads=heads, channels=3 * heads * 128)
    scratch = KDAScratch.allocate(65, heads, "cpu")
    assert state.recurrent.shape == (6, heads, 128, 128)
    assert state.convolution.shape == (6, 3 * heads * 128, 3)
    assert scratch.rows(1).projected_qkv.data_ptr() == scratch.projected_qkv.data_ptr()
    assert scratch.rows(1).output.shape == (1, 4096)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                   reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
@pytest.mark.parametrize("world,tokens", [(2, 65), (4, 129), (4, 4096)])
@torch.no_grad()
def test_actual_kda_native_prefill_and_decode(world, tokens):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoints-masters/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    manifest = inspect_indexed_checkpoint(snapshot)
    model = create_model_tree(snapshot)
    module = model.model.language_model.layers[0].self_attn
    device = torch.device("cuda", 0)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, world),
                         prefixes=(f"{TEXT_PREFIX}.layers.0.self_attn.",)) as loader:
        weights = loader.load_attention(0)
    heads = 64 // world
    state = DeltaState.allocate(6, device, heads=heads, channels=3 * heads * 128)
    arena = ScratchArena(device)
    metadata = {mode: DeltaMetadata.allocate(2, device) for mode in ("prefill", "decode")}
    inputs = {mode: torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
              for mode, capacity in (("prefill", tokens), ("decode", 2))}
    scratch = {mode: KDAScratch.allocate(source.shape[0], heads, device) for mode, source in inputs.items()}
    graphs, operators, compiled = {}, {}, {}
    worker = None
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                    workload_schema="glm-kda-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        for mode, source in inputs.items():
            operators[mode] = PreparedGlmKDA(module, weights, source, scratch[mode], state, metadata[mode],
                mode=mode, session=session, name=f"kda.{mode}", workspace=arena, checkpoint_export=mode == "prefill")
        session.tune()
        try:
            for mode, operator in operators.items():
                source = inputs[mode]

                def forward(operator=operator, source=source):
                    operator(source)

                compiled[mode] = compile_forward(forward)
                compiled[mode]()
                torch.cuda.synchronize()
                graphs[mode] = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graphs[mode]):
                    compiled[mode]()
                torch.cuda.synchronize()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs=graphs,
                buffers={"convolution": state.convolution, "recurrent": state.recurrent},
                resources=(compiled, operators, arena, session, inputs, metadata, scratch))
            worker = bundle.start_worker()
            generator = torch.Generator().manual_seed(53)
            for mode, count, repetitions in (("prefill", tokens, 1), ("decode", 1, 8), ("prefill", 3, 1)):
                source, meta = inputs[mode], metadata[mode]
                source.zero_()
                source[:count].copy_((torch.randn((count, 4096), generator=generator) * .1).to(torch.bfloat16))
                meta.query_start_loc.copy_(torch.tensor([0, count, count], dtype=torch.int32))
                meta.convolution_slots.copy_(torch.tensor([1, -1], dtype=torch.int64))
                meta.fresh.fill_(mode == "prefill" and count == tokens)
                meta.state_indices[0, 0] = 1
                meta.initial_state_indices[0] = 0 if count == tokens else 1
                meta.final_state_indices[0] = 1
                meta.checkpoint_state_indices[0] = 2 if count >= 16 else 0
                meta.checkpoint_offsets[0] = 16 if count >= 16 else 0
                meta.num_seqs.fill_(1)
                meta.num_tokens.fill_(count)
                initial = (state.convolution.clone(), state.recurrent.clone())
                for _ in range(repetitions):
                    operators[mode](source)
                expected = (scratch[mode].output.clone(), state.convolution.clone(), state.recurrent.clone())
                state.convolution.copy_(initial[0])
                state.recurrent.copy_(initial[1])
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices[mode], repetitions=repetitions)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for actual, reference in zip((scratch[mode].output, state.convolution, state.recurrent), expected, strict=True):
                    assert bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert not state.recurrent[0].count_nonzero()
                assert not state.recurrent[3:].count_nonzero()
            worker.close()
            worker = None
        finally:
            if worker is not None:
                worker.close()
            for graph in graphs.values():
                graph.reset()
