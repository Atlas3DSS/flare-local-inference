"""GLM KDA and dense/MoE layers in compiled TP graphs with native rank replay."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world, rendezvous, snapshot, mode, capacity, layer_index):
    from flare.components.parallel import TensorParallelShard
    from flare.components.recurrent import DeltaMetadata, DeltaState
    from flare.preparation.checkpoint import inspect_indexed_checkpoint
    from flare.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from flare.preparation.compilation import compile_forward
    from flare.preparation.graphs import PreparedGraphBundle
    from flare.preparation.scratch import ScratchArena
    from flare.preparation.tuning import RecordedPreparationSession
    from flare.recipes.glm53.decoder import LayerScratch, PreparedGlmLayer
    from flare.recipes.glm53.model import create_model_tree
    from flare.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = create_model_tree(Path(snapshot))
    manifest = inspect_indexed_checkpoint(snapshot)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(rank, world),
                         prefixes=(f"{TEXT_PREFIX}.layers.{layer_index}.",)) as loader:
        attention, mlp, residual = loader.load_attention(layer_index), loader.load_mlp(layer_index), loader.load_residual(layer_index)
        assert loader._session.torch_copy_bytes == 0
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_all_reduce_runtime(group, device, max_tokens=capacity, hidden_size=4096)
    worker = None
    graph = torch.cuda.CUDAGraph()
    try:
        count = capacity if mode == "prefill" else 1
        heads = 64 // world
        state = DeltaState.allocate(6, device, heads=heads, channels=3 * heads * 128)
        metadata = DeltaMetadata.allocate(2, device)
        scratch = LayerScratch.allocate(capacity, world, device, sparse=layer_index >= 3)
        source = torch.zeros((capacity, 4, 4096), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(source)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                        workload_schema="glm-layer-tp-v1", numerical_policy="default",
                                        coordinator=coordinator) as session:
            session.register_scratch(arena)
            layer = PreparedGlmLayer(model.model.language_model.layers[layer_index], attention, mlp, residual,
                source, output, scratch, state, metadata, mode=mode, runtime=runtime, coordinator=coordinator,
                session=session, name=f"layer{layer_index}", workspace=arena)
            session.tune()

            def forward():
                layer(source)
                arena.tensor.fill_(255)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                compiled()
            torch.cuda.synchronize(device)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs={mode: graph},
                buffers={"source": source, "output": output, "state": state.recurrent, "history": state.convolution},
                resources=(compiled, layer, runtime, session, arena, stream, metadata))
            worker = bundle.start_worker()
            generator = torch.Generator().manual_seed(530)
            source[:count].copy_((torch.randn((count, 4, 4096), generator=generator) * .1).to(torch.bfloat16))
            metadata.query_start_loc.copy_(torch.tensor([0, count, count], dtype=torch.int32))
            metadata.convolution_slots.copy_(torch.tensor([1, -1], dtype=torch.int64))
            metadata.state_indices[0, 0] = 1
            metadata.initial_state_indices[0] = 1
            metadata.final_state_indices[0] = 1
            metadata.checkpoint_state_indices[0] = 2 if count >= 16 else 0
            metadata.checkpoint_offsets[0] = 16 if count >= 16 else 0
            metadata.num_seqs.fill_(1)
            metadata.num_tokens.fill_(count)
            stages = {
                "attention_norm": scratch.attention_hc.normalized,
                "attention": scratch.attention_output,
                "mlp_input": scratch.mlp_hc.normalized,
            }
            if layer_index >= 3:
                stages.update(router_logits=scratch.mlp.router_logits, ids=scratch.mlp.router.ids,
                              weights=scratch.mlp.router.weights, shared=scratch.mlp.shared.output)
            for repetitions in (1, 4):
                initial = (state.recurrent.clone(), state.convolution.clone())
                for _ in range(repetitions):
                    forward()
                expected = (output.clone(), state.recurrent.clone(), state.convolution.clone())
                expected_stages = {name: value.clone() for name, value in stages.items()}
                state.recurrent.copy_(initial[0])
                state.convolution.copy_(initial[1])
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(0, repetitions=repetitions)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for name, value in stages.items():
                    torch.testing.assert_close(value, expected_stages[name], rtol=0, atol=0)
                for actual, reference in zip((state.recurrent, state.convolution), expected[1:], strict=True):
                    assert bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert not state.recurrent[0].count_nonzero()
                gathered = [None] * world
                dist.all_gather_object(gathered, output.cpu(), group=group)
                for other in gathered:
                    torch.testing.assert_close(other, output.cpu(), rtol=0, atol=0)
                if layer_index >= 3:
                    dist.all_gather_object(gathered, scratch.mlp.router.ids.cpu(), group=group)
                    for other in gathered:
                        torch.testing.assert_close(other, scratch.mlp.router.ids.cpu(), rtol=0, atol=0)
                assert bool(torch.isfinite(output).all())
                torch.testing.assert_close(output, expected[0],
                    rtol=.01 if layer_index >= 3 else 0,
                    atol=.001 if layer_index >= 3 else 0)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("layer_index", [0, 4])
@pytest.mark.parametrize("world,mode,capacity", [(2, "decode", 2), (4, "decode", 2),
                                               (2, "prefill", 65), (4, "prefill", 4096)])
def test_actual_layer_native_tp_graph(world, mode, capacity, layer_index, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoints-masters/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), mode, capacity, layer_index),
             nprocs=world, join=True)
