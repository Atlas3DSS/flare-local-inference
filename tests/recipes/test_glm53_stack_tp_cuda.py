"""Complete checkpoint text forwards through compiled native TP2/TP4 graphs."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS") != "1",
    reason="requires four assigned GPUs and FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS=1")


def _inputs(metadata, tokens, start):
    device = metadata.token_ids.device
    count = len(tokens)
    metadata.token_ids.zero_()
    metadata.token_ids[:count].copy_(torch.tensor(tokens, dtype=torch.int64, device=device))
    metadata.sample_indices.fill_(count - 1)
    delta, index = metadata.recurrent, metadata.indexed
    delta.num_seqs.fill_(1)
    delta.num_tokens.fill_(count)
    delta.query_start_loc[1:].fill_(count)
    delta.convolution_slots[0] = 1
    delta.fresh[0] = start == 0
    delta.state_indices[0, 0] = 1
    delta.initial_state_indices[0] = 1
    delta.final_state_indices[0] = 1
    index.state_slots[0] = 1
    index.query_start_loc[1:].fill_(count)
    index.block_table[0].copy_(torch.arange(index.block_table.shape[1] - 1, -1, -1,
                                           dtype=torch.int32, device=device))
    index.positions.fill_(-1)
    index.positions[:count].copy_(torch.arange(start, start + count, dtype=torch.int64, device=device))
    index.token_slots.fill_(-1)
    index.token_slots[:count].copy_(index.block_table[0, index.positions[:count] // 256].long()
                                   * 256 + index.positions[:count] % 256)


def _rank(rank, world, rendezvous, snapshot, capacity):
    from transformers import AutoTokenizer

    from flare.components.parallel import TensorParallelShard
    from flare.preparation.checkpoint import inspect_indexed_checkpoint
    from flare.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from flare.preparation.compilation import compile_forward
    from flare.preparation.graphs import PreparedGraphBundle
    from flare.preparation.scratch import ScratchArena
    from flare.preparation.tuning import RecordedPreparationSession
    from flare.recipes.glm53.stack import DecoderScratch, PreparedDecoder
    from flare.recipes.glm53.state import DecoderMetadata, DecoderState
    from flare.recipes.glm53.weights import ModelWeights

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    def progress(message):
        if rank == 0:
            print(f"TP{world}: {message}", flush=True)
    shard = TensorParallelShard(rank, world)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress)
    assert len(weights.layers) == 45
    assert weights.tree.lm_head.weight.is_meta == (rank != 0)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=15))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group, timeout=timedelta(minutes=15))
    runtime = create_all_reduce_runtime(group, device, max_tokens=capacity, hidden_size=4096)
    worker = None
    graphs = {mode: torch.cuda.CUDAGraph() for mode in ("prefill", "decode")}
    try:
        state = DecoderState.allocate(weights.tree.config.text_config.layer_types, world, 3, 64, device)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                workload_schema="glm-text-stack-tp-v1", numerical_policy="default", coordinator=coordinator) as session:
            session.register_scratch(arena)
            decoders, metadata, scratch = {}, {}, {}
            for mode, tokens in (("prefill", capacity), ("decode", 1)):
                metadata[mode] = DecoderMetadata.allocate(tokens, 1, 64, device)
                scratch[mode] = DecoderScratch.allocate(tokens, 64, mode, device, shard=shard, sample_rows=1)
                decoders[mode] = PreparedDecoder(weights, scratch[mode], state, metadata[mode], mode=mode,
                    runtime=runtime, coordinator=coordinator, session=session, name=mode, workspace=arena,
                    progress=progress)
            progress("autotuning complete text forwards")
            session.tune()
            compiled = {}
            stream = torch.cuda.Stream(device=device)
            for mode, decoder in decoders.items():
                progress(f"compiling {mode}")
                compiled[mode] = compile_forward(decoder)
                compiled[mode]()
                torch.cuda.synchronize(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graphs[mode], stream=stream):
                    compiled[mode]()
                torch.cuda.synchronize(device)
            session.freeze()
            buffers = dict(state.named_tensors())
            buffers.update({mode + ".hidden": value.hidden for mode, value in scratch.items()})
            bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers=buffers,
                resources=(weights, compiled, decoders, state, metadata, scratch, session, arena, runtime, stream))
            worker = bundle.start_worker()
            state.clear_for_preparation()
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            seed = tokenizer.encode("Explain how tensor parallelism divides a language model across GPUs.")
            tokens = (seed * ((capacity + 3 + len(seed) - 1) // len(seed)))[:capacity + 3]
            for index in range(4):
                mode = "prefill" if index == 0 else "decode"
                start = 0 if index == 0 else capacity + index - 1
                _inputs(metadata[mode], tokens[:capacity] if index == 0 else tokens[start:start + 1], start)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(bundle.graph_indices[mode])
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                hidden = scratch[mode].hidden
                assert torch.isfinite(hidden).all() and hidden.count_nonzero()
                gathered = [None] * world
                dist.all_gather_object(gathered, hidden.cpu(), group=group)
                for value in gathered:
                    torch.testing.assert_close(value, hidden.cpu(), rtol=0, atol=0)
                if rank == 0:
                    logits = scratch[mode].logits
                    assert torch.isfinite(logits).all() and logits.count_nonzero()
                for layer_state in state.layers:
                    if hasattr(layer_state, "recurrent"):
                        assert not layer_state.recurrent[0].count_nonzero()
                        assert torch.isfinite(layer_state.recurrent[1]).all()
                progress(f"native {mode} completed at position {start}")
            progress(f"weight bytes {weights.allocated_bytes}; operator scratch bytes {arena.nbytes}")
            worker.close()
            worker = None
            for graph in graphs.values():
                graph.reset()
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("capacity", [65, 4096])
def test_complete_text_stack_native_tp_replay(world, capacity, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoints-masters/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), capacity),
             nprocs=world, join=True)
