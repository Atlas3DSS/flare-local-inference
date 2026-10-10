"""Sharded vocabulary projection and column gathering through native graph replay."""

import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


@torch.no_grad()
def _rank(rank, world, rendezvous):
    from flare.components.vocabulary import PreparedVocabularyProjection
    from flare.preparation.collectives import (
        CollectivePreparation,
        create_all_gather_runtime,
    )
    from flare.preparation.compilation import compile_forward
    from flare.preparation.execution import GraphSignature, PreparedExecution
    from flare.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    vocab, hidden, capacity = 154880, 256, 4
    width = vocab // world
    torch.manual_seed(42)
    full_weight = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device) * .0625
    weight = full_weight[rank * width:(rank + 1) * width].clone()
    seed = torch.randn(capacity, hidden, dtype=torch.bfloat16, device=device) * .125
    source = seed.clone()
    local = torch.empty(capacity, width, dtype=torch.bfloat16, device=device)
    output = torch.empty(capacity, vocab, dtype=torch.bfloat16, device=device)
    runtime = create_all_gather_runtime(group, device, max_rows=capacity, columns=vocab)
    graphs, heads, compiled = {}, {}, {}
    control = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
               "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
               "completion_records": torch.zeros((2, 4, 8), dtype=torch.int64, device=device)}
    worker = None
    try:
        with RecordedPreparationSession(device=device, checkpoint="vocabulary-test",
                workload_schema="column-gather-v1", numerical_policy="default", coordinator=coordinator) as session:
            for rows in (4, 1):
                heads[rows] = PreparedVocabularyProjection(source[:rows], weight, local[:rows], output[:rows],
                    runtime=runtime, session=session, coordinator=coordinator, name=f"head{rows}")
            session.tune()
            stream = torch.cuda.Stream(device=device)
            pool = torch.cuda.graph_pool_handle()
            for rows, head in heads.items():
                compiled[rows] = compile_forward(lambda head=head, rows=rows: head(source[:rows]))
                compiled[rows]()
                torch.cuda.synchronize(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph, stream=stream, pool=pool):
                    compiled[rows]()
                graphs[f"head{rows}"] = graph
            session.freeze()
            bundle = PreparedExecution(device=rank, graphs=graphs,
                signatures={f"head{rows}": GraphSignature("decode", rows, 1) for rows in heads},
                buffers=control,
                resources=(heads, compiled, runtime, session, stream, pool, weight))
            worker = bundle.start_executor()
            for index, (rows, repetitions) in enumerate(((1, 13), (4, 8), (1, 31), (4, 17))):
                source.copy_(seed + index * .03125)
                expected = (source[:rows].float() @ full_weight.float().T).bfloat16()
                output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                for _ in range(repetitions):
                    ticket = worker.submit(bundle.graph_indices[f"head{rows}"], [])
                    assert ticket is not None
                    deadline = time.monotonic() + 120
                    completion = worker.poll()
                    while completion is None and time.monotonic() < deadline:
                        time.sleep(.001)
                        completion = worker.poll()
                    assert completion is not None and completion.fence == ticket
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                torch.testing.assert_close(output[:rows], expected, rtol=.016, atol=.002)
                torch.testing.assert_close(output[:rows, rank * width:(rank + 1) * width],
                                           local[:rows], rtol=0, atol=0)
                if rows < capacity:
                    assert output[rows:].isnan().all()
                dist.barrier(group=group)
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
def test_sharded_vocabulary_native_replay(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned CUDA devices")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
