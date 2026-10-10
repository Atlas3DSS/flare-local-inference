"""Compiled b12x TP2/TP4 graphs replayed by native workers on every rank."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world_size, rendezvous, rows):
    from b12x.comm.pcie import AllReduce
    from b12x.preparation import PreparationSession

    from flare.components.collectives import PreparedAllReduce
    from flare.preparation.collectives import CollectivePreparation
    from flare.preparation.compilation import compile_forward
    from flare.preparation.graphs import PreparedGraphBundle

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    source = torch.zeros(rows, 4096, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    seed = torch.full_like(source, rank + 1)
    result = torch.empty_like(source)
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    runtime = AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True)
    graph = torch.cuda.CUDAGraph()
    worker = None
    try:
        with PreparationSession(device=device, autotune=True, compile_workers=0,
                                collective_barrier=coordinator.barrier) as session:
            reduction = PreparedAllReduce(source, output, runtime=runtime, session=session,
                coordinator=coordinator, name=f"reduce{rows}")

            def forward():
                source.copy_(seed + counter.float() * .125)
                result.copy_(reduction(source) * 2)
                counter.add_(1)

            dist.barrier(group=group)
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture():
                with torch.cuda.graph(graph, stream=stream):
                    compiled()
            torch.cuda.synchronize(device)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs={"reduce": graph},
                buffers={"source": source, "output": output, "seed": seed,
                         "result": result, "counter": counter},
                resources=(compiled, reduction, runtime, session, stream))
            worker = bundle.start_worker()
            for repetitions, offset in ((8, 0.), (31, .5)):
                counter.zero_()
                seed.fill_(rank + 1 + offset)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(0, repetitions=repetitions)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket
                assert completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                expected = 2 * (world_size * (world_size + 1) / 2
                                + world_size * (offset + (repetitions - 1) * .125))
                torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)
                assert counter.item() == repetitions
                dist.barrier(group=group)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("rows", [1, 8])
def test_native_compiled_all_reduce(world_size, rows, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=(world_size, (tmp_path / "rendezvous").as_uri(), rows),
             nprocs=world_size, join=True)
