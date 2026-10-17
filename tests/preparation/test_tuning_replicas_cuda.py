"""Candidate sharding and graph replay through the shared b12x preparation path."""

import multiprocessing
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires two explicitly selected CUDA GPUs")


def _worker(directory, replica, connection):
    import torch
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.tuning_replicas import TuningComplete, TuningReplica

    torch.cuda.set_device(replica)
    device = torch.device("cuda", replica)
    generator = torch.Generator().manual_seed(47)
    source = torch.randn((4, 768), generator=generator, dtype=torch.bfloat16).to(device)
    weight = torch.randn((512, 768), generator=generator, dtype=torch.bfloat16).to(device)
    output = torch.empty((4, 512), device=device, dtype=torch.bfloat16)
    role = TuningReplica(str(Path(directory) / "gloo"), replica, 2, 1)
    session = RecordedPreparationSession(device=device, checkpoint="replica-cuda-test",
        workload_schema="projection", numerical_policy="bf16", compile_workers=4,
        namespace={"case": directory}, tuning_replica=role)
    projection = UnquantizedProjection(source, weight, output, session=session, name="projection")
    try:
        session.tune()
    except TuningComplete:
        assert replica == 1
    else:
        assert replica == 0
        expected = source.float() @ weight.float().T
        projection(source)
        torch.testing.assert_close(output.float(), expected, rtol=0.01, atol=0.25)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            projection(source)
        for _ in range(4):
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output.float(), expected, rtol=0.01, atol=0.25)
    report = session.tuning_report
    connection.send({key: report[key] for key in ("compile_workers", "benchmarked_candidates",
                     "parent_cute_compilations", "parent_triton_compilations", "selections")})
    connection.close()
    session.close()


def test_two_gpus_share_candidates_and_replay_the_selected_plan(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes, connections = [], []
    try:
        for replica in range(2):
            root, peer = context.Pipe(duplex=False)
            process = context.Process(target=_worker, args=(str(tmp_path), replica, peer))
            process.start()
            peer.close()
            processes.append(process)
            connections.append(root)
        for process in processes:
            process.join(timeout=180)
        assert [process.exitcode for process in processes] == [0, 0]
        reports = [connection.recv() for connection in connections]
        assert all(report["compile_workers"] == 2 for report in reports)
        assert all(report["benchmarked_candidates"] > 0 for report in reports)
        assert all(report["parent_cute_compilations"] == report["parent_triton_compilations"] == 0
                   for report in reports)
        assert reports[0]["selections"][0]["config"] == reports[1]["selections"][0]["config"]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for connection in connections:
            connection.close()


def _tp_worker(directory, member, connection):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.tuning_replicas import TuningComplete, TuningReplica

    replica, rank = divmod(member, 2)
    torch.cuda.set_device(member)
    device = torch.device('cuda', member)
    dist.init_process_group('gloo', init_method=(Path(directory) / f'tp-{replica}').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4, hidden_size=768)
    role = TuningReplica(str(Path(directory) / 'replicas'), replica, 2, 2)
    session = RecordedPreparationSession(device=device, checkpoint='replica-tp-test',
        workload_schema='projection-reduction', numerical_policy='bf16', compile_workers=4,
        namespace={'case': directory}, tuning_replica=role, coordinator=coordinator)
    source = torch.full((4, 768), rank + 1, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    projection = None
    if rank == 0:
        weight = torch.ones((512, 768), device=device, dtype=torch.bfloat16)
        projected = torch.empty((4, 512), device=device, dtype=torch.bfloat16)
        projection = UnquantizedProjection(source, weight, projected, session=session, name='projection')
    reduction = PreparedAllReduce(source, output, runtime=runtime, session=session,
                                  coordinator=coordinator, name='reduce')
    try:
        session.tune()
    except TuningComplete:
        assert replica == 1
    else:
        assert replica == 0
        reduction(source)
        torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
            reduction(source)
            if projection is not None:
                projection(source)
        for _ in range(4):
            output.fill_(float('nan'))
            dist.barrier()
            graph.replay()
            torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)
            if projection is not None:
                torch.testing.assert_close(projected, torch.full_like(projected, 768), rtol=0, atol=0)
    connection.send(session.tuning_report['selections'])
    connection.close()
    dist.barrier()
    session.close()
    runtime.close()
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                    reason='requires four explicitly selected CUDA GPUs')
def test_tp_replicas_preserve_uneven_rank_work_and_collectives(tmp_path):
    context = multiprocessing.get_context('spawn')
    processes, connections = [], []
    try:
        for member in range(4):
            root, peer = context.Pipe(duplex=False)
            process = context.Process(target=_tp_worker, args=(str(tmp_path), member, peer))
            process.start()
            peer.close()
            processes.append(process)
            connections.append(root)
        for process in processes:
            process.join(timeout=180)
        assert [process.exitcode for process in processes] == [0] * 4
        selections = [connection.recv() for connection in connections]
        for rank in range(2):
            assert [(item['name'], item['config']) for item in selections[rank]] == [
                (item['name'], item['config']) for item in selections[rank + 2]]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for connection in connections:
            connection.close()
