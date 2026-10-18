"""Rank-zero sampling decisions survive divergent logits and native graph replay."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires explicitly assigned tensor-parallel GPUs")


class SamplingRound:
    def __init__(self, proposals, verify, ids, probabilities, valid, sampled, offset):
        self.proposals, self.verify = proposals, verify
        self.ids, self.probabilities, self.valid = ids, probabilities, valid
        self.sampled, self.offset = sampled, offset

    def __call__(self):
        for index, proposal in enumerate(self.proposals):
            self.offset.fill_(index * 4)
            proposal()
            self.ids[:, index].copy_(self.sampled)
            self.probabilities[:, index].copy_(proposal.probabilities[:, 0])
            self.valid[:, index].copy_(proposal.valid[:, 0])
        self.verify()

    def results(self):
        return (self.ids, self.probabilities, self.valid, *vars(self.verify.output).values())


def _rank(rank, world, rendezvous):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.parallel_draft import prepare_sampling_broadcasts, prepare_sampling_scratch
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.speculation import PreparedStochasticVerification, SpeculativeRngState
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    # TP2 exercises multirow messages; TP16 uses Kimi's vocabulary channel geometry.
    runtime = create_all_gather_runtime(dist.group.WORLD, device, max_rows=64,
                                        columns=128 if world == 2 else 163840)
    graphs, owners, references, calls = {}, [], [], []
    worker = None
    try:
        state = DeviceRequestState.allocate(device, vocab_size=257, capacity=4)
        for slot in range(4):
            state.admit(slot, RequestAdmission(generation=7, seed=93 + slot, max_new_tokens=100,
                                               top_p=.95, eos_token_ids=()), [1, 2])
        slots = torch.tensor([3, 0, 2, 1], dtype=torch.int32, device=device)
        epochs = torch.tensor([7, 7, 7, 99], dtype=torch.int64, device=device)
        buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        with RecordedPreparationSession(device=device, checkpoint="sampling-broadcast",
                workload_schema="rank-sampling-v1", numerical_policy="default", coordinator=coordinator) as session:
            for batch, width in ((4, 3), (1, 1), (4, 8)):
                live = torch.tensor([batch], dtype=torch.int32, device=device)
                cursor = buffers.cursor()
                workspace = prepare_sampling_scratch(batch, width, 257, device, buffers=cursor)
                ids = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
                probabilities = cursor.zeros((batch, width - 1, 257), torch.float32)
                valid = torch.zeros((batch, width - 1), dtype=torch.bool, device=device)
                seed = torch.arange(batch, dtype=torch.int64, device=device) + 51
                offset, sampled = torch.zeros_like(seed), torch.zeros_like(seed)
                logits = torch.randn((batch, 1, 257), generator=torch.Generator().manual_seed(17 + rank)).to(device)
                proposals = tuple(PreparedDraftDistribution(logits, state, slots[:batch], epochs[:batch], live, ids,
                    prefix_offset=index, seed=seed, offset=offset, sampled=sampled,
                    workspace=workspace["proposal"]) for index in range(width - 1))
                for proposal in proposals:
                    proposal.prepare()
                target = torch.zeros((batch, width, 257), device=device)
                target[:, :, rank + 19] = 1.
                verify = PreparedStochasticVerification(probabilities, ids, target, state,
                    slots[:batch], epochs[:batch], live, rng=SpeculativeRngState.allocate(device),
                    draft_valid=valid, workspace=workspace["verify"])
                verify.prepare()
                operation = SamplingRound(proposals, verify, ids, probabilities, valid, sampled, offset)
                operation()
                reference = [tensor.cpu().clone() for tensor in operation.results()]
                dist.broadcast_object_list(reference, src=0)
                references.append(reference)
                prepare_sampling_broadcasts(proposals, verify,
                    collectives=dict(vocabulary_runtime=runtime, coordinator=coordinator), session=session,
                    name=f"sampling{batch}x{width}", buffers=cursor)
                owners.append(operation)
            session.tune(show_progress=False)
            for index, operation in enumerate(owners):
                call = compile_forward(operation)
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                graphs[str(index)] = graph
                calls.append(call)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers={},
                resources=(owners, calls, session, runtime, state, buffers))
            worker = bundle.start_worker()
            for index in (0, 1, 2, 0, 2, 1):
                # Poison shared scratch before switching shape; every replay must initialize it.
                for chunks in buffers.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else 255)
                torch.cuda.synchronize()
                dist.barrier()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                for actual, expected in zip(owners[index].results(), references[index], strict=True):
                    assert torch.equal(actual.cpu(), expected)
            worker.close()
            worker = None
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 16])
def test_native_sampling_broadcast(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() >= world
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
