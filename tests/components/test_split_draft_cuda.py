"""Diagnostic rejection reports failure without touching device request state."""

import os

import pytest
import torch

from ignition.components.split_draft import _reject_stale
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_captured_rejection_preserves_replacement_and_inactive_requests():
    valid = torch.zeros(1, dtype=torch.bool, device="cuda")
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    epoch = torch.tensor([9], dtype=torch.int64, device="cuda")
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    generation = epoch.clone()
    consumed = torch.tensor([127], device="cuda")
    generated = torch.tensor([3], device="cuda")
    records = torch.zeros((2, 1, 24), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.tensor([19], device="cuda")

    def forward():
        torch.library.wrap_triton(_reject_stale)[(1,)](valid, slot, epoch, live, generation,
            consumed, generated, records, frame, sequence, 1, 1)

    operation = compile_forward(forward)
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    bundle = PreparedGraphBundle(device=0, graphs={"reject": graph}, buffers={},
        resources=(valid, slot, epoch, live, generation, consumed, generated, records,
                   frame, sequence, operation))
    worker = bundle.start_worker()
    try:
        for admitted, active, request_epoch in ((False, 1, 9), (True, 1, 9), (False, 0, 9), (False, 1, 10)):
            valid.fill_(admitted)
            live.fill_(active)
            generation.fill_(request_epoch)
            records.fill_(71)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0)
            completion = worker.poll(10000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert generation.item() == request_epoch and consumed.item() == 127 and generated.item() == 3
            if admitted or not active:
                assert records.eq(71).all()
            else:
                record = records[0, 0].tolist()
                assert record[:8] == [9, 19, 127 if request_epoch == 9 else 0,
                                      3 if request_epoch == 9 else 0, -1, 3, 256, 0]
                assert record[8] == 0 and record[16:] == [-1] * 8
    finally:
        worker.close()
        graph.reset()


@torch.no_grad()
def test_native_proposal_feedback_copies_host_mapped_confidence_without_allocating():
    import struct
    import time
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.package_storage import PackageStorage

    storage = PackageStorage(world=1)
    confidence = storage.host_allocate("proposal_confidence", (1, 7), torch.float32)
    values = torch.tensor([[.9, .8, .3, .1, .4, .5, .6]], device="cuda")
    records = torch.zeros((2, 1, 24), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        confidence.device_view.copy_(values)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        confidence.device_view.copy_(values)
    model = PreparedExecution(device=0, max_active=1, graphs={"proposal": graph},
        signatures={"proposal": GraphSignature("proposal", 1, 8, ("values",), speculation_enabled=True)},
        buffers=dict(frame_index=frame, submission_sequence=sequence, completion_records=records,
                     proposal_confidence=confidence.device_view, values=values), resources=(storage, confidence))
    executor = model.start_executor()
    try:
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for value in (.25, .75, 0., 1.) * 8:
            fence = executor.submit(0, [("values", 0, struct.pack("7f", *([value]*7)))], True)
            deadline = time.monotonic() + 10
            while (completed := executor.poll()) is None:
                assert time.monotonic() < deadline
            assert completed.fence == fence
            assert list(completed.proposal_confidence) == [value] * 7
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    finally:
        executor.close()
        graph.reset()
        for region in storage.host_regions:
            region.close()
