"""Queued single-rank decode consumes GPU-owned pending tokens."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState, RequestAdmission, RUNNING, ERROR
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_queued_decode_uses_pending_device_token_and_rejects_invalid_state():
    device = torch.device("cuda", 0)
    requests = DeviceRequestState.allocate(device, vocab_size=256, capacity=2)
    inputs = BatchInputs.allocate(2, 2, device, capacity=2)
    metadata = Metadata.allocate(2, 2, 2, device)
    pool = SimpleNamespace(memory_capacity=torch.tensor([3, 4], device=device))
    batch = PreparedBatch(inputs, requests, metadata, mode="decode",
                          shard=TensorParallelShard(0, 1), pool=pool)

    def step():
        batch.begin(batch.validate())
        batch.finish()
        # A deterministic device sampler supplies the next token without a host round trip.
        requests.pending_token.copy_(torch.where(requests.status == RUNNING,
                                                 metadata.token_ids + 1, requests.pending_token))

    compiled = compile_forward(step)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    requests.admit(0, RequestAdmission(generation=1, seed=0), [7], consumed_tokens=1)
    requests.pending_token[0] = 11
    inputs.request_state_rows[0, 0] = 1
    inputs.row_slot_ids[0] = 0
    inputs.row_generations[0] = 1
    inputs.live_rows.fill_(1)
    inputs.page_tables[0].copy_(torch.tensor([0, 1], device=device))
    inputs.token_ids.fill_(11)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    assert requests.status[0].item() == RUNNING
    assert requests.consumed_tokens[0].item() == 5
    assert requests.pending_token[0].item() == 15
    assert metadata.token_ids.tolist() == [14, 0]
    assert requests.consumed_tokens[1].item() == 0
    requests.pending_token[0] = 256
    graph.replay()
    torch.cuda.synchronize()
    assert requests.status[0].item() == ERROR
    assert requests.error_code[0].item() == 32
    assert requests.consumed_tokens[0].item() == 5
    graph.reset()
