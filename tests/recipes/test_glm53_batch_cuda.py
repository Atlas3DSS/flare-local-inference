"""Native GLM request packing, physical page mapping, and mutation guards."""

import os

import pytest
import torch

from flare.components.parallel import TensorParallelShard
from flare.components.sampling import DeviceRequestState, RequestAdmission
from flare.preparation.compilation import compile_forward
from flare.preparation.graphs import PreparedGraphBundle
from flare.recipes.glm53.batch import BatchInputs, PreparedBatch
from flare.recipes.glm53.state import DecoderMetadata

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rank,world", [(0, 2), (3, 4)])
@pytest.mark.parametrize("mode,capacity", [("prefill", 65), ("prefill", 4096), ("decode", 4)])
@torch.no_grad()
def test_native_glm_metadata_and_consumption(rank, world, mode, capacity):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    rows = 1 if mode == "prefill" else capacity
    inputs = BatchInputs.allocate(capacity, 32, "cuda")
    requests = DeviceRequestState.allocate("cuda", vocab_size=154880)
    metadata = DecoderMetadata.allocate(capacity, rows, 32, "cuda")
    batch = PreparedBatch(inputs, requests, metadata, mode=mode,
        shard=TensorParallelShard(rank, world), pages=64)

    def forward():
        batch.begin(batch.validate() * world)
        batch.finish()

    call = compile_forward(forward)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"batch": graph},
        buffers={"tokens": metadata.token_ids, "slots": metadata.indexed.token_slots},
        resources=(batch, inputs, requests, metadata, call))
    worker = bundle.start_worker()
    try:
        faults = (None, "partial", "stale", "page", "token", "pending", "context")
        if mode == "decode":
            faults += ("duplicate",)
        for fault in faults:
            slots = [3] if mode == "prefill" else [3, 1, 0, 2]
            lengths = [capacity if fault != "partial" else 13] if mode == "prefill" else [1] * rows
            positions = [0, 63, 255, 256]
            if mode == "prefill" and fault is None:
                positions[3] = 0
            if fault == "context":
                positions[3] = 8192
            for slot in range(4):
                requests.admit(slot, RequestAdmission(7, 13, max_new_tokens=10000),
                    [3] * positions[slot], consumed_tokens=positions[slot])
            page_table = torch.stack([(torch.arange(32) + slot * 7) % 64 for slot in range(4)]).int()
            if fault == "page":
                page_table[3, 0] = 64
            token_ids = list(range(10, 10 + capacity))
            if fault == "token":
                token_ids[0] = 154880
            active = [True] * rows
            if mode == "decode" and rank == 0:
                for row, slot in enumerate(slots):
                    requests.pending_token[slot] = token_ids[row]
            if fault == "pending":
                requests.pending_token[3] = 99
            if fault == "duplicate":
                slots[1] = 3
                active[1] = False
            epochs = [7] * rows
            if fault == "stale":
                epochs[0] = 6
            if fault not in (None, "partial"):
                active[0] = False
            inputs.row_slot_ids.fill_(-1)
            inputs.row_slot_ids[:rows].copy_(torch.tensor(slots, dtype=torch.int32))
            inputs.row_generations.zero_()
            inputs.row_generations[:rows].copy_(torch.tensor(epochs))
            inputs.query_lengths[:rows].copy_(torch.tensor(lengths, dtype=torch.int32))
            inputs.live_rows.fill_(rows)
            inputs.token_ids.copy_(torch.tensor(token_ids))
            inputs.page_tables.copy_(page_table)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            starts, expected_tokens, expected_positions, expected_requests, expected_slots = [0], [], [], [], []
            expected_consumed = positions.copy()
            for row, slot in enumerate(slots):
                count = lengths[row] if active[row] else 0
                starts.append(starts[-1] + count)
                expected_consumed[slot] += count
                expected_tokens.extend(token_ids[:count] if mode == "prefill" else token_ids[row:row + count])
                for position in range(positions[slot], positions[slot] + count):
                    expected_positions.append(position)
                    expected_requests.append(row)
                    expected_slots.append(int(page_table[slot, position // 256]) * 256 + position % 256)
            count = len(expected_tokens)
            g, index = metadata.recurrent, metadata.indexed
            assert g.query_start_loc.cpu().tolist() == starts
            assert index.query_start_loc.cpu().tolist() == starts
            assert metadata.token_ids[:count].cpu().tolist() == expected_tokens
            assert not metadata.token_ids[count:].count_nonzero()
            assert index.positions[:count].cpu().tolist() == expected_positions
            assert (index.positions[count:] == -1).all()
            assert index.token_slots[:count].cpu().tolist() == expected_slots
            assert (index.token_slots[count:] == -1).all()
            assert index.requests[:count].cpu().tolist() == expected_requests
            assert g.convolution_slots.cpu().tolist() == [slot + 1 if live else -1 for slot, live in zip(slots, active, strict=True)]
            assert g.initial_state_indices.cpu().tolist() == [slot + 1 if live and positions[slot] > 0 else 0 for slot, live in zip(slots, active, strict=True)]
            assert metadata.sample_indices.cpu().tolist() == [max(end - 1, 0) for end in starts[1:]]
            assert g.num_tokens.item() == count and g.num_seqs.item() == rows
            assert requests.consumed_tokens.cpu().tolist() == expected_consumed
            assert requests.status[3] == (3 if fault not in (None, "partial", "stale") else 1)
            assert requests.error_code[3] == (32 if fault not in (None, "partial", "stale") else 0)
            assert batch.token_state.history is None
    finally:
        worker.close()
        graph.reset()
