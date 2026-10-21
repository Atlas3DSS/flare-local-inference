"""Exact native-offset dense draft checkpoints through captured maintenance."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.sampling import DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.recipes.qwen38_dense.pool import ContinuationPool, continuation_layout, page_specs
from ignition.recipes.qwen38_dense.speculative import PreparedDenseStateTransfer


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


def _commands(commands, entries):
    commands.request_slots.fill_(-1)
    commands.source_slots.fill_(-1)
    commands.destination_slots.fill_(-1)
    commands.expected_consumed.fill_(-1)
    commands.operation.zero_()
    for index, (slot, source, destination, endpoint) in enumerate(entries):
        commands.request_slots[index] = slot
        commands.source_slots[index] = source
        commands.destination_slots[index] = destination
        commands.expected_consumed[index] = endpoint
    commands.num_ops.fill_(len(entries))


def _replay(graph):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
        assert before[key] == after[key], key


@pytest.fixture
def transfer():
    assert torch.cuda.get_device_capability() == (7, 0)
    config = SimpleNamespace(layers=("full_attention",), hidden=16,
        text={"num_key_value_heads": 1, "head_dim": 8})
    layout = continuation_layout(config, dtype=torch.float32, element_bytes=2, draft_tokens=3)
    pool = ContinuationPool.allocate(layout, page_specs(config, element_bytes=2, mtp=True), 8, 16, "cuda")
    commands = StateCommands.allocate("cuda")
    tables = torch.tensor([[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]], dtype=torch.int32, device="cuda")
    draft = tables + 1
    states = torch.arange(1, 5, device="cuda")[:, None].expand(4, 8).contiguous()
    requests = DeviceRequestState.allocate("cuda", vocab_size=128, consumed_tokens=pool.region("consumed_tokens")[1:5, 0])
    operation = PreparedDenseStateTransfer(pool, commands, tables,
        pages=((name, cache) for name, cache in pool.pages.items() if not name.startswith("draft.")),
        draft_tables=draft, page_tokens=128, state_rows=states, requests=requests)
    operation.prepare()
    for compiled in (operation._draft_validate.compiled, operation._draft_copy.compiled):
        assert compiled.asm["cubin"]
        assert ".target sm_70" in compiled.asm["ptx"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.validate()
        operation.apply()
    try:
        yield pool, commands, tables, draft, graph
    finally:
        graph.reset()


@pytest.mark.parametrize("endpoint", [1, 2, 127, 128, 129, 130, 255, 256, 257, 258])
def test_shifted_checkpoint_snapshot_restore_exact_bytes(transfer, endpoint):
    pool, commands, tables, draft, graph = transfer
    pool.backing[1].random_(0, 256)
    pool.region("consumed_tokens")[1] = endpoint
    for cache in pool.pages.values():
        cache.random_(0, 256)
    row = pool.backing[1].clone()
    expected_backup = {}
    n, remainder = divmod(endpoint, 128)
    for name, cache in pool.pages.items():
        if not name.startswith("draft."):
            if remainder:
                expected_backup[name] = cache[int(tables[0, n])].clone()
        else:
            if endpoint > 1 and n > 0:
                expected_backup[name] = cache[int(draft[0, n - 1])].clone()
            if endpoint > 1 and remainder >= 2:
                expected_backup[name + ".extra"] = cache[int(draft[0, n])].clone()
    _commands(commands, [(0, 1, 5, endpoint)])
    _replay(graph)
    assert not commands.errors.count_nonzero()
    for region in pool.layout.regions:
        expected = expected_backup.get(region.name, row[region.offset:region.offset + region.nbytes])
        assert torch.equal(pool.backing[5, region.offset:region.offset + region.nbytes], expected), region.name
    immutable = pool.backing[5].clone()

    # Match native prefix sharing: only N physical target pages are shared;
    # draft logical j occupies physical allocation j+1.
    allocation = list(range(n)) + list(range(8, 12))
    tables[1].copy_(torch.tensor(allocation[:3], device="cuda"))
    draft[1].copy_(torch.tensor(allocation[1:4], device="cuda"))
    for cache in pool.pages.values():
        cache[8:12].fill_(173)
    expected_cache = {name: cache.clone() for name, cache in pool.pages.items()}
    for name, expected in expected_cache.items():
        if name.startswith("draft."):
            if endpoint > 1 and n > 0:
                expected[int(draft[1, n - 1])].copy_(expected_backup[name])
            if endpoint > 1 and remainder >= 2:
                expected[int(draft[1, n])].copy_(expected_backup[name + ".extra"])
        elif remainder:
            expected[int(tables[1, n])].copy_(expected_backup[name])
    _commands(commands, [(1, 5, 2, endpoint)])
    _replay(graph)
    assert not commands.errors.count_nonzero()
    assert torch.equal(pool.backing[2], immutable)
    assert torch.equal(pool.backing[5], immutable)
    for name, cache in pool.pages.items():
        assert torch.equal(cache, expected_cache[name]), name

    # Replay remains allocation-free and leaves every physical byte alone with
    # zero commands, including the immutable snapshot and inactive live rows.
    before = pool.backing.clone()
    commands.num_ops.zero_()
    _replay(graph)
    assert torch.equal(pool.backing, before)
    for name, cache in pool.pages.items():
        assert torch.equal(cache, expected_cache[name]), name


@pytest.mark.parametrize("fault", ["missing_full", "missing_partial", "out_of_range", "same_request_alias", "cross_part_alias"])
def test_invalid_draft_pages_reject_all_writes(transfer, fault):
    pool, commands, tables, draft, graph = transfer
    endpoint = 129 if fault == "missing_full" else 130
    pool.backing.random_(0, 256)
    pool.region("consumed_tokens").fill_(endpoint)
    for cache in pool.pages.values():
        cache.random_(0, 256)
    _commands(commands, [(0, 5, 1, endpoint)])
    if fault == "missing_full":
        draft[0, 0] = -1
    elif fault == "missing_partial":
        draft[0, 1] = -1
    elif fault == "out_of_range":
        draft[0, 0] = 16
    elif fault == "same_request_alias":
        draft[0, 1] = draft[0, 0]
    else:
        _commands(commands, [(0, 5, 1, endpoint), (1, 6, 2, endpoint)])
        draft[1, 1] = draft[0, 0]
    before = pool.backing.clone()
    caches = {name: cache.clone() for name, cache in pool.pages.items()}
    _replay(graph)
    assert (commands.errors[:int(commands.num_ops)] != 0).all()
    assert torch.equal(pool.backing, before)
    for name, cache in pool.pages.items():
        assert torch.equal(cache, caches[name]), name
