"""Immutable GLM continuations and partial-page isolation through native replay."""

import os

import pytest
import torch

from flare.components.state_transfer import PreparedStateTransfer, StateCommands
from flare.preparation.graphs import PreparedGraphBundle
from flare.recipes.glm53.pool import ContinuationPool

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.fixture
def transfers():
    pool = ContinuationPool.allocate(("linear_attention", "indexed_attention"), 2, 8, 5, "cuda:0")
    commands = StateCommands.allocate("cuda:0")
    tables = torch.full((4, 3), -1, dtype=torch.int32, device="cuda:0")
    operation = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=256)
    operation.prepare()
    peer_errors = torch.zeros(4, dtype=torch.int32, device="cuda:0")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.validate()
        torch.maximum(commands.errors, peer_errors, out=commands.errors)
        operation.apply()
    buffers = {"state": pool.backing, "tables": tables, "peer_errors": peer_errors}
    buffers.update({name: getattr(commands, name) for name in commands.__dataclass_fields__})
    buffers.update(dict(pool.partial_pages()))
    bundle = PreparedGraphBundle(device=0, graphs={"state": graph}, buffers=buffers,
                                resources=(pool, commands, operation, tables, peer_errors))
    worker = bundle.start_worker()
    try:
        yield pool, commands, tables, peer_errors, worker
    finally:
        worker.close()
        graph.reset()


def _commands(commands, rows):
    commands.source_slots.fill_(-1)
    commands.destination_slots.fill_(-1)
    commands.expected_consumed.fill_(-1)
    commands.operation.zero_()
    for index, (source, destination, operation, endpoint) in enumerate(rows):
        commands.source_slots[index] = source
        commands.destination_slots[index] = destination
        commands.operation[index] = operation
        commands.expected_consumed[index] = endpoint
    commands.num_ops.fill_(len(rows))


def _replay(worker):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    ticket = worker.submit(0)
    completion = worker.poll(timeout_ms=120000)
    assert completion is not None and completion.ticket == ticket and completion.error is None
    after = torch.cuda.memory_stats()
    for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
        assert before[key] == after[key], key


@pytest.mark.parametrize("endpoint", [1, 3, 4, 255, 256, 257, 511, 512, 513])
def test_exact_snapshot_branch_restore_and_reset(transfers, endpoint):
    pool, commands, tables, _, worker = transfers
    pool.backing[1].random_(0, 256)
    pool.consumed_tokens[1] = endpoint
    source = pool.backing[1].clone()
    tables[0].copy_(torch.tensor([3, 0, 4], dtype=torch.int32))
    tables[1].fill_(2)
    tables[2].fill_(1)
    for _, cache in pool.partial_pages():
        cache.random_(0, 256)
    page = tables[0, endpoint // 256].item() if endpoint % 256 else None
    expected_pages = {name: cache[page].clone() for name, cache in pool.partial_pages()} if page is not None else {}
    _commands(commands, [(1, 5, 0, endpoint)])
    _replay(worker)
    assert not commands.errors.count_nonzero()
    for region in pool.layout.regions:
        expected = expected_pages.get(region.name, source[region.offset:region.offset + region.nbytes])
        actual = pool.backing[5, region.offset:region.offset + region.nbytes]
        assert torch.equal(actual, expected)
    immutable = pool.backing[5].clone()
    pool.backing[1].zero_()
    _commands(commands, [(5, 2, 0, endpoint), (5, 3, 0, endpoint)])
    before_cache = {name: cache.clone() for name, cache in pool.partial_pages()}
    _replay(worker)
    assert not commands.errors.count_nonzero()
    assert torch.equal(pool.backing[2], immutable) and torch.equal(pool.backing[3], immutable)
    assert torch.equal(pool.backing[5], immutable)
    for name, cache in pool.partial_pages():
        expected = before_cache[name]
        if page is not None:
            expected[1].copy_(expected_pages[name])
            expected[2].copy_(expected_pages[name])
        assert torch.equal(cache, expected)
    _commands(commands, [(-1, 3, 1, 0)])
    _replay(worker)
    assert not commands.errors.count_nonzero() and not pool.backing[3].count_nonzero()
    assert torch.equal(pool.backing[2], immutable) and torch.equal(pool.backing[5], immutable)
    assert not pool.backing[[0, 1, 4, 6, 7]].count_nonzero()


@pytest.mark.parametrize("fault", ["operation", "direction", "source", "endpoint", "count", "page",
                                      "duplicate", "read_write", "page_alias", "peer"])
def test_rejected_commands_preserve_every_state_and_cache_byte(transfers, fault):
    pool, commands, tables, peer_errors, worker = transfers
    pool.backing.random_(0, 256)
    pool.consumed_tokens.fill_(257)
    tables.fill_(1)
    _commands(commands, [(5, 1, 0, 257)])
    if fault == "operation":
        commands.operation[0] = 8
    elif fault == "direction":
        commands.destination_slots[0] = 6
    elif fault == "source":
        commands.source_slots[0] = 500
    elif fault == "endpoint":
        commands.expected_consumed[0] = 255
    elif fault == "count":
        commands.num_ops.fill_(5)
    elif fault == "page":
        tables[0, 1] = 5
    elif fault == "duplicate":
        _commands(commands, [(5, 1, 0, 257), (6, 1, 0, 257), (7, 1, 0, 257)])
    elif fault == "read_write":
        _commands(commands, [(5, 1, 0, 257), (1, 6, 0, 257)])
    elif fault == "page_alias":
        _commands(commands, [(5, 1, 0, 257), (6, 2, 0, 257)])
    elif fault == "peer":
        peer_errors[0] = 7
    before = pool.backing.clone()
    caches = {name: cache.clone() for name, cache in pool.partial_pages()}
    _replay(worker)
    assert (commands.errors[:commands.num_ops.clamp(max=4).item()] != 0).all()
    assert torch.equal(pool.backing, before)
    for name, cache in pool.partial_pages():
        assert torch.equal(cache, caches[name])
