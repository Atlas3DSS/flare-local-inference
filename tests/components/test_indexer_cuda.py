"""Tuned sparse index selection preserves caller storage through native replay."""

import os

import pytest
import torch
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference

from ignition.components.indexer import PreparedSparseIndexer
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


def _assert_selection(actual, actual_scores, expected, expected_scores):
    actual_ids, actual_order = actual.sort()
    expected_ids, expected_order = expected.sort()
    actual_scores = actual_scores.gather(1, actual_order)
    expected_scores = expected_scores.gather(1, expected_order)
    same = actual_ids == expected_ids
    torch.testing.assert_close(actual_scores[same], expected_scores[same], rtol=0, atol=0)
    for row in (~same).any(1).nonzero().flatten().tolist():
        a = dict(zip(actual_ids[row].tolist(), actual_scores[row].tolist()))
        b = dict(zip(expected_ids[row].tolist(), expected_scores[row].tolist()))
        assert len(a) == len(b)
        for index in a.keys() & b.keys():
            assert a[index] == b[index]
        cutoff = min(value for index, value in b.items() if index >= 0)
        assert all(a[index] == cutoff for index in a.keys() - b.keys())
        assert all(b[index] == cutoff for index in b.keys() - a.keys())


@pytest.mark.parametrize("mode,rows", [("decode", 2), ("prefill", 4096)])
@torch.no_grad()
def test_tuned_sparse_indexer_native_replay(mode, rows):
    device = torch.device("cuda:0")
    pages, heads, topk = 32, 32, 512
    query = torch.randn((rows, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, heads), device=device)
    cache = pack_index_k_cache_reference(torch.randn((pages * 64, 128), device=device))
    table = torch.randperm(pages, device=device, dtype=torch.int32).unsqueeze(0).expand(rows, -1)
    if mode == "decode":
        table = table.contiguous()
    lengths = torch.full((rows,), pages * 64, dtype=torch.int32, device=device)
    width = torch.tensor([pages * 64], dtype=torch.int32, device=device)
    output = torch.empty((rows, topk), dtype=torch.int32, device=device)
    scores = torch.empty((rows, topk), device=device)
    observed, observed_scores = torch.empty_like(output), torch.empty_like(scores)
    arena = ScratchArena(device)
    worker = None
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint="sparse-indexer-component",
            workload_schema="glm-indexer-v1", numerical_policy="default") as session:
        try:
            session.register_scratch(arena)
            caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=rows,
                max_page_table_width=pages, topk=topk, mode=mode)
            indexer = PreparedSparseIndexer(query, weights, cache, table, lengths, width, output,
                caps=caps, session=session, name="indexer", workspace=arena, scores=scores)
            old_storage = arena.tensor
            arena.reserve(arena.nbytes + 1024)
            saved = tuple(tensor.clone() for tensor in (query, weights, cache, table, lengths, width))
            session.tune()
            for actual, expected in zip((query, weights, cache, table, lengths, width), saved, strict=True):
                torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
            assert indexer.binding.runtime.scratch.shared_scratch.data_ptr() == arena.tensor.data_ptr()
            assert arena.tensor.data_ptr() != old_storage.data_ptr()
            del old_storage, saved

            def forward():
                observed.copy_(indexer())
                observed_scores.copy_(scores)
                arena.tensor.fill_(255)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={mode: graph},
                buffers={"query": query, "cache": cache, "output": observed},
                resources=(indexer, compiled, arena, session))
            worker = bundle.start_worker()
            for count in (256, 1024, pages * 64):
                query.copy_(torch.randn(query.shape, device=device).to(query.dtype))
                weights.normal_()
                cache.copy_(pack_index_k_cache_reference(torch.randn((pages * 64, 128), device=device)))
                lengths.copy_((torch.arange(rows, device=device, dtype=torch.int32) // 4 + count).clamp_max(pages * 64))
                lengths[0] = 0
                forward()
                expected, expected_scores = observed.clone(), observed_scores.clone()
                observed.fill_(-42)
                observed_scores.fill_(42)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                _assert_selection(observed, observed_scores, expected, expected_scores)
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
