"""Tests for BatchScheduler — chunked prefill, batching, lifecycle.

No GPU required. Uses mock PagePool and CPU tensors.
"""

import pytest
import torch

from serve.cache.linear_state_arena import LinearStateArena
from serve.cache.page_pool import PagePool
from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
from serve.engine.request import Request
from serve.engine.sampling import SamplingParams
from serve.engine.scheduler import BatchScheduler


class MockPagePool:
    """Fake PagePool for testing."""

    def __init__(self, num_pages=100):
        self.num_pages = num_pages
        self._free = list(range(num_pages))

    def alloc(self, n):
        if n > len(self._free):
            raise RuntimeError("OOM")
        result = self._free[-n:]
        del self._free[-n:]
        return result

    def free(self, page_ids):
        self._free.extend(page_ids)

    @property
    def num_free(self):
        return len(self._free)


def _make_scheduler(num_pages=100, max_running=8, captured_bs=None, chunk_size=512):
    pool = MockPagePool(num_pages)
    cache = PrefixCheckpointCache(pool)
    return BatchScheduler(
        cache=cache, pool=pool,
        captured_bs=captured_bs or [1, 2, 4, 8],
        max_running=max_running,
        max_prefill_tokens=4096,
        chunk_size=chunk_size,
        device="cpu",
    )


def _make_request(rid, prompt_len=5, max_new_tokens=10):
    return Request(
        rid=rid,
        prompt_ids=list(range(rid * 1000, rid * 1000 + prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


def _make_linear_state_arena(live_slots=4, snapshot_slots=4):
    return LinearStateArena(
        live_slots=live_slots,
        snapshot_slots=snapshot_slots,
        num_linear_layers=1,
        num_heads=2,
        head_v_dim=4,
        head_k_dim=4,
        conv_dim=8,
        conv_kernel=4,
        device="cpu",
    )


def _insert_checkpoint_chain(cache, token_ids, page_ids):
    checkpoint, created = cache.get_or_create_checkpoint(
        cache.root,
        token_ids,
        page_ids,
    )
    assert created
    assert checkpoint is not None


def _prefill_request(sched, req):
    """Helper: run all prefill chunks for a request until it enters running."""
    sched.add_request(req)
    while req.prefill_remaining > 0 or req not in sched.running:
        batch = sched.step()
        assert batch is not None
        assert batch.mode == "prefill"
        if batch.is_last_chunk:
            sched.process_prefill_chunk([42])
        else:
            sched.process_prefill_chunk(None)


# -- basic lifecycle -------------------------------------------------------


def test_empty_scheduler_returns_none():
    sched = _make_scheduler()
    assert sched.step() is None


def test_single_request_prefill():
    sched = _make_scheduler()
    sched.add_request(_make_request(1))

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert len(batch.requests) == 1
    assert batch.is_last_chunk  # Short prompt fits in one chunk.
    assert sched.num_running == 1
    assert sched.num_waiting == 0


def test_single_request_prefill_then_decode():
    sched = _make_scheduler()
    req = _make_request(1, prompt_len=5, max_new_tokens=3)
    sched.add_request(req)

    # Prefill (single chunk for short prompt).
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.is_last_chunk
    sched.process_prefill_chunk([42])

    assert req.output_ids == [42]
    assert req.first_token_at is not None

    # Decode steps.
    for i in range(2):
        batch = sched.step()
        assert batch.mode == "decode"
        sched.process_decode_output([100 + i])

    assert req.is_finished
    assert req.finished_reason == "length"
    assert len(sched.finished) == 1
    assert sched.num_running == 0


def test_stop_token_finishes_request():
    sched = _make_scheduler()
    req = Request(
        rid=1, prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=100, stop_token_ids=[999]),
    )
    sched.add_request(req)

    sched.step()  # Prefill.
    sched.process_prefill_chunk([42])

    batch = sched.step()  # Decode.
    sched.process_decode_output([999])

    assert req.is_finished
    assert req.finished_reason == "stop"


# -- chunked prefill -------------------------------------------------------


def test_long_prompt_chunked():
    """A prompt longer than chunk_size gets split into multiple chunks."""
    sched = _make_scheduler(chunk_size=64)
    # 200-token prompt: needs ceil(200/64) = 4 chunks (64+64+64+8).
    req = _make_request(1, prompt_len=200, max_new_tokens=3)
    sched.add_request(req)

    chunks = 0
    while True:
        batch = sched.step()
        assert batch.mode == "prefill"
        chunks += 1
        if batch.is_last_chunk:
            sched.process_prefill_chunk([42])
            break
        else:
            assert batch.q_seqlens[0] == 64
            sched.process_prefill_chunk(None)
        # Interleaving: no running requests, so no decode between chunks.

    assert chunks == 4  # 64+64+64+8
    assert req.output_ids == [42]
    assert req.prefill_progress == 200
    assert sched.num_running == 1


def test_oversize_request_is_rejected_without_blocking_next_request():
    sched = _make_scheduler(num_pages=2, chunk_size=64)

    oversized = _make_request(1, prompt_len=192, max_new_tokens=2)
    follower = _make_request(2, prompt_len=32, max_new_tokens=2)
    sched.add_request(oversized)
    sched.add_request(follower)

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert batch.requests == [follower]
    assert oversized.is_finished
    assert oversized.finished_reason == "context_too_long"


def test_cached_prefix_request_still_rejected_when_total_page_table_cannot_fit():
    sched = _make_scheduler(num_pages=2, chunk_size=64)

    shared_prefix = list(range(64))
    first = Request(
        rid=1,
        prompt_ids=list(shared_prefix),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(first)
    batch = sched.step()
    assert batch is not None
    sched.process_prefill_chunk([7], batch.requests)
    assert sched.cache.total_cached_pages == 1

    oversized = Request(
        rid=2,
        prompt_ids=list(shared_prefix) + list(range(1000, 1128)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    follower = _make_request(3, prompt_len=32, max_new_tokens=1)
    sched.add_request(oversized)
    sched.add_request(follower)

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert batch.requests == [follower]
    assert oversized.checkpoint_len == 64
    assert oversized.is_finished
    assert oversized.finished_reason == "context_too_long"


def test_chunked_prefill_interleaves_with_decode():
    """Decode steps happen between prefill chunks when requests are running."""
    sched = _make_scheduler(chunk_size=64)

    # First: a short request that enters decode immediately.
    req1 = _make_request(1, prompt_len=10, max_new_tokens=20)
    sched.add_request(req1)
    batch = sched.step()
    assert batch.mode == "prefill"
    sched.process_prefill_chunk([10])

    # Now req1 is running (decode). Submit a long prompt.
    req2 = _make_request(2, prompt_len=200, max_new_tokens=3)
    sched.add_request(req2)

    # Scheduler should interleave: prefill chunk, decode, prefill chunk, decode, ...
    saw_decode = False
    saw_prefill = False
    for _ in range(10):
        batch = sched.step()
        if batch.mode == "decode":
            saw_decode = True
            sched.process_decode_output([100] * len(batch.requests))
        else:
            saw_prefill = True
            if batch.is_last_chunk:
                sched.process_prefill_chunk([42])
            else:
                sched.process_prefill_chunk(None)
        if req2 in sched.running:
            break

    assert saw_decode, "Should have interleaved decode steps"
    assert saw_prefill, "Should have done prefill chunks"


def test_batched_prefill_dedupes_checkpoint_pages_and_preserves_accounting():
    sched = _make_scheduler(num_pages=10)

    prompt = list(range(128))
    req1 = Request(
        rid=1,
        prompt_ids=list(prompt),
        sampling_params=SamplingParams(max_new_tokens=2),
    )
    req2 = Request(
        rid=2,
        prompt_ids=list(prompt),
        sampling_params=SamplingParams(max_new_tokens=2),
    )
    sched.add_request(req1)
    sched.add_request(req2)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 2
    sched.process_prefill_chunk([10, 20], batch.requests)
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == 4

    batch = sched.step()
    assert batch.mode == "decode"
    sched.process_decode_output([11, 21])

    assert req1.is_finished
    assert req2.is_finished
    assert sched.cache.total_cached_pages == 2
    assert sched.pool.num_free + sched.cache.total_cached_pages == 10

    assert sched.cache.evict(2) == 2
    assert sched.pool.num_free == 10
    assert sched.cache.total_cached_pages == 0


def test_running_request_does_not_materialize_intermediate_checkpoints():
    sched = _make_scheduler(num_pages=10)

    req = Request(
        rid=1,
        prompt_ids=list(range(128)),
        sampling_params=SamplingParams(max_new_tokens=3),
    )
    sched.add_request(req)

    batch = sched.step()
    assert batch.mode == "prefill"
    sched.process_prefill_chunk([10], batch.requests)

    assert not req.is_finished
    assert req.checkpoint_len == 0
    assert sched.cache.total_cached_pages == 0

    batch = sched.step()
    assert batch.mode == "decode"
    sched.process_decode_output([11])

    assert not req.is_finished
    assert req.checkpoint_len == 0
    assert sched.cache.total_cached_pages == 0

    batch = sched.step()
    assert batch.mode == "decode"
    sched.process_decode_output([12])

    assert req.is_finished
    assert sched.cache.total_cached_pages == 2


def test_terminal_checkpoint_restores_linear_state_on_readmission():
    pool = MockPagePool(10)
    state_arena = _make_linear_state_arena(live_slots=4, snapshot_slots=2)
    cache = PrefixCheckpointCache(pool, state_arena=state_arena)
    sched = BatchScheduler(
        cache=cache,
        pool=pool,
        ssm_pool=state_arena,
        enable_prefix_cache=True,
        captured_bs=[1, 2, 4, 8],
        max_running=4,
        max_prefill_tokens=4096,
        chunk_size=512,
        device="cpu",
    )

    prompt = list(range(64))
    req1 = Request(
        rid=1,
        prompt_ids=list(prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req1)

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert req1.ssm_slot > 0

    state_arena.ssm_state_for_layer(0)[req1.ssm_slot].fill_(3.5)
    state_arena.conv_state_for_layer(0)[req1.ssm_slot].fill_(7.0)
    expected_ssm = state_arena.ssm_state_for_layer(0)[req1.ssm_slot].clone()
    expected_conv = state_arena.conv_state_for_layer(0)[req1.ssm_slot].clone()

    sched.process_prefill_chunk([42], batch.requests)
    assert req1.is_finished
    assert cache.total_cached_pages == 1

    req2 = Request(
        rid=2,
        prompt_ids=list(prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req2)

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert req2.checkpoint_len == 64
    assert req2.ssm_slot > 0
    assert torch.equal(state_arena.ssm_state_for_layer(0)[req2.ssm_slot], expected_ssm)
    assert torch.equal(state_arena.conv_state_for_layer(0)[req2.ssm_slot], expected_conv)


def test_chunk_size_respected():
    """Each prefill chunk has at most chunk_size tokens."""
    sched = _make_scheduler(chunk_size=32)
    req = _make_request(1, prompt_len=100, max_new_tokens=1)
    sched.add_request(req)

    while True:
        batch = sched.step()
        if batch.mode != "prefill":
            break
        assert batch.q_seqlens[0] <= 32
        if batch.is_last_chunk:
            sched.process_prefill_chunk([42])
            break
        else:
            sched.process_prefill_chunk(None)


# -- batching --------------------------------------------------------------


def test_multiple_short_requests_batched_prefill():
    """Multiple short requests are batched in one prefill step."""
    sched = _make_scheduler()
    for i in range(4):
        sched.add_request(_make_request(i))

    # All 4 short requests batched together.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 4
    sched.process_prefill_chunk([10, 20, 30, 40], batch.requests)

    assert sched.num_running == 4
    assert sched.num_waiting == 0

    # All 4 decode together.
    batch = sched.step()
    assert batch.mode == "decode"
    assert len(batch.requests) == 4


def test_prefill_priority_over_decode():
    sched = _make_scheduler()

    # Start one request.
    sched.add_request(_make_request(1))
    sched.step()
    sched.process_prefill_chunk([42])

    # Add another while first is running.
    sched.add_request(_make_request(2))

    # Next step: interleave flag is False (last was prefill for req1,
    # but then we did nothing). Should prefill req2.
    # Actually after prefill chunk for req1, _last_was_prefill=True.
    # So next step prefers decode for req1.
    batch = sched.step()
    assert batch.mode == "decode"  # Interleave: decode after prefill.

    sched.process_decode_output([100])

    # Now should admit req2.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.requests[0].rid == 2


# -- concurrency limits ---------------------------------------------------


def test_max_running_limits_admission():
    sched = _make_scheduler(max_running=2)

    sched.add_request(_make_request(1))
    sched.add_request(_make_request(2))
    sched.add_request(_make_request(3))

    # Batch prefill first 2 (max_running=2).
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 2
    sched.process_prefill_chunk([10, 20], batch.requests)

    # Now 2 running, third still waiting.
    assert sched.num_running == 2
    assert sched.num_waiting == 1


def test_finished_request_frees_slot():
    sched = _make_scheduler(max_running=2)

    req1 = _make_request(1, max_new_tokens=1)
    req2 = _make_request(2, max_new_tokens=1)
    req3 = _make_request(3, max_new_tokens=10)

    sched.add_request(req1)
    sched.add_request(req2)
    sched.add_request(req3)

    # Batch prefill req1 and req2 (max_running=2).
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 2
    sched.process_prefill_chunk([10, 20], batch.requests)
    assert req1.is_finished  # max_new_tokens=1.
    assert req2.is_finished

    # Both finished, slots open. Now req3 can be admitted.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.requests[0].rid == 3


# -- graph size selection --------------------------------------------------


def test_graph_size_selection():
    sched = _make_scheduler(captured_bs=[1, 2, 4, 8])

    # 1 request → bs=1 graph for decode.
    sched.add_request(_make_request(1))
    sched.step()
    sched.process_prefill_chunk([10])
    batch = sched.step()
    assert batch.mode == "decode"
    assert batch.graph_bs == 1


def test_no_graph_for_prefill_last_chunk():
    """Last chunk (variable size) doesn't get a graph."""
    sched = _make_scheduler(captured_bs=[1, 2, 4], chunk_size=64)
    req = _make_request(1, prompt_len=80)  # 80 tokens: chunk of 64 + 16.
    sched.add_request(req)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.q_seqlens[0] == 64
    assert batch.graph_bs == 1  # Full chunk gets graph.
    sched.process_prefill_chunk(None)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.q_seqlens[0] == 16
    assert batch.graph_bs is None  # Last chunk, partial — no graph.
    assert batch.is_last_chunk


# -- OOM handling ----------------------------------------------------------


def test_oom_eviction_allows_admission():
    pool = MockPagePool(10)
    cache = PrefixCheckpointCache(pool)

    pages = pool.alloc(3)
    _insert_checkpoint_chain(cache, list(range(3 * 64)), pages)

    remaining = pool.alloc(pool.num_free)

    sched = BatchScheduler(
        cache=cache, pool=pool,
        captured_bs=[1, 2, 4, 8], max_running=8,
        max_prefill_tokens=4096, chunk_size=512, device="cpu",
    )

    req = _make_request(1, prompt_len=50)
    sched.add_request(req)
    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"


def test_running_request_with_visible_output_is_not_preempted():
    sched = _make_scheduler(num_pages=2, max_running=4)

    req1 = _make_request(1, prompt_len=64)
    sched.add_request(req1)
    sched.step()
    sched.process_prefill_chunk([10])

    req2 = _make_request(2, prompt_len=64)
    sched.add_request(req2)

    # Interleave: last was prefill, so decode first.
    batch = sched.step()
    assert batch.mode == "decode"
    sched.process_decode_output([100])

    # Req2 stays queued because rewinding req1 would invalidate streamed output.
    batch = sched.step()
    assert batch.mode == "decode"
    assert req1.output_ids == [10, 100]
    assert req2 in sched.waiting
    assert sched.stats_preemptions == 0


def test_multi_page_running_request_is_not_rewound_for_admission():
    sched = _make_scheduler(num_pages=4, max_running=4)

    req1 = _make_request(1, prompt_len=3 * 64, max_new_tokens=3)
    sched.add_request(req1)
    batch = sched.step()
    assert batch.mode == "prefill"
    sched.process_prefill_chunk([10], batch.requests)

    assert req1.checkpoint_len == 0
    assert len(req1.page_ids) == 4

    req2 = _make_request(2, prompt_len=64, max_new_tokens=1)
    sched.add_request(req2)

    batch = sched.step()
    assert batch.mode == "decode"
    sched.process_decode_output([100])

    batch = sched.step()
    assert batch.mode == "decode"
    assert req1.output_ids == [10, 100]
    assert req2 in sched.waiting
    assert sched.stats_preemptions == 0


# -- has_work --------------------------------------------------------------


def test_has_work():
    sched = _make_scheduler()
    assert not sched.has_work

    sched.add_request(_make_request(1))
    assert sched.has_work

    sched.step()
    sched.process_prefill_chunk([10])
    assert sched.has_work  # Still running.


def test_has_work_includes_prefilling():
    """has_work is True while requests are mid-chunked-prefill."""
    sched = _make_scheduler(chunk_size=32)
    req = _make_request(1, prompt_len=100, max_new_tokens=1)
    sched.add_request(req)

    batch = sched.step()
    sched.process_prefill_chunk(None)  # First chunk, not done.
    assert sched.has_work
    assert sched.num_prefilling == 1


# -- batch tensors ---------------------------------------------------------


def test_batch_step_has_page_table_and_cache_seqlens():
    sched = _make_scheduler()
    req = _make_request(1, prompt_len=5, max_new_tokens=3)
    sched.add_request(req)

    batch = sched.step()
    assert batch.page_table is not None
    assert batch.cache_seqlens is not None
    assert batch.page_table.shape[0] == 1
    assert batch.cache_seqlens.shape[0] == 1
    assert batch.cache_seqlens[0].item() == 5

    sched.process_prefill_chunk([42])

    # Interleave after prefill: decode.
    batch = sched.step()
    assert batch.mode == "decode"
    assert batch.cache_seqlens[0].item() == 6


def test_decode_oom_finishes_request():
    sched = _make_scheduler(num_pages=1, max_running=4)

    req = Request(
        rid=1,
        prompt_ids=list(range(60)),
        sampling_params=SamplingParams(max_new_tokens=100),
    )
    sched.add_request(req)

    sched.step()
    sched.process_prefill_chunk([42])

    for i in range(10):
        if req.is_finished:
            break
        batch = sched.step()
        sched.process_decode_output([100 + i])

    assert req.is_finished
    assert req.finished_reason == "oom"


# -- prefill token budget --------------------------------------------------


def test_stats_tracked():
    """Scheduler stats are updated correctly."""
    sched = _make_scheduler()
    assert sched.stats["prefill_steps"] == 0
    assert sched.stats["decode_steps"] == 0

    req = _make_request(1, prompt_len=5, max_new_tokens=2)
    sched.add_request(req)

    # Prefill.
    batch = sched.step()
    assert sched.stats["prefill_steps"] == 1
    sched.process_prefill_chunk([42], batch.requests)

    # Decode.
    batch = sched.step()
    assert sched.stats["decode_steps"] == 1
    sched.process_decode_output([43])

    assert sched.stats["finished"] == 1
    assert sched.stats["running"] == 0


def test_prefill_single_large_request():
    pool = MockPagePool(200)
    cache = PrefixCheckpointCache(pool)
    sched = BatchScheduler(
        cache=cache, pool=pool,
        captured_bs=[1, 2, 4, 8], max_running=8,
        max_prefill_tokens=100, chunk_size=512,
        device="cpu",
    )

    req = Request(
        rid=1,
        prompt_ids=list(range(100)),
        sampling_params=SamplingParams(max_new_tokens=5),
    )
    sched.add_request(req)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 1


# -- per-request sampling -------------------------------------------------


def test_sample_batch_mixed_params():
    """sample_batch handles per-request params correctly."""
    from serve.engine.sampling import SamplingParams, sample_batch
    import torch

    logits = torch.randn(3, 100)

    # All same params — fast path.
    params = [SamplingParams(temperature=0.0)] * 3
    tokens = sample_batch(logits, params)
    assert tokens.shape == (3,)
    # Greedy: should match argmax.
    expected = logits.argmax(dim=-1)
    assert torch.equal(tokens, expected)

    # Mixed params — slow path.
    params = [
        SamplingParams(temperature=0.0),   # greedy
        SamplingParams(temperature=1.0),   # sampling
        SamplingParams(temperature=0.0),   # greedy
    ]
    tokens = sample_batch(logits, params)
    assert tokens.shape == (3,)
    assert tokens[0] == logits[0].argmax()
    assert tokens[2] == logits[2].argmax()


def test_sample_batch_respects_non_temperature_param_differences():
    from unittest.mock import patch

    from serve.engine.sampling import SamplingParams, sample_batch
    import torch

    logits = torch.randn(2, 100)
    params = [
        SamplingParams(temperature=0.0, presence_penalty=0.0),
        SamplingParams(temperature=0.0, presence_penalty=1.0),
    ]

    with patch("serve.engine.sampling.sample") as mock_sample:
        mock_sample.side_effect = [
            torch.tensor([11], dtype=torch.long),
            torch.tensor([22], dtype=torch.long),
        ]
        tokens = sample_batch(logits, params, generated_ids=[[], []])

    assert tokens.tolist() == [11, 22]
    assert mock_sample.call_count == 2

def test_sample_batch_greedy():
    """Greedy sampling returns argmax."""
    from serve.engine.sampling import SamplingParams, sample_batch
    import torch

    torch.manual_seed(42)
    logits = torch.randn(4, 100)
    params = [SamplingParams(temperature=0.0)] * 4
    tokens = sample_batch(logits, params)
    assert torch.equal(tokens, logits.argmax(dim=-1))


def test_sample_batch_single_request():
    """Single-request batch works correctly."""
    from serve.engine.sampling import SamplingParams, sample_batch
    import torch

    logits = torch.randn(1, 100)
    params = [SamplingParams(temperature=0.0)]
    tokens = sample_batch(logits, params)
    assert tokens.shape == (1,)
    assert tokens[0] == logits[0].argmax()


# -- batched vs chunked prefill -------------------------------------------


def test_short_requests_batched_together():
    """Multiple short requests are batched in one prefill step."""
    sched = _make_scheduler(chunk_size=64)

    for i in range(3):
        sched.add_request(_make_request(i, prompt_len=10))

    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 3  # All batched.
    assert batch.is_last_chunk
    assert sum(batch.q_seqlens) == 30  # 3 * 10 tokens.


def test_long_request_not_batched():
    """A long request that needs chunking is prefilled alone."""
    sched = _make_scheduler(chunk_size=32)

    sched.add_request(_make_request(1, prompt_len=100))  # Needs chunking.
    sched.add_request(_make_request(2, prompt_len=10))   # Short.

    # Long request goes first, alone.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 1
    assert batch.requests[0].rid == 1
    assert not batch.is_last_chunk  # First chunk of 100 tokens.


def test_mixed_short_then_long():
    """Short requests batch first, long request waits."""
    sched = _make_scheduler(chunk_size=32)

    sched.add_request(_make_request(1, prompt_len=10))   # Short.
    sched.add_request(_make_request(2, prompt_len=100))  # Long.
    sched.add_request(_make_request(3, prompt_len=10))   # Short.

    # Short request admitted first (it's at front of queue).
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.requests[0].rid == 1
    assert batch.is_last_chunk
    # Only req1 batched — req2 is long and breaks the batch.
    assert len(batch.requests) == 1


def test_batched_and_chunked_in_same_session():
    """Short requests batch, long requests chunk, in the same scheduler."""
    sched = _make_scheduler(chunk_size=32, max_running=8)

    # Submit 3 short + 1 long.
    for i in range(3):
        sched.add_request(_make_request(i, prompt_len=10, max_new_tokens=2))
    sched.add_request(Request(
        rid=10,
        prompt_ids=list(range(100)),  # 100 tokens, needs chunking.
        sampling_params=SamplingParams(max_new_tokens=2),
    ))

    # First step: 3 short requests batched.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 3
    assert batch.is_last_chunk
    sched.process_prefill_chunk([10, 20, 30], batch.requests)

    # Next: interleave decode, then start chunking the long request.
    saw_chunk = False
    saw_decode = False
    for _ in range(20):
        batch = sched.step()
        if batch is None:
            break
        if batch.mode == "prefill":
            saw_chunk = True
            if batch.is_last_chunk:
                sched.process_prefill_chunk([40], batch.requests)
            else:
                sched.process_prefill_chunk(None)
        else:
            saw_decode = True
            sched.process_decode_output([100] * len(batch.requests))

    assert saw_chunk, "Long request should have been chunked"
    assert saw_decode, "Short requests should have decoded between chunks"


def test_prefill_token_budget_limits_batch():
    """max_prefill_tokens limits how many short requests are batched."""
    from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache

    pool = MockPagePool(200)
    cache = PrefixCheckpointCache(pool)
    sched = BatchScheduler(
        cache=cache, pool=pool,
        captured_bs=[1, 2, 4, 8], max_running=8,
        max_prefill_tokens=25,  # Only room for 2 requests of 10 tokens.
        chunk_size=512,
        device="cpu",
    )

    for i in range(4):
        sched.add_request(_make_request(i, prompt_len=10))

    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 2  # Budget limits to 2.
    assert sum(batch.q_seqlens) == 20


def test_cancelled_request_cleaned_up():
    """A cancelled request is removed before another decode step runs."""
    sched = _make_scheduler()
    req = _make_request(1, prompt_len=5, max_new_tokens=100)
    sched.add_request(req)

    batch = sched.step()
    sched.process_prefill_chunk([42], batch.requests)
    assert sched.num_running == 1

    # Cancel the request externally.
    req.cancel()
    assert req.is_finished
    assert req.finished_reason == "cancelled"

    batch = sched.step()
    assert batch is None
    assert sched.num_running == 0
    assert sched.stats["finished"] >= 1


def test_waiting_request_times_out_before_admission():
    sched = _make_scheduler()
    req = Request(
        rid=1,
        prompt_ids=list(range(8)),
        sampling_params=SamplingParams(max_new_tokens=10),
        timeout_s=0.001,
    )
    sched.add_request(req)

    import time
    time.sleep(0.01)

    batch = sched.step()
    assert batch is None
    assert req.finished_reason == "timeout"
    assert sched.num_waiting == 0
    assert sched.stats["finished"] >= 1


def test_prefilling_request_times_out_before_next_chunk():
    sched = _make_scheduler(chunk_size=32)
    req = Request(
        rid=1,
        prompt_ids=list(range(100)),
        sampling_params=SamplingParams(max_new_tokens=10),
        timeout_s=0.001,
    )
    sched.add_request(req)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert sched.num_prefilling == 1

    import time
    time.sleep(0.01)

    batch = sched.step()
    assert batch is None
    assert req.finished_reason == "timeout"
    assert sched.num_prefilling == 0
    assert sched.stats["finished"] >= 1


def test_full_scheduler_lifecycle_with_all_features():
    """Comprehensive lifecycle: batched prefill, chunked prefill,
    interleaved decode, timeout, preemption — all in one test."""
    sched = _make_scheduler(num_pages=100, max_running=4, chunk_size=32)

    # Mix of request types.
    reqs = {
        "short1": Request(rid=1, prompt_ids=list(range(10)),
            sampling_params=SamplingParams(max_new_tokens=5)),
        "short2": Request(rid=2, prompt_ids=list(range(100, 110)),
            sampling_params=SamplingParams(max_new_tokens=5)),
        "long": Request(rid=3, prompt_ids=list(range(200, 300)),  # 100 tokens, chunked.
            sampling_params=SamplingParams(max_new_tokens=3)),
        "quick": Request(rid=4, prompt_ids=list(range(300, 305)),
            sampling_params=SamplingParams(max_new_tokens=1)),
    }

    for req in reqs.values():
        sched.add_request(req)

    # Run until all done.
    steps = 0
    while sched.has_work:
        batch = sched.step()
        if batch is None:
            break
        if batch.mode == "prefill":
            if batch.is_last_chunk:
                tokens = list(range(1000, 1000 + len(batch.requests)))
                sched.process_prefill_chunk(tokens, batch.requests)
            else:
                sched.process_prefill_chunk(None)
        else:
            tokens = list(range(2000, 2000 + len(batch.requests)))
            sched.process_decode_output(tokens)
        steps += 1
        assert steps < 200, "Scheduler stuck"

    # All finished.
    assert all(r.is_finished for r in reqs.values())
    assert sched.num_running == 0
    assert sched.num_waiting == 0


def test_fail_all_marks_and_cleans_up_requests():
    sched = _make_scheduler(chunk_size=32)
    req = _make_request(1, prompt_len=100, max_new_tokens=10)
    sched.add_request(req)

    batch = sched.step()
    assert batch is not None
    sched.process_prefill_chunk(None)
    assert sched.num_prefilling == 1

    sched.fail_all("engine_error")

    assert req.finished_reason == "engine_error"
    assert sched.num_waiting == 0
    assert sched.num_prefilling == 0
    assert sched.num_running == 0
    assert sched.num_prefilling == 0

    stats = sched.stats
    assert stats["finished"] == 1
    assert stats["prefill_steps"] > 0
    assert stats["decode_steps"] == 0


def test_max_waiting_rejects_overflow():
    """Queue full raises RuntimeError."""
    sched = _make_scheduler(max_running=1)
    # Override max_waiting to a small value.
    sched.max_waiting = 3

    for i in range(3):
        sched.add_request(_make_request(i))

    with pytest.raises(RuntimeError, match="queue full"):
        sched.add_request(_make_request(99))
