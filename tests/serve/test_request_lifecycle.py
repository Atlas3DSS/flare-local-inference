"""Tests for Request lifecycle and BatchScheduler integration.

Exercises the full submit → prefill → decode → finish → cleanup cycle
with mock page pools. No GPU required.
"""

import time
import pytest

from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
from serve.engine.request import Request
from serve.engine.sampling import SamplingParams
from serve.engine.scheduler import BatchScheduler


class MockPagePool:
    def __init__(self, num_pages=200):
        self.num_pages = num_pages
        self._free = list(range(num_pages))

    def alloc(self, n):
        if n > len(self._free):
            raise RuntimeError("OOM")
        r = self._free[-n:]
        del self._free[-n:]
        return r

    def free(self, ids):
        self._free.extend(ids)

    @property
    def num_free(self):
        return len(self._free)


def _make_scheduler(num_pages=200, max_running=8, captured_bs=None, chunk_size=512):
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


# -- request finish conditions ---------------------------------------------


def test_request_finish_max_tokens():
    req = Request(
        rid=0, prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=3),
    )
    assert not req.is_finished

    req.output_ids = [10, 20]
    req.check_finished()
    assert not req.is_finished

    req.output_ids.append(30)
    req.check_finished()
    assert req.is_finished
    assert req.finished_reason == "length"


def test_request_finish_stop_token():
    req = Request(
        rid=0, prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=100, stop_token_ids=[999]),
    )
    req.output_ids = [10, 20]
    req.check_finished()
    assert not req.is_finished

    req.output_ids.append(999)
    req.check_finished()
    assert req.is_finished
    assert req.finished_reason == "stop"


def test_request_finish_multiple_stop_tokens():
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=100, stop_token_ids=[100, 200, 300]),
    )
    req.output_ids = [10, 200]
    req.check_finished()
    assert req.is_finished
    assert req.finished_reason == "stop"


def test_request_finish_multi_token_stop_sequence_strips_output():
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=100, stop_sequences=[[10, 20, 30]]),
    )
    req.output_ids = [10, 20]
    req.check_finished()
    assert not req.is_finished

    req.output_ids.append(30)
    req.check_finished()
    assert req.is_finished
    assert req.finished_reason == "stop"
    assert req.output_ids == []


def test_request_stop_sequence_requires_full_suffix_match():
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=100, stop_sequences=[[20, 30]]),
    )
    req.output_ids = [30]
    req.check_finished()
    assert not req.is_finished
    assert req.output_ids == [30]


def test_request_not_finished_without_stop():
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=100, stop_token_ids=[999]),
    )
    req.output_ids = [10, 20, 30]
    req.check_finished()
    assert not req.is_finished


def test_request_timeout():
    req = Request(
        rid=0, prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=100),
        timeout_s=0.001,  # 1ms timeout.
    )
    req.output_ids = [10]
    time.sleep(0.01)  # Exceed timeout.
    req.check_finished()
    assert req.is_finished
    assert req.finished_reason == "timeout"
    assert req._done_event.is_set()


def test_request_no_timeout_when_not_set():
    req = Request(
        rid=0, prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=100),
    )
    req.output_ids = [10]
    req.check_finished()
    assert not req.is_finished


# -- timing ----------------------------------------------------------------


def test_request_tracks_timing():
    req = Request(rid=0, prompt_ids=[1, 2, 3])
    assert req.ttft_ms is None

    req.record_first_token()
    assert req.first_token_at is not None
    assert req.ttft_ms is not None
    assert req.ttft_ms >= 0

    # Recording again doesn't change it.
    first = req.first_token_at
    time.sleep(0.01)
    req.record_first_token()
    assert req.first_token_at == first


def test_request_total_time():
    req = Request(rid=0, prompt_ids=[1])
    assert req.total_time_ms is None

    req.finished_at = time.monotonic()
    assert req.total_time_ms is not None
    assert req.total_time_ms >= 0


# -- full lifecycle through scheduler -------------------------------------


def test_full_lifecycle_single_request():
    sched = _make_scheduler()
    req = Request(
        rid=1, prompt_ids=[10, 20, 30, 40, 50],
        sampling_params=SamplingParams(max_new_tokens=5),
    )
    sched.add_request(req)

    # Step 1: prefill.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 1
    assert batch.requests[0] is req
    sched.process_prefill_chunk([100])
    assert req.output_ids == [100]
    assert req.first_token_at is not None

    # Steps 2-5: decode.
    for i in range(4):
        batch = sched.step()
        assert batch.mode == "decode"
        sched.process_decode_output([200 + i])

    assert req.is_finished
    assert req.finished_reason == "length"
    assert len(req.output_ids) == 5
    assert sched.num_running == 0
    assert len(sched.finished) == 1


def test_full_lifecycle_concurrent_requests():
    sched = _make_scheduler(max_running=4)

    reqs = [
        Request(rid=i, prompt_ids=list(range(i*100, i*100+10)),
                sampling_params=SamplingParams(max_new_tokens=3))
        for i in range(4)
    ]
    for r in reqs:
        sched.add_request(r)

    # Prefill all 4 (short requests may be batched together).
    steps = 0
    while not all(r.first_token_at is not None for r in reqs):
        batch = sched.step()
        assert batch is not None, f"step() returned None with {sched.num_waiting} waiting, {sched.num_running} running"
        if batch.mode == "prefill":
            if batch.is_last_chunk:
                tokens = [10 + r.rid for r in batch.requests]
                sched.process_prefill_chunk(tokens, batch.requests)
            else:
                sched.process_prefill_chunk(None)
        else:
            sched.process_decode_output([500] * len(batch.requests))
        steps += 1
        assert steps < 30

    # All should have first token.
    for r in reqs:
        assert len(r.output_ids) >= 1
        assert r.first_token_at is not None

    # Decode until all finish.
    while not all(r.is_finished for r in reqs):
        batch = sched.step()
        assert batch.mode == "decode"
        sched.process_decode_output([600] * len(batch.requests))
        steps += 1
        assert steps < 50

    assert all(r.is_finished for r in reqs)
    assert sched.num_running == 0


def test_interleaved_admit_and_decode():
    """New requests arrive while others are decoding."""
    sched = _make_scheduler(max_running=4)

    # Start 2 short requests — batched together.
    req1 = Request(rid=1, prompt_ids=[1,2,3],
                   sampling_params=SamplingParams(max_new_tokens=5))
    req2 = Request(rid=2, prompt_ids=[4,5,6],
                   sampling_params=SamplingParams(max_new_tokens=5))
    sched.add_request(req1)
    sched.add_request(req2)

    # Both admitted in one batched prefill.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 2
    sched.process_prefill_chunk([10, 20], batch.requests)

    # Both running now. Decode.
    batch = sched.step()
    assert batch.mode == "decode"
    assert len(batch.requests) == 2
    sched.process_decode_output([11, 21])

    # Add a new request mid-flight.
    req3 = Request(rid=3, prompt_ids=[7,8,9],
                   sampling_params=SamplingParams(max_new_tokens=5))
    sched.add_request(req3)

    # Prefill req3 (after interleave decode).
    for _ in range(5):
        batch = sched.step()
        if batch.mode == "prefill":
            assert batch.requests[0].rid == 3
            sched.process_prefill_chunk([30], batch.requests)
            break
        else:
            sched.process_decode_output([100] * len(batch.requests))

    # All 3 running.
    assert sched.num_running == 3


def test_request_pages_freed_on_finish():
    """Full aligned pages remain cached when a request finishes."""
    sched = _make_scheduler(num_pages=50)
    initial_free = sched.pool.num_free

    req = Request(rid=1, prompt_ids=[1,2,3],
                  sampling_params=SamplingParams(max_new_tokens=1))
    sched.add_request(req)

    # Prefill allocates pages.
    sched.step()
    sched.process_prefill_chunk([42])
    assert req.is_finished

    # Fully materialized checkpoint pages remain cached, not free.
    # Total pages accounted for = free + cached.
    assert sched.pool.num_free + sched.cache.total_cached_pages <= initial_free


def test_prefix_sharing_two_requests():
    """Second request with same prefix reuses cached pages."""
    sched = _make_scheduler(num_pages=100)

    shared_prompt = list(range(128))

    # First request.
    req1 = Request(rid=1, prompt_ids=list(shared_prompt),
                   sampling_params=SamplingParams(max_new_tokens=1))
    sched.add_request(req1)
    sched.step()
    sched.process_prefill_chunk([42])
    assert req1.is_finished

    # Full aligned pages are now in the checkpoint cache.
    assert sched.cache.total_cached_pages > 0

    # Second request with same prefix.
    req2 = Request(rid=2, prompt_ids=list(shared_prompt) + [9, 10],
                   sampling_params=SamplingParams(max_new_tokens=1))
    sched.add_request(req2)

    batch = sched.step()
    assert batch.mode == "prefill"
    # req2 should have matched a prefix.
    assert req2.checkpoint_len > 0
    # The extend should only cover the non-cached tokens.
    assert batch.q_seqlens[0] < len(req2.prompt_ids)


def test_stress_many_requests():
    """Submit many requests, verify all complete and resources are clean."""
    sched = _make_scheduler(num_pages=500, max_running=8)

    n_requests = 20
    reqs = []
    for i in range(n_requests):
        req = Request(
            rid=i,
            prompt_ids=list(range(i * 100, i * 100 + 10)),
            sampling_params=SamplingParams(max_new_tokens=3),
        )
        sched.add_request(req)
        reqs.append(req)

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
        if steps > 500:
            pytest.fail("scheduler stuck — infinite loop")

    assert all(r.is_finished for r in reqs)
    assert sched.num_running == 0
    assert sched.num_waiting == 0
    assert len(sched.finished) == n_requests


def test_max_running_queues_excess():
    """Excess requests wait in queue until slots open."""
    sched = _make_scheduler(max_running=2)

    reqs = [
        Request(rid=i, prompt_ids=[i * 10 + j for j in range(5)],
                sampling_params=SamplingParams(max_new_tokens=3))
        for i in range(5)
    ]
    for r in reqs:
        sched.add_request(r)

    # First step batches req 0 and 1 (max_running=2).
    batch = sched.step()
    assert batch.mode == "prefill"
    assert len(batch.requests) == 2
    sched.process_prefill_chunk([10, 20], batch.requests)

    # 2 running, 3 waiting.
    assert sched.num_running == 2
    assert sched.num_waiting == 3


# -- done event -----------------------------------------------------------


def test_done_event_set_on_max_tokens():
    """The _done_event is signaled when request finishes via max_new_tokens."""
    req = Request(
        rid=0, prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=2),
    )
    assert not req._done_event.is_set()

    req.output_ids = [10]
    req.check_finished()
    assert not req._done_event.is_set()

    req.output_ids.append(20)
    req.check_finished()
    assert req.is_finished
    assert req._done_event.is_set()


def test_done_event_set_on_stop_token():
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=100, stop_token_ids=[999]),
    )
    req.output_ids = [10, 999]
    req.check_finished()
    assert req._done_event.is_set()


def test_done_event_set_on_oom():
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=100),
    )
    req._mark_finished("oom")
    assert req._done_event.is_set()
    assert req.finished_reason == "oom"


def test_done_event_through_scheduler():
    """Full lifecycle: done_event is set when scheduler finishes a request."""
    sched = _make_scheduler()
    req = Request(
        rid=1, prompt_ids=[10, 20, 30],
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req)
    assert not req._done_event.is_set()

    sched.step()  # Prefill.
    sched.process_prefill_chunk([42])

    assert req.is_finished
    assert req._done_event.is_set()


def test_token_event_pulsed_on_each_token():
    """_token_event is set when tokens are appended via append_token()."""
    req = Request(
        rid=0, prompt_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=5),
    )
    assert not req._token_event.is_set()

    req.append_token(10)
    assert req._token_event.is_set()
    assert req.output_ids == [10]

    # Consumer clears the event.
    req._token_event.clear()
    assert not req._token_event.is_set()

    req.append_token(20)
    assert req._token_event.is_set()


def test_concurrent_submit_and_step():
    """Simulate concurrent submit + stepping from different threads."""
    import threading

    sched = _make_scheduler(max_running=4)
    lock = threading.Lock()

    # Submit 4 requests from different threads.
    reqs = []
    def submit_one(rid):
        req = Request(
            rid=rid, prompt_ids=list(range(rid * 10, rid * 10 + 5)),
            sampling_params=SamplingParams(max_new_tokens=2),
        )
        with lock:
            sched.add_request(req)
        reqs.append(req)

    threads = [threading.Thread(target=submit_one, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(reqs) == 4
    assert sched.num_waiting == 4

    # Step through to completion (single thread, like the server loop).
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
        if steps > 100:
            break

    assert all(r.is_finished for r in reqs)
    assert all(r._done_event.is_set() for r in reqs)


# -- chunked prefill lifecycle --------------------------------------------


def test_long_prompt_lifecycle():
    """A long prompt gets chunked, completes, and resources are clean."""
    sched = _make_scheduler(num_pages=200, max_running=4, chunk_size=32)
    req = Request(
        rid=1,
        prompt_ids=list(range(100)),  # 100 tokens, 4 chunks of 32 + 1 of 4.
        sampling_params=SamplingParams(max_new_tokens=3),
    )
    sched.add_request(req)

    steps = 0
    while sched.has_work:
        batch = sched.step()
        if batch is None:
            break
        if batch.mode == "prefill":
            if batch.is_last_chunk:
                sched.process_prefill_chunk([42], batch.requests)
            else:
                sched.process_prefill_chunk(None)
        else:
            sched.process_decode_output([100 + steps])
        steps += 1
        assert steps < 50

    assert req.is_finished
    assert req.finished_reason == "length"
    assert len(req.output_ids) == 3
    assert sched.num_running == 0
    assert sched.num_prefilling == 0


def test_prefill_progress_tracked():
    """prefill_progress advances correctly through chunks."""
    sched = _make_scheduler(chunk_size=32)
    req = Request(
        rid=1,
        prompt_ids=list(range(80)),  # 80 tokens: 2 full chunks + 1 partial.
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req)

    # Chunk 1: tokens 0-31.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.q_seqlens[0] == 32
    assert not batch.is_last_chunk
    assert req.prefill_progress == 32
    sched.process_prefill_chunk(None)

    # Chunk 2: tokens 32-63.
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.q_seqlens[0] == 32
    assert not batch.is_last_chunk
    assert req.prefill_progress == 64
    sched.process_prefill_chunk(None)

    # Chunk 3: tokens 64-79 (16 tokens, last chunk).
    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.q_seqlens[0] == 16
    assert batch.is_last_chunk
    assert req.prefill_progress == 80
    sched.process_prefill_chunk([42], batch.requests)

    assert req.output_ids == [42]
    assert req.is_finished  # max_new_tokens=1, done after first token.
    assert sched.num_prefilling == 0


def test_single_token_prompt():
    """A 1-token prompt works correctly."""
    sched = _make_scheduler()
    req = Request(
        rid=1,
        prompt_ids=[42],
        sampling_params=SamplingParams(max_new_tokens=2),
    )
    sched.add_request(req)

    batch = sched.step()
    assert batch.mode == "prefill"
    assert batch.is_last_chunk
    assert batch.q_seqlens[0] == 1
    sched.process_prefill_chunk([10], batch.requests)

    batch = sched.step()
    assert batch.mode == "decode"
    sched.process_decode_output([20])

    assert req.is_finished
    assert len(req.output_ids) == 2


def test_no_tokens_during_nonfinal_chunks():
    """Non-final prefill chunks don't append tokens or fire events."""
    sched = _make_scheduler(chunk_size=32)
    req = Request(
        rid=1,
        prompt_ids=list(range(80)),  # 3 chunks.
        sampling_params=SamplingParams(max_new_tokens=5),
    )
    sched.add_request(req)

    # Chunk 1.
    batch = sched.step()
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert len(req.output_ids) == 0
    assert not req._token_event.is_set()
    assert req.first_token_at is None

    # Chunk 2.
    batch = sched.step()
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert len(req.output_ids) == 0
    assert not req._token_event.is_set()

    # Chunk 3 (last).
    batch = sched.step()
    assert batch.is_last_chunk
    sched.process_prefill_chunk([42], batch.requests)
    assert len(req.output_ids) == 1
    assert req._token_event.is_set()
    assert req.first_token_at is not None


def test_chunked_prefill_with_prefix_sharing():
    """Prefix sharing works correctly with chunked prefill."""
    sched = _make_scheduler(num_pages=200, chunk_size=64)

    shared_prompt = list(range(100))  # 100 tokens.

    # First request — no prefix cache.
    req1 = Request(
        rid=1, prompt_ids=list(shared_prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req1)

    # Process all chunks until req1 finishes.
    for _ in range(20):
        if req1.is_finished:
            break
        batch = sched.step()
        assert batch is not None
        if batch.mode == "prefill":
            if batch.is_last_chunk:
                sched.process_prefill_chunk([42], batch.requests)
            else:
                sched.process_prefill_chunk(None)
        else:
            sched.process_decode_output([100] * len(batch.requests))

    assert req1.is_finished

    # Second request with same prefix — should match cache.
    req2 = Request(
        rid=2, prompt_ids=list(shared_prompt) + [999],
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req2)

    batch = sched.step()
    # With interleaving, might get a decode first if req1 was still running.
    # But req1 finished, so should go straight to prefill for req2.
    while batch.mode != "prefill" or batch.requests[0].rid != 2:
        if batch.mode == "decode":
            sched.process_decode_output([100] * len(batch.requests))
        batch = sched.step()

    assert req2.checkpoint_len > 0  # Got a cache hit.
    assert batch.q_seqlens[0] < len(req2.prompt_ids)  # Only extend non-cached.


def test_request_prefill_remaining():
    """prefill_remaining property is correct."""
    req = Request(
        rid=1,
        prompt_ids=list(range(100)),
        sampling_params=SamplingParams(max_new_tokens=5),
    )
    assert req.prefill_remaining == 100

    req.prefill_progress = 32
    assert req.prefill_remaining == 68

    req.prefill_progress = 100
    assert req.prefill_remaining == 0


def test_server_loop_simulation():
    """Simulate the background step loop with concurrent submissions."""
    import threading

    sched = _make_scheduler(max_running=4, chunk_size=64)
    lock = threading.Lock()
    work_event = threading.Event()

    # Simulate server loop in a thread.
    loop_running = True
    step_count = [0]

    def server_loop():
        while loop_running:
            with lock:
                has_work = sched.has_work
            if has_work:
                with lock:
                    batch = sched.step()
                if batch is not None:
                    step_count[0] += 1
                    if batch.mode == "prefill":
                        with lock:
                            if batch.is_last_chunk:
                                sched.process_prefill_chunk(
                                    [42 + step_count[0]], batch.requests)
                            else:
                                sched.process_prefill_chunk(None)
                    else:
                        with lock:
                            sched.process_decode_output(
                                [100 + step_count[0]] * len(batch.requests))
            else:
                work_event.wait(timeout=0.01)
                work_event.clear()

    loop_thread = threading.Thread(target=server_loop, daemon=True)
    loop_thread.start()

    # Submit 3 requests from the main thread.
    reqs = []
    for i in range(3):
        req = Request(
            rid=i,
            prompt_ids=list(range(i * 100, i * 100 + 10)),
            sampling_params=SamplingParams(max_new_tokens=3),
        )
        with lock:
            sched.add_request(req)
        work_event.set()
        reqs.append(req)

    # Wait for all to finish.
    for req in reqs:
        req._done_event.wait(timeout=5.0)
        assert req._done_event.is_set(), f"Request {req.rid} didn't finish"

    loop_running = False
    loop_thread.join(timeout=1.0)

    assert all(r.is_finished for r in reqs)
    assert all(len(r.output_ids) == 3 for r in reqs)
    assert step_count[0] > 0


def test_stress_mixed_prompt_lengths():
    """Stress test: mix of short and long prompts with chunked prefill."""
    sched = _make_scheduler(num_pages=500, max_running=8, chunk_size=32)

    reqs = []
    for i in range(15):
        prompt_len = 5 + (i * 17) % 100  # Varying lengths: 5 to 100.
        req = Request(
            rid=i,
            prompt_ids=list(range(i * 200, i * 200 + prompt_len)),
            sampling_params=SamplingParams(max_new_tokens=3),
        )
        sched.add_request(req)
        reqs.append(req)

    steps = 0
    while sched.has_work:
        batch = sched.step()
        if batch is None:
            break
        if batch.mode == "prefill":
            if batch.is_last_chunk:
                sched.process_prefill_chunk(
                    [1000 + steps] * len(batch.requests), batch.requests)
            else:
                sched.process_prefill_chunk(None)
        else:
            sched.process_decode_output(
                [2000 + steps] * len(batch.requests))
        steps += 1
        if steps > 1000:
            pytest.fail("scheduler stuck")

    assert all(r.is_finished for r in reqs), \
        f"Unfinished: {[r.rid for r in reqs if not r.is_finished]}"
    assert sched.num_running == 0
    assert sched.num_waiting == 0
    assert sched.num_prefilling == 0


def test_request_cancel():
    """Cancelling a request marks it finished and signals events."""
    req = Request(
        rid=1,
        prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=100),
    )
    assert not req.is_finished

    req.cancel()
    assert req.is_finished
    assert req.finished_reason == "cancelled"
    assert req._done_event.is_set()

    # Cancelling again is a no-op.
    req.cancel()
    assert req.finished_reason == "cancelled"


def test_running_request_keeps_events_and_output_when_not_preempted():
    """Visible output is preserved when admission applies backpressure."""
    sched = _make_scheduler(num_pages=2, max_running=4)

    req1 = Request(
        rid=1,
        prompt_ids=list(range(64)),
        sampling_params=SamplingParams(max_new_tokens=10),
    )
    sched.add_request(req1)

    # Prefill req1.
    batch = sched.step()
    sched.process_prefill_chunk([42], batch.requests)
    assert req1.output_ids == [42]
    assert req1._token_event.is_set()

    # Now submit a request that needs all pages.
    req2 = Request(
        rid=2,
        prompt_ids=list(range(100, 164)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(req2)

    # Interleave: decode first.
    batch = sched.step()
    if batch.mode == "decode":
        sched.process_decode_output([100])
    # Then try to admit req2.
    batch = sched.step()
    assert batch.mode == "decode"

    assert req1.output_ids == [42, 100]
    assert req1.checkpoint_len == 0
    assert req1.prefill_progress == 64
    assert req1.cache_len == 66
    assert req1._token_event.is_set()
    assert req1.finished_reason is None
    assert req2 in sched.waiting


def test_mixed_workload_simulation():
    """Simulate a realistic API workload: concurrent requests with
    different params, some finishing quickly, some timing out."""
    import threading

    sched = _make_scheduler(num_pages=500, max_running=8, chunk_size=64)
    lock = threading.Lock()
    work_event = threading.Event()
    loop_running = True

    def server_loop():
        while loop_running:
            with lock:
                has_work = sched.has_work
            if has_work:
                with lock:
                    batch = sched.step()
                if batch is not None:
                    if batch.mode == "prefill":
                        with lock:
                            if batch.is_last_chunk:
                                tokens = list(range(len(batch.requests)))
                                sched.process_prefill_chunk(tokens, batch.requests)
                            else:
                                sched.process_prefill_chunk(None)
                    else:
                        with lock:
                            tokens = list(range(len(batch.requests)))
                            sched.process_decode_output(tokens)
            else:
                work_event.wait(timeout=0.01)
                work_event.clear()

    loop = threading.Thread(target=server_loop, daemon=True)
    loop.start()

    # Submit a mix of requests.
    reqs = []
    configs = [
        (5, 3),    # Short prompt, few tokens.
        (50, 10),  # Medium prompt, medium tokens.
        (5, 1),    # Short prompt, instant finish.
        (100, 5),  # Long prompt (chunked), few tokens.
        (5, 5),    # Short prompt.
        (10, 3),   # Short prompt.
    ]
    for i, (prompt_len, max_tokens) in enumerate(configs):
        req = Request(
            rid=i,
            prompt_ids=list(range(i * 200, i * 200 + prompt_len)),
            sampling_params=SamplingParams(max_new_tokens=max_tokens),
        )
        with lock:
            sched.add_request(req)
        work_event.set()
        reqs.append(req)

    # Wait for all to finish.
    for req in reqs:
        assert req._done_event.wait(timeout=10.0), \
            f"Request {req.rid} (prompt={len(req.prompt_ids)}, max={req.sampling_params.max_new_tokens}) didn't finish"

    loop_running = False
    loop.join(timeout=2.0)

    assert all(r.is_finished for r in reqs)
    assert all(r.finished_reason in ("length", "stop") for r in reqs)
    assert sched.num_running == 0
    assert sched.num_prefilling == 0
    assert sched.stats["finished"] == len(reqs)


def test_timeout_during_prefill_unblocks_stream():
    """A request that times out during prefill should unblock
    any thread waiting on _token_event."""
    import threading

    req = Request(
        rid=1,
        prompt_ids=list(range(100)),
        sampling_params=SamplingParams(max_new_tokens=50),
        timeout_s=0.01,  # 10ms timeout.
    )

    # Simulate a streaming consumer waiting on token event.
    unblocked = threading.Event()

    def consumer():
        req._token_event.wait(timeout=5.0)
        unblocked.set()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    # Simulate prefill taking too long.
    time.sleep(0.05)
    req.check_finished()  # Should trigger timeout.

    assert req.is_finished
    assert req.finished_reason == "timeout"
    assert unblocked.wait(timeout=1.0), "Consumer should be unblocked"


def test_finish_reasons_all_propagate():
    """All finish reasons are correctly set."""
    # Length.
    req = Request(rid=1, prompt_ids=[1],
                  sampling_params=SamplingParams(max_new_tokens=1))
    req.append_token(10)
    req.check_finished()
    assert req.finished_reason == "length"

    # Stop.
    req = Request(rid=2, prompt_ids=[1],
                  sampling_params=SamplingParams(max_new_tokens=100, stop_token_ids=[99]))
    req.append_token(99)
    req.check_finished()
    assert req.finished_reason == "stop"

    # Timeout.
    req = Request(rid=3, prompt_ids=[1],
                  sampling_params=SamplingParams(max_new_tokens=100),
                  timeout_s=0.001)
    time.sleep(0.01)
    req.check_finished()
    assert req.finished_reason == "timeout"

    # Cancel.
    req = Request(rid=4, prompt_ids=[1],
                  sampling_params=SamplingParams(max_new_tokens=100))
    req.cancel()
    assert req.finished_reason == "cancelled"

    # OOM.
    req = Request(rid=5, prompt_ids=[1],
                  sampling_params=SamplingParams(max_new_tokens=100))
    req._mark_finished("oom")
    assert req.finished_reason == "oom"
