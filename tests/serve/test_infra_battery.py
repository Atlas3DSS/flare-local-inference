"""Infrastructure-focused stress tests for the serving stack.

These tests target failure handling, cleanup paths, and invariants that are
easy to miss in happy-path request/response coverage.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

from serve.cache.linear_state_arena import LinearStateArena
from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache
from serve.engine.request import Request
from serve.engine.sampling import SamplingParams
from serve.engine.scheduler import BatchScheduler
from serve.engine.serving import ServingEngine


class MockPagePool:
    def __init__(self, num_pages=64):
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


def _make_scheduler(num_pages=64, max_running=8, chunk_size=64):
    pool = MockPagePool(num_pages)
    cache = PrefixCheckpointCache(pool)
    return BatchScheduler(
        cache=cache,
        pool=pool,
        captured_bs=[1, 2, 4, 8],
        max_running=max_running,
        max_prefill_tokens=4096,
        chunk_size=chunk_size,
        device="cpu",
    )


def _make_request(rid, prompt_len=64, *, max_new_tokens=8, timeout_s=None):
    return Request(
        rid=rid,
        prompt_ids=list(range(rid * 1000, rid * 1000 + prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        timeout_s=timeout_s,
    )


def _make_minimal_engine(scheduler: BatchScheduler) -> ServingEngine:
    engine = ServingEngine.__new__(ServingEngine)
    engine.scheduler = scheduler
    engine._lock = threading.Lock()
    engine._work_event = threading.Event()
    engine._loop_running = False
    engine._loop_error = None
    engine._next_rid = 0
    engine.world_size = 1
    engine.device = "cpu"
    engine._runtime_policy = {}
    engine._tp_health = None
    tokenizer = MagicMock()
    tokenizer.eos_token_id = None
    tokenizer.unk_token_id = -1
    tokenizer.convert_tokens_to_ids.return_value = -1
    engine.tokenizer = tokenizer
    return engine


def _make_running_request_with_visible_output(sched: BatchScheduler, rid: int = 1) -> Request:
    req = _make_request(rid, prompt_len=64, max_new_tokens=8)
    sched.add_request(req)
    while req not in sched.running:
        batch = sched.step()
        assert batch is not None
        assert batch.mode == "prefill"
        if batch.is_last_chunk:
            sched.process_prefill_chunk([42], batch.requests)
        else:
            sched.process_prefill_chunk(None)
    assert req.output_ids == [42]
    assert req in sched.running
    return req


def test_cancelled_running_request_restores_page_budget_without_caching():
    sched = _make_scheduler(num_pages=2)
    req = _make_running_request_with_visible_output(sched)

    req.cancel()
    batch = sched.step()

    assert batch is None
    assert req.finished_reason == "cancelled"
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == sched.pool.num_pages
    assert sched.num_running == 0


def test_timed_out_prefilling_request_restores_page_budget_without_caching():
    sched = _make_scheduler(num_pages=2, chunk_size=32)
    req = _make_request(1, prompt_len=100, timeout_s=0.001)
    sched.add_request(req)

    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert req in sched.prefilling
    assert sched.pool.num_free == 0

    time.sleep(0.01)
    batch = sched.step()

    assert batch is None
    assert req.finished_reason == "timeout"
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == sched.pool.num_pages
    assert sched.num_prefilling == 0


def test_fail_all_releases_mixed_request_states_without_creating_checkpoints():
    sched = _make_scheduler(num_pages=8, chunk_size=32)

    running = _make_running_request_with_visible_output(sched, rid=1)

    prefilling = _make_request(2, prompt_len=100, max_new_tokens=4)
    sched.add_request(prefilling)
    batch = sched.step()
    assert batch is not None
    if batch.mode == "decode":
        sched.process_decode_output([99])
        batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert not batch.is_last_chunk
    sched.process_prefill_chunk(None)
    assert prefilling in sched.prefilling

    waiting = _make_request(3, prompt_len=16, max_new_tokens=2)
    sched.add_request(waiting)

    sched.fail_all("engine_error")

    for req in (running, prefilling, waiting):
        assert req.finished_reason == "engine_error"
        assert req._done_event.is_set()
    assert sched.cache.total_cached_pages == 0
    assert sched.pool.num_free == sched.pool.num_pages
    assert sched.num_running == 0
    assert sched.num_prefilling == 0
    assert sched.num_waiting == 0


def test_server_loop_crash_marks_engine_unhealthy_and_fails_inflight_work():
    sched = _make_scheduler()
    req = _make_request(1, prompt_len=16, max_new_tokens=2)
    sched.add_request(req)
    engine = _make_minimal_engine(sched)

    crashed = threading.Event()

    def crash_step():
        crashed.set()
        raise RuntimeError("synthetic crash")

    engine._step = crash_step
    engine.start_server_loop()

    assert crashed.wait(timeout=1.0)
    engine._loop_thread.join(timeout=1.0)
    assert not engine._loop_thread.is_alive()

    health = engine.server_loop_health()
    assert health["status"] == "fatal"
    assert health["healthy"] is False
    assert health["alive"] is False
    assert "synthetic crash" in health["last_error"]

    assert req.finished_reason == "engine_error"
    assert req._done_event.is_set()
    assert sched.num_waiting == 0


def test_submit_rejects_new_work_after_server_loop_crash():
    sched = _make_scheduler()
    engine = _make_minimal_engine(sched)
    engine._loop_error = "RuntimeError: synthetic crash"

    with pytest.raises(RuntimeError, match="server loop unhealthy"):
        engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4))

    assert engine._next_rid == 0
    assert sched.num_waiting == 0


def test_submit_rejects_new_work_after_tp_worker_failure():
    sched = _make_scheduler()
    engine = _make_minimal_engine(sched)
    engine.world_size = 2

    with patch(
        "serve.engine.serving.get_active_tp_health",
        return_value={
            "healthy": False,
            "fatal": True,
            "summary": "rank 1 exited with code 17",
            "workers": [{"rank": 1, "exitcode": 17}],
        },
    ):
        with pytest.raises(RuntimeError, match="tp workers unhealthy"):
            engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4))


def test_abort_paths_do_not_contaminate_subsequent_prefix_cache_reuse():
    sched = _make_scheduler(num_pages=4)
    shared_prompt = list(range(2000, 2128))

    cancelled = _make_running_request_with_visible_output(sched, rid=1)
    cancelled.cancel()
    assert sched.step() is None
    assert sched.cache.total_cached_pages == 0

    good = Request(
        rid=2,
        prompt_ids=list(shared_prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(good)
    while not good.is_finished:
        batch = sched.step()
        assert batch is not None
        assert batch.mode == "prefill"
        if batch.is_last_chunk:
            sched.process_prefill_chunk([77], batch.requests)
        else:
            sched.process_prefill_chunk(None)
    assert good.finished_reason == "length"
    assert sched.cache.total_cached_pages == 2

    follower = Request(
        rid=3,
        prompt_ids=list(shared_prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    sched.add_request(follower)
    batch = sched.step()
    assert batch is not None
    assert batch.mode == "prefill"
    assert follower.checkpoint_len == 128
    assert batch.q_seqlens == [1]


def test_ssm_control_ops_replay_snapshot_lifecycle_on_follower():
    pool = MockPagePool(num_pages=4)
    leader_arena = LinearStateArena(
        live_slots=2,
        snapshot_slots=2,
        num_linear_layers=1,
        num_heads=2,
        head_v_dim=4,
        head_k_dim=4,
        conv_dim=8,
        conv_kernel=4,
        device="cpu",
    )
    follower_arena = LinearStateArena(
        live_slots=2,
        snapshot_slots=2,
        num_linear_layers=1,
        num_heads=2,
        head_v_dim=4,
        head_k_dim=4,
        conv_dim=8,
        conv_kernel=4,
        device="cpu",
    )
    cache = PrefixCheckpointCache(pool, state_arena=leader_arena)
    sched = BatchScheduler(
        cache=cache,
        pool=pool,
        ssm_pool=leader_arena,
        captured_bs=[1],
        chunk_size=64,
        device="cpu",
    )
    follower_engine = ServingEngine.__new__(ServingEngine)
    follower_engine.runner = SimpleNamespace(ssm_pool=follower_arena)

    req = Request(rid=1, prompt_ids=list(range(64)), sampling_params=SamplingParams(max_new_tokens=1))
    sched.add_request(req)
    batch = sched.step()
    ServingEngine._apply_ssm_control_ops(follower_engine, sched.drain_ssm_control_ops())

    leader_slot = req.ssm_slot
    leader_arena.live.state.ssm[:, leader_slot].fill_(3.25)
    leader_arena.live.state.conv[0][leader_slot].fill_(2.0)
    follower_arena.live.state.ssm[:, leader_slot].copy_(leader_arena.live.state.ssm[:, leader_slot])
    follower_arena.live.state.conv[0][leader_slot].copy_(leader_arena.live.state.conv[0][leader_slot])

    sched.process_prefill_chunk([42], batch.requests)
    ServingEngine._apply_ssm_control_ops(follower_engine, sched.drain_ssm_control_ops())

    checkpoint = next(iter(cache.by_prefix.values()), None)
    assert checkpoint is not None
    assert checkpoint.state_snapshot_slot >= 0
    snapshot_slot = checkpoint.state_snapshot_slot
    assert torch.equal(
        leader_arena.snapshot.state.ssm[:, snapshot_slot],
        follower_arena.snapshot.state.ssm[:, snapshot_slot],
    )
    assert torch.equal(
        leader_arena.snapshot.state.conv[0][snapshot_slot],
        follower_arena.snapshot.state.conv[0][snapshot_slot],
    )


def test_scheduler_churn_preserves_page_budget():
    sched = _make_scheduler(num_pages=8, chunk_size=32)
    for rid in range(12):
        req = _make_request(rid, prompt_len=64 if rid % 2 == 0 else 16, max_new_tokens=2)
        sched.add_request(req)
        while not req.is_finished and (req in sched.waiting or req in sched.prefilling or req in sched.running):
            batch = sched.step()
            assert batch is not None
            if batch.mode == "prefill":
                if batch.is_last_chunk:
                    sched.process_prefill_chunk([90], batch.requests)
                else:
                    sched.process_prefill_chunk(None)
            else:
                sched.process_decode_output([91] * len(batch.requests))
    assert sched.pool.num_free + sched.cache.total_cached_pages == sched.pool.num_pages
