"""Integration-focused tests for the HTTP API server."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from serve.api.server import ServingApp
from serve.engine.serving import GenerationResult


class _FakeInputIds:
    def __init__(self, values):
        self.input_ids = [self]
        self._values = values

    def __getitem__(self, idx):
        return self

    def tolist(self):
        return list(self._values)


class _FakeTokenizer:
    eos_token_id = None
    unk_token_id = -1

    def __call__(self, text, return_tensors="pt"):
        del return_tensors
        return _FakeInputIds([len(text), len(text) + 1, len(text) + 2])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
        del tokenize, add_generation_prompt
        return "|".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)

    def convert_tokens_to_ids(self, _name):
        return -1


class _FakeRequest:
    def __init__(self, prompt_ids, *, tokens=None, finish_reason="stop", delay_s=0.0):
        self.prompt_ids = list(prompt_ids)
        self.output_ids = []
        self.finished_reason = None
        self.created_at = time.monotonic()
        self.first_token_at = None
        self.finished_at = None
        self._done_event = threading.Event()
        self._token_event = threading.Event()
        self._delay_s = delay_s
        self._cancelled = threading.Event()
        if tokens is not None:
            worker = threading.Thread(
                target=self._emit_tokens,
                args=(list(tokens), finish_reason),
                daemon=True,
            )
            worker.start()

    @property
    def is_finished(self):
        return self.finished_reason is not None

    @property
    def ttft_ms(self):
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.created_at) * 1000

    @property
    def total_time_ms(self):
        if self.finished_at is None:
            return None
        return (self.finished_at - self.created_at) * 1000

    def cancel(self):
        self._cancelled.set()
        self._mark_finished("cancelled")

    def finish_immediately(self, reason):
        self._mark_finished(reason)

    def _emit_tokens(self, tokens, finish_reason):
        for token in tokens:
            if self._cancelled.is_set():
                return
            time.sleep(self._delay_s)
            self.output_ids.append(token)
            if self.first_token_at is None:
                self.first_token_at = time.monotonic()
            self._token_event.set()
        self._mark_finished(finish_reason)

    def _mark_finished(self, reason):
        if self.finished_reason is not None:
            return
        self.finished_reason = reason
        self.finished_at = time.monotonic()
        self._token_event.set()
        self._done_event.set()


class _FakeEngine:
    def __init__(self):
        self.model_path = "/data/models/test-model"
        self.scheduler = SimpleNamespace(
            stats={
                "waiting": 0,
                "prefilling": 0,
                "running": 0,
                "finished": 5,
                "prefill_steps": 10,
                "decode_steps": 50,
                "preemptions": 0,
                "cache_pages": 100,
                "free_pages": 200,
            }
        )
        self.tokenizer = _FakeTokenizer()
        self._health = {
            "status": "ok",
            "running": True,
            "alive": True,
            "ready": True,
            "healthy": True,
            "degraded": False,
            "fatal": False,
            "last_error": None,
            "tp_workers": None,
        }
        self._runtime_policy = {
            "graph_batch_sizes": [1, 2, 4, 8],
            "prefill_chunk_size": 512,
            "max_running": 8,
            "max_waiting": 256,
            "max_prefill_tokens": 4096,
            "prefix_cache_enabled": True,
        }
        self._submit_plans = deque()
        self.submissions = []
        self.requests = []

    def queue_submit_plan(self, **plan):
        self._submit_plans.append(plan)

    def submit(self, input_ids, params, timeout_s=None):
        self.submissions.append(
            {
                "input_ids": list(input_ids),
                "params": params,
                "timeout_s": timeout_s,
            }
        )
        plan = self._submit_plans.popleft() if self._submit_plans else {}
        request = _FakeRequest(
            input_ids,
            tokens=plan.get("tokens", [ord("o"), ord("k")]),
            finish_reason=plan.get("finish_reason", "stop"),
            delay_s=plan.get("delay_s", 0.0),
        )
        self.requests.append(request)
        immediate_reason = plan.get("immediate_reason")
        if immediate_reason is not None:
            request.finish_immediately(immediate_reason)
        return request

    def _to_result(self, request):
        return GenerationResult(
            request_id=0,
            prompt_ids=list(request.prompt_ids),
            generated_ids=list(request.output_ids),
            finished=request.is_finished,
            finish_reason=request.finished_reason,
            time_to_first_token_ms=request.ttft_ms or 0.0,
            total_time_ms=request.total_time_ms or 0.0,
        )

    def server_loop_health(self):
        return dict(self._health)

    def runtime_policy(self):
        return dict(self._runtime_policy)


@pytest.fixture
def engine():
    return _FakeEngine()


@pytest.fixture
def client(engine):
    return TestClient(ServingApp(engine).app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-model"
    assert data["runtime_policy"]["max_running"] == 8
    assert data["server_loop"]["healthy"] is True


def test_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "test-model"


def test_webui_disabled_by_default(engine):
    client = TestClient(ServingApp(engine).app)
    resp = client.get("/ui")
    assert resp.status_code == 404


def test_webui_enabled_returns_html(engine):
    client = TestClient(ServingApp(engine, enable_webui=True).app)
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "b12x chat" in resp.text
    assert "test-model" in resp.text


def test_webui_root_redirects_when_enabled(engine):
    client = TestClient(ServingApp(engine, enable_webui=True).app)
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/ui"


def test_completions_passes_extended_sampling_params(client, engine):
    resp = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.2,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.25,
            "timeout": 30.0,
        },
    )
    assert resp.status_code == 200
    params = engine.submissions[-1]["params"]
    assert params.min_p == 0.2
    assert params.presence_penalty == 0.5
    assert params.frequency_penalty == 0.25
    assert engine.submissions[-1]["timeout_s"] == 30.0


def test_stop_strings_converted_to_token_sequences(client, engine):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 10, "stop": ["\n", "END"]},
    )
    assert resp.status_code == 200
    params = engine.submissions[-1]["params"]
    assert params.stop_sequences == [[10], [69, 78, 68]]


def test_unsupported_seed_and_logprobs_return_400(client):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 10, "seed": 123, "logprobs": 5},
    )
    assert resp.status_code == 400
    assert "unsupported request fields" in resp.json()["detail"]


def test_context_too_long_maps_to_http_400(client, engine):
    engine.queue_submit_plan(tokens=None, immediate_reason="context_too_long")
    resp = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 10})
    assert resp.status_code == 400
    assert "KV-cache budget" in resp.json()["detail"]


def test_chat_completions(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["usage"]["total_tokens"] >= 3


def test_streaming_completion_uses_real_sse(client, engine):
    engine.queue_submit_plan(tokens=[ord("A"), ord("B"), ord("C")], delay_s=0.01)
    with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 10, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    assert any('"text": "A"' in line for line in lines)
    assert any('"text": "B"' in line for line in lines)
    assert lines[-1] == "data: [DONE]"


def test_stream_generator_close_cancels_request(engine):
    engine.queue_submit_plan(tokens=[ord("A"), ord("B"), ord("C")], delay_s=0.05)
    app = ServingApp(engine)
    async def _exercise():
        stream = app._stream_completion("Hello", app._to_sampling_params(SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=10,
            stop=None,
            timeout=None,
            seed=None,
            logprobs=None,
        )))
        first_chunk = await anext(stream)
        assert first_chunk.startswith("data: ")
        await stream.aclose()
        await asyncio.sleep(0.1)

    asyncio.run(_exercise())
    assert engine.requests[-1].finished_reason == "cancelled"


def test_concurrent_requests_complete_through_same_app(engine):
    app = ServingApp(engine)
    client = TestClient(app.app)
    engine.queue_submit_plan(tokens=[ord("1")], delay_s=0.02)
    engine.queue_submit_plan(tokens=[ord("2")], delay_s=0.02)

    def _post(prompt):
        return client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 5})

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(_post, "one")
        fut2 = pool.submit(_post, "two")
        resp1 = fut1.result(timeout=2.0)
        resp2 = fut2.result(timeout=2.0)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert len(engine.submissions) == 2


def test_health_reports_fatal_loop_state(client, engine):
    engine._health = {
        "status": "fatal",
        "running": False,
        "alive": False,
        "ready": False,
        "healthy": False,
        "degraded": False,
        "fatal": True,
        "last_error": "RuntimeError: boom",
        "tp_workers": {"fatal": True, "summary": "rank 1 exited"},
    }
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "fatal"
    assert data["server_loop"]["tp_workers"]["summary"] == "rank 1 exited"
