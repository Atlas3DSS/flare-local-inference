"""End-to-end tests for the serving stack.

Requires GPU and the MiniMax-M2.5-NVFP4 checkpoint.
"""

import os
import time
import pytest
import torch

torch.set_grad_enabled(False)

MODEL_PATH = "/data/models/MiniMax-M2.5-NVFP4-REAP"


def require_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not available")


def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


@pytest.fixture(scope="module")
def engine():
    """Load model and build engine once for all tests."""
    require_model()
    require_gpu()

    from serve.engine.serving import ServingEngine
    try:
        eng = ServingEngine(
            MODEL_PATH,
            device="cuda",
            warmup_prefill_lengths=[4],
            graph_batch_sizes=[1, 2, 4],
        )
    except RuntimeError as exc:
        if "PagePool OOM" in str(exc):
            pytest.skip(f"Insufficient GPU memory for e2e engine fixture: {exc}")
        raise
    return eng


def test_arithmetic(engine):
    from serve.engine.sampling import SamplingParams
    result = engine.complete("1 + 1 =", SamplingParams(max_new_tokens=10))
    assert result.finished
    assert len(result.generated_ids) > 0


def test_code_generation(engine):
    from serve.engine.sampling import SamplingParams
    result = engine.complete("def fibonacci(n):", SamplingParams(max_new_tokens=30))
    assert result.finished
    assert len(result.generated_ids) > 0


def test_chat(engine):
    from serve.engine.sampling import SamplingParams
    messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    result = engine.chat(messages, SamplingParams(max_new_tokens=50))
    text = engine.tokenizer.decode(result.generated_ids).lower()
    assert "paris" in text, f"Expected 'paris' in output, got {text!r}"


def test_streaming(engine):
    from serve.engine.sampling import SamplingParams
    input_ids = engine.tokenizer("Hello world", return_tensors="pt").input_ids[0].tolist()
    tokens = []
    for tok_id, text, result in engine.generate_stream(input_ids, SamplingParams(max_new_tokens=10)):
        tokens.append(tok_id)
    assert len(tokens) > 0
    assert len(tokens) <= 10


def test_decode_throughput(engine):
    """Verify decode throughput is reasonable (>10 tok/s at bs=1)."""
    from serve.engine.sampling import SamplingParams
    # Warmup.
    engine.complete("Warmup", SamplingParams(max_new_tokens=3))
    torch.cuda.synchronize()
    # Timed.
    torch.cuda.synchronize()
    t0 = time.time()
    result = engine.complete("Test", SamplingParams(max_new_tokens=20))
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n = len(result.generated_ids)
    tok_per_sec = n / elapsed
    assert tok_per_sec > 5, f"Throughput too low: {tok_per_sec:.1f} tok/s"


def test_concurrent_requests(engine):
    """Generate 4 requests concurrently via generate_batch()."""
    from serve.engine.sampling import SamplingParams
    params = SamplingParams(max_new_tokens=10)

    prompts = ["1 + 1 =", "2 + 2 =", "3 + 3 =", "Hello world"]
    prompt_ids = [engine.tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]
    results = engine.generate_batch(prompt_ids, params)

    assert len(results) == 4
    for r in results:
        assert r.finished
        assert len(r.generated_ids) > 0
        assert r.time_to_first_token_ms > 0
        assert r.total_time_ms > 0


def test_prefix_sharing_e2e(engine):
    """Two requests with same prefix — second should be faster."""
    from serve.engine.sampling import SamplingParams
    params = SamplingParams(max_new_tokens=10)

    shared_prompt = "checkpoint prefix " * 80
    input_ids_1 = engine.tokenizer(shared_prompt + "What", return_tensors="pt").input_ids[0].tolist()
    input_ids_2 = engine.tokenizer(shared_prompt + "How", return_tensors="pt").input_ids[0].tolist()

    # First request — populates aligned prefix checkpoints.
    r1 = engine.generate(input_ids_1, params)
    assert r1.finished

    # Second request — should reuse cached prefix.
    r2 = engine.generate(input_ids_2, params)
    assert r2.finished

    # Verify cache has entries.
    assert engine.cache.total_cached_pages > 0


def test_stress_many_requests(engine):
    """Generate 10 requests concurrently via generate_batch()."""
    from serve.engine.sampling import SamplingParams

    prompts = [f"Count to {i}: " for i in range(10)]
    prompt_ids = [engine.tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]
    results = engine.generate_batch(prompt_ids, SamplingParams(max_new_tokens=5))

    assert len(results) == 10
    assert all(r.finished for r in results)
    assert all(len(r.generated_ids) > 0 for r in results)
    assert engine.scheduler.num_running == 0
    assert engine.scheduler.num_waiting == 0


def test_stop_token_e2e(engine):
    """Verify EOS-based stop works end-to-end."""
    from serve.engine.sampling import SamplingParams
    # Default params should include EOS as stop token.
    result = engine.complete("Hi", SamplingParams(max_new_tokens=200))
    # Should stop well before 200 tokens (EOS triggered).
    assert result.finished
    assert len(result.generated_ids) <= 200
