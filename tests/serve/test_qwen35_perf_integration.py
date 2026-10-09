"""End-to-end Qwen3.5 serve battery focused on long-context performance.

This test intentionally loads the real checkpoint and sweeps a fixed matrix of
batch sizes and prompt lengths. Each prompt carries a request-specific
verification code near the end so correctness remains easy to check even at
64k-token context.

Environment variables:
    B12X_SERVE_QWEN35_MODEL_PATH:
        Override the default checkpoint path.
    B12X_SERVE_QWEN35_GPU_IDS:
        Comma-separated GPU IDs for TP launch. Defaults to ``0,1,2,3``.
    B12X_SERVE_QWEN35_MIN_CASE_TOK_S_JSON:
        Optional JSON object mapping ``bs{B}_ctx{N}`` to minimum aggregate
        tokens/sec.
    B12X_SERVE_QWEN35_MAX_CASE_TTFT_MS_JSON:
        Optional JSON object mapping ``bs{B}_ctx{N}`` to maximum mean TTFT.
    B12X_SERVE_QWEN35_BATCH_SIZES:
        Optional comma-separated batch subset for debug reruns.
    B12X_SERVE_QWEN35_CONTEXT_LENGTHS:
        Optional comma-separated context-length subset for debug reruns.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest
import torch

from serve.tp.launch import launch_tp

torch.set_grad_enabled(False)

MODEL_PATH = "/data/models/Qwen3.5-397B-A17B-NVFP4-BF16shared"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_CONTEXT_LENGTHS = (128, 4096, 16384, 32768, 65536)
GRAPH_BATCH_SIZES = [1, 2, 4, 8]
MAX_NEW_TOKENS = 16


def _parse_gpu_ids() -> list[int]:
    raw = os.environ.get("B12X_SERVE_QWEN35_GPU_IDS", "0,1,2,3")
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def _load_threshold_map(env_name: str) -> dict[str, float]:
    raw = os.environ.get(env_name)
    if not raw:
        return {}
    data = json.loads(raw)
    return {str(key): float(value) for key, value in data.items()}


def _parse_int_list_env(env_name: str, defaults: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(env_name)
    if not raw:
        return defaults
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    return tuple(values)


def _battery_batch_sizes() -> tuple[int, ...]:
    return _parse_int_list_env("B12X_SERVE_QWEN35_BATCH_SIZES", DEFAULT_BATCH_SIZES)


def _battery_context_lengths() -> tuple[int, ...]:
    return _parse_int_list_env("B12X_SERVE_QWEN35_CONTEXT_LENGTHS", DEFAULT_CONTEXT_LENGTHS)


def _verification_code(batch_size: int, context_length: int, req_idx: int) -> str:
    context_index = _battery_context_lengths().index(context_length)
    code_num = batch_size * 100 + context_index * 10 + req_idx
    return f"{code_num:04d}"


def _build_prompt_ids(tokenizer, *, target_tokens: int, code: str) -> list[int]:
    if os.environ.get("B12X_SERVE_QWEN35_PROMPT_DEBUG"):
        print(
            f"[qwen35-perf] build_prompt_ids start target={target_tokens} code={code}",
            flush=True,
        )
    intro_ids = tokenizer.encode(
        "Long-context serving validation. The body below is filler.\n",
        add_special_tokens=False,
    )
    content_suffix_ids = tokenizer.encode(
        (
            f"\nIgnore the filler. The verification code is {code}. "
            f"Reply with exactly {code}.\nAnswer:"
        ),
        add_special_tokens=False,
    )
    wrapper_prefix_ids, wrapper_suffix_ids = _chat_wrapper_ids(tokenizer)
    fixed_tokens = (
        len(wrapper_prefix_ids)
        + len(intro_ids)
        + len(content_suffix_ids)
        + len(wrapper_suffix_ids)
    )
    if fixed_tokens > target_tokens:
        raise ValueError(
            f"target context {target_tokens} is too small for the verification template"
        )
    filler_unit_ids = tokenizer.encode(
        " filler block data for throughput measurement.",
        add_special_tokens=False,
    )
    filler_tokens = target_tokens - fixed_tokens
    repeats = (filler_tokens + len(filler_unit_ids) - 1) // len(filler_unit_ids)
    filler_ids = (filler_unit_ids * max(1, repeats))[:filler_tokens]
    prompt_ids = (
        wrapper_prefix_ids
        + intro_ids
        + filler_ids
        + content_suffix_ids
        + wrapper_suffix_ids
    )
    assert len(prompt_ids) == target_tokens
    if os.environ.get("B12X_SERVE_QWEN35_PROMPT_DEBUG"):
        print(
            f"[qwen35-perf] build_prompt_ids done target={target_tokens} code={code}",
            flush=True,
        )
    return prompt_ids


def _chat_wrapper_ids(tokenizer) -> tuple[list[int], list[int]]:
    cached = getattr(tokenizer, "_b12x_qwen35_perf_chat_wrapper_ids", None)
    if cached is not None:
        return cached
    if os.environ.get("B12X_SERVE_QWEN35_PROMPT_DEBUG"):
        print("[qwen35-perf] deriving chat wrapper ids", flush=True)

    def _chat_ids(content: str) -> list[int]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer.encode(rendered, add_special_tokens=False)

    a_ids = _chat_ids("A")
    b_ids = _chat_ids("B")
    prefix_len = 0
    while prefix_len < min(len(a_ids), len(b_ids)) and a_ids[prefix_len] == b_ids[prefix_len]:
        prefix_len += 1
    suffix_len = 0
    while (
        suffix_len < min(len(a_ids) - prefix_len, len(b_ids) - prefix_len)
        and a_ids[len(a_ids) - 1 - suffix_len] == b_ids[len(b_ids) - 1 - suffix_len]
    ):
        suffix_len += 1

    prefix_ids = a_ids[:prefix_len]
    suffix_ids = a_ids[len(a_ids) - suffix_len :]
    if a_ids[prefix_len : len(a_ids) - suffix_len] != tokenizer.encode("A", add_special_tokens=False):
        raise AssertionError("failed to isolate chat-template content prefix for Qwen3.5")
    if b_ids[prefix_len : len(b_ids) - suffix_len] != tokenizer.encode("B", add_special_tokens=False):
        raise AssertionError("failed to isolate chat-template content suffix for Qwen3.5")

    tokenizer._b12x_qwen35_perf_chat_wrapper_ids = (prefix_ids, suffix_ids)
    if os.environ.get("B12X_SERVE_QWEN35_PROMPT_DEBUG"):
        print(
            f"[qwen35-perf] derived chat wrapper ids prefix={len(prefix_ids)} suffix={len(suffix_ids)}",
            flush=True,
        )
    return prefix_ids, suffix_ids


def _build_case_specs(tokenizer) -> list[dict]:
    cases: list[dict] = []
    for batch_size in _battery_batch_sizes():
        for context_length in _battery_context_lengths():
            requests = []
            for req_idx in range(batch_size):
                code = _verification_code(batch_size, context_length, req_idx)
                requests.append(
                    {
                        "expected_code": code,
                        "prompt_ids": _build_prompt_ids(
                            tokenizer,
                            target_tokens=context_length,
                            code=code,
                        ),
                    }
                )
            cases.append(
                {
                    "case_id": f"bs{batch_size}_ctx{context_length}",
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "requests": requests,
                }
            )
    return cases


def _warmup_prompts(tokenizer) -> list[dict]:
    warmups = []
    for batch_size in _battery_batch_sizes():
        prompts = []
        for req_idx in range(batch_size):
            prompts.append(
                _build_prompt_ids(
                    tokenizer,
                    target_tokens=128,
                    code=f"{9000 + batch_size * 10 + req_idx:04d}",
                )
            )
        warmups.append({"batch_size": batch_size, "prompts": prompts})
    return warmups


def _write_partial_results(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def _run_qwen35_perf_battery(tp_group, model_path: str, results_path: str) -> None:
    rank = tp_group.rank if tp_group else 0
    device = f"cuda:{tp_group.device.index}" if tp_group else "cuda"

    from serve.engine.sampling import SamplingParams
    from serve.engine.serving import ServingEngine

    if rank == 0:
        print(f"[qwen35-perf] initializing engine on {device}", flush=True)

    engine = ServingEngine(
        model_path,
        device=device,
        tp_group=tp_group,
        warmup_prefill_lengths=[128],
        graph_batch_sizes=GRAPH_BATCH_SIZES,
    )

    if rank != 0:
        engine.run_follower()
        return

    try:
        print("[qwen35-perf] engine ready", flush=True)
        warmup_cases = _warmup_prompts(engine.tokenizer)
        for warmup in warmup_cases:
            print(
                f"[qwen35-perf] warmup batch={warmup['batch_size']} count={len(warmup['prompts'])}",
                flush=True,
            )
            engine.generate_batch(
                warmup["prompts"],
                SamplingParams.greedy(max_new_tokens=2),
            )
        torch.cuda.synchronize()

        battery_results = {
            "model_path": model_path,
            "tp": tp_group.world_size if tp_group else 1,
            "gpu_ids": _parse_gpu_ids(),
            "graph_batch_sizes": GRAPH_BATCH_SIZES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "cases": [],
        }

        for case in _build_case_specs(engine.tokenizer):
            print(
                f"[qwen35-perf] case {case['case_id']} starting",
                flush=True,
            )
            prompts = [request["prompt_ids"] for request in case["requests"]]
            params = SamplingParams.greedy(max_new_tokens=MAX_NEW_TOKENS)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = engine.generate_batch(prompts, params)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0

            request_results = []
            total_generated_tokens = 0
            total_prompt_tokens = 0
            verified = True
            ttft_values = []

            for request_spec, result in zip(case["requests"], results, strict=True):
                decoded = engine.tokenizer.decode(
                    result.generated_ids,
                    skip_special_tokens=True,
                )
                request_verified = _normalize_text(decoded).find(
                    request_spec["expected_code"]
                ) != -1
                verified = verified and request_verified
                total_generated_tokens += len(result.generated_ids)
                total_prompt_tokens += len(request_spec["prompt_ids"])
                ttft_values.append(result.time_to_first_token_ms)
                request_results.append(
                    {
                        "expected_code": request_spec["expected_code"],
                        "verified": request_verified,
                        "prompt_tokens": len(request_spec["prompt_ids"]),
                        "generated_tokens": len(result.generated_ids),
                        "decoded_text": decoded,
                        "finish_reason": result.finish_reason,
                        "ttft_ms": result.time_to_first_token_ms,
                        "total_time_ms": result.total_time_ms,
                    }
                )

            battery_results["cases"].append(
                {
                    "case_id": case["case_id"],
                    "batch_size": case["batch_size"],
                    "context_length": case["context_length"],
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_generated_tokens": total_generated_tokens,
                    "wall_elapsed_s": elapsed_s,
                    "aggregate_tok_per_s": (
                        total_generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
                    ),
                    "mean_ttft_ms": sum(ttft_values) / len(ttft_values),
                    "all_verified": verified,
                    "requests": request_results,
                }
            )
            _write_partial_results(results_path, battery_results)
            print(
                f"[qwen35-perf] case {case['case_id']} done "
                f"tok/s={battery_results['cases'][-1]['aggregate_tok_per_s']:.2f} "
                f"verified={verified}",
                flush=True,
            )

        _write_partial_results(results_path, battery_results)
    finally:
        engine.shutdown()


def _require_battery_prereqs() -> None:
    model_path = os.environ.get("B12X_SERVE_QWEN35_MODEL_PATH", MODEL_PATH)
    if not Path(model_path).exists():
        pytest.skip(f"Qwen3.5 checkpoint not available: {model_path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    gpu_ids = _parse_gpu_ids()
    if torch.cuda.device_count() < len(gpu_ids):
        pytest.skip(
            f"Need {len(gpu_ids)} visible GPUs for TP battery, found {torch.cuda.device_count()}"
        )


@pytest.fixture(scope="module")
def qwen35_perf_battery(tmp_path_factory) -> dict:
    _require_battery_prereqs()
    results_path = tmp_path_factory.mktemp("qwen35-perf") / "battery.json"
    model_path = os.environ.get("B12X_SERVE_QWEN35_MODEL_PATH", MODEL_PATH)
    gpu_ids = _parse_gpu_ids()
    launch_tp(
        _run_qwen35_perf_battery,
        world_size=len(gpu_ids),
        args=(model_path, str(results_path)),
        gpu_ids=gpu_ids,
    )
    return json.loads(results_path.read_text())


def test_prompt_builder_hits_exact_context_length():
    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [sum(ord(ch) for ch in token) for token in text.split()]

        def apply_chat_template(
            self,
            messages,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
            return_tensors=None,
        ):
            del add_generation_prompt, enable_thinking, return_tensors
            assert tokenize is False
            content = messages[0]["content"]
            return "PREFIX " + content + " SUFFIX"

    prompt_ids = _build_prompt_ids(_FakeTokenizer(), target_tokens=128, code="ABC123")
    assert len(prompt_ids) == 128


def test_qwen35_perf_battery_verifies_outputs_and_emits_metrics(qwen35_perf_battery):
    assert qwen35_perf_battery["graph_batch_sizes"] == GRAPH_BATCH_SIZES
    assert len(qwen35_perf_battery["cases"]) == (
        len(_battery_batch_sizes()) * len(_battery_context_lengths())
    )

    min_tok_s = _load_threshold_map("B12X_SERVE_QWEN35_MIN_CASE_TOK_S_JSON")
    max_ttft_ms = _load_threshold_map("B12X_SERVE_QWEN35_MAX_CASE_TTFT_MS_JSON")

    failures: list[str] = []
    for case in qwen35_perf_battery["cases"]:
        assert case["all_verified"], f"{case['case_id']} failed verification"
        assert case["wall_elapsed_s"] > 0
        assert case["total_generated_tokens"] > 0
        assert case["aggregate_tok_per_s"] > 0
        for request in case["requests"]:
            assert request["verified"], (
                f"{case['case_id']} failed for {request['expected_code']}: "
                f"{request['decoded_text']!r}"
            )
            assert request["finish_reason"] is not None
            assert request["generated_tokens"] > 0
            assert request["ttft_ms"] >= 0
            assert request["total_time_ms"] >= request["ttft_ms"]

        case_id = case["case_id"]
        if case_id in min_tok_s and case["aggregate_tok_per_s"] < min_tok_s[case_id]:
            failures.append(
                f"{case_id} aggregate throughput {case['aggregate_tok_per_s']:.2f} tok/s "
                f"< required {min_tok_s[case_id]:.2f} tok/s"
            )
        if case_id in max_ttft_ms and case["mean_ttft_ms"] > max_ttft_ms[case_id]:
            failures.append(
                f"{case_id} mean TTFT {case['mean_ttft_ms']:.2f} ms "
                f"> allowed {max_ttft_ms[case_id]:.2f} ms"
            )

    assert not failures, "\n".join(failures)
