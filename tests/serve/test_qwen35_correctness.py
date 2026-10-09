#!/usr/bin/env python3
"""Correctness test for Qwen3.5-397B on TP=4.

Runs the actual model on the exact raw completion prompt IDs used by the
sglang baseline and writes a comparable trace file on rank 0.

Usage:
    python tests/serve/test_qwen35_correctness.py
    python tests/serve/test_qwen35_correctness.py --gpu-ids 0,1,2,3
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from tests.serve.qwen35_trace import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT,
    RAW_COMPLETION_PROMPT_IDS,
    build_prompt_spec,
    write_trace,
)

torch.set_grad_enabled(False)

TP = 4


def _run(tp_group, model_path: str, trace_out: str) -> None:
    rank = tp_group.rank if tp_group else 0
    device = f"cuda:{tp_group.device.index}" if tp_group else "cuda"

    from serve.engine.sampling import SamplingParams
    from serve.engine.serving import ServingEngine

    engine = ServingEngine(
        model_path,
        device=device,
        tp_group=tp_group,
        graph_batch_sizes=[1, 2, 4, 8],
    )

    if rank != 0:
        engine.run_follower()
        return

    prompt_spec = build_prompt_spec(
        model_path=model_path,
        prompt=DEFAULT_PROMPT,
        prompt_ids=RAW_COMPLETION_PROMPT_IDS,
        chat=False,
    )
    prompt_ids = prompt_spec["prompt_ids"]
    greedy = SamplingParams(temperature=0.0, max_new_tokens=30)
    failures: list[str] = []
    trace = {
        "backend": "b12x",
        "prompt": prompt_spec,
        "tests": [],
    }

    print("Prompt under test:", prompt_spec["decoded_prompt"], flush=True)
    print(f"Prompt IDs: {prompt_ids}", flush=True)

    print("Test 1: Raw token completion (greedy)...", flush=True)
    result = engine.generate(prompt_ids, greedy)
    text = engine.tokenizer.decode(result.generated_ids, skip_special_tokens=True)
    trace["tests"].append(
        {
            "name": "raw_completion_greedy",
            "generated_ids": result.generated_ids,
            "generated_text": text,
            "finish_reason": result.finish_reason,
            "ttft_ms": result.time_to_first_token_ms,
            "total_time_ms": result.total_time_ms,
        }
    )
    print(f"  Output: {text!r}")
    if "Paris" not in text and "paris" not in text.lower():
        failures.append(f"Test 1 FAIL: expected 'Paris' in output, got {text!r}")
    else:
        print("  PASS")

    print("Test 2: Greedy determinism...", flush=True)
    r1 = engine.generate(prompt_ids, SamplingParams(temperature=0.0, max_new_tokens=20))
    r2 = engine.generate(prompt_ids, SamplingParams(temperature=0.0, max_new_tokens=20))
    t1 = engine.tokenizer.decode(r1.generated_ids, skip_special_tokens=True)
    t2 = engine.tokenizer.decode(r2.generated_ids, skip_special_tokens=True)
    trace["tests"].append(
        {
            "name": "greedy_determinism",
            "run_1_ids": r1.generated_ids,
            "run_1_text": t1,
            "run_2_ids": r2.generated_ids,
            "run_2_text": t2,
        }
    )
    if r1.generated_ids != r2.generated_ids:
        failures.append(
            "Test 2 FAIL: greedy not deterministic\n"
            f"  Run 1: {t1!r}\n"
            f"  Run 2: {t2!r}"
        )
    else:
        print("  PASS")

    engine.shutdown()

    trace["failures"] = failures
    write_trace(trace_out, trace)
    print(f"Trace written to {trace_out}")

    if failures:
        print(f"\n{'=' * 60}")
        print(f"FAILED: {len(failures)} test(s)")
        for failure in failures:
            print(f"  {failure}")
        print(f"{'=' * 60}")
        sys.exit(1)

    print("\nAll tests passed.")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--trace-out",
        type=str,
        default="/tmp/b12x_qwen35_correctness_trace.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    assert len(gpu_ids) == TP, f"Need exactly {TP} GPUs, got {len(gpu_ids)}"

    from serve.tp.launch import launch_tp

    launch_tp(
        _run,
        world_size=TP,
        args=(args.model, args.trace_out),
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    main()
