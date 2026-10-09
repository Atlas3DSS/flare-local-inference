#!/usr/bin/env python3
"""Run the canonical sglang baseline trace for Qwen3.5.

Uses the same exact raw completion prompt IDs as the b12x correctness test
and writes a comparable trace file.

Usage:
    python tests/serve/run_sglang_baseline.py
    python tests/serve/run_sglang_baseline.py --gpu-ids 4,5,6,7
"""

from __future__ import annotations

import argparse
import os

from tests.serve.qwen35_trace import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT,
    RAW_COMPLETION_PROMPT_IDS,
    build_prompt_spec,
    run_sglang,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--out", default="/tmp/sglang_qwen35_baseline_trace.json")
    args = parser.parse_args()

    os.environ.setdefault("SGLANG_STEP_LOG", "1")

    prompt_spec = build_prompt_spec(
        model_path=args.model,
        prompt=DEFAULT_PROMPT,
        prompt_ids=RAW_COMPLETION_PROMPT_IDS,
        chat=False,
    )
    print("Prompt under test:", prompt_spec["decoded_prompt"], flush=True)
    print(f"Prompt IDs: {prompt_spec['prompt_ids']}", flush=True)

    run_sglang(
        model_path=args.model,
        prompt_spec=prompt_spec,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_path=args.out,
        tp=len(args.gpu_ids.split(",")),
        logprobs=args.logprobs,
        visible_devices=args.gpu_ids,
    )
    print(f"Trace written to {args.out}")

    if os.path.exists("/tmp/sglang_step_log.json"):
        print("Step log at /tmp/sglang_step_log.json")


if __name__ == "__main__":
    main()
