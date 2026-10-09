"""Batched throughput benchmark for the serving stack.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python tests/serve/bench_throughput.py \
        /data/models/MiniMax-M2.5-NVFP4 --tp 2 --gpu-ids 0,1
"""

import json
import sys
import time
import torch
torch.set_grad_enabled(False)

from serve.engine.serving import ServingEngine
from serve.engine.sampling import SamplingParams
from serve.tp.launch import launch_tp


def _record_metric(metrics: dict, name: str, *, tokens: int, elapsed_s: float, ttft_ms: float | None = None):
    metrics[name] = {
        "tokens": tokens,
        "elapsed_s": elapsed_s,
        "tok_per_s": tokens / elapsed_s,
        "ttft_ms": ttft_ms,
    }


def run(tp_group, model_path, json_out=None, min_bs1_tok_s=None, min_bs4_tok_s=None, min_bs8_tok_s=None):
    rank = tp_group.rank if tp_group else 0
    device = f"cuda:{tp_group.device.index}" if tp_group else "cuda"
    engine = ServingEngine(model_path, device=device, tp_group=tp_group,
        warmup_prefill_lengths=[4, 64],
        graph_batch_sizes=[1, 2, 4, 8])

    if rank != 0:
        engine.run_follower()
        return

    metrics = {}

    # Warmup: cover the shapes that will be used in timed runs.
    engine.complete("Warmup short", SamplingParams(max_new_tokens=5))
    engine.complete("Warmup longer generation to trigger autotuner", SamplingParams(max_new_tokens=100))
    engine.complete("Warmup final", SamplingParams(max_new_tokens=5))
    torch.cuda.synchronize()

    # bs=1.
    torch.cuda.synchronize()
    t0 = time.time()
    r = engine.generate(
        engine.tokenizer("Test prompt here", return_tensors="pt").input_ids[0].tolist(),
        SamplingParams(max_new_tokens=100))
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n = len(r.generated_ids)
    _record_metric(metrics, "bs1", tokens=n, elapsed_s=elapsed, ttft_ms=r.time_to_first_token_ms)
    print(f"bs=1: {n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s, TTFT={r.time_to_first_token_ms:.0f}ms")

    # bs=4.
    prompts_text = ["Tell me about stars", "What is math", "Explain physics", "Write a poem"]
    prompt_ids = [engine.tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts_text]
    torch.cuda.synchronize()
    t0 = time.time()
    results = engine.generate_batch(prompt_ids, SamplingParams(max_new_tokens=50))
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    total = sum(len(r.generated_ids) for r in results)
    _record_metric(metrics, "bs4", tokens=total, elapsed_s=elapsed)
    print(f"bs=4: {total} tokens in {elapsed:.2f}s = {total/elapsed:.1f} tok/s")

    # bs=8.
    prompts_text = [f"Topic {i}: explain briefly" for i in range(8)]
    prompt_ids = [engine.tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts_text]
    torch.cuda.synchronize()
    t0 = time.time()
    results = engine.generate_batch(prompt_ids, SamplingParams(max_new_tokens=50))
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    total = sum(len(r.generated_ids) for r in results)
    _record_metric(metrics, "bs8", tokens=total, elapsed_s=elapsed)
    print(f"bs=8: {total} tokens in {elapsed:.2f}s = {total/elapsed:.1f} tok/s")

    # Long prompt (chunked prefill).
    long_prompt = " ".join(["The"] * 200)  # ~200 tokens.
    long_ids = engine.tokenizer(long_prompt, return_tensors="pt").input_ids[0].tolist()
    torch.cuda.synchronize()
    t0 = time.time()
    r = engine.generate(long_ids, SamplingParams(max_new_tokens=20))
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n = len(r.generated_ids)
    chunks = (len(long_ids) + 511) // 512  # Approximate chunk count.
    _record_metric(metrics, "long_prompt", tokens=n, elapsed_s=elapsed, ttft_ms=r.time_to_first_token_ms)
    print(f"long prompt ({len(long_ids)} tokens, ~{chunks} chunks): {n} decode tokens in {elapsed:.2f}s, TTFT={r.time_to_first_token_ms:.0f}ms")

    if json_out is not None:
        with open(json_out, "w") as f:
            json.dump(metrics, f, indent=2)

    thresholds = {
        "bs1": min_bs1_tok_s,
        "bs4": min_bs4_tok_s,
        "bs8": min_bs8_tok_s,
    }
    failures = []
    for name, threshold in thresholds.items():
        if threshold is None:
            continue
        actual = metrics[name]["tok_per_s"]
        if actual < threshold:
            failures.append(f"{name} throughput {actual:.1f} tok/s < required {threshold:.1f} tok/s")

    engine.shutdown()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--min-bs1-tok-s", type=float, default=None)
    parser.add_argument("--min-bs4-tok-s", type=float, default=None)
    parser.add_argument("--min-bs8-tok-s", type=float, default=None)
    args = parser.parse_args()

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    launch_tp(
        run,
        world_size=args.tp,
        args=(
            args.model_path,
            args.json_out,
            args.min_bs1_tok_s,
            args.min_bs4_tok_s,
            args.min_bs8_tok_s,
        ),
        gpu_ids=gpu_ids,
    )
