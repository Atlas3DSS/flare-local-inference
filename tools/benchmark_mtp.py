#!/usr/bin/env python3
"""Benchmark an already running native Flare MTP server on fixed coding prompts.

No server is started. Run once for each externally configured depth, for example:
  python benchmark_mtp.py --depth 2 --output native-depth2.json
Native token IDs, token logprobs and cumulative usage determine token counts;
SSE message counts never stand in for generated tokens. Repeated prompts may
reuse prefixes, and every request records the server's reported cache reuse.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.error
import urllib.request


DEFAULT_IDS = ("code-00", "code-01", "code-02", "code-03")
TAU_UNAVAILABLE = (
    "Native verify_steps counts graphs, including terminal no-ops, rather than progressing request rounds. "
    "draft_rounds excludes width-one/no-offer request rounds. generated-minus-accepted also misses "
    "budget-clipped rounds with no bonus. Use measure_mtp.measure on retained graphs for exact tau."
)


def load_prompts(path, prompt_ids=DEFAULT_IDS):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("prompt IDs must be unique")
    selected = rows if prompt_ids is None else [by_id[name] for name in prompt_ids]
    if not selected or any(not isinstance(row.get("messages"), list) for row in selected):
        raise ValueError("selected prompts must contain chat messages")
    return selected


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def native_logprob_tokens(logprobs):
    """Count native records in completion or chat-completion logprob schemas."""
    if logprobs is None:
        return 0
    tokens, content = logprobs.get("tokens"), logprobs.get("content")
    if tokens is not None and not isinstance(tokens, list):
        raise ValueError("logprobs.tokens must be a native token list")
    if content is not None and (not isinstance(content, list) or any(
            not isinstance(record, dict) or not (type(record.get("id")) is int
            or isinstance(record.get("token"), str)) for record in content)):
        raise ValueError("logprobs.content must contain native token records")
    if tokens is not None and content is not None and len(tokens) != len(content):
        raise ValueError("native logprob token schemas disagree")
    return len(content if content is not None else tokens or [])


class TokenObservations:
    def __init__(self, output_tokens):
        self.limit = output_tokens
        self.ids, self.timestamps, self.events, self.samples = [], [], [], []
        self.logprob_count = self.usage_count = self.observed = self.first_count = 0
        self.usage = self.first = self.last = None
        self.text, self.reasoning, self.finish_reasons = [], [], []

    def consume(self, event, elapsed):
        self.events.append({"seconds": elapsed, "event": event})
        if "error" in event:
            raise RuntimeError(event["error"])
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("benchmark expects one completion per request")
            ids = choice.get("token_ids", [])
            if not isinstance(ids, list) or any(type(token) is not int for token in ids):
                raise ValueError("token_ids must contain native integer IDs")
            self.ids.extend(ids)
            self.timestamps.extend(choice.get("token_timestamps_ns", []))
            self.logprob_count += native_logprob_tokens(choice.get("logprobs"))
            delta = choice.get("delta", {})
            self.text.append(choice.get("text", "") or delta.get("content", "") or "")
            self.reasoning.append(delta.get("reasoning_content", "") or "")
            if choice.get("finish_reason") is not None:
                self.finish_reasons.append(choice["finish_reason"])
        if event.get("usage") is not None:
            self.usage = event["usage"]
            count = self.usage.get("completion_tokens", 0)
            if type(count) is not int or count < self.usage_count:
                raise ValueError("completion usage must be a monotonic native token count")
            self.usage_count = count
        count = max(len(self.ids), self.logprob_count, self.usage_count)
        if count > self.limit:
            raise ValueError("native output-token count exceeds requested workload")
        if count > self.observed:
            if self.first is None:
                self.first, self.first_count = elapsed, count
            self.last, self.observed = elapsed, count
            self.samples.append([elapsed, count])

    def result(self):
        duration = None if self.first is None or self.last <= self.first else self.last - self.first
        return {"output_tokens": self.observed, "first_token_seconds": self.first,
            "first_observed_tokens": self.first_count, "last_token_seconds": self.last,
            "decode_tokens_per_second": None if duration is None else (self.observed - self.first_count) / duration,
            "usage": self.usage, "native_logprob_tokens": self.logprob_count,
            "token_ids": self.ids, "token_timestamps_ns": self.timestamps,
            "observations": self.samples, "raw_events": self.events,
            "output_text": "".join(self.text), "reasoning_text": "".join(self.reasoning),
            "finish_reasons": self.finish_reasons,
            "cached_tokens": None if self.usage is None else self.usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)}


def request_body(model, prompt, output_tokens, seed, *, logprobs):
    body = {"model": model, "messages": prompt["messages"], "max_tokens": output_tokens,
        "stream": True, "ignore_eos": True, "seed": seed,
        "temperature": .7, "top_p": .95, "top_k": 20, "min_p": 0.,
        "repetition_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.,
        "return_token_ids": True, "return_token_timestamps": True,
        "chat_template_kwargs": {"enable_thinking": bool(prompt.get("thinking", False))},
        "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    if logprobs:
        body.update(logprobs=True, top_logprobs=1)
    if prompt.get("tools"):
        body.update(tools=prompt["tools"], tool_choice="auto")
    return body


def completion(base, body, prompt_id, barrier, timeout):
    observed = TokenObservations(body["max_tokens"])
    result = {"prompt_id": prompt_id, "seed": body["seed"], "raw_request": body, "status": "failed"}
    barrier.wait()
    started = time.perf_counter()
    result["started_monotonic_seconds"] = started
    done = False
    try:
        request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result["http_status"] = response.status
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                observed.consume(json.loads(payload), time.perf_counter() - started)
        if not done:
            raise RuntimeError("stream ended without [DONE]")
        if observed.usage is None or observed.usage_count != body["max_tokens"]:
            raise RuntimeError(f"server did not complete the exact requested workload: {observed.usage}")
        if observed.ids and len(observed.ids) != observed.usage_count:
            raise RuntimeError("native token IDs disagree with final completion usage")
        if observed.first is None or observed.last <= observed.first:
            raise RuntimeError("insufficient native token observations for decode timing")
        result["status"] = "complete"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, urllib.error.HTTPError):
            result["http_status"] = error.code
            result["error"]["response_body"] = error.read().decode(errors="replace")
    finally:
        result.update(observed.result(), elapsed_seconds=time.perf_counter() - started, stream_done=done)
    return result


def metrics(base):
    result = {"observed_at": datetime.now(timezone.utc).isoformat(), "errors": []}
    try:
        result["json"] = get_json(base + "/metrics?format=json")
    except Exception as error:
        result["errors"].append({"endpoint": "json", "message": str(error)})
    try:
        with urllib.request.urlopen(base + "/metrics?format=prometheus", timeout=30) as response:
            raw = response.read().decode()
        relevant = ("flare_speculative_", "flare_proposal_", "flare_generated_tokens_total",
                    "flare_accepted_draft_tokens_total", "flare_draft_rounds_total")
        result["prometheus"] = {match[1]: int(match[2]) for line in raw.splitlines()
            if line.startswith(relevant)
            and (match := re.fullmatch(r"(\S+)\s+(\d+)", line))}
    except Exception as error:
        result["errors"].append({"endpoint": "prometheus", "message": str(error)})
    return result


def metric_delta(before, after):
    left, right = before.get("prometheus", {}), after.get("prometheus", {})
    delta = {name: right[name] - value for name, value in left.items() if name in right}
    if any(value < 0 for value in delta.values()):
        return {"error": "metrics counters reset during the measured interval", "counters": delta}
    totals = {}
    for key in ("rounds_total", "gpu_nanoseconds_total", "emitted_tokens_total", "offered_drafts_total", "accepted_drafts_total"):
        totals[key] = sum(value for name, value in delta.items() if name.startswith("flare_speculative_" + key + "{"))
    rounds, offered = totals["rounds_total"], totals["offered_drafts_total"]
    return {"counters": delta, "combined_speculative_graphs": totals,
        "combined_graph_mean_gpu_ms": totals["gpu_nanoseconds_total"] / rounds / 1e6 if rounds else None,
        "draft_acceptance_rate": totals["accepted_drafts_total"] / offered if offered else None,
        "tau": None, "tau_unavailable_reason": TAU_UNAVAILABLE,
        "stage_timing_note": "Combined speculative graph CUDA event time includes drafting, verification, commit and absorption."}


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(runs, concurrency):
    selected = [run for run in runs if run["requested_concurrency"] == concurrency]
    all_requests = [request for run in selected for request in run["requests"]]
    success = [request for request in all_requests if request["status"] == "complete"]
    rates = [request["decode_tokens_per_second"] for request in success]
    per_agent = []
    for prompt_id in sorted({request["prompt_id"] for request in all_requests}):
        requests = [request for request in success if request["prompt_id"] == prompt_id]
        per_agent.append({"prompt_id": prompt_id, "successful_requests": len(requests),
            "median_decode_tokens_per_second": statistics.median(r["decode_tokens_per_second"] for r in requests) if requests else None})
    wall = sum(run["wall_seconds"] for run in selected)
    output = sum(request["output_tokens"] for request in success)
    return {"concurrency": concurrency, "successful_requests": len(success),
        "failed_requests": len(all_requests) - len(success), "per_agent": per_agent,
        "median_per_agent_decode_tokens_per_second": statistics.median(rates) if rates else None,
        "aggregate_output_tokens_per_second": output / wall if wall else None,
        "aggregate_basis": "successful native output tokens / measured batch wall time, including prefill and HTTP",
        "median_aggregate_decode_window_tokens_per_second": statistics.median(
            run["aggregate_decode_window_tokens_per_second"] for run in selected
            if "aggregate_decode_window_tokens_per_second" in run) if success else None,
        "decode_window_basis": "sum of tokens after each request's first observation / earliest first to latest last observation",
        "latency_seconds": {"p50": percentile([r["elapsed_seconds"] for r in success], .5),
                            "p95": percentile([r["elapsed_seconds"] for r in success], .95)},
        "first_token_seconds": {"p50": percentile([r["first_token_seconds"] for r in success], .5),
                                "p95": percentile([r["first_token_seconds"] for r in success], .95)},
        "tau": None, "tau_unavailable_reason": TAU_UNAVAILABLE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8180")
    parser.add_argument("--model")
    parser.add_argument("--depth", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=list(DEFAULT_IDS))
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7035)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.concurrencies, args.repetitions) < 1 or args.output_tokens < 2:
        parser.error("positive concurrency/repetitions and at least two output tokens are required")
    prompts = load_prompts(args.prompts, args.prompt_ids)
    if any(len(prompts) % concurrency for concurrency in args.concurrencies):
        parser.error("prompt count must divide into complete batches at each concurrency")
    with args.output.open("x"):
        pass
    report = {"status": "running", "depth_label": args.depth,
        "depth_note": "Selected externally by the loaded native package/runtime; the client does not alter draft depth.",
        "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sampling": {"temperature": .7, "top_k": 20, "top_p": .95, "ignore_eos": True},
        "cache_policy": "Unmodified prompts; repeated requests may reuse native prefixes; per-request cache usage retained.",
        "runs": [], "summary": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        models = get_json(args.base_url.rstrip("/") + "/v1/models")
        report["models"] = models
        card = next((item for item in models["data"] if args.model is None or item["id"] == args.model), None)
        if card is None:
            raise ValueError("requested model is not served")
        logprobs = card.get("flare", {}).get("max_top_logprobs", 0) > 0
        for concurrency in args.concurrencies:
            for repetition in range(args.repetitions):
                for offset in range(0, len(prompts), concurrency):
                    group = prompts[offset:offset + concurrency]
                    before = metrics(args.base_url.rstrip("/"))
                    barrier = threading.Barrier(concurrency + 1)
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [executor.submit(completion, args.base_url.rstrip("/"),
                            request_body(card["id"], prompt, args.output_tokens,
                                args.seed + repetition * len(prompts) + offset + index, logprobs=logprobs),
                            prompt["id"], barrier, args.timeout) for index, prompt in enumerate(group)]
                        started = time.perf_counter()
                        barrier.wait()
                        results = [future.result() for future in futures]
                        wall = time.perf_counter() - started
                    after = metrics(args.base_url.rstrip("/"))
                    run = {"repetition": repetition, "requested_concurrency": concurrency,
                        "wall_seconds": wall, "requests": results, "metrics_before": before,
                        "metrics_after": after, "metrics_delta": metric_delta(before, after)}
                    successful = [r for r in results if r["status"] == "complete"]
                    if successful:
                        first = min(r["started_monotonic_seconds"] + r["first_token_seconds"] for r in successful)
                        last = max(r["started_monotonic_seconds"] + r["last_token_seconds"] for r in successful)
                        run["aggregate_decode_window_tokens_per_second"] = (
                            sum(r["output_tokens"] - r["first_observed_tokens"] for r in successful) / (last - first))
                    report["runs"].append(run)
                    report["summary"] = [summarize(report["runs"], c) for c in args.concurrencies]
                    save()
                    print(f"depth={args.depth} concurrency={concurrency} repetition={repetition + 1} "
                          f"completed={len(successful)}/{concurrency} wall={wall:.3f}s", flush=True)
        report["status"] = "complete" if all(r["status"] == "complete"
            for run in report["runs"] for r in run["requests"]) else "failed"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        save()
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
