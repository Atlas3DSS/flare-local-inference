"""`flare bench`: run the fixed HTTP workload against a running server.

The workload and its stream analysis come from `qualify.performance`, which a
source checkout provides: exact-length cold prompts, a cached-prefix request,
and decode batches at concurrency 1, 2, and 4. Figures describe what the server
delivered to this client.
"""

import http.client
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from flare.errors import FlareError, UsageError
from flare.recipes import resolve

from . import display
from .config import Configuration
from .status import fetch, server_url

_REPETITIONS = 5
_TIMEOUT_SECONDS = 300
_TTFT_KINDS = (("ttft-512", "cold 512-token prompt"), ("ttft-4096", "cold 4,096-token prompt"),
               ("prefix-hit", "cached 4,096-token prefix"))


def _stream(url, key, request):
    """Post one streaming completion and timestamp each server-sent event at arrival."""
    target = urlsplit(url)
    connection = (http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection)(
        target.hostname, target.port, timeout=_TIMEOUT_SECONDS)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    events, data = [], []
    try:
        connection.connect()
        started = time.perf_counter_ns()
        connection.request("POST", target.path.rstrip("/") + "/v1/chat/completions",
                           body=json.dumps(request).encode(), headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise FlareError(f"HTTP {response.status}: {response.read(4096).decode(errors='replace')}")
        while line := response.readline(1 << 20):
            stripped = line.rstrip(b"\r\n")
            if stripped.startswith(b"data:"):
                data.append(stripped[5:].removeprefix(b" "))
            elif not stripped and data:
                value = b"\n".join(data).decode()
                data.clear()
                events.append({"elapsed_ns": time.perf_counter_ns() - started,
                               "data": value if value == "[DONE]" else json.loads(value)})
                if value == "[DONE]":
                    break
    finally:
        connection.close()
    return events


def measure(url, key, corpus, analyze):
    """Run every workload group except the 65,536-token prefill and analyze each stream."""
    cases = {case["id"]: case for case in corpus["cases"]}
    results = {}

    def run(case, barrier):
        request = {**case["request"], "stream": True, "return_token_ids": True, "return_token_timestamps": True,
                   "stream_options": {"include_usage": True}}
        barrier.wait(timeout=_TIMEOUT_SECONDS)
        return analyze(_stream(url, key, request), case, backend="flare")

    for group in corpus["groups"]:
        if group["kind"] == "prefill-65536":
            continue
        barrier = threading.Barrier(group["batch"])
        with ThreadPoolExecutor(group["batch"]) as pool:
            analyses = list(pool.map(lambda name: run(cases[name], barrier), group["cases"]))
        if group["measured"]:
            results.setdefault(group["kind"], []).append(analyses)
    return results


def summarize(results) -> dict:
    """Reduce analyzed streams to latency and throughput figures with their request counts."""
    failures = sorted({name for groups in results.values() for analyses in groups for analysis in analyses
                       for name, passed in analysis["checks"].items() if not passed})
    if failures:
        raise FlareError(f"the server's responses failed workload checks: {', '.join(failures)}")
    summary = {"time_to_first_token": {}, "decode": {}}
    for kind, _ in _TTFT_KINDS:
        values = [analysis["delivery_ttft_ns"] / 1e6 for analyses in results.get(kind, ()) for analysis in analyses]
        summary["time_to_first_token"][kind] = {"median_ms": statistics.median(values), "requests": len(values)}
    for batch in (1, 2, 4):
        groups = results.get(f"decode-b{batch}", ())
        intervals = sorted(interval / 1e6 for analyses in groups for analysis in analyses
                           for interval in analysis["core_itl_ns"])
        # Each batch's throughput sums its requests' token rates between first and last completion.
        rates = [sum((len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])
                     for stamps in (analysis["core_token_timestamps_ns"] for analysis in analyses))
                 for analyses in groups]
        summary["decode"][batch] = {
            "inter_token_median_ms": statistics.median(intervals),
            "inter_token_p99_ms": intervals[-(-99 * len(intervals) // 100) - 1],
            "tokens_per_second": statistics.median(rates), "requests": sum(map(len, groups))}
    return summary


def report(summary: dict, model: str, url: str):
    """Tabulate time to first token by prompt and decode latency and throughput by concurrency."""
    first_token = [(label, f"{figure['median_ms']:.1f} ms", str(figure["requests"]))
                   for label, figure in ((label, summary["time_to_first_token"][kind]) for kind, label in _TTFT_KINDS)]
    decode = [(f"concurrency {batch}", f"{figure['inter_token_median_ms']:.1f} ms",
               f"{figure['inter_token_p99_ms']:.1f} ms", f"{figure['tokens_per_second']:.1f}",
               str(figure["requests"])) for batch, figure in summary["decode"].items()]
    return display.section(
        display.title(model, url),
        display.table(("time to first token", "median", "requests"), first_token, numeric={1, 2}),
        display.table(("decode", "inter-token median", "p99", "tokens/s", "requests"), decode,
                      numeric={1, 2, 3, 4}))


def bench(args) -> int:
    configuration = Configuration.load(args.config)
    url = server_url(configuration, args.url)
    names = [entry["id"] for entry in json.loads(fetch(configuration, url, "/v1/models"))["data"]]
    try:
        from qualify.performance import MODEL_ID, analyze_events, build_corpus
    except ImportError as error:
        raise FlareError("bench runs the workload in qualify/performance.py, which requires a source checkout"
                         f" and the preparation environment: {error}") from error
    if MODEL_ID not in names:
        raise UsageError(f"bench has a workload only for {resolve(MODEL_ID).checkpoint.alias};"
                         f" {url} serves {', '.join(names)}")
    snapshot = resolve(MODEL_ID).snapshot(configuration.directory("hub_cache"))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    except Exception as error:  # noqa: BLE001 - A missing snapshot and a missing library read the same to the user.
        raise FlareError(f"bench renders prompts with the tokenizer in {snapshot}: {error}") from error
    summary = summarize(measure(url, configuration.get("server.api_key").value,
                                build_corpus(tokenizer, repetitions=_REPETITIONS), analyze_events))
    if args.json:
        print(json.dumps({"url": url, "model": MODEL_ID, **summary}, indent=2))
        return 0
    display.show(report(summary, resolve(MODEL_ID).checkpoint.alias, url))
    return 0
