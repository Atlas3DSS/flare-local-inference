"""`flare bench` figures from analyzed streams."""

import pytest

from flare.cli import bench
from flare.errors import FlareError

MS = 1_000_000


def analysis(ttft_ms=10.0, intervals_ms=(), passed=True):
    stamps = [0]
    for interval in intervals_ms:
        stamps.append(stamps[-1] + int(interval * MS))
    return {"checks": {"no_stream_error": True, "exact_output_length": passed},
            "delivery_ttft_ns": int(ttft_ms * MS), "core_itl_ns": [int(value * MS) for value in intervals_ms],
            "core_token_timestamps_ns": stamps}


def results(**overrides):
    values = {
        "ttft-512": [[analysis(ttft_ms=value)] for value in (40, 44, 42)],
        "ttft-4096": [[analysis(ttft_ms=value)] for value in (210, 190)],
        "prefix-hit": [[analysis(ttft_ms=18)]],
        # One request emits five tokens in 20 ms; a pair then emits at half that rate each.
        "decode-b1": [[analysis(intervals_ms=(5, 5, 5, 5))]],
        "decode-b2": [[analysis(intervals_ms=(10, 10, 10, 10))] * 2],
        "decode-b4": [[analysis(intervals_ms=(1, 1, 1, 97))] * 4, [analysis(intervals_ms=(25,) * 4)] * 4],
    }
    values.update(overrides)
    return values


def test_figures_carry_medians_and_request_counts():
    summary = bench.summarize(results())
    assert summary["time_to_first_token"] == {
        "ttft-512": {"median_ms": 42, "requests": 3}, "ttft-4096": {"median_ms": 200, "requests": 2},
        "prefix-hit": {"median_ms": 18, "requests": 1}}
    assert summary["decode"][1] == {"inter_token_median_ms": 5, "inter_token_p99_ms": 5,
                                    "tokens_per_second": 200, "requests": 1}
    assert summary["decode"][2]["tokens_per_second"] == 200 and summary["decode"][2]["requests"] == 2
    assert summary["decode"][4]["inter_token_p99_ms"] == 97 and summary["decode"][4]["requests"] == 8
    assert summary["decode"][4]["tokens_per_second"] == 160


def test_failed_workload_checks_are_runtime_errors():
    with pytest.raises(FlareError, match="failed workload checks: exact_output_length"):
        bench.summarize(results(**{"ttft-512": [[analysis(passed=False)]]}))


def test_measure_skips_the_long_prefill_and_unmeasured_groups(monkeypatch):
    corpus = {"cases": [{"id": name, "request": {"model": "m"}} for name in ("w", "a", "b", "p")],
              "groups": [{"kind": "decode", "batch": 1, "measured": False, "cases": ["w"]},
                         {"kind": "decode-b2", "batch": 2, "measured": True, "cases": ["a", "b"]},
                         {"kind": "prefill-65536", "batch": 1, "measured": True, "cases": ["p"]}]}
    sent = []

    def stream(url, key, request):
        sent.append((url, key, request))
        return ["event"]

    monkeypatch.setattr(bench, "_stream", stream)
    measured = bench.measure("http://host", "secret", corpus, lambda events, case, backend: (case["id"], events, backend))
    assert measured == {"decode-b2": [[("a", ["event"], "flare"), ("b", ["event"], "flare")]]}
    assert len(sent) == 3 and all(key == "secret" and request["stream"] and request["return_token_timestamps"]
                                  for _, key, request in sent)
