"""Finite synthetic workloads that expose interference and acceptance heterogeneity."""

from .model import Request


WORKLOADS = ("prefill_burst", "decode_contention", "acceptance_groups", "short_arrivals", "long_context", "slot_pressure")


def workload(name: str) -> list[Request]:
    if name == "prefill_burst":
        return [Request("large-prompt", 0, 8192, 64)] + [
            Request(f"small-{i}", 12 + 17 * i, 64, 16) for i in range(8)
        ]
    if name == "decode_contention":
        return [Request(f"decode-{i}", 0, 0, 160, (0.85,) * 5) for i in range(8)]
    if name == "acceptance_groups":
        return [Request(f"accept-{i}", 0, 0, 128, ((0.97 if i < 4 else 0.1),) * 5)
                for i in range(8)]
    if name == "short_arrivals":
        return [Request(f"long-{i}", 0, 128, 200) for i in range(3)] + [
            Request(f"short-{i}", 15 + 20 * i, 64, 4) for i in range(10)
        ]
    if name == "long_context":
        return [Request(f"context-{i}", 0, 128, 64, (0.8,) * 5,
                        cached_tokens=32768 if i < 2 else 512) for i in range(6)]
    if name == "slot_pressure":
        return [Request(f"resident-{i}", 0, 64, 200) for i in range(2)] + [
            Request(f"queued-{i}", 10 + 10 * i, 64, 4) for i in range(4)
        ]
    raise ValueError(f"unknown workload: {name}")
