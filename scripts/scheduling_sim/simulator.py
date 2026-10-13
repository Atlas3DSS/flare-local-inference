"""Nonpreemptive actions, boundary admission, and censored speculative outcomes."""

from dataclasses import dataclass, field
from random import Random

from .model import Action, Config, CostModel, Request, View, Work, delivery_reference, expected_tokens
from .scheduler import Scheduler


@dataclass
class State:
    request: Request
    index: int
    uniforms: list[tuple[float, ...]]
    prefill_credit: float
    decode_credit: float
    prefetched: int = 0
    generated: int = 0
    admitted: float | None = None
    completed: float | None = None
    first_output: float | None = None
    last_output: float | None = None
    last_service: float = 0.0
    entitlement: float = 0.0
    progress: float = 0.0
    debt: float = 0.0
    max_lag: float = 0.0
    max_gap: float = 0.0
    output_gaps: list[float] = field(default_factory=list)
    successes: list[float] = field(default_factory=list)
    failures: list[float] = field(default_factory=list)
    offered: int = 0
    accepted: int = 0
    decode_rounds: int = 0
    max_delivery_slowdown: float = 0.0
    worst_delivery_token: int = 0
    service_gap_violations: int = 0

    @property
    def first_token_reference(self) -> float:
        return (self.request.prompt_tokens * self.prefill_credit
                if self.request.prompt_tokens else self.decode_credit)

    def view(self, config: Config) -> View:
        probabilities = self.request.acceptance
        if config.acceptance_mode == "learned":
            probabilities = tuple(s / (s + f) for s, f in zip(self.successes, self.failures))
        return View(
            self.index, self.request.arrival_ms,
            self.request.prompt_tokens - self.prefetched,
            self.request.cached_tokens + self.prefetched + self.generated,
            self.last_service, self.debt,
            probabilities, self.prefill_credit, self.decode_credit,
            self.generated, self.first_token_reference,
        )


def reference_costs(request: Request, config: Config, model: CostModel) -> tuple[float, float]:
    """Freeze normalization before scheduling; fragmentation cannot earn extra credit."""
    view = View(0, request.arrival_ms, 0, request.cached_tokens + request.prompt_tokens,
                0, 0, request.acceptance, 0, 0, 0, 0)
    depths = (0,) if config.reference == "ordinary" else range(config.max_drafts + 1)
    decode = min(
        model.duration(Action((Work(0, drafts=k),)), {0: view}) / expected_tokens(request.acceptance, k)
        for k in depths
    )
    position, prefill = 0, 0.0
    while position < request.prompt_tokens:
        remaining = request.prompt_tokens - position
        view = View(0, 0, remaining, request.cached_tokens + position, 0, 0, (), 0, 0, 0, 0)
        options = []
        for amount in sorted({min(c, remaining) for c in config.chunks}):
            duration = model.duration(Action((Work(0, prefill=amount),)), {0: view})
            if duration <= config.quantum_ms + 1e-9:
                options.append((amount / duration, amount, duration))
        if not options:
            raise ValueError(f"prefill for {request.name} cannot fit the execution quantum")
        _, amount, duration = max(options)
        position += amount
        prefill += duration
    return prefill / request.prompt_tokens if request.prompt_tokens else 0, decode


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


@dataclass
class Result:
    summary: dict
    requests: list[dict]
    trace: list[dict]


def run(
    requests: list[Request], policy: str, config: Config = Config(),
    model: CostModel = CostModel(), seed: int = 0, keep_trace: bool = False,
    max_steps: int = 100_000,
) -> Result:
    if not requests or len({r.name for r in requests}) != len(requests):
        raise ValueError("workload must contain uniquely named requests")
    if any(len(r.acceptance) < config.max_drafts for r in requests):
        raise ValueError("acceptance profiles must cover the configured draft depths")
    states = []
    for index, request in enumerate(requests):
        rng = Random(seed * 1_000_003 + index)
        uniforms = [tuple(rng.random() for _ in request.acceptance)
                    for _ in range(request.output_tokens)]
        prefill, decode = reference_costs(request, config, model)
        states.append(State(request, index, uniforms, prefill, decode,
                            successes=[1.0] * config.max_drafts,
                            failures=[1.0] * config.max_drafts))
    scheduler = Scheduler(policy, config, model)
    now = min(r.arrival_ms for r in requests)
    start = now
    trace = []
    rounds = mixed_rounds = busy_ms = max_action_ms = 0
    while any(s.completed is None for s in states):
        if rounds >= max_steps:
            raise RuntimeError(f"simulation exceeded {max_steps} actions")
        resident = [s for s in states if s.admitted is not None and s.completed is None]
        waiting = sorted(
            (s for s in states if s.admitted is None and s.request.arrival_ms <= now),
            key=lambda s: (s.request.arrival_ms, s.index),
        )
        for state in waiting[:config.resident_limit - len(resident)]:
            state.admitted = state.last_service = now
            resident.append(state)
        if not resident:
            now = min(s.request.arrival_ms for s in states if s.admitted is None)
            continue
        views = [s.view(config) for s in resident]
        action, duration = scheduler.choose(now, views)
        end = now + duration
        # All resident requests remain runnable until the action completes.
        for state in resident:
            state.entitlement += duration / len(resident)
            state.debt += duration / len(resident)
        changes = []
        for work in action.work:
            state = states[work.index]
            previous_progress = state.progress
            gap = end - state.last_service
            state.max_gap = max(state.max_gap, gap)
            if config.service_gap_target_ms is not None and gap > config.service_gap_target_ms + 1e-9:
                state.service_gap_violations += 1
            state.last_service = end
            emitted = 0
            observations = []
            if work.prefill:
                state.prefetched += work.prefill
                state.progress += work.prefill * state.prefill_credit
                if state.prefetched == state.request.prompt_tokens:
                    emitted = 1
            else:
                emitted = 1
                state.decode_rounds += 1
                state.offered += work.drafts
                for depth in range(work.drafts):
                    if state.generated + emitted == state.request.output_tokens:
                        break
                    accepted = state.uniforms[state.generated][depth] < state.request.acceptance[depth]
                    observations.append(accepted)
                    state.successes[depth] *= 0.95
                    state.failures[depth] *= 0.95
                    if accepted:
                        state.successes[depth] += 1
                        state.accepted += 1
                        emitted += 1
                    else:
                        state.failures[depth] += 1
                        break
                state.progress += emitted * state.decode_credit
            state.debt = max(0, state.debt - (state.progress - previous_progress))
            first_position = state.generated + 1
            delivery_slowdown = None
            if emitted:
                delivery_slowdown = (end - state.request.arrival_ms) / delivery_reference(
                    state.first_token_reference, state.decode_credit, first_position)
                if delivery_slowdown > state.max_delivery_slowdown:
                    state.max_delivery_slowdown = delivery_slowdown
                    state.worst_delivery_token = first_position
            state.generated += emitted
            if emitted:
                if state.first_output is None:
                    state.first_output = end
                if state.last_output is not None:
                    state.output_gaps.append(end - state.last_output)
                state.last_output = end
            if state.generated == state.request.output_tokens:
                state.completed = end
            assert 0 <= state.generated <= state.request.output_tokens
            assert 0 <= state.prefetched <= state.request.prompt_tokens
            changes.append({
                "request": state.request.name, "prefill": work.prefill, "drafts": work.drafts,
                "emitted": emitted, "observations": observations,
                "first_output_position": first_position if emitted else None,
                "delivery_slowdown": delivery_slowdown,
            })
        for state in resident:
            state.max_lag = max(state.max_lag, state.debt)
        if keep_trace:
            trace.append({"start_ms": now, "end_ms": end, "mixed": action.mixed,
                          "resident": len(resident), "work": changes,
                          "rescue_lateness_ms": scheduler.last_rescue_lateness_ms})
        rounds += 1
        mixed_rounds += int(action.mixed)
        busy_ms += duration
        max_action_ms = max(max_action_ms, duration)
        now = end
    rows = []
    for state in states:
        request = state.request
        demand = (request.prompt_tokens * state.prefill_credit
                  + (request.output_tokens - int(request.prompt_tokens > 0)) * state.decode_credit)
        rows.append({
            "request": request.name, "arrival_ms": request.arrival_ms,
            "prompt_tokens": request.prompt_tokens, "output_tokens": state.generated,
            "queue_ms": state.admitted - request.arrival_ms,
            "ttft_ms": state.first_output - request.arrival_ms,
            "completion_ms": state.completed - request.arrival_ms,
            "reference_demand_ms": demand,
            "credited_progress_ms": state.progress,
            "reference_slowdown": (state.completed - request.arrival_ms) / demand,
            "first_token_reference_ms": state.first_token_reference,
            "decode_reference_ms": state.decode_credit,
            "max_delivery_slowdown": state.max_delivery_slowdown,
            "worst_delivery_token": state.worst_delivery_token,
            "max_service_gap_ms": state.max_gap,
            "service_gap_violations": state.service_gap_violations,
            "max_service_gap_excess_ms": (
                max(0, state.max_gap - config.service_gap_target_ms)
                if config.service_gap_target_ms is not None else None
            ),
            "max_output_gap_ms": max(state.output_gaps, default=0),
            "max_lag_ms": state.max_lag,
            "offered_drafts": state.offered, "accepted_drafts": state.accepted,
            "decode_rounds": state.decode_rounds,
        })
    gaps = [gap for state in states for gap in state.output_gaps]
    summary = {
        "policy": policy, "seed": seed, "requests": len(rows),
        "latency_guard_enabled": policy in ("interactive", "delivery") and config.service_gap_target_ms is not None,
        "makespan_ms": now - start, "busy_ms": busy_ms, "rounds": rounds,
        "mixed_rounds": mixed_rounds, "max_action_ms": max_action_ms,
        "output_tokens_per_second": sum(r.output_tokens for r in requests) * 1000 / (now - start),
        "mean_ttft_ms": sum(r["ttft_ms"] for r in rows) / len(rows),
        "p95_ttft_ms": quantile([r["ttft_ms"] for r in rows], 0.95),
        "mean_completion_ms": sum(r["completion_ms"] for r in rows) / len(rows),
        "p95_completion_ms": quantile([r["completion_ms"] for r in rows], 0.95),
        "p95_output_gap_ms": quantile(gaps, 0.95),
        "max_service_gap_ms": max(r["max_service_gap_ms"] for r in rows),
        "max_queue_ms": max(r["queue_ms"] for r in rows),
        "max_lag_ms": max(r["max_lag_ms"] for r in rows),
        "max_reference_slowdown": max(r["reference_slowdown"] for r in rows),
        "max_delivery_slowdown": max(r["max_delivery_slowdown"] for r in rows),
        "p95_delivery_slowdown": quantile([r["max_delivery_slowdown"] for r in rows], 0.95),
        "service_gap_violations": sum(r["service_gap_violations"] for r in rows),
        "max_service_gap_excess_ms": (
            max(r["max_service_gap_excess_ms"] for r in rows)
            if config.service_gap_target_ms is not None else None
        ),
        "uncertified_decisions": scheduler.uncertified_decisions,
    }
    return Result(summary, rows, trace)
