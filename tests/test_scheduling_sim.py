from dataclasses import replace
from math import prod
from itertools import permutations
from random import Random
import csv
import json
import subprocess
import sys

import pytest

from scripts.scheduling_sim.model import (
    Action, Config, CostModel, Request, View, Work, expected_tokens, token_outcomes,
)
from scripts.scheduling_sim.frontier import max_min_pair
from scripts.scheduling_sim.latency import rescue_latenesses
from scripts.scheduling_sim.scheduler import InfeasibleQuantum, Scheduler, candidates, delivery_pressures, POLICIES
from scripts.scheduling_sim.simulator import reference_costs, run


def view(index=0, prefill=0, context=512, age=0, lag=0, probabilities=(0.8,) * 5):
    return View(index, 0, prefill, context, age, lag, probabilities, 0.025, 3.2,
                0, prefill * 0.025 if prefill else 3.2)


def test_survival_expectation_by_enumerating_every_outcome():
    probabilities = (0.9, 0.6, 0.2, 0.8)
    exact = sum((i + 1) * prod(probabilities[:i]) * (1 - probabilities[i])
                for i in range(len(probabilities)))
    exact += (len(probabilities) + 1) * prod(probabilities)
    assert expected_tokens(probabilities, 4) == pytest.approx(exact)
    assert expected_tokens((0.0,) * 5, 5) == 1
    assert expected_tokens((1.0,) * 5, 5) == 6


def test_monte_carlo_matches_conditional_survival():
    rng = Random(41)
    probabilities = (0.9, 0.6, 0.2)
    total = 0
    for _ in range(40_000):
        emitted = 1
        for probability in probabilities:
            if rng.random() >= probability:
                break
            emitted += 1
        total += emitted
    assert total / 40_000 == pytest.approx(expected_tokens(probabilities, 3), abs=0.015)


def test_prefix_survival_cannot_be_replaced_by_average_acceptance():
    assert expected_tokens((1.0, 0.0), 2) == 2
    assert expected_tokens((0.0, 1.0), 2) == 1


def test_mixing_can_help_or_hurt_for_identical_work():
    views = {0: view(0, prefill=256), 1: view(1)}
    prefill = Action((Work(0, prefill=256),))
    decode = Action((Work(1),))
    mixed = Action(prefill.work + decode.work)
    model = CostModel()
    separate = model.duration(prefill, views) + model.duration(decode, views)
    assert model.duration(mixed, views) < separate
    assert replace(model, mixed_penalty_ms=10).duration(mixed, views) > separate


def test_padding_and_heterogeneous_depth_costs():
    views = {i: view(i) for i in range(3)}
    action = Action((Work(0, drafts=5), Work(1), Work(2)))
    rectangular = CostModel(target_token_ms=1)
    ragged = replace(rectangular, shape="ragged")
    assert rectangular.duration(action, views) > ragged.duration(action, views)


def test_context_changes_legal_prefill_chunk():
    config, model = Config(), CostModel()
    def largest(context):
        options = candidates([view(prefill=1024, context=context)], config, model)
        return max(a.work[0].prefill for a, _ in options)
    assert largest(32768) < largest(512)


def test_impossible_quantum_is_rejected():
    with pytest.raises(InfeasibleQuantum, match="no action within"):
        Scheduler("interactive", Config(quantum_ms=1), CostModel()).choose(0, [view()])


def test_oldest_guard_overrides_high_expected_throughput():
    config = Config(max_batch=1, urgent_after_ms=20)
    views = [view(0, age=0, probabilities=(0.0,) * 5),
             view(1, age=39, probabilities=(1.0,) * 5)]
    action, _ = Scheduler("interactive", config, CostModel()).choose(40, views)
    assert action.indices == {0}


@pytest.mark.parametrize("policy", ("interactive", "delivery"))
@pytest.mark.parametrize("seed", range(8))
def test_bounded_service_gap_with_churn_and_random_acceptance(policy, seed):
    rng = Random(seed)
    requests = [Request(f"request-{i}", i * 3, rng.choice((0, 64, 256)),
                        rng.randint(3, 24), (rng.random(),) * 5) for i in range(12)]
    config = Config(resident_limit=4, max_batch=2, urgent_after_ms=8, quantum_ms=12)
    result = run(requests, policy, config, seed=seed)
    bound = config.urgent_after_ms + (config.resident_limit + 1) * config.quantum_ms
    assert result.summary["max_service_gap_ms"] <= bound + 1e-8
    assert result.summary["max_action_ms"] <= config.quantum_ms + 1e-8
    assert [r["output_tokens"] for r in result.requests] == [r.output_tokens for r in requests]


def test_arrival_during_action_waits_until_boundary_and_is_not_in_batch():
    requests = [Request("running", 0, 0, 8), Request("arrival", 0.001, 0, 2)]
    result = run(requests, "interactive", keep_trace=True)
    first = result.trace[0]
    assert [w["request"] for w in first["work"]] == ["running"]
    assert result.requests[1]["queue_ms"] == pytest.approx(first["end_ms"] - 0.001)


def test_prefill_emits_first_token_and_single_token_request_completes():
    result = run([Request("one-token", 0, 128, 1)], "interactive", keep_trace=True)
    assert len(result.trace) == 1
    assert result.requests[0]["ttft_ms"] == result.requests[0]["completion_ms"]
    assert result.requests[0]["decode_rounds"] == 0


def test_eos_censors_unobserved_drafts():
    result = run([Request("short", 0, 0, 2, (1.0,) * 5)], "throughput",
                 Config(acceptance_mode="oracle"), keep_trace=True)
    assert result.trace[0]["work"][0]["drafts"] > 1
    assert result.trace[0]["work"][0]["observations"] == [True]
    assert result.requests[0]["output_tokens"] == 2


def test_slot_queue_is_separate_from_resident_service_guarantee():
    requests = [Request("resident", 0, 0, 20), Request("queued", 1, 0, 1)]
    result = run(requests, "interactive", Config(resident_limit=1), keep_trace=True)
    assert result.requests[1]["queue_ms"] == pytest.approx(result.requests[0]["completion_ms"] - 1)
    assert result.requests[1]["queue_ms"] > result.requests[1]["max_service_gap_ms"]


def test_cost_reference_does_not_depend_on_actual_fragmentation(monkeypatch):
    request = Request("prompt", 0, 1024, 1)
    whole = run([request], "interactive")
    def small_chunks(self, now, views):
        return min(candidates(views, self.config, self.model), key=lambda item: item[0].work[0].prefill)
    monkeypatch.setattr(Scheduler, "choose", small_chunks)
    fragmented = run([request], "interactive")
    assert fragmented.summary["busy_ms"] > whole.summary["busy_ms"]
    assert fragmented.requests[0]["credited_progress_ms"] == pytest.approx(
        whole.requests[0]["credited_progress_ms"])


def test_realized_progress_can_fall_below_speculative_reference():
    probabilities = (0.99,) * 5
    config = Config(reference="speculative")
    model = CostModel()
    _, per_token = reference_costs(Request("high-acceptance", 0, 0, 100, probabilities), config, model)
    action_time = model.duration(Action((Work(0, drafts=5),)), {0: view()})
    assert per_token < action_time
    assert 1 - probabilities[0] > 0
    # Repeated first-draft rejection has positive probability and yields one token per round.
    rounds = 1000
    assert rounds * action_time - rounds * per_token > 1000


def test_request_views_do_not_disclose_remaining_decode():
    assert "output_tokens" not in View.__dataclass_fields__
    assert "remaining_output" not in View.__dataclass_fields__


def test_reproducibility_includes_policy_decisions_and_output_events():
    requests = [Request(f"r{i}", i * 2, 64, 16) for i in range(3)]
    assert run(requests, "interactive", seed=7, keep_trace=True) == run(
        requests, "interactive", seed=7, keep_trace=True)


@pytest.mark.parametrize("policy", POLICIES)
def test_policies_finish_prefill_and_decode_without_token_loss(policy):
    requests = [Request("prefill", 0, 160, 5), Request("decode", 1, 0, 9, (0.0,) * 5)]
    result = run(requests, policy, keep_trace=True)
    assert [r["output_tokens"] for r in result.requests] == [5, 9]
    assert sum(w["prefill"] for e in result.trace for w in e["work"]) == 160


@pytest.mark.parametrize("kwargs", ({"quantum_ms": float("nan")}, {"max_batch": 0},
                                    {"chunks": (128, 64)}, {"lag_scale_ms": -1},
                                    {"service_gap_target_ms": 0}, {"service_gap_target_ms": float("inf")}))
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_exact_fair_frontier_beats_any_single_action():
    rate, mixture = max_min_pair([(1.5, 0.5), (0.5, 1.5), (0.9, 0.9)])
    assert rate == pytest.approx(1)
    assert mixture == [(0, 0.5), (1, 0.5)]


def test_fair_frontier_weights_are_time_fractions_not_round_fractions():
    # Actions delivering (4, 0) in 2 ms and (0, 1) in 1 ms need a 1:2 time split.
    rate, mixture = max_min_pair([(2, 0), (0, 1)])
    assert rate == pytest.approx(2 / 3)
    assert dict(mixture)[0] == pytest.approx(1 / 3)


def test_exact_frontier_dominates_dense_bruteforce_mixtures():
    points = [(0.2, 2.5), (1.8, 0.4), (1.1, 1.2), (0.6, 1.7)]
    optimum, _ = max_min_pair(points)
    for first in range(21):
        for second in range(21 - first):
            for third in range(21 - first - second):
                weights = [first, second, third, 20 - first - second - third]
                rate = min(sum(w * p[k] / 20 for w, p in zip(weights, points)) for k in (0, 1))
                assert optimum >= rate - 1e-10


def test_greedy_expected_progress_can_worsen_finite_workload_makespan():
    requests = [Request(f"r{i}", 0, 0, 32, (1.0 if i < 4 else 0.0,) * 5) for i in range(8)]
    config = Config(acceptance_mode="oracle")
    speculative = run(requests, "throughput", config, keep_trace=True)
    ordinary = run(requests, "throughput", replace(config, max_drafts=0))
    assert any(w["drafts"] for w in speculative.trace[0]["work"])
    assert speculative.summary["makespan_ms"] > ordinary.summary["makespan_ms"]


def test_cli_custom_workload_config_and_trace(tmp_path):
    workload = tmp_path / "workload.json"
    config = tmp_path / "config.json"
    output = tmp_path / "results"
    workload.write_text(json.dumps([
        {"name": "first", "arrival_ms": 0, "prompt_tokens": 64, "output_tokens": 5},
        {"name": "second", "arrival_ms": 0, "prompt_tokens": 0, "output_tokens": 2},
    ]))
    config.write_text(json.dumps({"config": {"resident_limit": 1, "max_drafts": 0,
                                            "service_gap_target_ms": 20},
                                  "model": {"weight_ms": 2}}))
    subprocess.run([
        sys.executable, "-m", "scripts.scheduling_sim", "--workload-file", str(workload),
        "--config", str(config), "--policies", "interactive", "--resident-limit", "2",
        "--gap-targets", "none,10",
        "--output", str(output), "--trace",
    ], check=True, capture_output=True, text=True)
    with (output / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
        row = rows[0]
    assert row["resident_limit"] == "2"
    assert row["cost_weight_ms"] == "2"
    assert row["max_drafts"] == "0"
    assert [r["service_gap_target_ms"] for r in rows] == ["", "10.0"]
    assert [r["latency_guard_enabled"] for r in rows] == ["False", "True"]
    assert all(json.loads(line)["resident"] <= 2 for line in (output / "trace.jsonl").read_text().splitlines())
    with (output / "requests.csv").open() as handle:
        assert sum(int(r["output_tokens"]) for r in csv.DictReader(handle)) == 14


def test_unobserved_depth_is_not_silently_assumed():
    with pytest.raises(ValueError):
        expected_tokens((0.9,), 2)


def test_deficit_metric_is_measured_after_committed_progress():
    result = run([Request("one", 0, 0, 4)], "round_robin", Config(max_drafts=0),
                 CostModel(attention_ms_per_ktoken=0))
    assert result.summary["max_lag_ms"] == pytest.approx(0, abs=1e-12)


def test_burst_delivery_slowdown_uses_first_token_and_preserves_historical_peak(monkeypatch):
    model = CostModel(launch_ms=0, weight_ms=10, target_token_ms=0,
                      attention_ms_per_ktoken=0, draft_step_ms=18, draft_token_ms=0)
    depths = iter((0, 5, 0))
    def prescribed(self, now, views):
        action = Action((Work(0, drafts=next(depths)),))
        return action, self.model.duration(action, {v.index: v for v in views})
    monkeypatch.setattr(Scheduler, "choose", prescribed)
    result = run([Request("burst", 0, 0, 8, (1.0,) * 5)], "interactive",
                 Config(quantum_ms=100), model, keep_trace=True)
    row = result.requests[0]
    assert row["max_delivery_slowdown"] == pytest.approx(110 / 20)
    assert row["worst_delivery_token"] == 2
    assert row["reference_slowdown"] == pytest.approx(120 / 80)
    assert row["max_output_gap_ms"] == 100
    assert [e["work"][0]["first_output_position"] for e in result.trace] == [1, 2, 8]


def test_delivery_slowdown_matches_every_emitted_token_in_trace():
    requests = [Request("prefill", 0, 320, 19), Request("decode", 2, 0, 23)]
    result = run(requests, "delivery", keep_trace=True)
    for row in result.requests:
        ratios = []
        count = 0
        for event in result.trace:
            for work in event["work"]:
                if work["request"] != row["request"]:
                    continue
                for token in range(count + 1, count + work["emitted"] + 1):
                    reference = row["first_token_reference_ms"] + (token - 1) * row["decode_reference_ms"]
                    ratios.append((event["end_ms"] - row["arrival_ms"]) / reference)
                count += work["emitted"]
        assert row["max_delivery_slowdown"] == pytest.approx(max(ratios))
        assert row["max_delivery_slowdown"] >= row["reference_slowdown"] - 1e-12


def test_delivery_reference_includes_prefill_once_and_queue_delay():
    model = CostModel(launch_ms=0, weight_ms=10, target_token_ms=0,
                      attention_ms_per_ktoken=0, draft_step_ms=0, draft_token_ms=0)
    result = run([Request("resident", 0, 0, 3), Request("queued", 0, 64, 2)],
                 "round_robin", Config(resident_limit=1, max_drafts=0), model)
    row = result.requests[1]
    assert row["first_token_reference_ms"] == 10
    assert row["reference_demand_ms"] == 20
    assert row["queue_ms"] == 30
    assert row["max_delivery_slowdown"] == 4
    assert row["reference_slowdown"] == 2.5


def test_conditional_outcome_distribution_and_nonlinear_delivery_pressure():
    profile = (0.5, 0.25)
    outcomes = token_outcomes(profile, 2)
    assert outcomes == ((1, 0.5), (2, 0.375), (3, 0.125))
    assert sum(probability for _, probability in outcomes) == 1
    assert sum(count * probability for count, probability in outcomes) == expected_tokens(profile, 2)
    state = view(probabilities=profile)
    action = Action((Work(0, drafts=2),))
    model = CostModel(launch_ms=0, weight_ms=3.2, attention_ms_per_ktoken=0)
    pressure = delivery_pressures(10, action, 5, [state], model)[state.index]
    exact = sum(probability * 18.2 / ((count + 1) * 3.2) for count, probability in outcomes)
    assert pressure == pytest.approx(exact)
    assert pressure > 18.2 / ((expected_tokens(profile, 2) + 1) * 3.2)


def test_rescue_deadline_order_minimizes_next_visit_lateness():
    views = [view(i, age=last) for i, last in enumerate((9, 1, 3, 7, 2))]
    costs = {0: 8, 1: 2, 2: 5, 3: 1, 4: 3}
    action, duration, now, target = Action((Work(0),)), 8, 10, 20
    predicted = max(rescue_latenesses(now, action, duration, views, costs, target).values())
    for order in permutations(views[1:]):
        end = now + duration
        worst = end - views[0].last_service_ms - target
        for state in order:
            end += costs[state.index]
            worst = max(worst, end - state.last_service_ms - target)
        assert predicted <= worst + 1e-12


@pytest.mark.parametrize("policy", ("interactive", "delivery"))
def test_latency_guard_anticipates_a_deadline_before_urgency_threshold(policy):
    config = Config(service_gap_target_ms=40, explore_every=0)
    views = [view(0, context=0, age=0), view(1, prefill=512, context=0, age=35)]
    scheduler = Scheduler(policy, config, CostModel())
    action, duration = scheduler.choose(35, views)
    assert 0 in action.indices
    assert duration <= 5
    assert scheduler.last_rescue_lateness_ms <= 0
    assert scheduler.uncertified_decisions == 0


@pytest.mark.parametrize("policy", ("interactive", "delivery"))
def test_impossible_gap_target_is_reported_and_not_relaxed(policy):
    result = run([Request("decode", 0, 0, 2)], policy, Config(service_gap_target_ms=1, max_drafts=0),
                 CostModel(launch_ms=0, weight_ms=3, attention_ms_per_ktoken=0), keep_trace=True)
    assert result.summary["uncertified_decisions"] == 2
    assert result.summary["service_gap_violations"] == 2
    assert result.summary["max_service_gap_excess_ms"] == 2
    assert all(e["rescue_lateness_ms"] == 2 for e in result.trace)


@pytest.mark.parametrize("policy", ("interactive", "delivery"))
@pytest.mark.parametrize("seed", range(3))
def test_latency_guard_on_feasible_resident_churn(policy, seed):
    rng = Random(seed)
    requests = [Request(f"r{i}", i * 5, rng.choice((0, 64, 512)), 16,
                        (rng.random(),) * 5) for i in range(12)]
    result = run(requests, policy, Config(resident_limit=4, max_batch=4, service_gap_target_ms=20),
                 seed=seed)
    assert result.summary["uncertified_decisions"] == 0
    assert result.summary["service_gap_violations"] == 0


def test_draft_depth_limit_does_not_shift_random_outcomes(monkeypatch):
    def one_draft(self, now, views):
        action = Action((Work(views[0].index, drafts=1),))
        return action, self.model.duration(action, {v.index: v for v in views})
    monkeypatch.setattr(Scheduler, "choose", one_draft)
    requests = [Request("decode", 0, 0, 40, (0.7,) * 5)]
    shallow = run(requests, "interactive", Config(max_drafts=2), seed=3, keep_trace=True)
    deep = run(requests, "interactive", Config(max_drafts=5), seed=3, keep_trace=True)
    assert shallow == deep


def test_delivery_policy_services_a_bottleneck_when_two_prefills_share_pressure():
    config, model = Config(explore_every=0), CostModel()
    decoding = replace(view(0, context=132, age=25), generated_tokens=4)
    prefills = [replace(view(i, prefill=64, context=64, age=5), first_token_reference_ms=3.18)
                for i in (1, 2)]
    action, _ = Scheduler("delivery", config, model).choose(28, [decoding, *prefills])
    assert any(w.prefill for w in action.work)


def test_uncertified_singleton_rescue_does_not_mean_batch_schedule_is_infeasible():
    model = CostModel(launch_ms=0, weight_ms=3, target_token_ms=0,
                      attention_ms_per_ktoken=0, draft_step_ms=0, draft_token_ms=0)
    requests = [Request(f"r{i}", 0, 0, 3) for i in range(8)]
    result = run(requests, "interactive", Config(max_batch=2, max_drafts=0, service_gap_target_ms=12),
                 model)
    assert result.summary["uncertified_decisions"] > 0
    assert result.summary["service_gap_violations"] == 0
    assert result.summary["max_service_gap_ms"] == 12


def test_overdue_recovery_avoids_singleton_lock_in_from_worst_lateness():
    views = [view(i, age=i * 10) for i in range(3)]
    model = CostModel()
    config = Config(service_gap_target_ms=8, max_drafts=0)
    solo = Action((Work(0),))
    batch = Action(tuple(Work(i) for i in range(3)))
    by_id = {v.index: v for v in views}
    costs = {v.index: model.duration(Action((Work(v.index),)), by_id) for v in views}
    solo_delays = rescue_latenesses(40, solo, model.duration(solo, by_id), views, costs, 8)
    batch_delays = rescue_latenesses(40, batch, model.duration(batch, by_id), views, costs, 8)
    assert max(solo_delays.values()) < max(batch_delays.values())
    assert sum(solo_delays.values()) > sum(batch_delays.values())
    scheduler = Scheduler("interactive", config, model)
    chosen, _ = scheduler.choose(40, views)
    assert chosen.indices == {0, 1, 2}
    assert scheduler.uncertified_decisions == 1
