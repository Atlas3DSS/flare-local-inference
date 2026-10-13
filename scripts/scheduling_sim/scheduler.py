"""Batch selection with a bounded-age guard and useful-progress service lag."""

from itertools import combinations
from dataclasses import replace

from .latency import rescue_latenesses, singleton_costs
from .model import (
    Action, Config, CostModel, View, Work, credit, delivery_reference,
    expected_tokens, token_outcomes,
)


POLICIES = ("fifo", "decode_first", "round_robin", "throughput", "interactive", "delivery")


class InfeasibleQuantum(ValueError):
    """At least one resident request cannot make progress within the quantum."""


def cohorts(decodes: list[View], capacity: int) -> list[tuple[View, ...]]:
    if capacity <= 0:
        return []
    groups = {}

    def add(group):
        ordered = tuple(sorted(group, key=lambda v: v.index))
        groups[tuple(v.index for v in ordered)] = ordered

    if len(decodes) <= 5:
        for size in range(1, min(capacity, len(decodes)) + 1):
            for group in combinations(decodes, size):
                add(group)
    else:
        keys = (
            lambda v: (v.last_service_ms, v.index),
            lambda v: (-v.lag_ms, v.index),
            lambda v: (-expected_tokens(v.probabilities, len(v.probabilities)), v.index),
            lambda v: (expected_tokens(v.probabilities, len(v.probabilities)), v.index),
            lambda v: (v.context, v.index),
        )
        for key in keys:
            ordered = sorted(decodes, key=key)
            for size in sorted({1, 2, 4, capacity}):
                if size <= capacity:
                    add(ordered[:size])
        for view in decodes:
            add((view,))
    return list(groups.values())


def candidates(views: list[View], config: Config, model: CostModel) -> list[tuple[Action, float]]:
    by_id = {v.index: v for v in views}
    prefills = [v for v in views if v.remaining_prefill]
    decodes = [v for v in views if not v.remaining_prefill]
    actions = {}

    def add(work):
        action = Action(tuple(sorted(work, key=lambda w: w.index)))
        if action in actions:
            return
        duration = model.duration(action, by_id)
        if duration <= config.quantum_ms + 1e-9:
            actions[action] = duration

    decode_options = []
    for group in cohorts(decodes, config.max_batch):
        for depth in range(config.max_drafts + 1):
            work = tuple(Work(v.index, drafts=depth) for v in group)
            decode_options.append(work)
            add(work)
        if model.shape == "ragged":
            work = []
            for view in group:
                depth = max(range(config.max_drafts + 1), key=lambda k:
                    expected_tokens(view.probabilities, k)
                    / model.duration(Action((Work(view.index, drafts=k),)), by_id))
                work.append(Work(view.index, drafts=depth))
            decode_options.append(tuple(work))
            add(work)
    for view in prefills:
        for amount in sorted({min(c, view.remaining_prefill) for c in config.chunks}):
            add((Work(view.index, prefill=amount),))
    if config.mixing:
        # Limit mixed candidates while preserving every request's singleton action.
        chosen = {
            min(prefills, key=lambda v: (v.remaining_prefill, v.index)).index,
            min(prefills, key=lambda v: (v.last_service_ms, v.index)).index,
            max(prefills, key=lambda v: (v.lag_ms, -v.index)).index,
        } if prefills else set()
        for view in prefills:
            if view.index not in chosen:
                continue
            for amount in sorted({min(c, view.remaining_prefill) for c in config.chunks}):
                for work in decode_options:
                    if len(work) < config.max_batch:
                        add((Work(view.index, prefill=amount), *work))
    covered = set().union(*(a.indices for a in actions)) if actions else set()
    missing = set(by_id) - covered
    if missing:
        raise InfeasibleQuantum(f"no action within {config.quantum_ms:g} ms for requests {sorted(missing)}")
    return list(actions.items())


def delivery_pressures(
    now: float, action: Action | None, duration: float, views: list[View], model: CostModel,
) -> dict[int, float]:
    """Expected next-token pressure after an action, or at the dispatch boundary."""
    work_by_id = {w.index: w for w in action.work} if action else {}
    pressures = {}
    for view in views:
        work = work_by_id.get(view.index)
        prefill = work.prefill if work else 0
        remaining = view.remaining_prefill - prefill
        elapsed = now + duration - view.arrival_ms
        if remaining:
            pressures[view.index] = ((elapsed + remaining * view.prefill_credit_ms)
                                    / view.first_token_reference_ms)
            continue
        outcomes = ((0, 1.0),) if work is None else (
            ((1, 1.0),) if prefill else token_outcomes(view.probabilities, work.drafts)
        )
        pressure = 0.0
        for emitted, probability in outcomes:
            if probability == 0:
                continue
            following = replace(view, remaining_prefill=0, context=view.context + prefill + emitted)
            next_cost = model.duration(Action((Work(view.index),)), {view.index: following})
            reference = delivery_reference(view.first_token_reference_ms, view.decode_credit_ms,
                                           view.generated_tokens + emitted + 1)
            pressure += probability * (elapsed + next_cost) / reference
        pressures[view.index] = pressure
    return pressures


class Scheduler:
    def __init__(self, policy: str, config: Config, model: CostModel):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        self.policy, self.config, self.model = policy, config, model
        self.decisions = 0
        self.uncertified_decisions = 0
        self.last_rescue_lateness_ms = None

    def choose(self, now: float, views: list[View]) -> tuple[Action, float]:
        self.decisions += 1
        self.last_rescue_lateness_ms = None
        options = candidates(views, self.config, self.model)
        rescue_cost = singleton_costs(options)
        by_id = {v.index: v for v in views}
        oldest = min(views, key=lambda v: (v.last_service_ms, v.index))
        protected = self.policy in ("interactive", "delivery")
        anchor = None
        if self.policy == "fifo":
            anchor = min(views, key=lambda v: (v.arrival_ms, v.index)).index
            options = [(a, t) for a, t in options if len(a.work) == 1]
        elif self.policy == "round_robin":
            anchor = oldest.index
            options = [(a, t) for a, t in options if not a.mixed and all(w.drafts == 0 for w in a.work)]
        elif self.policy == "decode_first":
            if any(not v.remaining_prefill for v in views):
                options = [(a, t) for a, t in options if all(not w.prefill for w in a.work)]
            else:
                anchor = min(views, key=lambda v: (v.arrival_ms, v.index)).index
        elif protected:
            if now - oldest.last_service_ms >= self.config.urgent_after_ms:
                anchor = oldest.index
        if anchor is not None:
            options = [(a, t) for a, t in options if anchor in a.indices]

        target = self.config.service_gap_target_ms
        if protected and target is not None:
            lateness = {a: rescue_latenesses(now, a, t, views, rescue_cost, target) for a, t in options}
            certified = [(a, t) for a, t in options if max(lateness[a].values()) <= 1e-9]
            if certified:
                options = certified
            else:
                self.uncertified_decisions += 1
                tardiness = {a: sum(max(0, value) for value in values.values())
                             for a, values in lateness.items()}
                best = min(tardiness.values())
                options = [(a, t) for a, t in options if tardiness[a] <= best + 1e-9]

        if self.policy == "interactive" and anchor is None:
            owed = max(views, key=lambda v: (v.lag_ms, -v.index))
            owed_options = [(a, t) for a, t in options if owed.index in a.indices]
            if owed.lag_ms >= self.config.lag_scale_ms and owed_options:
                anchor = owed.index
                options = owed_options
        can_explore = anchor is None
        if self.policy == "delivery" and anchor is None:
            pressure = delivery_pressures(now, None, 0, views, self.model)
            urgent = max(views, key=lambda v: (pressure[v.index], -v.index))
            urgent_options = [(a, t) for a, t in options if urgent.index in a.indices]
            if urgent_options:
                options = urgent_options
        if (protected and can_explore and self.config.acceptance_mode == "learned"
              and self.config.explore_every and self.decisions % self.config.explore_every == 0):
            slack = self.config.urgent_after_ms - (now - oldest.last_service_ms)
            probes = [(a, t) for a, t in options
                      if t <= slack and any(w.drafts == self.config.max_drafts and w.drafts for w in a.work)]
            if probes:
                options = probes

        def score(item):
            action, duration = item
            benefit = 0.0
            for work in action.work:
                view = by_id[work.index]
                weight = 1.0
                if self.policy == "interactive":
                    weight += max(0, view.lag_ms) / self.config.lag_scale_ms
                benefit += weight * credit(work, view)
                if self.policy == "interactive" and work.prefill == view.remaining_prefill and work.prefill:
                    benefit += self.config.lag_scale_ms
            return benefit / duration, -duration, -sum(w.drafts for w in action.work)

        if self.policy == "delivery":
            selected = min(options, key=lambda item: (
                tuple(sorted(delivery_pressures(now, *item, views, self.model).values(), reverse=True)),
                tuple(-value for value in score(item)),
            ))
        else:
            selected = max(options, key=score)
        if protected and target is not None:
            self.last_rescue_lateness_ms = max(lateness[selected[0]].values())
        return selected
