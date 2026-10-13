"""Exact two-request max-min allocation over a finite, stationary action catalog."""

import argparse
from dataclasses import asdict
from itertools import product
import json
from math import isfinite

from .model import Action, Config, CostModel, Request, View, Work, credit
from .simulator import reference_costs


def max_min_pair(rates: list[tuple[float, float]]) -> tuple[float, list[tuple[int, float]]]:
    """Return the maximum minimum rate and action wall-time fractions."""
    if not rates or any(len(pair) != 2 or any(not isfinite(x) or x < 0 for x in pair) for pair in rates):
        raise ValueError("rate vectors must be finite and nonnegative")
    best = max(range(len(rates)), key=lambda i: min(rates[i]))
    value, mixture = min(rates[best]), [(best, 1.0)]
    for left, a in enumerate(rates):
        for right in range(left + 1, len(rates)):
            b = rates[right]
            da, db = a[0] - a[1], b[0] - b[1]
            if da == db:
                continue
            fraction = -db / (da - db)
            if 0 < fraction < 1:
                rate = min(fraction * a[k] + (1 - fraction) * b[k] for k in (0, 1))
                if rate > value:
                    value, mixture = rate, [(left, fraction), (right, 1 - fraction)]
    return value, mixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", default="0.97,0.1")
    parser.add_argument("--contexts", default="512,512")
    parser.add_argument("--shape", choices=("rectangular", "ragged"), default="rectangular")
    parser.add_argument("--quantum", type=float, default=20)
    parser.add_argument("--draft-depth", type=int, default=5)
    parser.add_argument("--reference", choices=("ordinary", "speculative"), default="ordinary")
    args = parser.parse_args()
    probabilities = [float(x) for x in args.acceptance.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    if len(probabilities) != 2 or len(contexts) != 2:
        parser.error("exactly two acceptance probabilities and contexts are required")
    config = Config(quantum_ms=args.quantum, max_batch=2, max_drafts=args.draft_depth,
                    acceptance_mode="oracle", reference=args.reference)
    model = CostModel(shape=args.shape)
    views = []
    for i, (probability, context) in enumerate(zip(probabilities, contexts)):
        request = Request(str(i), 0, 0, 1, (probability,) * config.max_drafts, context)
        _, decode = reference_costs(request, config, model)
        views.append(View(i, 0, 0, context, 0, 0, request.acceptance, 0, decode, 0, decode))
    options = []
    for depths in product(range(-1, config.max_drafts + 1), repeat=2):
        work = tuple(Work(i, drafts=k) for i, k in enumerate(depths) if k >= 0)
        if not work:
            continue
        action = Action(work)
        duration = model.duration(action, {v.index: v for v in views})
        if duration <= config.quantum_ms:
            options.append((action, duration))
    rates = []
    for action, duration in options:
        progress = {w.index: credit(w, views[w.index]) / duration for w in action.work}
        rates.append((progress.get(0, 0), progress.get(1, 0)))
    optimum, mixture = max_min_pair(rates)
    print(json.dumps({
        "model": asdict(model), "config": asdict(config),
        "acceptance": probabilities, "contexts": contexts,
        "max_min_reference_rate": optimum,
        "mixture": [{"wall_time_fraction": fraction, "work": [asdict(w) for w in options[i][0].work],
                     "duration_ms": options[i][1], "rates": rates[i]} for i, fraction in mixture],
        "stationary_assumptions": "fixed contexts, known acceptance, infinite backlog; no gap guarantee",
    }, indent=2))


if __name__ == "__main__":
    main()
