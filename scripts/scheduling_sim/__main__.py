"""Run synthetic scheduling comparisons and Cartesian parameter sweeps."""

import argparse
import csv
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

from .model import Config, CostModel, Request
from .scheduler import POLICIES
from .simulator import run
from .workloads import WORKLOADS, workload


def numbers(value, kind):
    return [kind(part) for part in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=(*WORKLOADS, "all"), default="all")
    parser.add_argument("--workload-file", type=Path, help="JSON list of Request fields")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--seeds", default="0", help="comma-separated integer seeds")
    parser.add_argument("--config", type=Path, help="JSON object with config and model fields")
    parser.add_argument("--quantums", help="comma-separated action limits in ms")
    parser.add_argument("--gap-targets", help="comma-separated resident service-gap targets in ms, or none")
    parser.add_argument("--batch-sizes", help="comma-separated request capacities")
    parser.add_argument("--draft-depths", help="comma-separated maximum draft lengths")
    parser.add_argument("--shapes", help="rectangular,ragged")
    parser.add_argument("--chunks", help="comma-separated prepared prefill capacities")
    parser.add_argument("--resident-limit", type=int)
    parser.add_argument("--acceptance", choices=("learned", "oracle"))
    parser.add_argument("--reference", choices=("ordinary", "speculative"))
    parser.add_argument("--no-mixing", action="store_true")
    parser.add_argument("--output", type=Path, help="write summary.csv and requests.csv")
    parser.add_argument("--trace", action="store_true", help="also write per-action trace.jsonl")
    args = parser.parse_args()
    if args.trace and not args.output:
        parser.error("--trace requires --output")
    data = json.loads(args.config.read_text()) if args.config else {}
    settings = data.get("config", {})
    if "chunks" in settings:
        settings["chunks"] = tuple(settings["chunks"])
    config, model = Config(**settings), CostModel(**data.get("model", {}))
    overrides = {}
    for option, field in (("resident_limit", "resident_limit"), ("acceptance", "acceptance_mode"),
                          ("reference", "reference")):
        if getattr(args, option) is not None:
            overrides[field] = getattr(args, option)
    if args.chunks:
        overrides["chunks"] = tuple(numbers(args.chunks, int))
    if args.no_mixing:
        overrides["mixing"] = False
    config = replace(config, **overrides)
    policies = args.policies.split(",")
    if set(policies) - set(POLICIES):
        parser.error(f"policies must be among {POLICIES}")
    if args.workload_file:
        entries = json.loads(args.workload_file.read_text())
        requests = [Request(**(entry | {"acceptance": tuple(entry.get("acceptance", (0.75,) * 5))}))
                    for entry in entries]
        workloads = [(str(args.workload_file), requests)]
    else:
        names = WORKLOADS if args.workload == "all" else (args.workload,)
        workloads = [(name, workload(name)) for name in names]
    summary_rows, request_rows = [], []
    trace_file = None
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.trace:
            trace_file = (args.output / "trace.jsonl").open("w")
    print("Synthetic durations; output is a model experiment, not a GPU benchmark.")
    try:
        for quantum, target, batch, drafts, shape, seed, policy, (name, requests) in product(
            numbers(args.quantums, float) if args.quantums else [config.quantum_ms],
            [None if value == "none" else float(value) for value in args.gap_targets.split(",")]
            if args.gap_targets else [config.service_gap_target_ms],
            numbers(args.batch_sizes, int) if args.batch_sizes else [config.max_batch],
            numbers(args.draft_depths, int) if args.draft_depths else [config.max_drafts],
            args.shapes.split(",") if args.shapes else [model.shape],
            numbers(args.seeds, int), policies, workloads,
        ):
            selected = replace(config, quantum_ms=quantum, service_gap_target_ms=target,
                               max_batch=batch, max_drafts=drafts)
            if name == "slot_pressure" and args.resident_limit is None and "resident_limit" not in settings:
                selected = replace(selected, resident_limit=2)
            costs = replace(model, shape=shape)
            result = run(requests, policy, selected, costs, seed, args.trace)
            case = {"workload": name, "policy": policy, "seed": seed,
                    **asdict(selected), **{f"cost_{k}": v for k, v in asdict(costs).items()}}
            summary_rows.append(case | result.summary)
            for request, row in zip(requests, result.requests):
                request_rows.append(case | row | {"acceptance_profile": request.acceptance,
                                                  "cached_tokens": request.cached_tokens})
            if trace_file:
                for event in result.trace:
                    trace_file.write(json.dumps(case | event) + "\n")
            s = result.summary
            print(f"{name:19} {policy:12} seed={seed} Q={quantum:g} B={batch} K={drafts} {shape:11} "
                  f"tok/s={s['output_tokens_per_second']:7.1f} "
                  f"TTFT95={s['p95_ttft_ms']:7.1f}ms gap={s['max_service_gap_ms']:6.1f}ms "
                  f"delivery={s['max_delivery_slowdown']:5.2f} "
                  f"G={target} misses={s['service_gap_violations']} "
                  f"uncertified={s['uncertified_decisions']} mixed={s['mixed_rounds']}")
    finally:
        if trace_file:
            trace_file.close()
    if args.output:
        for filename, rows in (("summary.csv", summary_rows), ("requests.csv", request_rows)):
            with (args.output / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
