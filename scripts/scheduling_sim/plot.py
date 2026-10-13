"""Plot raw simulation seeds against throughput and latency; requires Matplotlib."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["workload"] == args.workload]
    if not rows:
        parser.error("no rows match the requested workload")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    groups = dict.fromkeys((r["policy"], r["service_gap_target_ms"]) for r in rows)
    for policy, target in groups:
        selected = [r for r in rows if (r["policy"], r["service_gap_target_ms"]) == (policy, target)]
        for axis, metric, label in zip(
            axes, ("p95_ttft_ms", "max_service_gap_ms", "max_delivery_slowdown"),
            ("P95 time to first output (ms)", "Maximum resident service gap (ms)",
             "Worst token-delivery slowdown (ratio)"),
        ):
            axis.scatter([float(r[metric]) for r in selected],
                         [float(r["output_tokens_per_second"]) for r in selected],
                         label=f"{policy}, G={target or 'off'}", s=42, alpha=0.8)
            axis.set_xlabel(f"{label}; lower is better")
            axis.set_ylabel("Output tokens/s (higher is better)")
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"{args.workload}: synthetic execution model; each point is one run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
