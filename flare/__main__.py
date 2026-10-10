"""Load and capture a supported checkpoint, then start the native HTTP server."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from time import monotonic


def main(argv=None):
    started = monotonic()
    from flare.preparation.qwen38 import MODEL_ID, REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path.home() / ".cache/huggingface/hub" /
        f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / REVISION)
    parser.add_argument("--bind", default="127.0.0.1:8000")
    parser.add_argument("--device", type=int, default=0, help="process-local CUDA ordinal")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=1,
        help="tensor-parallel ranks on consecutive CUDA ordinals beginning at --device; GLM requires 2 or 4")
    parser.add_argument("--max-active", type=int, choices=(1, 2, 3, 4), default=4,
        help="active request capacity for GLM graph preparation (default: 4)")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096,
        help="maximum tokens per captured prefill graph; 64-aligned and no larger than context (default: 4096)")
    parser.add_argument("--pages", type=int,
        help="KV pages; Qwen defaults to remaining GPU budget, GLM to context capacity per active request")
    parser.add_argument("--mtp", action="store_true", help="prepare native multi-token prediction graphs")
    parser.add_argument("--mtp-proposal-source", choices=("mtp", "copy"), default="mtp",
        help="proposal source for speculative graphs; copy loads no MTP weights")
    parser.add_argument("--mtp-widths", type=int, nargs="+", default=list(range(1, 9)),
        help="verification widths including the pending input; batch times width is at most eight (default: 1 through 8)")
    parser.add_argument("--mtp-fixed-width", type=int, choices=range(1, 9),
        help="select a fixed prepared width instead of adaptive verification")
    parser.add_argument("--history-drafts", action="store_true",
        help="enable draft continuations from request history during speculation (default: disabled)")
    parser.add_argument("--snapshot-budget-gib", type=float, default=4)
    parser.add_argument("--headroom-gib", type=float, default=4)
    parser.add_argument("--template-defaults", type=Path, help="JSON object of default chat template arguments")
    parser.add_argument("--memory-report", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="validate the graph catalog and release it without binding HTTP")
    args = parser.parse_args(argv)
    defaults = {}
    if args.template_defaults is not None:
        defaults = json.loads(args.template_defaults.read_text())
        if not isinstance(defaults, dict):
            parser.error("template defaults must be a JSON object")
    architecture = json.loads((args.snapshot / "config.json").read_text()).get("model_type")
    progress = lambda text: print(f"{monotonic() - started:6.1f}s {text}", file=sys.stderr, flush=True)
    budgets = {"snapshot_budget_bytes": int(args.snapshot_budget_gib * (1 << 30)),
               "headroom_bytes": int(args.headroom_gib * (1 << 30))}
    if architecture == "glm5_next":
        if args.tp not in (2, 4) or args.mtp:
            parser.error("GLM requires --tp 2 or --tp 4 and has no prepared MTP catalog")
        from flare.preparation.rank_group import prepare_group
        from flare.recipes.glm53.bootstrap import PreparationConfig, prepare_rank

        config = PreparationConfig(devices=tuple(range(args.device, args.device + args.tp)),
            context=args.context, pages=args.pages, prefill_chunk_size=args.prefill_chunk_size,
            max_active=args.max_active, **budgets)
        model = prepare_group(args.snapshot, config, prepare_rank, progress=progress)
    elif architecture == "qwen4_exp":
        if args.tp != 1 or args.max_active != 4:
            parser.error("Qwen requires --tp 1 and --max-active 4")
        from flare.recipes.qwen38.bootstrap import PreparationConfig, prepare_model

        config = PreparationConfig(device=args.device, context=args.context, pages=args.pages,
            prefill_chunk_size=args.prefill_chunk_size,
            mtp=args.mtp, speculative_widths=tuple(args.mtp_widths),
            speculative_proposal_source=args.mtp_proposal_source,
            speculative_fixed_width=args.mtp_fixed_width, history_drafts=args.history_drafts, **budgets)
        model = prepare_model(args.snapshot, config, progress=progress)
    else:
        parser.error(f"unsupported checkpoint model_type {architecture!r}")
    with model:
        report = json.dumps(model.memory_report, indent=2) + "\n"
        if args.memory_report is not None:
            args.memory_report.write_text(report)
        print(report, file=sys.stderr, end="", flush=True)
        if args.prepare_only:
            model.manifest()
            return 0
        server = model.start_server(bind=args.bind, template_defaults=defaults)
        print(f"Listening on {server.address}; startup {monotonic() - started:.2f}s", file=sys.stderr, flush=True)
        def stop(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            while server.wait(1000):
                pass
            raise RuntimeError("native server stopped unexpectedly")
        except KeyboardInterrupt:
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
