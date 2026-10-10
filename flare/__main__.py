"""Load and capture a supported checkpoint, then start the native HTTP server."""

from __future__ import annotations

import argparse
import json
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
    catalog = parser.add_mutually_exclusive_group()
    catalog.add_argument("--mtp-max-draft-tokens", type=int, choices=range(1, 8),
        help="maximum draft tokens per request; adaptive speculation may use fewer (Qwen default: 3)")
    catalog.add_argument("--mtp-draft-tokens", type=int, nargs="+", choices=range(8),
        help="explicit increasing draft-token counts including zero; overrides the default draft limit")
    parser.add_argument("--mtp-fixed-draft-tokens", type=int, choices=range(8),
        help="fixed draft-token count instead of adaptive selection; expands the default limit if necessary")
    parser.add_argument("--mtp-graph-budget", type=int, choices=range(3, 25), default=8,
        help="maximum speculative graph captures; prioritize smaller batches and larger draft counts (default: 8)")
    parser.add_argument("--history-drafts", action="store_true",
        help="enable draft continuations from request history during speculation (default: disabled)")
    parser.add_argument("--snapshot-budget-gib", type=float, default=4)
    parser.add_argument("--headroom-gib", type=float, default=4)
    parser.add_argument("--no-compile", action="store_true",
        help="skip torch.compile for faster development startup; still capture CUDA graphs")
    parser.add_argument("--no-dashboard", action="store_true",
        help="disable the native terminal metrics dashboard (automatically disabled when stderr is redirected)")
    parser.add_argument("--template-defaults", type=Path, help="JSON object of default chat template arguments")
    parser.add_argument("--memory-report", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="validate the graph catalog and release it without binding HTTP")
    args = parser.parse_args(argv)
    if args.mtp_draft_tokens is None:
        maximum = args.mtp_max_draft_tokens
        if maximum is None:
            maximum = max(3, args.mtp_fixed_draft_tokens or 0)
        args.mtp_draft_tokens = list(range(maximum + 1))
    if args.mtp_draft_tokens[0] != 0 or sorted(set(args.mtp_draft_tokens)) != args.mtp_draft_tokens:
        parser.error("--mtp-draft-tokens must be increasing unique counts including zero")
    if args.mtp_fixed_draft_tokens is not None and args.mtp_fixed_draft_tokens not in args.mtp_draft_tokens:
        parser.error("--mtp-fixed-draft-tokens must fit the explicit maximum or belong to --mtp-draft-tokens")
    args.mtp_widths = [count + 1 for count in args.mtp_draft_tokens]
    args.mtp_fixed_width = None if args.mtp_fixed_draft_tokens is None else args.mtp_fixed_draft_tokens + 1
    import flare_helios

    with flare_helios.CliSignals():
        return _run(args, parser, started)


def _run(args, parser, started):
    from flare.progress import StartupProgress, phase

    progress = StartupProgress()
    defaults = {}
    if args.template_defaults is not None:
        defaults = json.loads(args.template_defaults.read_text())
        if not isinstance(defaults, dict):
            parser.error("template defaults must be a JSON object")
    architecture = json.loads((args.snapshot / "config.json").read_text()).get("model_type")
    budgets = {"snapshot_budget_bytes": int(args.snapshot_budget_gib * (1 << 30)),
               "headroom_bytes": int(args.headroom_gib * (1 << 30))}
    if architecture == "glm5_next":
        if args.tp not in (2, 4) or args.mtp:
            parser.error("GLM requires --tp 2 or --tp 4 and has no prepared MTP catalog")
        with phase(progress, "Initialize model runtime"):
            from flare.preparation.rank_group import prepare_group
            from flare.recipes.glm53.bootstrap import PreparationConfig, prepare_rank

        config = PreparationConfig(devices=tuple(range(args.device, args.device + args.tp)),
            context=args.context, pages=args.pages, prefill_chunk_size=args.prefill_chunk_size,
            max_active=args.max_active, compile_decoder=not args.no_compile, **budgets)
        model = prepare_group(args.snapshot, config, prepare_rank, progress=progress)
    elif architecture == "qwen4_exp":
        if args.tp != 1 or args.max_active != 4:
            parser.error("Qwen requires --tp 1 and --max-active 4")
        with phase(progress, "Initialize model runtime"):
            from flare.recipes.qwen38.bootstrap import PreparationConfig, prepare_model

        config = PreparationConfig(device=args.device, context=args.context, pages=args.pages,
            prefill_chunk_size=args.prefill_chunk_size,
            mtp=args.mtp, speculative_widths=tuple(args.mtp_widths),
            speculative_proposal_source=args.mtp_proposal_source,
            speculative_fixed_width=args.mtp_fixed_width, speculative_graph_budget=args.mtp_graph_budget,
            history_drafts=args.history_drafts, **budgets)
        model = prepare_model(args.snapshot, config, progress=progress, compile_decoder=not args.no_compile)
    else:
        parser.error(f"unsupported checkpoint model_type {architecture!r}")
    with model:
        if args.memory_report is not None:
            args.memory_report.write_text(json.dumps(model.memory_report, indent=2) + "\n")
        if args.prepare_only:
            model.manifest()
            progress("preparation complete")
            return 0
        with phase(progress, "Start Helios"):
            server = model.start_server(bind=args.bind, template_defaults=defaults)
        print(f"Listening on {server.address}; startup {monotonic() - started:.2f}s", file=sys.stderr, flush=True)
        if not args.no_dashboard:
            try:
                server.start_dashboard()
            except RuntimeError as error:
                print(f"Cannot start Helios dashboard: {error}", file=sys.stderr)
        while server.wait(1000):
            pass
        raise RuntimeError("native server stopped unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
