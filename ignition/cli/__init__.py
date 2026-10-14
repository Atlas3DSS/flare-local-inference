"""The `ignition` preparation command line and development package launcher."""

import argparse
from pathlib import Path

from ignition.errors import FlareError


def _parser() -> argparse.ArgumentParser:
    from . import serve

    parser = argparse.ArgumentParser(
        prog="ignition", allow_abbrev=False,
        description="Prepare native runtime packages for NVIDIA SM103, SM120, and SM121 GPUs.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, handler, summary):
        subparser = subparsers.add_parser(name, help=summary, description=summary, allow_abbrev=False)
        subparser.set_defaults(handler=handler)
        subparser.add_argument("--config", type=Path, metavar="PATH",
                               help="configuration file (default: ~/.config/flare/config.toml)")
        return subparser

    running = command("run", serve.run, "cache a runtime package and launch Flare")
    exporting = command("export", serve.export, "prepare a native model package")
    exporting.add_argument("--output", type=Path, required=True, metavar="FILE",
                           help="destination .safetensors file; must not already exist")
    for preparing in (running, exporting):
        _preparation_arguments(preparing)
    running.add_argument("--publisher-keys", type=Path, action="append", default=[], metavar="PATH")
    running.add_argument("--trust-store", type=Path, metavar="PATH")
    running.add_argument("--verify-weights", action="store_true")
    serving = command("serve", serve.serve, "prepare in this process and serve over HTTP, with peer ranks on other hosts")
    for listening in (running, serving):
        listening.add_argument("--host", metavar="ADDR", help="listening address (default: server.host, else 127.0.0.1)")
        listening.add_argument("--port", type=int, metavar="N", help="listening port (default: server.port, else 8000)")
    _preparation_arguments(serving)
    for spanning in (serving, exporting):
        hosts = spanning.add_argument_group("multi-host tensor parallelism")
        hosts.add_argument("--ranks", type=int, default=1, choices=(1, 2, 4), metavar="N",
                           help="tensor-parallel ranks, one per host; this process is rank zero (default: 1);"
                                " an export writes each host's own rank file at --output on that host")
        hosts.add_argument("--rendezvous", metavar="ADDR:PORT",
                           help="IPv4 address and port on which rank zero admits its peers")
    serving.add_argument("--prefix-cache-states", type=int, metavar="N",
                         help="advanced: cached request states kept resident for prefix reuse; fewer leave more"
                              " memory for KV pages (default: sized from free memory)")
    joining = command("join", serve.join, "run a peer rank for rank zero's rendezvous on another host")
    joining.add_argument("rendezvous", metavar="ADDR:PORT", help="rank zero's rendezvous address")
    joining.add_argument("--rank", type=int, required=True, metavar="R", help="this host's rank, 1 through N-1")
    joining.add_argument("--ranks", type=int, required=True, choices=(2, 4), metavar="N", help="tensor-parallel ranks")
    joining.add_argument("--gpu", type=int, default=0, metavar="INDEX", help="this host's GPU (default: 0)")
    return parser


def _preparation_arguments(running):
    running.add_argument("--official-source", metavar="URL", help="published checkpoint repository URL")
    running.add_argument("model", nargs="?", metavar="MODEL",
                         help="recipe alias, repository id, or checkpoint directory (default: defaults.model)")
    running.add_argument("--gpus", metavar="LIST",
                         help="GPU indices as NVML enumerates them, such as 0, 0,1, or 0-3;"
                              " their count is the tensor-parallel size (default: first GPUs that fit)")
    running.add_argument("--max-concurrency", type=int, metavar="N",
                         help="maximum simultaneously active requests (default: defaults.max_concurrency,"
                              " else the recipe default)")
    running.add_argument("--verbose", action="store_true", help="per-phase detail on stderr")
    advanced = running.add_argument_group("advanced options")
    advanced.add_argument("--prefill-chunk-size", type=int, metavar="N",
                          help="most prompt tokens per prefill step; larger chunks reach a long prompt's first"
                               " token sooner and pause other requests longer (default: the recipe's)")
    advanced.add_argument("--no-mtp", action="store_true",
                          help="disable multi-token speculation: load no draft model and decode one token"
                               " per step")
    advanced.add_argument("--mtp-draft-tokens", type=int, metavar="N",
                          help="most draft tokens verified in one round, 0 through 7; rounds adapt among"
                               " 0 through N (default: the recipe's)")
    advanced.add_argument("--no-mtp-adaptive-verification", action="store_true",
                          help="default to fixed verification at runtime; capture still includes every smaller width")
    advanced.add_argument("--mtp-proposal-source", choices=("mtp", "copy"),
                          help="where drafts come from: the checkpoint's MTP module, or copy for continuations"
                               " of the request's own history, which loads no draft model (default: mtp)")
    advanced.add_argument("--history-drafts", action="store_true",
                          help="also draft continuations copied from the request's own history")
    advanced.add_argument("--no-context-parallel", action="store_true",
                          help="keep each rank's attention context whole instead of sharding it across"
                               " tensor-parallel ranks; fewer cached tokens (default: the recipe's)")
    advanced.add_argument("--no-stream-overlap", action="store_true",
                          help="run every concurrent branch (shared experts, projection chains, query chains)"
                               " serially on the main stream, to diagnose scheduling-dependent faults; slower"
                               " (default: overlap)")
    advanced.add_argument("--no-compile", action="store_true",
                          help="development: capture graphs without compiling forwards")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args) or 0
    except FlareError as error:
        from . import display

        display.error(str(error))
        return error.exit_code
