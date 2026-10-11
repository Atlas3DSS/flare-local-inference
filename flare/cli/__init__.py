"""The `flare` command line: argument parsing and dispatch to command handlers."""

import argparse
from pathlib import Path

from flare.errors import FlareError


def _parser() -> argparse.ArgumentParser:
    from . import bench, commands, serve, status

    parser = argparse.ArgumentParser(
        prog="flare", allow_abbrev=False,
        description="Serve qualified checkpoints on NVIDIA SM120 and SM121 GPUs.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, handler, summary, *, configured=True, json=False):
        subparser = subparsers.add_parser(name, help=summary, description=summary, allow_abbrev=False)
        subparser.set_defaults(handler=handler)
        if configured:
            subparser.add_argument("--config", type=Path, metavar="PATH",
                                   help="configuration file (default: ~/.config/flare/config.toml)")
        if json:
            subparser.add_argument("--json", action="store_true", help="print JSON")
        return subparser

    running = command("run", serve.run, "prepare and serve")
    running.add_argument("model", nargs="?", metavar="MODEL",
                         help="recipe alias, repository id, or checkpoint directory (default: defaults.model)")
    running.add_argument("--gpus", metavar="LIST",
                         help="GPU indices as NVML enumerates them, such as 0, 0,1, or 0-3;"
                              " their count is the tensor-parallel size (default: first GPUs that fit)")
    running.add_argument("--max-concurrency", type=int, metavar="N",
                         help="maximum simultaneously active requests (default: defaults.max_concurrency,"
                              " else the recipe default)")
    running.add_argument("--host", metavar="ADDR", help="listening address (default: server.host, else 127.0.0.1)")
    running.add_argument("--port", type=int, metavar="N", help="listening port (default: server.port, else 8000)")
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
                          help="verify exactly the draft-token count every round instead of adapting")
    advanced.add_argument("--mtp-graph-budget", type=int, metavar="N",
                          help="most speculative graphs captured across batch sizes, small batches and deep"
                               " drafts first (default: the recipe's)")
    advanced.add_argument("--mtp-proposal-source", choices=("mtp", "copy"),
                          help="where drafts come from: the checkpoint's MTP module, or copy for continuations"
                               " of the request's own history, which loads no draft model (default: mtp)")
    advanced.add_argument("--history-drafts", action="store_true",
                          help="also draft continuations copied from the request's own history")
    advanced.add_argument("--no-compile", action="store_true",
                          help="development: capture graphs without compiling forwards")
    command("models", commands.models, "list supported checkpoints and what fits on this host", json=True)
    command("pull", commands.pull, "download the pinned revision").add_argument("model", metavar="MODEL")
    command("status", status.status, "summarize a running server", json=True).add_argument(
        "--url", metavar="URL", help="server address (default: the configured host and port)")
    command("bench", bench.bench, "measure a running server", json=True).add_argument(
        "--url", metavar="URL", help="server address (default: the configured host and port)")
    command("config", commands.config, "print the effective configuration and file locations", json=True)
    command("skill", commands.skill, "print the agent guide", configured=False)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args) or 0
    except FlareError as error:
        from . import display

        display.error(str(error))
        return error.exit_code
