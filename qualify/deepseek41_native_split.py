"""Serve DS4.1 CSF graphs through Helios with explicit concurrency and cache budgets.

Requires a qualification build of Heliopause. Both modes retain Engram tables
in shared resident RAM and hand graphs directly to the native HTTP server.
No runtime package is exported. Split execution remains research-only.

python -m qualify.deepseek41_native_split --gpus 0,1,2,3 --mode split --output /tmp/native-split
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace


def main(args):
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply

    configuration = Configuration.load(None)
    apply(configuration, [SimpleNamespace(uuid=gpu) for gpu in args.gpus.split(",")])

    import torch
    import flashinfer
    from ignition.planning import CACHE_STATES_PER_REQUEST
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.rank_group import prepare_group
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig, prepare_rank
    from ignition.recipes.deepseek41.recipe import RECIPE

    checkpoint = RECIPE.checkpoints[0]
    snapshot = configuration.directory("hub_cache") / ("models--" + checkpoint.repository.replace("/", "--")) / "snapshots" / checkpoint.revision
    if args.mode == "split" and args.max_concurrency != 1:
        raise ValueError("split speculation is qualified only at concurrency one")
    snapshot_rows = args.snapshot_rows or CACHE_STATES_PER_REQUEST * args.max_concurrency
    config = PreparationConfig(max_active=args.max_concurrency, speculative_widths=tuple(range(1, 9)),
        speculative_fixed_width=args.fixed_width, split_speculation=args.mode == "split",
        snapshot_budget_bytes=snapshot_rows * RECIPE.ranks[4].state_row_bytes,
        kv_pages=args.kv_pages,
        headroom_bytes=args.headroom_mib << 20, prefill_chunk_size=4096,
        retain_graphs=False, compile_decoder=True, kernel_cache=configuration.directory("kernel_cache"))
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(preparation=asdict(config), mode=args.mode, devices=args.gpus,
        proposal_confidence_threshold=args.threshold, embedding_table_storage="ram",
        sampling=dict(temperature=1.0, top_p=0.95, top_k=0), checkpoint=asdict(checkpoint),
        flashinfer=dict(version=flashinfer.__version__, path=flashinfer.__file__))
    (args.output / "configuration.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
    print(json.dumps(settings, default=str), flush=True)
    identity = dict(pid=os.getpid(), start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split(") ")[1].split()[19],
        cwd=str(Path.cwd()), command=Path(f"/proc/{os.getpid()}/cmdline").read_bytes().decode().replace("\0", " "))
    (args.output / "process.json").write_text(json.dumps(identity, indent=2) + "\n")
    started = time.monotonic()
    def progress(message):
        print(f"{time.monotonic()-started:8.1f}s {message}", flush=True)
    torch.cuda.set_device(0)
    storage = PackageStorage(world=4)
    try:
        with torch.no_grad(), storage:
            group = prepare_group(snapshot, config, prepare_rank, package_storage=storage, progress=progress)
        with group:
            report = {**group.memory_report, "ranks": [
                {key: value for key, value in rank.items() if key != "tuning"}
                for rank in group.memory_report["ranks"]]}
            (args.output / "memory.json").write_text(json.dumps(report, default=str, indent=2) + "\n")
            identity["peers"] = [process.pid for process in group.processes]
            (args.output / "process.json").write_text(json.dumps(identity, indent=2) + "\n")
            server = group.start_server(bind=args.bind, options={
                "proposal_confidence_threshold": args.threshold,
                "queue_capacity": 4 * args.max_concurrency, "frontend_workers": args.max_concurrency})
            (args.output / "ready.json").write_text(json.dumps(dict(address=server.address, startup_seconds=time.monotonic()-started)) + "\n")
            print(f"Listening on {server.address}", flush=True)
            def stop(signum, frame):
                raise KeyboardInterrupt
            previous = signal.signal(signal.SIGTERM, stop)
            try:
                while server.wait(1000):
                    pass
                raise RuntimeError("native HTTP server stopped unexpectedly")
            except KeyboardInterrupt:
                pass
            finally:
                signal.signal(signal.SIGTERM, previous)
    finally:
        for region in storage.host_regions:
            region.close()
    print("Every rank drained.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--mode", choices=("baseline", "split"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1:8121")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--snapshot-rows", type=int, help="defaults to eight retained states per request")
    parser.add_argument("--kv-pages", type=int, help="fixed per-rank page count; omitted uses available GPU memory")
    parser.add_argument("--headroom-mib", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--fixed-width", type=int, choices=range(1, 9))
    main(parser.parse_args())
