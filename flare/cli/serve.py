"""`flare run` and `flare prepare`: fetch, prepare, measure, and serve one checkpoint."""

import ipaddress
import json
import signal
import socket
import traceback
from dataclasses import replace

from flare import planning
from flare.errors import CapacityError, FlareError, PreparationError, UsageError

from . import environment
from .commands import Host, derive, fetch, resolve_model
from .config import Configuration


def _address(host: str, port: int) -> str:
    """Validate a listening address and confirm that nothing else holds it."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise UsageError(f"--host must be an IP address; got {host!r}") from error
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    try:
        socket.create_server((host, port), family=family).close()
    except OSError as error:
        raise UsageError(f"cannot listen on {host} port {port}: {error.strerror or error}") from error
    return f"[{host}]:{port}" if address.version == 6 else f"{host}:{port}"


def _measure(configuration, plan, usage):
    """Record what preparation held on each rank and return the plan with measured figures."""
    costs = plan.costs
    rows = plan.recipe.live_rows(plan.concurrency) + plan.cache_states
    for rank in usage:
        if rank.page_bytes != costs.page_bytes or rank.state_bytes != rows * costs.state_row_bytes:
            raise PreparationError(
                f"the {plan.recipe.name} recipe declares {costs.page_bytes} bytes per page and"
                f" {rows * costs.state_row_bytes} bytes of state; preparation allocated {rank.page_bytes} and"
                f" {rank.state_bytes}")
    # The rank holding the most memory outside state rows and pages bounds every rank's supply.
    held, rank = max(((gpu.free_bytes - rank.free_bytes - rank.state_bytes - rank.pages * rank.page_bytes, rank)
                      for gpu, rank in zip(plan.gpus, usage, strict=True)), key=lambda item: item[0])
    measurement = planning.Measurement(rank.weights_bytes, max(0, held - rank.weights_bytes))
    planning.write_measurement(
        planning.measurement_path(configuration.directory("kernel_cache"), plan.checkpoint, plan.world_size,
                                  plan.concurrency, plan.prefill_chunk), plan.gpus[0].name, measurement)
    measured = replace(plan, pages=min(rank.pages for rank in usage), weights_bytes=measurement.weights_bytes,
                       graph_bytes=measurement.graph_bytes, measured=True)
    # Headroom has served its purpose once capture ends; only the reserve must remain free.
    free = min(rank.free_bytes for rank in usage)
    if free < plan.reserve_bytes:
        gib = 1 << 30
        raise CapacityError(
            f"prepared graphs and scratch hold {(measurement.graph_bytes - plan.graph_bytes) / gib:.1f} GiB more"
            f" than planned and leave {free / gib:.1f} GiB free;\n"
            f"[memory] reserve requires {plan.reserve_bytes / gib:.1f} GiB.\n"
            "The measured sizes are recorded: running again plans within them.")
    return measured


def _server_options(configuration, model, plan) -> dict:
    """Derive the native server's names, HTTP limits, and rejection hints from the plan."""
    recipe, checkpoint = plan.recipe, plan.checkpoint
    return {
        "model_id": model.model_id, "served_names": [checkpoint.alias],
        "api_key": configuration.get("server.api_key").value,
        "queue_capacity": 4 * plan.concurrency, "frontend_workers": plan.concurrency,
        "max_body_bytes": recipe.max_images * recipe.image_bytes + (1 << 20),
        "context_hint": (
            "this is the checkpoint's maximum context" if plan.context == recipe.context else
            "a lower --max-concurrency leaves more KV pages for each request"),
        "capacity_hint": "--max-concurrency sets the request limit; a lower [memory] reserve leaves more KV pages",
    }


def _tuning(prepared) -> dict:
    report = prepared.memory_report
    tuning = report.get("ranks", (report,))[0].get("tuning") or {}
    return {key: tuning.get(key) for key in ("requests", "cache_hits", "benchmarked_candidates", "selection_counts")}


def _serve(args, *, listen: bool) -> int:
    from flare.progress import StartupProgress, phase

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if args.verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    progress = Progress()
    progress.splash()
    configuration = Configuration.load(args.config)
    bind = None
    if listen:
        bind = _address(configuration.get("server.host", args.host).value,
                        configuration.get("server.port", args.port).value)
    hub_cache = configuration.directory("hub_cache")
    with phase(progress, "Resolve checkpoint") as stage:
        model = resolve_model(configuration, args.model)
        cached = model.cached(hub_cache)
        stage.update(f"{model.checkpoint.alias} · {model.source} · {'cached' if cached else 'not cached'}")
    with phase(progress, "Plan memory") as stage:
        concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                    args.max_concurrency)
        plan = derive(configuration, model, Host.discover(), gpus=args.gpus, concurrency=concurrency,
                      prefill_chunk=args.prefill_chunk_size)
        stage.update(f"{plan.world_size} GPU{'s' if plan.world_size > 1 else ''} · concurrency {plan.concurrency}"
                     f" · {plan.tokens:,} KV tokens")
    if listen and args.config is None and not configuration.exists:
        configuration.write_initial(model.checkpoint.alias, concurrency.value)
    if not cached:
        if model.directory is not None:
            raise UsageError(f"{model.directory} lacks tensor files named by its safetensors index")
        with phase(progress, "Download checkpoint"):
            fetch(model, hub_cache, progress_bars=False)
            model.verify(hub_cache)

    environment.apply(configuration, plan.gpus)
    import flare_helios

    # Preparation runs native CUDA work that cannot unwind; a signal ends the process at once.
    signals = flare_helios.CliSignals()
    signals.__enter__()
    try:
        try:
            prepared = model.recipe.prepare(
                model.snapshot(hub_cache), plan, compile_forwards=not args.no_compile,
                kernel_cache=configuration.directory("kernel_cache"), progress=progress)
        except FlareError:
            raise
        except MemoryError as error:
            raise CapacityError(
                f"{error};\nmeasured graphs and scratch leave fewer than the minimum {plan.minimum_pages:,} pages"
                f" for {plan.concurrency} requests of {planning.MINIMUM_REQUEST_TOKENS:,} tokens.") from error
        except Exception as error:  # noqa: BLE001 - Every preparation failure maps to one exit code.
            if args.verbose:
                traceback.print_exc()
            raise PreparationError(f"preparation failed: {type(error).__name__}: {error}") from error
        with prepared:
            with phase(progress, "Measure memory") as stage:
                measured = _measure(configuration, plan, model.recipe.usage(prepared))
                stage.update(f"{measured.pages:,} pages · {measured.tokens:,} tokens"
                             f" · {measured.cache_states} cached states")
            tuning = _tuning(prepared)
            if not listen:
                prepared.manifest()
                if args.json:
                    print(json.dumps({**measured.to_dict(), "tuning": tuning}, indent=2))
                else:
                    print(planning.render(measured))
                    counts = tuning["selection_counts"] or {}
                    print(f"\nkernel selections   {', '.join(f'{count:,} {source}' for source, count in counts.items())}"
                          f"   {tuning['benchmarked_candidates'] or 0:,} candidates measured")
                return 0
            with phase(progress, "Start Helios") as stage:
                try:
                    server = prepared.start_server(bind=bind, options=_server_options(configuration, model, plan))
                except Exception as error:  # noqa: BLE001 - Report native startup failures by exit code.
                    if "cannot bind HTTP listener" in str(error):
                        raise UsageError(f"cannot listen on {bind}: {error}") from error
                    raise FlareError(f"the native server did not start: {error}") from error
                stage.update(f"listening on {server.address}")
            try:
                server.start_dashboard()
            except RuntimeError as error:
                progress.console.print(f"Helios dashboard unavailable: {error}", highlight=False)
            # Serving stops gracefully: closing the prepared model ends admission and drains replay.
            signals.__exit__(None, None, None)

            def interrupt(signum, frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, interrupt)
            try:
                while server.wait(1000):
                    pass
            except KeyboardInterrupt:
                return 0
            raise FlareError("the native server stopped unexpectedly")
    finally:
        signals.__exit__(None, None, None)


def run(args) -> int:
    return _serve(args, listen=True)


def prepare(args) -> int:
    return _serve(args, listen=False)
