"""Prepare one checkpoint for native serving or runtime-package export."""

import ipaddress
import json
import os
import socket
import sys
import tempfile
import traceback

from ignition import planning
from ignition.errors import CapacityError, FlareError, PreparationError, UsageError

from . import display, environment
from .preparation import Host, context_parallel, derive, fetch, resolve_model, speculation, stream_overlap
from .config import Configuration

# Request bytes allowed per context token: JSON-escaped text (a \uXXXX escape takes six bytes per
# UTF-16 unit) and message structure stay well within this for prompts that fill the context.
_BODY_BYTES_PER_TOKEN = 32


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


def _keep_reserve(plan, usage) -> None:
    """Require that preparation left the configured reserve free on every GPU."""
    free = min(rank.free_bytes for rank in usage)
    if free < plan.reserve_bytes:
        gib = 1 << 30
        raise CapacityError(f"preparation left {free / gib:.1f} GiB free;\n"
                            f"[memory] reserve requires {plan.reserve_bytes / gib:.1f} GiB.")


def _server_options(configuration, model, plan) -> dict:
    """Derive native server settings independently of preparation's physical KV backing."""
    recipe, checkpoint = plan.recipe, plan.checkpoint
    return {
        "model_id": model.model_id, "served_names": [checkpoint.alias],
        "reserve_bytes": plan.reserve_bytes,
        "api_key": configuration.get("server.api_key").value,
        "queue_capacity": 4 * plan.concurrency, "frontend_workers": plan.concurrency,
        "max_body_bytes": recipe.max_images * recipe.image_bytes + recipe.context * _BODY_BYTES_PER_TOKEN,
        "context_hint": "context is limited by the checkpoint and available cache capacity",
        "capacity_hint": "--max-concurrency sets the request limit; a lower [memory] reserve leaves more KV pages",
    }


def _exhausted(error, plan) -> CapacityError | None:
    """Describe a preparation that ran out of GPU memory, or return None for any other failure."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(error, torch.OutOfMemoryError):
        return None
    relief = ["fewer concurrent requests", "more GPUs"]
    if plan.speculation is not None:
        relief.insert(1, "--no-mtp")
    return CapacityError(
        f"preparation ran out of GPU memory: {str(error).split(' Including non-PyTorch memory')[0]}\n"
        f"{', '.join(relief[:-1]).capitalize()}, or {relief[-1]} need less memory.")


def _publisher(configuration, args):
    from ignition.preparation.protection import Publisher
    values = configuration.values.get("publishing", {})
    try:
        if not values and not args.official_source:
            from ignition.development import publisher
            return publisher(configuration)
        return Publisher.load(values["signing_key"], values["namespace"],
                              args.official_source or values["official_source"])
    except (KeyError, ValueError, OSError) as error:
        raise UsageError(f"publishing requires an Ed25519 signing_key, namespace, and official_source: {error}") from error


def run(args) -> int:
    from .runtime import binary, cached_package

    configuration = Configuration.load(args.config)
    publisher = _publisher(configuration, args)
    model = resolve_model(configuration, args.model)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                args.max_concurrency)
    plan = derive(configuration, model, Host.discover(), gpus=args.gpus, concurrency=concurrency,
                  speculation=speculation(args, model), prefill_chunk=args.prefill_chunk_size,
                  context_parallel=context_parallel(args, model), stream_overlap=stream_overlap(args, model))
    runtime = binary()
    bind = _address(configuration.get("server.host", args.host).value,
                    configuration.get("server.port", args.port).value)
    with cached_package(configuration, model, plan, args.no_compile, publisher) as package:
        if not package.exists():
            _prepare(args, export_path=package)
        else:
            print(f"Load cached runtime package. {package}", flush=True)
    environment.apply(configuration, plan.gpus)
    command = [str(runtime), "run", str(package), "--gpus", ",".join(str(rank) for rank in range(plan.world_size)), "--bind", bind]
    command += configuration.prefix_cache_flags()
    for path in args.publisher_keys:
        command += ["--publisher-keys", str(path)]
    if publisher.development:
        from ignition.development import profile
        command += ["--publisher-keys", str(profile(configuration, publisher))]
    command += ["--dangerously-approve-key", publisher.fingerprint.hex(),
                "--dangerously-accept-publisher", publisher.namespace]
    if args.trust_store is not None:
        command += ["--trust-store", str(args.trust_store)]
    if args.verify_weights:
        command += ["--verify-weights"]
    key = configuration.get("server.api_key").value
    if key:
        key_file = tempfile.TemporaryFile()
        key_file.write(key.encode())
        key_file.flush()
        descriptor = key_file.fileno()
        os.set_inheritable(descriptor, True)
        command += ["--api-key-file", f"/proc/self/fd/{descriptor}"]
    os.execv(command[0], command)
    raise AssertionError("execv returned")


def export(args) -> int:
    if args.output.suffix != ".safetensors":
        args.output = args.output.with_name(args.output.name + ".safetensors")
    if args.output.exists():
        raise UsageError(f"export destination already exists: {args.output}")
    if args.ranks > 1:
        return _export_hosts(args, export_path=args.output)
    return _prepare(args, export_path=args.output)


def _export_hosts(args, *, export_path) -> int:
    """Prepare rank zero here and one peer rank per other host; every host writes its own rank's file.

    Each peer host runs `ignition join` for this rendezvous. Every rank exports to `export_path` on
    its own host; `flare run FILE --rendezvous ADDR:PORT` then serves each rank's file on its host.
    """
    import uuid

    from ignition.progress import phase
    from ignition.preparation.host_ranks import HostRendezvous, local_topology, source_identity, topology_gpu

    if args.rendezvous is None:
        raise UsageError("--ranks greater than one requires --rendezvous ADDRESS:PORT for the peer hosts")
    progress = _serving_progress(args.verbose)
    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    model = resolve_model(configuration, args.model)
    if not model.cached(hub_cache):
        raise UsageError(f"{model.checkpoint.alias} is not cached; a single-host export downloads it")
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                args.max_concurrency)
    host = Host.discover()
    selected = planning.parse_gpus(args.gpus or "0")
    if len(selected) != 1 or selected[0] >= len(host.gpus):
        raise UsageError("multi-host ranks each use one GPU; --gpus names this host's")
    local = (host.gpus[selected[0]],)
    rendezvous = HostRendezvous(args.rendezvous, args.ranks, identity=source_identity(),
                                topology=local_topology(local[0]))
    with phase(progress, "Wait for tensor-parallel hosts") as stage:
        topologies = rendezvous.accept(report=stage.update)
    gpus = tuple(topology_gpu(topology, index) for index, topology in enumerate(topologies))
    plan = planning.plan(model.recipe, model.checkpoint, gpus, selected=tuple(range(len(gpus))),
                         concurrency=concurrency.value, reserve_bytes=configuration.reserve_bytes,
                         speculation=speculation(args, model), prefill_chunk=args.prefill_chunk_size,
                         context_parallel=context_parallel(args, model), stream_overlap=stream_overlap(args, model))
    environment.apply(configuration, local)
    options = _server_options(configuration, model, plan)
    options.pop("api_key")
    print(json.dumps({"plan": plan.to_dict()}), file=sys.stderr, flush=True)
    import flare_heliopause

    signals = flare_heliopause.CliSignals()
    signals.__enter__()
    try:
        import torch

        from ignition.preparation.module_images import ModuleImages
        from ignition.preparation.package_storage import PackageStorage

        recorder = ModuleImages()
        with recorder:
            torch.cuda.set_device(0)
            storage = PackageStorage(world=plan.world_size)
            with storage:
                prepared = model.recipe.prepare(
                    model.snapshot(hub_cache), plan, compile_forwards=not args.no_compile,
                    kernel_cache=configuration.directory("kernel_cache"), progress=progress,
                    retain_graphs=True, package_storage=storage, ranks=rendezvous)
        with prepared:
            with phase(progress, "Export native packages") as stage:
                prepared.export(export_path, options=options, export_id=uuid.uuid4().hex, images=recorder,
                                storage=storage, publisher=_publisher(configuration, args),
                                progress=progress if args.verbose else None)
                stage.update(f"{export_path} on each of {plan.world_size} hosts")
        for region in storage.host_regions:
            region.close()
        return 0
    finally:
        signals.__exit__(None, None, None)


def _prepare(args, *, export_path) -> int:
    from ignition.progress import StartupProgress, phase

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if args.verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    progress = Progress()
    configuration = Configuration.load(args.config)
    publisher = _publisher(configuration, args)
    hub_cache = configuration.directory("hub_cache")
    with phase(progress, "Resolve checkpoint") as stage:
        model = resolve_model(configuration, args.model)
        cached = model.cached(hub_cache)
        stage.update(f"{model.checkpoint.alias} · {model.source} · {'cached' if cached else 'not cached'}")
    with phase(progress, "Select GPUs") as stage:
        concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                    args.max_concurrency)
        host = Host.discover()
        plan = derive(configuration, model, host, gpus=args.gpus, concurrency=concurrency,
                      speculation=speculation(args, model), prefill_chunk=args.prefill_chunk_size,
                      context_parallel=context_parallel(args, model), stream_overlap=stream_overlap(args, model))
        indices = planning.format_gpus(gpu.index for gpu in plan.gpus)
        # A caller's restriction changes which GPU each index names.
        restricted = "" if host.restriction is None else f" of CUDA_VISIBLE_DEVICES={host.restriction}"
        serial = "" if plan.stream_overlap or not plan.recipe.stream_overlap else " · branches serialized"
        stage.update(f"GPU{'s' if plan.world_size > 1 else ''} {indices}{restricted}"
                     f" · concurrency {plan.concurrency} · {display.speculation(plan.speculation, brief=True)}{serial}")
    if args.config is None and not configuration.exists:
        configuration.write_initial(model.checkpoint.alias, concurrency.value)
    if not cached:
        if model.directory is not None:
            raise UsageError(f"{model.directory} lacks tensor files named by its safetensors index")
        with phase(progress, "Download checkpoint"):
            fetch(model, hub_cache, progress_bars=False)
            model.verify(hub_cache)

    environment.apply(configuration, plan.gpus)
    import flare_heliopause

    # Preparation runs native CUDA work that cannot unwind; a signal ends the process at once.
    signals = flare_heliopause.CliSignals()
    signals.__enter__()
    try:
        try:
            from ignition.preparation.module_images import ModuleImages

            recorder = ModuleImages()
            preparation_options = {"retain_graphs": True}
            with recorder:
                import torch
                from ignition.preparation.package_storage import PackageStorage

                torch.cuda.set_device(0)
                storage = PackageStorage(world=plan.world_size)
                preparation_options["package_storage"] = storage
                with storage:
                    prepared = model.recipe.prepare(
                        model.snapshot(hub_cache), plan, compile_forwards=not args.no_compile,
                        kernel_cache=configuration.directory("kernel_cache"), progress=progress,
                        **preparation_options)
        except FlareError:
            raise
        except MemoryError as error:
            raise CapacityError(f"insufficient memory during model preparation: {error}") from error
        except Exception as error:  # noqa: BLE001 - Every other preparation failure maps to one exit code.
            if args.verbose:
                traceback.print_exc()
            raise (_exhausted(error, plan)
                   or PreparationError(f"preparation failed: {type(error).__name__}: {error}")) from error
        with prepared:
            usage = model.recipe.usage(prepared)
            _keep_reserve(plan, usage)
            options = _server_options(configuration, model, plan)
            from ignition.preparation.model_package import export_package

            with phase(progress, "Export native package") as stage:
                try:
                    options.pop("api_key")
                    export_package(prepared, recorder, storage, export_path, options=options, publisher=publisher,
                                 progress=progress if args.verbose else None)
                except Exception as error:
                    if args.verbose:
                        traceback.print_exc()
                    raise PreparationError(f"native package export failed: {error}") from error
                stage.update(str(export_path))
            return 0
    finally:
        signals.__exit__(None, None, None)


def _serving_progress(verbose):
    from ignition.progress import StartupProgress

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    return Progress()


def serve(args) -> int:
    """Prepare in this process and serve over HTTP; with --ranks, one peer rank per other host joins.

    Requires heliopause's qualification build, which embeds the HTTP server in Python. With
    --direct, Ignition's diagnostic step loop serves instead, without Helios or that build.
    """
    from ignition.progress import phase

    progress = _serving_progress(args.verbose)
    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    model = resolve_model(configuration, args.model)
    if not model.cached(hub_cache):
        raise UsageError(f"{model.checkpoint.alias} is not cached; ignition export or run downloads it")
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                args.max_concurrency)
    host = Host.discover()
    bind = _address(configuration.get("server.host", args.host).value,
                    configuration.get("server.port", args.port).value)
    rendezvous = None
    if args.ranks == 1:
        gpus = host.gpus if args.gpus is None else tuple(host.gpus[i] for i in planning.parse_gpus(args.gpus))
        local = gpus
    else:
        from ignition.preparation.host_ranks import HostRendezvous, local_topology, source_identity, topology_gpu

        if args.rendezvous is None:
            raise UsageError("--ranks greater than one requires --rendezvous ADDRESS:PORT for the peer hosts")
        selected = planning.parse_gpus(args.gpus or "0")
        if len(selected) != 1 or selected[0] >= len(host.gpus):
            raise UsageError("multi-host ranks each use one GPU; --gpus names this host's")
        local = (host.gpus[selected[0]],)
        rendezvous = HostRendezvous(args.rendezvous, args.ranks, identity=source_identity(),
                                    topology=local_topology(local[0]))
        with phase(progress, "Wait for tensor-parallel hosts") as stage:
            topologies = rendezvous.accept(report=stage.update)
        gpus = tuple(topology_gpu(topology, index) for index, topology in enumerate(topologies))
    plan = planning.plan(model.recipe, model.checkpoint, gpus,
                         selected=None if args.ranks == 1 and args.gpus is None else tuple(range(len(gpus))),
                         concurrency=concurrency.value, reserve_bytes=configuration.reserve_bytes,
                         speculation=speculation(args, model), prefill_chunk=args.prefill_chunk_size,
                         context_parallel=context_parallel(args, model), stream_overlap=stream_overlap(args, model),
                         cache_states=args.prefix_cache_states)
    if rendezvous is None:
        local = plan.gpus
    environment.apply(configuration, local)
    options = _server_options(configuration, model, plan)
    print(json.dumps({"plan": plan.to_dict()}), file=sys.stderr, flush=True)
    prepared = model.recipe.prepare(
        model.snapshot(hub_cache), plan, compile_forwards=not args.no_compile,
        kernel_cache=configuration.directory("kernel_cache"), progress=progress,
        **({} if rendezvous is None else {"ranks": rendezvous}))
    if args.direct:
        from ignition.direct.server import serve_direct

        code = serve_direct(prepared, bind=bind, verbose=args.verbose,
                            served_names=(*options["served_names"], options["model_id"]))
    else:
        code = _serve_prepared(prepared, bind, options)
    print("Stopped; every rank drained and released its resources.", flush=True)
    return code


def _serve_prepared(prepared, bind, options):
    import signal

    with prepared:
        ranks = [{key: value for key, value in rank.items() if key != "tuning"}
                 for rank in prepared.memory_report.get("ranks", (prepared.memory_report,))]
        print(json.dumps({"memory": ranks}, default=str), file=sys.stderr, flush=True)
        server = prepared.start_server(bind=bind, options=options)
        print(f"Listening on {server.address}", flush=True)

        def stop(signum, frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            while server.wait(1000):
                pass
            raise PreparationError("native HTTP server stopped unexpectedly")
        except KeyboardInterrupt:
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous)


def join(args) -> int:
    """Run one peer rank on this host for rank zero's rendezvous on another host."""
    import time

    from ignition.preparation.host_ranks import knock, local_topology, serve_peer, source_identity

    started = time.monotonic()

    def progress(message):
        print(f"{time.monotonic() - started:8.1f}s rank {args.rank}: {message}", flush=True)

    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    host = Host.discover()
    if not 0 <= args.gpu < len(host.gpus):
        raise UsageError(f"--gpu {args.gpu} names a GPU this host does not have")
    gpu = host.gpus[args.gpu]
    environment.apply(configuration, (gpu,))
    control, executor = knock(args.rendezvous, args.rank, args.ranks, identity=source_identity(),
                              topology=local_topology(gpu), report=progress)
    serve_peer(control, executor, args.rank, args.ranks, hub_cache=hub_cache, progress=progress)
    return 0
