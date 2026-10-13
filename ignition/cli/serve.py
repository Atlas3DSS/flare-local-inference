"""Prepare one checkpoint for native serving or runtime-package export."""

import ipaddress
import os
import socket
import sys
import tempfile
import traceback

from ignition import planning
from ignition.errors import CapacityError, FlareError, PreparationError, UsageError

from . import display, environment
from .preparation import Host, derive, fetch, resolve_model, speculation
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


def run(args) -> int:
    from .runtime import binary, cached_package

    configuration = Configuration.load(args.config)
    model = resolve_model(configuration, args.model)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                args.max_concurrency)
    plan = derive(configuration, model, Host.discover(), gpus=args.gpus, concurrency=concurrency,
                  speculation=speculation(args, model), prefill_chunk=args.prefill_chunk_size,
                  context_parallel=not args.no_context_parallel)
    runtime = binary()
    bind = _address(configuration.get("server.host", args.host).value,
                    configuration.get("server.port", args.port).value)
    with cached_package(configuration, model, plan, args.no_compile) as package:
        if not package.exists():
            _prepare(args, export_path=package)
        else:
            print(f"Load cached runtime package. {package}", flush=True)
    environment.apply(configuration, plan.gpus)
    command = [str(runtime), "run", str(package), "--gpus", ",".join(str(rank) for rank in range(plan.world_size)), "--bind", bind]
    command += configuration.prefix_cache_flags()
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
    if args.output.suffix != ".flaretensors":
        args.output = args.output.with_name(args.output.name + ".flaretensors")
    if args.output.exists():
        raise UsageError(f"export destination already exists: {args.output}")
    return _prepare(args, export_path=args.output)


def _prepare(args, *, export_path) -> int:
    from ignition.progress import StartupProgress, phase

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if args.verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    progress = Progress()
    configuration = Configuration.load(args.config)
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
                      context_parallel=not args.no_context_parallel)
        indices = planning.format_gpus(gpu.index for gpu in plan.gpus)
        # A caller's restriction changes which GPU each index names.
        restricted = "" if host.restriction is None else f" of CUDA_VISIBLE_DEVICES={host.restriction}"
        stage.update(f"GPU{'s' if plan.world_size > 1 else ''} {indices}{restricted}"
                     f" · concurrency {plan.concurrency} · {display.speculation(plan.speculation, brief=True)}")
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
                    export_package(prepared, recorder, storage, export_path, options=options,
                                 progress=progress if args.verbose else None)
                except Exception as error:
                    if args.verbose:
                        traceback.print_exc()
                    raise PreparationError(f"native package export failed: {error}") from error
                stage.update(str(export_path))
            return 0
    finally:
        signals.__exit__(None, None, None)
