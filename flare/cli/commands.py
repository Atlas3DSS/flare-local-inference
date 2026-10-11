"""Handlers for the commands that neither load weights nor contact a server."""

import json
import os
import sys
from dataclasses import dataclass
from importlib import resources

from flare import planning
from flare.errors import CapacityError, FetchError, UsageError
from flare.recipes import ResolvedModel, recipes, resolve

from .config import Configuration, Setting


@dataclass(frozen=True)
class Host:
    """GPUs as the command line indexes them, and the caller's restriction on them."""

    gpus: tuple[planning.Gpu, ...]
    restriction: str | None

    @classmethod
    def discover(cls) -> "Host":
        gpus = planning.discover_gpus()
        # The one environment variable Flare inspects: a caller's restriction
        # changes which GPU each index names, and the plan reports that.
        restriction = os.environ.get("CUDA_VISIBLE_DEVICES")
        if restriction is not None:
            gpus = planning.restrict_gpus(gpus, restriction)
        return cls(gpus, restriction)


def resolve_model(configuration: Configuration, name: str | None) -> ResolvedModel:
    name = configuration.model if name is None else name
    if name is None:
        raise UsageError("no MODEL given and defaults.model is not set in the configuration file")
    model = resolve(name, configuration.directory("hub_cache"))
    model.verify(configuration.directory("hub_cache"))
    return model


def derive(configuration: Configuration, model: ResolvedModel, host: Host, *, gpus: str | None,
           concurrency: Setting, prefill_chunk: int | None = None) -> planning.Plan:
    """Plan one checkpoint on this host from the configuration and command-line intent."""
    kernel_cache = configuration.directory("kernel_cache")

    def measured(world_size, limit, chunk, gpu_name):
        path = planning.measurement_path(kernel_cache, model.checkpoint, world_size, limit, chunk)
        return planning.read_measurement(path, gpu_name)

    return planning.plan(
        model.recipe, model.checkpoint, host.gpus, selected=None if gpus is None else planning.parse_gpus(gpus),
        concurrency=concurrency.value, reserve_bytes=configuration.reserve_bytes, prefill_chunk=prefill_chunk,
        measured=measured)


def plan(args) -> int:
    configuration = Configuration.load(args.config)
    model = resolve_model(configuration, args.model)
    host = Host.discover()
    if args.verbose:
        for gpu in host.gpus:
            print(f"gpu {gpu.index}  {gpu.name}  {gpu.free_bytes / (1 << 30):.1f} GiB free"
                  f"  SM{gpu.capability[0]}{gpu.capability[1]}  {gpu.uuid}", file=sys.stderr)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                args.max_concurrency)
    result = derive(configuration, model, host, gpus=args.gpus, concurrency=concurrency,
                    prefill_chunk=args.prefill_chunk_size)
    if args.json:
        print(json.dumps({**result.to_dict(), "cuda_visible_devices": host.restriction}, indent=2))
    else:
        print(planning.render(result, restriction=host.restriction))
    return 0


def models(args) -> int:
    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    host = Host.discover()
    entries = []
    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            model = ResolvedModel(recipe, checkpoint)
            entry = {"alias": checkpoint.alias, "repository": checkpoint.repository,
                     "revision": checkpoint.revision, "stored_bytes": checkpoint.stored_bytes,
                     "cached": model.cached(hub_cache), "plan": None, "error": None}
            try:
                fitted = derive(configuration, model, host, gpus=None,
                                concurrency=Setting(recipe.concurrency, "recipe"))
                entry.update(plan=fitted.to_dict(), fits=planning.summarize(fitted))
            except CapacityError as error:
                entry.update(error=str(error), fits="no")
            entries.append(entry)
    if args.json:
        print(json.dumps([{key: value for key, value in entry.items() if key != "fits"} for entry in entries],
                         indent=2))
        return 0
    width = max(len("ALIAS"), *(len(entry["alias"]) for entry in entries)) + 4
    print(f"{'ALIAS':<{width}}{'SIZE':<10}{'CACHED':<9}FITS HERE")
    for entry in entries:
        size = f"{round(entry['stored_bytes'] / (1 << 30))} GiB"
        print(f"{entry['alias']:<{width}}{size:<10}{'yes' if entry['cached'] else 'no':<9}{entry['fits']}")
    return 0


def pull(args) -> int:
    configuration = Configuration.load(args.config)
    model = resolve(args.model, configuration.directory("hub_cache"))
    if model.directory is not None:
        raise UsageError(f"{model.directory} is a local checkpoint directory; pull takes an alias or repository id")
    print(fetch(model, configuration.directory("hub_cache")))
    return 0


def fetch(model: ResolvedModel, hub_cache, *, progress_bars: bool = True) -> str:
    """Download the pinned revision's missing files and return the snapshot directory."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress_bars:
            disable_progress_bars()
        return snapshot_download(model.checkpoint.repository, revision=model.checkpoint.revision,
                                 cache_dir=hub_cache)
    except Exception as error:  # noqa: BLE001 - Network, authorization, and disk failures share one exit code.
        raise FetchError(f"cannot download {model.checkpoint.repository} at revision"
                         f" {model.checkpoint.revision[:8]}: {error}") from error


def config(args) -> int:
    configuration = Configuration.load(args.config)
    settings = configuration.settings()
    locations = {"configuration": str(configuration.path) + ("" if configuration.exists else " (absent)"),
                 "hub cache": str(configuration.directory("hub_cache")),
                 "kernel cache": str(configuration.directory("kernel_cache"))}

    def shown(key, setting):
        # The API key is a secret; report only whether one is set.
        if key == "server.api_key":
            return "(set)" if setting.value else "(empty)"
        return "" if setting.value is None else setting.value

    if args.json:
        print(json.dumps({"locations": locations, "settings": {
            key: {"value": shown(key, setting), "source": setting.source} for key, setting in settings.items()}},
            indent=2))
        return 0
    for name, location in locations.items():
        print(f"{name:<16}{location}")
    print()
    width = max(map(len, settings)) + 3
    for key, setting in settings.items():
        print(f"{key:<{width}}{str(shown(key, setting)):<28}{setting.source}")
    return 0


def skill(args) -> int:
    print(resources.files("flare.cli").joinpath("skill.md").read_text(), end="")
    return 0
