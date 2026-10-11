"""Handlers for the commands that neither load weights nor contact a server."""

import json
import os
from dataclasses import dataclass
from importlib import resources

from flare import planning
from flare.errors import CapacityError, FetchError, UsageError
from flare.recipes import ResolvedModel, recipes, resolve
from flare.recipes.declaration import Speculation

from . import display
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
           concurrency: Setting, speculation: Speculation | None, prefill_chunk: int | None = None) -> planning.Plan:
    """Select GPUs and estimate memory for one checkpoint from the configuration and command-line intent."""
    return planning.plan(
        model.recipe, model.checkpoint, host.gpus, selected=None if gpus is None else planning.parse_gpus(gpus),
        concurrency=concurrency.value, reserve_bytes=configuration.reserve_bytes, speculation=speculation,
        prefill_chunk=prefill_chunk)


def speculation(args, model: ResolvedModel) -> Speculation | None:
    """Apply the command line's speculation overrides to the recipe's settings; None disables speculation."""
    recipe, alias = model.recipe, model.checkpoint.alias
    default, limit = recipe.speculation, recipe.speculative_rows
    given = [flag for flag, value in (
        ("--mtp-draft-tokens", args.mtp_draft_tokens),
        ("--no-mtp-adaptive-verification", args.no_mtp_adaptive_verification or None),
        ("--mtp-graph-budget", args.mtp_graph_budget), ("--mtp-proposal-source", args.mtp_proposal_source),
        ("--history-drafts", args.history_drafts or None)) if value is not None]
    if default is None:
        if args.no_mtp or given:
            raise UsageError(f"{alias} has no multi-token speculation; {(given or ['--no-mtp'])[0]} does not apply")
        return None
    if args.no_mtp:
        if given:
            raise UsageError(f"--no-mtp disables speculation; {', '.join(given)} cannot accompany it")
        return None

    drafts = default.draft_tokens if args.mtp_draft_tokens is None else args.mtp_draft_tokens
    if not 0 <= drafts <= limit:
        raise UsageError(f"--mtp-draft-tokens must be 0 through {limit}; --no-mtp disables speculation")
    adaptive = default.adaptive and not args.no_mtp_adaptive_verification
    source = default.proposal_source if args.mtp_proposal_source is None else args.mtp_proposal_source
    history = default.history_drafts or args.history_drafts
    if (source == "copy" or history) and not recipe.history_copy:
        flag = "--history-drafts" if args.history_drafts else "--mtp-proposal-source copy"
        raise UsageError(f"{alias} drafts only with its MTP module; {flag} does not apply")
    budget = default.graph_budget if args.mtp_graph_budget is None else args.mtp_graph_budget
    return Speculation(drafts, adaptive, budget, source, history)


def _listed(values) -> str:
    """Join alternatives as `1`, `2 or 4`, or `1, 2, or 4`."""
    names = [str(value) for value in values]
    if len(names) < 3:
        return " or ".join(names)
    return f"{', '.join(names[:-1])}, or {names[-1]}"


def _model(entry: dict, recipe, fit: str | None):
    """Describe one recipe's default checkpoint, its serving range, and how it fits this host."""
    sizes = [f"{display.gib(entry['stored_bytes'])} GiB stored", f"{display.gib(entry['device_bytes'])} GiB on GPUs"]
    if entry["mapped_host_bytes"]:
        sizes.append(f"{display.gib(entry['mapped_host_bytes'])} GiB {recipe.mapped_host_name} in host memory")
    images = entry["max_images"]
    cached = (display.styled("yes", display.GOOD) if entry["cached"]
              else display.styled("no", display.CAUTION).append(" · downloaded by the first run", style="dim"))
    fits = (display.styled("no", display.BAD).append(f"\n{entry['error']}", style="dim") if fit is None
            else display.styled(fit, display.GOOD))
    return display.section(
        display.title(entry["alias"], "defaults.model" if entry["default"] else ""),
        display.fields([
            ("checkpoint", f"{entry['repository']}@{entry['revision'][:8]}"),
            ("architecture", entry["architecture"]),
            ("size", " · ".join(sizes)),
            ("context", f"{entry['context']:,} tokens"),
            ("input", f"text and up to {images} images per request" if images else "text"),
            ("speculation", "none" if recipe.speculation is None else display.speculation(recipe.speculation)),
            ("GPUs", _listed(entry["tensor_parallel_sizes"])),
            ("concurrency", f"{entry['default_concurrency']} by default · at most {entry['concurrency_limit']}"),
            ("cached", cached),
            ("fits here", fits),
        ]))


def models(args) -> int:
    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    host = Host.discover()
    entries, described = [], []
    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            model = ResolvedModel(recipe, checkpoint)
            entry = {
                "alias": checkpoint.alias, "default": configuration.model == checkpoint.alias,
                "recipe": recipe.name, "repository": checkpoint.repository, "revision": checkpoint.revision,
                "architecture": checkpoint.architecture, "stored_bytes": checkpoint.stored_bytes,
                "device_bytes": checkpoint.device_bytes, "mapped_host_bytes": checkpoint.mapped_host_bytes,
                "cached": model.cached(hub_cache), "context": recipe.context, "max_images": recipe.max_images,
                "speculation": None if recipe.speculation is None else {
                    "draft_tokens": recipe.speculation.draft_tokens, "adaptive": recipe.speculation.adaptive,
                    "graph_budget": recipe.speculation.graph_budget,
                    "proposal_source": recipe.speculation.proposal_source,
                    "history_drafts": recipe.speculation.history_drafts, "history_copy": recipe.history_copy},
                "tensor_parallel_sizes": list(recipe.tensor_parallel),
                "default_concurrency": recipe.concurrency, "concurrency_limit": recipe.max_concurrency,
                "prefill_chunk": recipe.prefill_chunk, "estimate": None, "error": None}
            fit = None
            try:
                fitted = derive(configuration, model, host, gpus=None,
                                concurrency=Setting(recipe.concurrency, "recipe"), speculation=recipe.speculation)
                entry["estimate"], fit = fitted.to_dict(), display.fit(fitted)
            except CapacityError as error:
                entry["error"] = str(error)
            entries.append(entry)
            described.append((entry, recipe, fit))
    if args.json:
        print(json.dumps(entries, indent=2))
        return 0
    display.show(*(_model(*model) for model in described), display.styled(
        "MODEL also accepts org/name[@revision] on Hugging Face or a checkpoint directory of a listed architecture."))
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
    display.show(
        display.fields(locations.items()),
        display.table(("setting", "value", "source"), [
            (key, str(shown(key, setting)), display.styled(setting.source)) for key, setting in settings.items()]))
    return 0


def skill(args) -> int:
    print(resources.files("flare.cli").joinpath("skill.md").read_text(), end="")
    return 0
