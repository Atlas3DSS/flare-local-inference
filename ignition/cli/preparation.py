"""Model resolution, memory planning, and downloads for preparation commands."""

import os
from dataclasses import dataclass, replace

from ignition import planning
from ignition.errors import FetchError, UsageError
from ignition.recipes import ResolvedModel, resolve
from ignition.recipes.declaration import Speculation

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
           concurrency: Setting, speculation: Speculation | None, prefill_chunk: int | None = None,
           context_parallel: bool = True) -> planning.Plan:
    """Select GPUs and estimate memory for one checkpoint from the configuration and command-line intent."""
    return planning.plan(
        model.recipe, model.checkpoint, host.gpus, selected=None if gpus is None else planning.parse_gpus(gpus),
        concurrency=concurrency.value, reserve_bytes=configuration.reserve_bytes, speculation=speculation,
        prefill_chunk=prefill_chunk, context_parallel=context_parallel)


def context_parallel(args, model: ResolvedModel, *, spans_hosts: bool = False) -> bool:
    """Whether a plan may shard attention context across its ranks, which ranks on separate hosts never do.

    Planning shards only recipes that declare context-parallel page costs, and
    --no-context-parallel applies only to those.
    """
    if args.no_context_parallel and not any(costs.page_tokens for costs in model.recipe.ranks.values()):
        raise UsageError(f"--no-context-parallel: {model.checkpoint.alias} does not shard attention context")
    return not args.no_context_parallel and not spans_hosts


def with_compiler_devices(host: Host, plan: planning.Plan, gpus: str | None, *, enabled=True):
    """Resolve compiler GPUs before preparation narrows the CUDA device mask."""
    if gpus is None:
        return plan
    if not enabled:
        raise UsageError("--compile-gpus requires forward compilation; omit --no-compile")
    by_index = {gpu.index: gpu for gpu in host.gpus}
    indices = planning.parse_gpus(gpus)
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise UsageError(f"--compile-gpus names unavailable GPUs: {planning.format_gpus(missing)}")
    selected = {gpu.uuid for gpu in plan.gpus}
    devices = []
    for index in indices:
        gpu = by_index[index]
        if (gpu.name, gpu.capability) != (plan.gpus[0].name, plan.gpus[0].capability):
            raise UsageError(f"--compile-gpus GPU {index} must match {plan.gpus[0].name} "
                             f"{plan.gpus[0].capability}; got {gpu.name} {gpu.capability}")
        if gpu.uuid not in selected and gpu.uuid not in devices:
            devices.append(gpu.uuid)
    return replace(plan, compile_devices=tuple(devices))


def speculation(args, model: ResolvedModel) -> Speculation | None:
    """Apply the command line's speculation overrides to the recipe's settings; None disables speculation."""
    recipe, alias = model.recipe, model.checkpoint.alias
    default, limit = recipe.speculation, recipe.max_draft_tokens
    given = [flag for flag, value in (
        ("--mtp-draft-tokens", args.mtp_draft_tokens),
        ("--no-mtp-adaptive-verification", args.no_mtp_adaptive_verification or None),
        ("--mtp-proposal-source", args.mtp_proposal_source),
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
    return Speculation(drafts, adaptive, source, history)


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
