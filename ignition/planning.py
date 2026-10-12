"""Derive a serving plan from a recipe's cost model and the host's free GPU memory.

Weights, graphs and scratch, live state, and a prefix cache sized from the
concurrency limit come first. KV pages take all remaining supply, the GPU's free
memory minus engine headroom and the configured reserve: the pool holds active
requests and every cached prefix. Active requests need at least 8,192 tokens
each; when supply cannot provide that, the prefix cache shrinks to its floor. A
plan that still does not fit is a capacity error naming the settings that would
make it fit.
"""

from dataclasses import dataclass, replace

from ignition.errors import CapacityError, UsageError
from ignition.recipes.declaration import Checkpoint, Recipe, Speculation

# Preparation headroom covers compiled warm-up and graph capture, measured at
# 0.83 GiB for one Qwen rank. Helios sizes its caches after graph upload with a
# separate driver margin. Graphs, scratch, and the CUDA context are planned separately.
HEADROOM_BYTES = 1 << 30
MINIMUM_REQUEST_TOKENS = 8192
# The prefix cache keeps about four recent exchanges per concurrent request
# resident; each exchange retains its prompt and terminal checkpoints.
CACHE_STATES_PER_REQUEST = 8
# The cache takes at most a quarter of the memory left after weights, graphs, and
# live state, so KV pages for active requests keep the larger share.
CACHE_SHARE = 4
CACHE_FLOOR = 8
SUPPORTED_CAPABILITIES = ((12, 0), (12, 1))
_GIB = 1 << 30


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    total_bytes: int
    free_bytes: int
    capability: tuple[int, int]

    @property
    def supported(self) -> bool:
        return self.capability in SUPPORTED_CAPABILITIES


def discover_gpus() -> tuple[Gpu, ...]:
    """Enumerate GPUs through NVML without creating a CUDA context."""
    try:
        from cuda.bindings import nvml

        nvml.init_v2()
    except Exception as error:  # noqa: BLE001 - Import and driver failures share one report.
        raise CapacityError(f"cannot enumerate GPUs through NVML: {error}") from error
    try:
        gpus = []
        for index in range(nvml.device_get_count_v2()):
            handle = nvml.device_get_handle_by_index_v2(index)
            memory = nvml.device_get_memory_info_v2(handle)
            gpus.append(Gpu(index, nvml.device_get_name(handle), nvml.device_get_uuid(handle), memory.total,
                            memory.free, tuple(nvml.device_get_cuda_compute_capability(handle))))
        return tuple(gpus)
    finally:
        nvml.shutdown()


def restrict_gpus(gpus: tuple[Gpu, ...], visible: str) -> tuple[Gpu, ...]:
    """Renumber GPUs as a caller's CUDA_VISIBLE_DEVICES value enumerates them.

    Integer entries are NVML indices and other entries are GPU UUID prefixes.
    """
    restricted = []
    for entry in filter(None, (value.strip() for value in visible.split(","))):
        if entry.isdecimal():
            matches = [gpu for gpu in gpus if gpu.index == int(entry)]
        else:
            matches = [gpu for gpu in gpus if gpu.uuid.startswith(entry)]
        if len(matches) != 1 or matches[0] in restricted:
            raise UsageError(f"CUDA_VISIBLE_DEVICES entry {entry!r} does not name one distinct GPU")
        restricted.append(matches[0])
    return tuple(replace(gpu, index=index) for index, gpu in enumerate(restricted))


def parse_gpus(text: str) -> tuple[int, ...]:
    """Parse `0`, `0,1`, or `0-3` into distinct GPU indices."""
    indices = []
    for part in text.split(","):
        first, dash, last = part.strip().partition("-")
        if not first.isdecimal() or (dash and not last.isdecimal()):
            raise UsageError(f"--gpus expects indices such as 0, 0,1, or 0-3; got {text!r}")
        indices.extend(range(int(first), int(last if dash else first) + 1))
    if not indices or len(set(indices)) != len(indices):
        raise UsageError(f"--gpus expects distinct indices in increasing ranges; got {text!r}")
    return tuple(indices)


def format_gpus(indices) -> str:
    indices = tuple(indices)
    if len(indices) > 1 and indices == tuple(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}-{indices[-1]}"
    return ",".join(map(str, indices))


@dataclass(frozen=True)
class Plan:
    """Per-rank memory demand and supply for one checkpoint on selected GPUs."""

    recipe: Recipe
    checkpoint: Checkpoint
    gpus: tuple[Gpu, ...]
    concurrency: int
    prefill_chunk: int
    speculation: Speculation | None
    cache_target: int
    cache_states: int
    pages: int
    reserve_bytes: int
    weights_bytes: int
    graph_bytes: int
    # Ranks shard each page's context when the recipe declares context-parallel page costs.
    context_parallel: bool = True

    @property
    def world_size(self) -> int:
        return len(self.gpus)

    @property
    def costs(self):
        return self.recipe.costs(self.world_size, self.speculation)

    @property
    def live_rows(self) -> int:
        return self.recipe.live_rows(self.concurrency, self.speculation)

    @property
    def live_bytes(self) -> int:
        return self.live_rows * self.costs.state_row_bytes

    @property
    def cache_bytes(self) -> int:
        return self.cache_states * self.costs.state_row_bytes

    @property
    def fixed_bytes(self) -> int:
        return self.weights_bytes + self.graph_bytes + self.live_bytes

    @property
    def demand_pages(self) -> int:
        return self.concurrency * (self.recipe.context // self.page_tokens)

    @property
    def page_tokens(self) -> int:
        return (self.costs.page_tokens if self.context_parallel else None) or self.recipe.page_tokens

    @property
    def minimum_pages(self) -> int:
        return self.concurrency * (MINIMUM_REQUEST_TOKENS // self.page_tokens)

    @property
    def free_bytes(self) -> int:
        return min(gpu.free_bytes for gpu in self.gpus)

    @property
    def supply_bytes(self) -> int:
        return self.free_bytes - HEADROOM_BYTES - self.reserve_bytes

    @property
    def page_bytes(self) -> int:
        return self.pages * self.costs.page_bytes

    @property
    def tokens(self) -> int:
        return self.pages * self.page_tokens

    @property
    def context(self) -> int:
        """The longest sequence one request may reach."""
        return min(self.recipe.context, self.tokens)

    @property
    def guaranteed_tokens(self) -> int:
        """Tokens every request can hold while all admitted requests are active."""
        return min(self.recipe.context, self.pages // self.concurrency * self.page_tokens)

    @property
    def snapshots(self) -> int:
        return self.recipe.snapshot_rows(self.cache_states, self.concurrency)

    @property
    def fits(self) -> bool:
        return (self.pages >= self.minimum_pages
                and self.fixed_bytes + self.cache_bytes + self.page_bytes <= self.supply_bytes)

    def to_dict(self) -> dict:
        return {
            "alias": self.checkpoint.alias, "repository": self.checkpoint.repository,
            "revision": self.checkpoint.revision, "architecture": self.checkpoint.architecture,
            "gpus": [{"index": gpu.index, "name": gpu.name, "uuid": gpu.uuid, "free_bytes": gpu.free_bytes,
                      "total_bytes": gpu.total_bytes} for gpu in self.gpus],
            "tensor_parallel": self.world_size, "max_concurrency": self.concurrency,
            "prefill_chunk": self.prefill_chunk, "context_parallel": self.context_parallel,
            "speculation": None if self.speculation is None else {
                "draft_tokens": self.speculation.draft_tokens, "adaptive": self.speculation.adaptive,
                "proposal_source": self.speculation.proposal_source,
                "history_drafts": self.speculation.history_drafts},
            "context": self.context, "checkpoint_context": self.recipe.context,
            "weights_bytes": self.weights_bytes, "mapped_host_bytes": self.checkpoint.mapped_host_bytes,
            "graph_bytes": self.graph_bytes,
            "live_bytes": self.live_bytes, "prefix_cache_states": self.cache_states,
            "prefix_cache_target": self.cache_target, "prefix_cache_bytes": self.cache_bytes,
            "snapshots": self.snapshots,
            "pages": self.pages, "demand_pages": self.demand_pages, "page_tokens": self.page_tokens,
            "page_bytes": self.page_bytes, "tokens": self.tokens, "guaranteed_tokens": self.guaranteed_tokens,
            "free_bytes": self.free_bytes, "headroom_bytes": HEADROOM_BYTES, "reserve_bytes": self.reserve_bytes,
            "supply_bytes": self.supply_bytes,
        }


def fit(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *, concurrency: int, reserve_bytes: int,
        speculation: Speculation | None, prefill_chunk: int | None = None, context_parallel: bool = True) -> Plan:
    """Size pages and the prefix cache to the supply; the result may not fit.

    Weights come from the checkpoint's stored tensors and graphs and scratch from
    the recipe's estimate.
    """
    costs = recipe.costs(len(gpus), speculation)
    plan = Plan(recipe, checkpoint, gpus, concurrency, recipe.prefill_chunk if prefill_chunk is None else prefill_chunk,
                speculation, 0, 0, 0, reserve_bytes,
                recipe.weights_bytes(checkpoint, len(gpus), speculation), costs.graph_bytes,
                context_parallel and costs.page_tokens is not None)
    available = plan.supply_bytes - plan.fixed_bytes
    target = min(CACHE_STATES_PER_REQUEST * concurrency,
                 max(CACHE_FLOOR, available // CACHE_SHARE // costs.state_row_bytes))

    def pages(kept):
        return (available - kept * costs.state_row_bytes) // costs.page_bytes

    kept = target
    if pages(kept) < plan.minimum_pages:
        affordable = (available - plan.minimum_pages * costs.page_bytes) // costs.state_row_bytes
        kept = min(target, max(CACHE_FLOOR, affordable))
    return replace(plan, cache_target=target, cache_states=kept, pages=max(pages(kept), plan.minimum_pages))


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def _per_gpu(plan: Plan) -> str:
    return " per GPU" if plan.world_size > 1 else ""


def _shortfall(plan: Plan, options: list[str], scope: str) -> CapacityError:
    need = plan.fixed_bytes + plan.cache_bytes + plan.page_bytes
    lines = [
        f"needs {_gib(plan.weights_bytes)} GiB weights + {_gib(plan.graph_bytes)} GiB graphs"
        f" + {_gib(plan.live_bytes)} GiB live state + {_gib(plan.cache_bytes)} GiB prefix cache",
        f"+ {_gib(plan.page_bytes)} GiB minimum pages ({plan.concurrency} × {MINIMUM_REQUEST_TOKENS:,} tokens)"
        f" = {_gib(need)} GiB{_per_gpu(plan)};",
        f"has {_gib(plan.free_bytes)} GiB free − {_gib(HEADROOM_BYTES)} GiB headroom"
        f" − {_gib(plan.reserve_bytes)} GiB reserve = {_gib(plan.supply_bytes)} GiB.",
        f"Options: {', or '.join(options)}." if options
        else f"No setting makes {plan.checkpoint.alias} fit on {scope}.",
    ]
    return CapacityError("\n".join(lines))


def plan(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *, selected: tuple[int, ...] | None,
         concurrency: int, reserve_bytes: int, speculation: Speculation | None,
         prefill_chunk: int | None = None, context_parallel: bool = True) -> Plan:
    """Select GPUs and size the plan, or raise a capacity error that lists working settings.

    Without `selected`, the plan uses the smallest supported tensor-parallel size
    for which enough GPUs fit, taking the first such GPUs in index order. Without
    `prefill_chunk`, the recipe's chunk applies. `speculation` is the drafting the
    plan prepares, or None for none. `context_parallel` shards each page's context
    across the ranks of recipes that declare context-parallel page costs.
    """
    if not 1 <= concurrency <= recipe.max_concurrency:
        raise UsageError(f"--max-concurrency for {checkpoint.alias} must be 1 through {recipe.max_concurrency}")
    chunk = recipe.prefill_chunk if prefill_chunk is None else prefill_chunk
    if chunk % 64 or not 64 <= chunk <= recipe.context:
        raise UsageError(f"--prefill-chunk-size for {checkpoint.alias} must be a multiple of 64"
                         f" from 64 through {recipe.context:,}")
    if not context_parallel and not any(costs.page_tokens for costs in recipe.ranks.values()):
        raise UsageError(f"--no-context-parallel: {checkpoint.alias} does not shard attention context")
    sizes = " or ".join(map(str, recipe.tensor_parallel))
    supported = [gpu for gpu in gpus if gpu.supported]
    chosen = None
    if selected is not None:
        if any(index >= len(gpus) for index in selected):
            raise UsageError(f"--gpus {format_gpus(selected)} names a GPU this host does not have;"
                             f" it has {len(gpus)}")
        chosen = tuple(gpus[index] for index in selected)
        if len(chosen) not in recipe.tensor_parallel:
            size = min(recipe.tensor_parallel, key=lambda size: (size < len(chosen), size))
            first = max(0, min(selected[0], len(gpus) - size))
            example = format_gpus(range(first, first + size))
            raise CapacityError(f"{checkpoint.alias} needs {sizes} GPUs; {len(chosen)} selected; use --gpus {example}")
        for gpu in chosen:
            if not gpu.supported:
                raise CapacityError(f"GPU {gpu.index} ({gpu.name}) is not an SM120 or SM121 device")
    elif len(supported) < recipe.tensor_parallel[0]:
        raise CapacityError(f"{checkpoint.alias} needs {sizes} SM120 or SM121 GPUs; this host has {len(supported)}")

    def fitted(group, *, concurrency=concurrency, reserve_bytes=reserve_bytes, speculation=speculation):
        return fit(recipe, checkpoint, group, concurrency=concurrency, reserve_bytes=reserve_bytes,
                   speculation=speculation, prefill_chunk=chunk, context_parallel=context_parallel)

    def automatic(world_size, **settings):
        """Plan the first GPUs that each fit, else the GPUs with the most free memory."""
        fitting = [gpu for gpu in supported if fitted((gpu,) * world_size, **settings).fits]
        if len(fitting) < world_size:
            fitting = sorted(sorted(supported, key=lambda gpu: -gpu.free_bytes)[:world_size],
                             key=lambda gpu: gpu.index)
        return fitted(tuple(fitting[:world_size]), **settings)

    def candidates(**settings):
        if chosen is not None:
            return [fitted(chosen, **settings)]
        return [automatic(size, **settings) for size in recipe.tensor_parallel if size <= len(supported)]

    def attempt(**settings):
        return next((candidate for candidate in candidates(**settings) if candidate.fits), None)

    result = attempt()
    if result is not None:
        if chunk > result.context:
            raise UsageError(f"--prefill-chunk-size {chunk:,} exceeds the {result.context:,}-token context"
                             " this plan allows")
        return result

    closest = min(candidates(), key=lambda candidate: (
        candidate.fixed_bytes + candidate.cache_bytes + candidate.page_bytes - candidate.supply_bytes))
    options = []
    if reserve_bytes > 0:
        for reserve in range((reserve_bytes - 1) // _GIB, -1, -1):
            if attempt(reserve_bytes=reserve * _GIB) is not None:
                options.append(f'[memory] reserve = "{reserve}G"' if reserve else '[memory] reserve = "0"')
                break
    for limit in range(concurrency - 1, 0, -1):
        if attempt(concurrency=limit) is not None:
            options.append(f"--max-concurrency {limit}")
            break
    if speculation is not None and attempt(speculation=None) is not None:
        options.append("--no-mtp")
    if chosen is not None:
        for size in recipe.tensor_parallel:
            if len(chosen) < size <= len(supported) and (larger := automatic(size)).fits:
                options.append(f"--gpus {format_gpus(gpu.index for gpu in larger.gpus)}")
                break
    scope = "this host" if chosen is None else f"GPU {format_gpus(gpu.index for gpu in chosen)}"
    raise _shortfall(closest, options, scope)
