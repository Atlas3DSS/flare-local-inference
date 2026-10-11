"""Derive a serving plan from a recipe's cost model and the host's free GPU memory.

Weights, graphs and scratch, live state, and a prefix cache sized from the
concurrency limit come first. KV pages take all remaining supply, the GPU's free
memory minus engine headroom and the configured reserve: the pool holds active
requests and every cached prefix. Active requests need at least 8,192 tokens
each; when supply cannot provide that, the prefix cache shrinks to its floor. A
plan that still does not fit is a capacity error naming the settings that would
make it fit.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from flare.errors import CapacityError, UsageError
from flare.recipes.declaration import Checkpoint, Recipe

# Free memory kept after the final KV allocation. Serving allocates nothing, so
# headroom covers the transient working memory of compiled warm-up and graph
# capture, measured at 0.83 GiB for one Qwen rank. Graphs and scratch, including
# the CUDA context, are planned separately.
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


def measurement_path(kernel_cache: Path, checkpoint: Checkpoint, world_size: int, concurrency: int,
                     prefill_chunk: int) -> Path:
    """Locate the weights and graph bytes measured by the last preparation of one plan shape."""
    return (Path(kernel_cache) / "measurements" / (f"{checkpoint.alias}-{checkpoint.revision[:8]}"
                                                   f"-tp{world_size}-c{concurrency}-p{prefill_chunk}.json"))


@dataclass(frozen=True)
class Measurement:
    """Per-rank device bytes a preparation held beyond state rows and KV pages."""

    weights_bytes: int
    graph_bytes: int


def read_measurement(path: Path, gpu_name: str) -> Measurement | None:
    try:
        record = json.loads(Path(path).read_text())
        measurement = Measurement(record["weights_bytes"], record["graph_bytes"])
        if record["gpu"] == gpu_name and all(type(value) is int and value >= 0 for value in (
                measurement.weights_bytes, measurement.graph_bytes)):
            return measurement
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def write_measurement(path: Path, gpu_name: str, measurement: Measurement) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpu": gpu_name, "weights_bytes": measurement.weights_bytes,
                                "graph_bytes": measurement.graph_bytes}) + "\n")


@dataclass(frozen=True)
class Plan:
    """Per-rank memory demand and supply for one checkpoint on selected GPUs."""

    recipe: Recipe
    checkpoint: Checkpoint
    gpus: tuple[Gpu, ...]
    concurrency: int
    prefill_chunk: int
    cache_target: int
    cache_states: int
    pages: int
    reserve_bytes: int
    weights_bytes: int
    graph_bytes: int
    measured: bool

    @property
    def world_size(self) -> int:
        return len(self.gpus)

    @property
    def costs(self):
        return self.recipe.ranks[self.world_size]

    @property
    def live_bytes(self) -> int:
        return self.recipe.live_rows(self.concurrency) * self.costs.state_row_bytes

    @property
    def cache_bytes(self) -> int:
        return self.cache_states * self.costs.state_row_bytes

    @property
    def fixed_bytes(self) -> int:
        return self.weights_bytes + self.graph_bytes + self.live_bytes

    @property
    def demand_pages(self) -> int:
        return self.concurrency * (self.recipe.context // self.recipe.page_tokens)

    @property
    def minimum_pages(self) -> int:
        return self.concurrency * (MINIMUM_REQUEST_TOKENS // self.recipe.page_tokens)

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
        return self.pages * self.recipe.page_tokens

    @property
    def context(self) -> int:
        """The longest sequence one request may reach."""
        return min(self.recipe.context, self.tokens)

    @property
    def guaranteed_tokens(self) -> int:
        """Tokens every request can hold while all admitted requests are active."""
        return min(self.recipe.context, self.pages // self.concurrency * self.recipe.page_tokens)

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
            "prefill_chunk": self.prefill_chunk,
            "context": self.context, "checkpoint_context": self.recipe.context,
            "weights_bytes": self.weights_bytes, "mapped_host_bytes": self.checkpoint.mapped_host_bytes,
            "graph_bytes": self.graph_bytes, "measured": self.measured,
            "live_bytes": self.live_bytes, "prefix_cache_states": self.cache_states,
            "prefix_cache_target": self.cache_target, "prefix_cache_bytes": self.cache_bytes,
            "snapshots": self.snapshots,
            "pages": self.pages, "demand_pages": self.demand_pages, "page_tokens": self.recipe.page_tokens,
            "page_bytes": self.page_bytes, "tokens": self.tokens, "guaranteed_tokens": self.guaranteed_tokens,
            "free_bytes": self.free_bytes, "headroom_bytes": HEADROOM_BYTES, "reserve_bytes": self.reserve_bytes,
            "supply_bytes": self.supply_bytes,
        }


def fit(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *, concurrency: int, reserve_bytes: int,
        prefill_chunk: int | None = None, measurement: Measurement | None = None) -> Plan:
    """Size pages and the prefix cache to the supply; the result may not fit.

    Without a measurement, weights come from the checkpoint's stored tensors and
    graphs and scratch from the recipe's estimate.
    """
    costs = recipe.ranks[len(gpus)]
    estimate = Measurement(recipe.weights_bytes(checkpoint, len(gpus)), costs.graph_bytes)
    plan = Plan(recipe, checkpoint, gpus, concurrency, recipe.prefill_chunk if prefill_chunk is None else prefill_chunk,
                0, 0, 0, reserve_bytes,
                *((estimate.weights_bytes, estimate.graph_bytes, False) if measurement is None
                  else (measurement.weights_bytes, measurement.graph_bytes, True)))
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
         concurrency: int, reserve_bytes: int, prefill_chunk: int | None = None, measured=None) -> Plan:
    """Select GPUs and size the plan, or raise a capacity error that lists working settings.

    Without `selected`, the plan uses the smallest supported tensor-parallel size
    for which enough GPUs fit, taking the first such GPUs in index order. Without
    `prefill_chunk`, the recipe's chunk applies. `measured(world_size,
    concurrency, prefill_chunk, gpu_name)` returns the `Measurement` recorded by
    an earlier preparation on this host, or None.
    """
    if not 1 <= concurrency <= recipe.max_concurrency:
        raise UsageError(f"--max-concurrency for {checkpoint.alias} must be 1 through {recipe.max_concurrency}")
    chunk = recipe.prefill_chunk if prefill_chunk is None else prefill_chunk
    if chunk % 64 or not 64 <= chunk <= recipe.context:
        raise UsageError(f"--prefill-chunk-size for {checkpoint.alias} must be a multiple of 64"
                         f" from 64 through {recipe.context:,}")
    sizes = " or ".join(map(str, recipe.tensor_parallel))
    supported = [gpu for gpu in gpus if gpu.supported]
    chosen = None
    if selected is not None:
        if any(index >= len(gpus) for index in selected):
            raise UsageError(f"--gpus {format_gpus(selected)} names a GPU this host does not have;"
                             f" it has {len(gpus)}")
        chosen = tuple(gpus[index] for index in selected)
        if len(chosen) not in recipe.ranks:
            size = min(recipe.tensor_parallel, key=lambda size: (size < len(chosen), size))
            first = max(0, min(selected[0], len(gpus) - size))
            example = format_gpus(range(first, first + size))
            raise CapacityError(f"{checkpoint.alias} needs {sizes} GPUs; {len(chosen)} selected; use --gpus {example}")
        for gpu in chosen:
            if not gpu.supported:
                raise CapacityError(f"GPU {gpu.index} ({gpu.name}) is not an SM120 or SM121 device")
    elif len(supported) < recipe.tensor_parallel[0]:
        raise CapacityError(f"{checkpoint.alias} needs {sizes} SM120 or SM121 GPUs; this host has {len(supported)}")

    def fitted(group, *, concurrency=concurrency, reserve_bytes=reserve_bytes):
        measurement = None if measured is None else measured(len(group), concurrency, chunk, group[0].name)
        return fit(recipe, checkpoint, group, concurrency=concurrency, reserve_bytes=reserve_bytes,
                   prefill_chunk=chunk, measurement=measurement)

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
    if chosen is not None:
        for size in recipe.tensor_parallel:
            if len(chosen) < size <= len(supported) and (larger := automatic(size)).fits:
                options.append(f"--gpus {format_gpus(gpu.index for gpu in larger.gpus)}")
                break
    scope = "this host" if chosen is None else f"GPU {format_gpus(gpu.index for gpu in chosen)}"
    raise _shortfall(closest, options, scope)


def _source(plan: Plan) -> str:
    checkpoint = plan.checkpoint
    if checkpoint in plan.recipe.checkpoints:
        return checkpoint.revision[:8]
    if Path(checkpoint.repository).is_absolute():
        return checkpoint.repository
    return f"{checkpoint.repository}@{checkpoint.revision[:8]}"


def render(plan: Plan, *, restriction: str | None = None) -> str:
    """Format a fitting plan as the table printed by `flare plan`."""
    recipe, checkpoint, per_gpu = plan.recipe, plan.checkpoint, _per_gpu(plan)
    names = {gpu.name.removeprefix("NVIDIA ") for gpu in plan.gpus}
    demand_bytes = plan.demand_pages * plan.costs.page_bytes
    target_bytes = plan.cache_target * plan.costs.state_row_bytes
    need = plan.fixed_bytes + target_bytes + demand_bytes
    reserve = _gib(plan.reserve_bytes) if plan.reserve_bytes else "0"
    cache_note = f"{plan.cache_states} states for prompt reuse"
    if plan.cache_states != plan.cache_target:
        cache_note += f"; {plan.cache_target} without memory pressure"
    rows = [("weights", plan.weights_bytes, f"device{per_gpu}")]
    if checkpoint.mapped_host_bytes:
        rows.append((recipe.mapped_host_name, checkpoint.mapped_host_bytes, "mapped host memory"))
    rows += [
        ("graphs and scratch", plan.graph_bytes,
         "measured on this host" if plan.measured else "estimate; never prepared on this host"),
        ("live state", plan.live_bytes, f"{plan.concurrency} active request{'s' if plan.concurrency > 1 else ''}"),
        ("prefix cache", plan.cache_bytes, cache_note),
        ("KV pages", plan.page_bytes, f"{plan.tokens:,} tokens"),
    ]
    lines = [
        f"{'model':<15}{checkpoint.alias}   {_source(plan)}   {checkpoint.architecture}",
        f"{'gpus':<15}{format_gpus(gpu.index for gpu in plan.gpus)}  {' / '.join(sorted(names))}"
        f"  {_gib(plan.free_bytes)} GiB free   tensor-parallel {plan.world_size}",
        f"{'concurrency':<15}{plan.concurrency:<16}context {recipe.context:,}   prefill chunk {plan.prefill_chunk:,}",
    ]
    if restriction is not None:
        lines.append(f"{'':<15}GPU indices follow CUDA_VISIBLE_DEVICES={restriction}")
    lines += [
        "",
        f"{'needs':<9}{_gib(plan.weights_bytes)} weights + {_gib(plan.graph_bytes)} graphs + {_gib(plan.live_bytes)} live"
        f" + {_gib(target_bytes)} cache ({plan.cache_target} states) + {_gib(demand_bytes)} pages"
        f" ({plan.concurrency} × {recipe.context:,}) = {_gib(need)} GiB{per_gpu}",
        f"{'has':<9}{_gib(plan.free_bytes)} free − {_gib(HEADROOM_BYTES)} headroom − {reserve} reserve"
        f" = {_gib(plan.supply_bytes)} GiB{per_gpu}",
        "",
        *(f"{name:<20}{_gib(value):>5} GiB   {note}" for name, value, note in rows),
        "",
        f"guarantees {plan.concurrency} × {plan.guaranteed_tokens:,} tokens at full concurrency;"
        f" a single request up to {plan.context:,}",
    ]
    return "\n".join(lines)


def summarize(plan: Plan) -> str:
    """Describe what fits in the form `1 GPU: 4 × 177K, 2 × 262K`."""
    def thousands(tokens):
        return f"{tokens // 1000}K"

    text = (f"{plan.world_size} GPU{'s' if plan.world_size > 1 else ''}:"
            f" {plan.concurrency} × {thousands(plan.guaranteed_tokens)}")
    whole = plan.tokens // plan.recipe.context
    if plan.guaranteed_tokens < plan.recipe.context and whole >= 1:
        text += f", {whole} × {thousands(plan.recipe.context)}"
    return text
