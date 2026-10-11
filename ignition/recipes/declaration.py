"""Recipe declarations: qualified checkpoints, serving ranges, and memory costs.

Declarations import no model code, so planning and model listing run without
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ignition.components.capacity import request_capacity

# State rows one admitted request can hold while it runs: the terminal, system,
# and prompt checkpoints that the scheduler reserves at admission.
ADMISSION_SNAPSHOTS = 3


@dataclass(frozen=True)
class Checkpoint:
    """One qualified checkpoint revision and where its stored tensors reside."""

    alias: str
    repository: str
    revision: str
    architecture: str
    stored_bytes: int
    device_bytes: int
    mapped_host_bytes: int = 0
    # The share of `device_bytes` that holds the draft model, loaded only for learned drafts.
    draft_bytes: int = 0


@dataclass(frozen=True)
class Speculation:
    """Multi-token speculation: how many draft tokens a round verifies and where drafts come from.

    A round verifies the pending input and up to `draft_tokens` drafts. With
    `adaptive` verification, graphs are captured for every count from zero
    through `draft_tokens` and rounds adapt among them from measured acceptance;
    without it every round verifies exactly `draft_tokens`. Preparation captures
    at most `graph_budget` speculative graphs across batch capacities, small
    batches and deep drafts first. Drafts come from the checkpoint's MTP module
    or, with the `copy` source, from continuations of the request's own history;
    `history_drafts` adds those continuations to MTP drafts.
    """

    draft_tokens: int
    adaptive: bool = True
    graph_budget: int = 8
    proposal_source: str = "mtp"
    history_drafts: bool = False

    @property
    def learned(self) -> bool:
        return self.proposal_source == "mtp"

    @property
    def widths(self) -> tuple[int, ...]:
        """Verification widths: each draft count from zero through `draft_tokens`, plus the pending input."""
        return tuple(range(1, self.draft_tokens + 2))

    @property
    def fixed_width(self) -> int | None:
        return None if self.adaptive else self.draft_tokens + 1


@dataclass(frozen=True)
class RankCosts:
    """Device bytes one tensor-parallel rank spends per unit of capacity."""

    state_row_bytes: int
    page_bytes: int
    graph_bytes: int
    # The share of `page_bytes` that stores the draft model's pages, present only with learned drafts.
    draft_page_bytes: int = 0


@dataclass(frozen=True)
class RankUsage:
    """What one rank allocated and left free after preparation."""

    pages: int
    free_bytes: int


@dataclass(frozen=True)
class Recipe:
    """One model family's qualified checkpoints, serving range, costs, and bootstrap.

    `prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress)` loads,
    tunes, and captures the plan's serving catalog and returns the prepared model.
    `usage(prepared)` reports each rank's device memory in rank order.

    `ranks` holds each tensor-parallel size's costs as the recipe serves by
    default. A recipe that declares `speculation` drafts with those settings
    unless a plan overrides them, and `plain_ranks` holds its costs with
    speculation disabled. `speculative_rows` is the state rows each request slot
    keeps for verification while speculating. `history_copy` states that drafts
    can be copied from request history.
    """

    name: str
    checkpoints: tuple[Checkpoint, ...]
    context: int
    page_tokens: int
    concurrency: int
    max_concurrency: int
    speculative_rows: int
    prefill_chunk: int
    ranks: Mapping[int, RankCosts]
    prepare: Callable
    usage: Callable
    mapped_host_name: str = ""
    max_images: int = 0
    image_bytes: int = 0
    speculation: Speculation | None = None
    plain_ranks: Mapping[int, RankCosts] = field(default_factory=lambda: MappingProxyType({}))
    history_copy: bool = False

    def __post_init__(self):
        if self.speculation is not None and set(self.plain_ranks) != set(self.ranks):
            raise ValueError("a recipe that speculates declares its costs without speculation for every rank count")

    @property
    def tensor_parallel(self) -> tuple[int, ...]:
        return tuple(sorted(self.ranks))

    def costs(self, world_size: int, speculation: Speculation | None) -> RankCosts:
        """Return one rank's costs under the given speculation settings."""
        if speculation is None:
            return (self.plain_ranks or self.ranks)[world_size]
        costs = self.ranks[world_size]
        return costs if speculation.learned else replace(
            costs, page_bytes=costs.page_bytes - costs.draft_page_bytes, draft_page_bytes=0)

    def weights_bytes(self, checkpoint: Checkpoint, world_size: int, speculation: Speculation | None) -> int:
        """Estimate one rank's device weights from the checkpoint's stored tensors."""
        learned = speculation is not None and speculation.learned
        return -(-(checkpoint.device_bytes - (0 if learned else checkpoint.draft_bytes)) // world_size)

    def live_rows(self, concurrency: int, speculation: Speculation | None) -> int:
        """State rows held for running requests: null, live, speculative, and admission rows."""
        slots = request_capacity(concurrency)
        speculative = self.speculative_rows if speculation is not None else 0
        return 1 + slots * (1 + speculative) + ADMISSION_SNAPSHOTS * concurrency

    def snapshot_rows(self, cache_states: int, concurrency: int) -> int:
        """Snapshot rows that keep `cache_states` cached states resident at full concurrency."""
        return cache_states + ADMISSION_SNAPSHOTS * concurrency
