"""Recipe declarations: qualified checkpoints, serving ranges, and memory costs.

Declarations import no model code, so planning and model listing run without
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
import math
from types import MappingProxyType

from ignition.components.capacity import request_capacity

# State rows one admitted request can hold while it runs: the terminal, system,
# and prompt checkpoints that the scheduler reserves at admission.
ADMISSION_SNAPSHOTS = 3


@dataclass(frozen=True)
class SamplingDefaults:
    """Complete serving parameters manually verified against the model card."""

    temperature: float
    top_p: float
    top_k: int
    seed: int = 0
    min_p: float = 0.
    repetition_penalty: float = 1.
    presence_penalty: float = 0.
    frequency_penalty: float = 0.

    def __post_init__(self):
        values = (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                  self.presence_penalty, self.frequency_penalty)
        if (any(not math.isfinite(v) for v in values) or self.temperature < 0
                or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1
                or self.repetition_penalty <= 0 or type(self.top_k) is not int or self.top_k < -1
                or type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("invalid recipe sampling defaults")

    def as_dict(self):
        return asdict(self)


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
    # Device bytes that loading adds beyond the stored tensors, such as block scales
    # expanded from a compressed encoding.
    expansion_bytes: int = 0


@dataclass(frozen=True)
class Speculation:
    """Multi-token speculation: how many draft tokens a round verifies and where drafts come from.

    A round verifies the pending input and up to `draft_tokens` drafts.
    Preparation captures every width from one through `draft_tokens + 1` at
    every prepared batch capacity. `adaptive` sets the serving default: choose
    widths from measured acceptance when true, or use the fixed maximum when
    false. Flare can reduce both capacities when loading the package. Drafts
    come from the checkpoint's MTP module or, with the `copy` source, from
    continuations of the request's own history; `history_drafts` adds those
    continuations to MTP drafts.
    """

    draft_tokens: int
    adaptive: bool = True
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
    # Global positions per page when ranks share each page's context; the recipe's page size otherwise.
    page_tokens: int | None = None


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
    speculation disabled. Each active request has one execution state row per
    selected verification position unless `verification_rows` gives another count
    of extra rows. `history_copy` states that drafts can be copied from request history.
    """

    name: str
    checkpoints: tuple[Checkpoint, ...]
    context: int
    page_tokens: int
    concurrency: int
    max_concurrency: int
    max_draft_tokens: int
    prefill_chunk: int
    ranks: Mapping[int, RankCosts]
    prepare: Callable
    usage: Callable
    sampling_defaults: SamplingDefaults
    mapped_host_name: str = ""
    max_images: int = 0
    image_bytes: int = 0
    speculation: Speculation | None = None
    plain_ranks: Mapping[int, RankCosts] = field(default_factory=lambda: MappingProxyType({}))
    history_copy: bool = False
    verification_rows: int | None = None

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
        """Estimate one rank's device weights from the checkpoint's stored tensors and their expansion."""
        learned = speculation is not None and speculation.learned
        return -(-(checkpoint.device_bytes + checkpoint.expansion_bytes
                   - (0 if learned else checkpoint.draft_bytes)) // world_size)

    def live_rows(self, concurrency: int, speculation: Speculation | None) -> int:
        """State rows held for running requests: null, live, speculative, and admission rows."""
        slots = request_capacity(concurrency)
        speculative = (0 if speculation is None else speculation.draft_tokens if self.verification_rows is None
                       else self.verification_rows)
        return 1 + slots * (1 + speculative) + ADMISSION_SNAPSHOTS * concurrency

    def snapshot_rows(self, cache_states: int, concurrency: int) -> int:
        """Snapshot rows that keep `cache_states` cached states resident at full concurrency."""
        return cache_states + ADMISSION_SNAPSHOTS * concurrency
