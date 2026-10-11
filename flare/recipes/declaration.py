"""Recipe declarations: qualified checkpoints, serving ranges, and memory costs.

Declarations import no model code, so planning and model listing run without
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from flare.components.capacity import request_capacity

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


@dataclass(frozen=True)
class RankCosts:
    """Device bytes one tensor-parallel rank spends per unit of capacity."""

    state_row_bytes: int
    page_bytes: int
    graph_bytes: int
    # Global positions per page when ranks share each page's context; the recipe's page size otherwise.
    page_tokens: int | None = None


@dataclass(frozen=True)
class RankUsage:
    """Device memory one rank holds after preparation."""

    weights_bytes: int
    state_bytes: int
    page_bytes: int
    pages: int
    free_bytes: int


@dataclass(frozen=True)
class Recipe:
    """One model family's qualified checkpoints, serving range, costs, and bootstrap.

    `prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress)` loads,
    tunes, and captures the plan's serving catalog and returns the prepared model.
    `usage(prepared)` reports each rank's device memory in rank order.
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

    @property
    def tensor_parallel(self) -> tuple[int, ...]:
        return tuple(sorted(self.ranks))

    def weights_bytes(self, checkpoint: Checkpoint, world_size: int) -> int:
        """Estimate one rank's device weights from the checkpoint's stored tensors."""
        return -(-checkpoint.device_bytes // world_size)

    def live_rows(self, concurrency: int) -> int:
        """State rows held for running requests: null, live, speculative, and admission rows."""
        slots = request_capacity(concurrency)
        return 1 + slots * (1 + self.speculative_rows) + ADMISSION_SNAPSHOTS * concurrency

    def snapshot_rows(self, cache_states: int, concurrency: int) -> int:
        """Snapshot rows that keep `cache_states` cached states resident at full concurrency."""
        return cache_states + ADMISSION_SNAPSHOTS * concurrency
