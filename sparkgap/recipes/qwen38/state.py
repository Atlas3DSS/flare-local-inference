"""Contiguous hybrid-model snapshots and captured indirect state transfers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

import torch
import triton
import triton.language as tl

from .gdn import GDNState

GDN_LAYERS = tuple(layer for layer in range(48) if layer % 4 != 3)
LIVE_BASE = 1
CANDIDATE_BASE = 5
SNAPSHOT_BASE = 9
EOS_TOKEN_ID = 248044
COPY = 0
RESET = 1


@dataclass(frozen=True)
class StateRegion:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class HybridStateLayout:
    """Byte layout for one complete consumed-prefix continuation state."""

    regions: tuple[StateRegion, ...]
    row_bytes: int
    speculative_tokens: int = 0
    draft_ring_capacity: int = 0

    @classmethod
    def build(cls, *, speculative_tokens: int = 0) -> HybridStateLayout:
        if type(speculative_tokens) is not int or not 0 <= speculative_tokens <= 7:
            raise ValueError("hybrid state supports zero through seven speculative tokens")
        regions = []
        offset = 0

        def add(name, shape, dtype):
            nonlocal offset
            offset = (offset + 255) // 256 * 256
            region = StateRegion(name, offset, shape, dtype)
            regions.append(region)
            offset += region.nbytes

        for layer in GDN_LAYERS:
            add(f"gdn.{layer}.recurrent", (48, 128, 128), torch.float32)
            add(f"gdn.{layer}.convolution", (10240, 3 + speculative_tokens), torch.bfloat16)
        add("ple_convolution", (10240, 9 + speculative_tokens), torch.bfloat16)
        add("last_hidden", (2560,), torch.bfloat16)
        add("consumed_tokens", (1,), torch.int64)
        add("ple_history", (2,), torch.int64)
        add("rope_delta", (1,), torch.int64)
        ring = 0
        if speculative_tokens:
            from b12x.attention import qsa

            geometry = qsa.cache_requirements(main_page_size=64, position_axes=3,
                                             max_speculative_tokens=speculative_tokens)
            ring = geometry.raw_ring_capacity
            add("target_residual", (10240,), torch.bfloat16)
            add("target_residual_position", (3,), torch.int64)
            add("draft_raw_keys", geometry.raw_k_ring_shape, torch.bfloat16)
            add("draft_raw_positions", geometry.raw_logical_positions_shape, torch.int64)
            add("draft_raw_rope", geometry.raw_rope_positions_shape, torch.int64)
            add("draft_raw_anchor", (1,), torch.int64)
            add("draft_boundary_key", geometry.main_k_page_shape, torch.bfloat16)
            add("draft_boundary_value", geometry.main_v_page_shape, torch.bfloat16)
            add("draft_boundary_selector", geometry.compressed_page_shape, torch.bfloat16)
        return cls(tuple(regions), (offset + 255) // 256 * 256, speculative_tokens, ring)

    def region(self, name: str) -> StateRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def snapshot_capacity(self, budget_bytes: int = 4 << 30) -> int:
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("snapshot budget must be a nonnegative byte count")
        return budget_bytes // self.row_bytes


@dataclass(frozen=True)
class HybridStatePool:
    """Typed disjoint views into one persistent byte row per state slot.

    Slot zero is a permanently zero null row. Native request slots 0..3 use
    model slots 1..4; their aligned candidates use slots 5..8. Published snapshot
    IDs map to slots ``SNAPSHOT_BASE + id``. A 4 GiB snapshot budget holds 37 rows,
    excluding null, live, and candidate storage.

    Candidate rows must contain the actual state at their consumed-token label.
    For a chunk crossing a 64-token boundary, recurrent export and convolution,
    history, and hidden-vector gathering target that boundary before publication.
    Copying the final chunk state while rounding its label down is invalid.
    Sampling RNG, penalties, pending tokens, and QSA raw rings are separate.
    """

    backing: torch.Tensor
    layout: HybridStateLayout
    gdn: Mapping[int, GDNState]
    ple_convolution: torch.Tensor
    last_hidden: torch.Tensor
    consumed_tokens: torch.Tensor
    ple_history: torch.Tensor
    rope_delta: torch.Tensor

    @classmethod
    def allocate(cls, slots: int, device, *, speculative_tokens: int = 0) -> HybridStatePool:
        if type(slots) is not int or slots < 2:
            raise ValueError("hybrid state requires null and live slots")
        layout = HybridStateLayout.build(speculative_tokens=speculative_tokens)
        backing = torch.zeros(
            (slots, layout.row_bytes), dtype=torch.uint8, device=device
        )

        def view(name):
            region = layout.region(name)
            return (
                backing[:, region.offset : region.offset + region.nbytes]
                .view(region.dtype)
                .view(slots, *region.shape)
            )

        gdn = MappingProxyType(
            {
                layer: GDNState(
                    view(f"gdn.{layer}.convolution"), view(f"gdn.{layer}.recurrent")
                )
                for layer in GDN_LAYERS
            }
        )
        history = view("ple_history")
        history[1:].fill_(EOS_TOKEN_ID)
        return cls(
            backing,
            layout,
            gdn,
            view("ple_convolution"),
            view("last_hidden"),
            view("consumed_tokens"),
            history,
            view("rope_delta"),
        )

    def region(self, name: str) -> torch.Tensor:
        region = self.layout.region(name)
        return self.backing[:, region.offset:region.offset + region.nbytes].view(
            region.dtype).view(self.backing.shape[0], *region.shape)

    def reset_for_preparation(self):
        """Restore empty unpublished model state after serialized benchmark use."""
        self.backing.zero_()
        self.ple_history[1:].fill_(EOS_TOKEN_ID)


@dataclass(frozen=True)
class StateTransferMetadata:
    """Device commands for complete-state copy/reset and aligned QSA restore.

    Negative source/destination slots and null slot zero are inactive. Operation
    codes are COPY=0 and RESET=1. ``qsa_slots`` maps a transfer to its separate raw
    selector slot; -1 omits invalidation. A QSA restore requires a nonnegative
    64-aligned consumed count and positive accepted-token count. The interval
    anchor becomes consumed minus accepted; empty initialization uses -1.

    ``errors`` contains 0=success/inactive, 1=operation, 2=slot bounds,
    3=raw-slot bounds, 4=unaligned restore, 5=accepted count, 6=batch alias conflict,
    7=command count outside capacity.
    The runtime must inspect errors before treating a transfer as successful.
    Batch destinations must be unique and cannot be another operation's source.
    """

    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    operation: torch.Tensor
    qsa_slots: torch.Tensor
    accepted_tokens: torch.Tensor
    num_ops: torch.Tensor
    errors: torch.Tensor

    @classmethod
    def allocate(cls, capacity: int, device) -> StateTransferMetadata:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("state-transfer capacity must be positive")
        return cls(
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.ones(capacity, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
        )


@triton.jit
def _transfer(
    Backing,
    Consumed,
    Sources,
    Destinations,
    Operations,
    RawSlots,
    Accepted,
    Count,
    Errors,
    ROW_BYTES: tl.constexpr,
    SLOTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    RAW_SLOTS: tl.constexpr,
    HISTORY_OFFSET: tl.constexpr,
    CONSUMED_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    OP_BLOCK: tl.constexpr,
):
    command, tile = tl.program_id(0), tl.program_id(1)
    count = tl.load(Count)
    source = tl.load(Sources + command)
    destination = tl.load(Destinations + command)
    operation = tl.load(Operations + command)
    raw_slot = tl.load(RawSlots + command)
    accepted = tl.load(Accepted + command)
    live = (command < count) & (destination > 0) & ((operation != 0) | (source > 0))
    error = tl.full((), 0, tl.int32)
    error = tl.where(live & ((operation < 0) | (operation > 1)), 1, error)
    error = tl.where(
        live & ((destination >= SLOTS) | ((operation == 0) & (source >= SLOTS))),
        2,
        error,
    )
    error = tl.where(live & (raw_slot >= RAW_SLOTS), 3, error)
    consumed = tl.load(
        Consumed + source * CONSUMED_STRIDE,
        mask=live & (operation == 0) & (source < SLOTS),
        other=0,
    )
    restoring = live & (raw_slot >= 0)
    error = tl.where(restoring & ((consumed < 0) | (consumed % 64 != 0)), 4, error)
    error = tl.where(restoring & (accepted <= 0), 5, error)
    others = tl.arange(0, OP_BLOCK)
    other_destination = tl.load(Destinations + others, others < CAPACITY, other=-1)
    other_source = tl.load(Sources + others, others < CAPACITY, other=-1)
    other_operation = tl.load(Operations + others, others < CAPACITY, other=-1)
    other_raw_slot = tl.load(RawSlots + others, others < CAPACITY, other=-1)
    other_live = (
        (others < count)
        & (others < CAPACITY)
        & (others != command)
        & (other_destination > 0)
    )
    other_live &= (other_operation == 1) | (other_source > 0)
    conflict = destination == other_destination
    conflict |= (operation == 0) & (source == other_destination)
    conflict |= (other_operation == 0) & (destination == other_source)
    conflict |= (raw_slot >= 0) & (raw_slot == other_raw_slot)
    error = tl.where(
        live & (tl.sum((other_live & conflict).to(tl.int32), 0) > 0), 6, error
    )
    error = tl.where((count < 0) | (count > CAPACITY), 7, error)
    if tile == 0:
        tl.store(Errors + command, error)
    valid = live & (error == 0)
    offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = valid & (offsets < ROW_BYTES)
    if operation == 1:
        history = (offsets >= HISTORY_OFFSET) & (offsets < HISTORY_OFFSET + 16)
        eos_byte = (
            tl.full((), 248044, tl.uint64) >> (((offsets - HISTORY_OFFSET) & 7) * 8)
        ) & 255
        value = tl.where(history, eos_byte, 0).to(tl.uint8)
        tl.store(Backing + destination * ROW_BYTES + offsets, value, mask)
    elif source != destination:
        value = tl.load(Backing + source * ROW_BYTES + offsets, mask, other=0)
        tl.store(Backing + destination * ROW_BYTES + offsets, value, mask)


@triton.jit
def _invalidate_qsa(
    Pointers,
    Consumed,
    Sources,
    Destinations,
    Operations,
    RawSlots,
    Accepted,
    Count,
    Errors,
    CONSUMED_STRIDE: tl.constexpr,
    SLOTS: tl.constexpr,
    RAW_SLOTS: tl.constexpr,
    RING: tl.constexpr,
    AXES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    command, layer = tl.program_id(0), tl.program_id(1)
    destination = tl.load(Destinations + command)
    source = tl.load(Sources + command)
    operation = tl.load(Operations + command)
    raw_slot = tl.load(RawSlots + command)
    live = (command < tl.load(Count)) & (tl.load(Errors + command) == 0)
    live &= (
        (destination > 0)
        & (destination < SLOTS)
        & (raw_slot >= 0)
        & (raw_slot < RAW_SLOTS)
    )
    live &= (operation == 1) | ((operation == 0) & (source > 0) & (source < SLOTS))
    logical = tl.load(Pointers + layer * 3).to(tl.pointer_type(tl.int64))
    rope = tl.load(Pointers + layer * 3 + 1).to(tl.pointer_type(tl.int64))
    anchor = tl.load(Pointers + layer * 3 + 2).to(tl.pointer_type(tl.int64))
    index = tl.arange(0, BLOCK)
    tl.store(logical + raw_slot * RING + index, -1, live & (index < RING))
    tl.store(rope + raw_slot * RING * AXES + index, -1, live & (index < RING * AXES))
    consumed = tl.load(Consumed + destination * CONSUMED_STRIDE, live, other=0)
    accepted = tl.load(Accepted + command)
    tl.store(anchor + raw_slot, consumed - accepted, live)


class PreparedStateTransfers:
    """Prepared indirect copy/reset with optional QSA raw-ring invalidation.

    Preparation requires zero live commands and primes every compiled kernel.
    CUDA graph replay reads only persistent device metadata. QSA invalidation
    writes tags, RoPE tags, and anchors; raw key payloads need not be cleared.
    """

    def __init__(
        self, pool: HybridStatePool, metadata: StateTransferMetadata, *, qsa_states=()
    ):
        device = pool.backing.device
        if device.type != "cuda":
            raise ValueError("prepared state transfers require CUDA")
        capacity = metadata.source_slots.numel()
        for name, dtype in (
            ("source_slots", torch.int64),
            ("destination_slots", torch.int64),
            ("operation", torch.int32),
            ("qsa_slots", torch.int64),
            ("accepted_tokens", torch.int32),
            ("num_ops", torch.int32),
            ("errors", torch.int32),
        ):
            tensor = getattr(metadata, name)
            shape = (1,) if name == "num_ops" else (capacity,)
            if (
                tensor.device != device
                or tensor.dtype != dtype
                or tensor.shape != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"state transfer {name} has incompatible shape, dtype, or layout"
                )
        if capacity <= 0:
            raise ValueError("state transfer needs positive command capacity")
        self.pool, self.metadata, self.qsa_states = pool, metadata, tuple(qsa_states)
        raw_slots, ring, axes = 0, 0, 0
        pointers = []
        for state in self.qsa_states:
            logical, rope, anchor = (
                state.raw_logical_positions,
                state.raw_rope_positions,
                state.raw_interval_start_positions,
            )
            if (
                logical.ndim != 2
                or rope.ndim != 3
                or rope.shape[:2] != logical.shape
                or anchor.shape != (logical.shape[0],)
            ):
                raise ValueError("QSA raw position tensors have incompatible shapes")
            shape = (logical.shape[0], logical.shape[1], rope.shape[2])
            if raw_slots and shape != (raw_slots, ring, axes):
                raise ValueError("QSA raw-state geometries must agree across layers")
            raw_slots, ring, axes = shape
            for tensor in (logical, rope, anchor):
                if (
                    tensor.device != device
                    or tensor.dtype != torch.int64
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "QSA raw positions must be contiguous int64 on the pool device"
                    )
            pointers.append([logical.data_ptr(), rope.data_ptr(), anchor.data_ptr()])
        self.qsa_pointers = (
            torch.tensor(pointers, dtype=torch.int64, device=device)
            if pointers
            else None
        )
        self._grid = (capacity, triton.cdiv(pool.layout.row_bytes, 4096), 1)
        self._transfer_args = (
            pool.backing,
            pool.consumed_tokens,
            metadata.source_slots,
            metadata.destination_slots,
            metadata.operation,
            metadata.qsa_slots,
            metadata.accepted_tokens,
            metadata.num_ops,
            metadata.errors,
            pool.layout.row_bytes,
            pool.backing.shape[0],
            capacity,
            raw_slots,
            pool.layout.region("ple_history").offset,
            pool.consumed_tokens.stride(0),
            4096,
            triton.next_power_of_2(capacity),
        )
        self._qsa_grid = (capacity, len(self.qsa_states), 1)
        self._qsa_args = (
            self.qsa_pointers,
            pool.consumed_tokens,
            metadata.source_slots,
            metadata.destination_slots,
            metadata.operation,
            metadata.qsa_slots,
            metadata.accepted_tokens,
            metadata.num_ops,
            metadata.errors,
            pool.consumed_tokens.stride(0),
            pool.backing.shape[0],
            raw_slots,
            ring,
            axes,
            triton.next_power_of_2(max(ring * axes, 1)),
        )
        self._prepared = False

    def prepare(self) -> None:
        if self.metadata.num_ops.item() != 0:
            raise ValueError("state transfer preparation requires zero live commands")
        compiled = _transfer.warmup(*self._transfer_args, grid=self._grid, num_warps=4)
        self._transfer = partial(compiled[self._grid], *self._transfer_args)
        self._qsa = None
        if self.qsa_states:
            compiled = _invalidate_qsa.warmup(
                *self._qsa_args, grid=self._qsa_grid, num_warps=4
            )
            self._qsa = partial(compiled[self._qsa_grid], *self._qsa_args)
        self._prepared = True
        self()

    def __call__(self) -> None:
        if not self._prepared:
            raise RuntimeError("state transfer prepare() must complete before capture")
        self._transfer()
        if self._qsa is not None:
            self._qsa()


__all__ = [
    "CANDIDATE_BASE",
    "COPY",
    "EOS_TOKEN_ID",
    "GDN_LAYERS",
    "LIVE_BASE",
    "RESET",
    "SNAPSHOT_BASE",
    "HybridStateLayout",
    "HybridStatePool",
    "PreparedStateTransfers",
    "StateRegion",
    "StateTransferMetadata",
]
