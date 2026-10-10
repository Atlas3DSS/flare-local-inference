"""Captured exact-prefix row transfers with model-owned partial KV pages."""

from collections import defaultdict
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .sampling import _check_tensor, _compile
from .state import StateStorage


@dataclass(frozen=True)
class StateCommands:
    """Copy (0) or reset (1) commands for live slots 1–4 and snapshots 5 onward.

    Copy requires one live and one immutable slot, and a positive exact consumed
    endpoint. Reset requires source -1 and endpoint zero. Errors are 1=operation,
    2=slot/direction, 3=endpoint, 4=alias conflict, 5=count, 6=page mapping.
    Native ownership must reserve snapshot destinations before publication.
    """

    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    operation: torch.Tensor
    expected_consumed: torch.Tensor
    num_ops: torch.Tensor
    errors: torch.Tensor

    @classmethod
    def allocate(cls, device):
        return cls(torch.full((4,), -1, dtype=torch.int64, device=device),
                   torch.full((4,), -1, dtype=torch.int64, device=device),
                   torch.zeros(4, dtype=torch.int32, device=device),
                   torch.full((4,), -1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device),
                   torch.zeros(4, dtype=torch.int32, device=device))


@triton.jit
def _validate(Sources, Destinations, Operations, Expected, Count, Errors, Consumed, Tables,
              SLOTS: tl.constexpr, CS: tl.constexpr, WIDTH: tl.constexpr,
              PAGES: tl.constexpr, PAGE_TOKENS: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(Count)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    operation, endpoint = tl.load(Operations + row), tl.load(Expected + row)
    live = row < count
    snapshot = (source >= 1) & (source < 5) & (destination >= 5) & (destination < SLOTS)
    restore = (source >= 5) & (source < SLOTS) & (destination >= 1) & (destination < 5)
    reset = (operation == 1) & (source == -1) & (destination >= 1) & (destination < 5)
    copying = operation == 0
    valid_slots = (copying & (snapshot | restore)) | reset
    consumed = tl.load(Consumed + source * CS, live & copying & (source > 0) & (source < SLOTS), other=-1)
    valid_endpoint = tl.where(copying, (endpoint > 0) & (endpoint == consumed)
                             & (endpoint <= WIDTH * PAGE_TOKENS), endpoint == 0)
    slot = tl.where(snapshot, source, destination) - 1
    tail = copying & (endpoint > 0) & (endpoint % PAGE_TOKENS != 0)
    logical = endpoint // PAGE_TOKENS
    page = tl.load(Tables + slot * WIDTH + logical,
                   live & tail & valid_slots & (logical >= 0) & (logical < WIDTH), other=-1)

    others = tl.arange(0, 4)
    other_sources, other_destinations = tl.load(Sources + others), tl.load(Destinations + others)
    other_operations, other_endpoints = tl.load(Operations + others), tl.load(Expected + others)
    other_copy = other_operations == 0
    other_snapshot = other_copy & (other_sources >= 1) & (other_sources < 5) & (other_destinations >= 5)
    other_restore = other_copy & (other_sources >= 5) & (other_destinations >= 1) & (other_destinations < 5)
    other_slots = tl.where(other_snapshot, other_sources, other_destinations) - 1
    other_tail = (other_snapshot | other_restore) & (other_endpoints > 0) & (other_endpoints % PAGE_TOKENS != 0)
    other_logical = other_endpoints // PAGE_TOKENS
    other_live = (others < count) & (others != row)
    other_pages = tl.load(Tables + other_slots * WIDTH + other_logical,
        other_live & other_tail & (other_slots >= 0) & (other_slots < 4)
        & (other_logical >= 0) & (other_logical < WIDTH), other=-1)
    conflict = destination == other_destinations
    conflict |= copying & (source == other_destinations)
    conflict |= other_copy & (destination == other_sources)
    conflict |= tail & other_tail & (page >= 0) & (page == other_pages) & (restore | other_restore)

    error = tl.full((), 0, tl.int32)
    error = tl.where(live & tail & ((page < 0) | (page >= PAGES)), 6, error)
    error = tl.where(live & (tl.sum((other_live & conflict).to(tl.int32), 0) > 0), 4, error)
    error = tl.where(live & ~valid_endpoint, 3, error)
    error = tl.where(live & ~valid_slots, 2, error)
    error = tl.where(live & ((operation < 0) | (operation > 1)), 1, error)
    error = tl.where((count < 0) | (count > 4), 5, error)
    tl.store(Errors + row, error)


@triton.jit
def _copy_rows(Backing, Sources, Destinations, Operations, Count, Errors,
               ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    copying = tl.load(Operations + row) == 0
    active = (row < tl.load(Count)) & (tl.load(Errors + row) == 0)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Backing + source * ROW_BYTES + index, active & copying & (index < ROW_BYTES), other=0)
    tl.store(Backing + destination * ROW_BYTES + index, value, active & (index < ROW_BYTES))


@triton.jit
def _copy_pages(Pointers, Backing, Tables, Sources, Destinations, Operations, Expected, Count, Errors,
                ROW_BYTES: tl.constexpr, PAGE_BYTES: tl.constexpr, WIDTH: tl.constexpr,
                PAGE_TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    row, region, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    endpoint = tl.load(Expected + row)
    snapshot = source < 5
    active = (row < tl.load(Count)) & (tl.load(Errors + row) == 0)
    active &= (tl.load(Operations + row) == 0) & (endpoint % PAGE_TOKENS != 0)
    slot = tl.where(snapshot, source, destination) - 1
    page = tl.load(Tables + slot * WIDTH + endpoint // PAGE_TOKENS, active, other=0)
    cache = tl.load(Pointers + region * 2).to(tl.pointer_type(tl.uint8))
    offset = tl.load(Pointers + region * 2 + 1)
    saved_slot = tl.where(snapshot, destination, source)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    saved = Backing + saved_slot * ROW_BYTES + offset + index
    external = cache + page.to(tl.int64) * PAGE_BYTES + index
    if snapshot:
        value = tl.load(external, active & (index < PAGE_BYTES), other=0)
        tl.store(saved, value, active & (index < PAGE_BYTES))
    else:
        value = tl.load(saved, active & (index < PAGE_BYTES), other=0)
        tl.store(external, value, active & (index < PAGE_BYTES))


class PreparedStateTransfer:
    """Validate then copy complete rows and each model's partial physical page.

    ``pages`` contains (state region name, contiguous uint8 [pages, page_bytes]
    cache) pairs. Cache owners and pointer tables remain alive until graph drain.
    These maintenance operations are captured directly. Tensor-parallel callers
    must combine validation failures across ranks between validate() and apply().
    A nonzero command error suppresses both row and external-page writes.
    """

    def __init__(self, pool: StateStorage, commands: StateCommands, tables, *, pages, page_tokens):
        device = pool.backing.device
        if device.type != "cuda" or pool.backing.shape[0] < 6:
            raise ValueError("state transfers require CUDA null/live/snapshot storage")
        if type(page_tokens) is not int or page_tokens <= 0:
            raise ValueError("state transfers require positive physical page size")
        for name in commands.__dataclass_fields__:
            dtype = torch.int32 if name in ("operation", "num_ops", "errors") else torch.int64
            _check_tensor(getattr(commands, name), (1,) if name == "num_ops" else (4,), dtype, device, name)
        if tables.ndim != 2 or tables.shape[0] != 4 or tables.shape[1] <= 0:
            raise ValueError("state transfers require four physical request page tables")
        _check_tensor(tables, tables.shape, torch.int32, device, "page_tables")
        consumed = pool.region("consumed_tokens")
        if consumed.shape != (pool.backing.shape[0], 1) or consumed.dtype != torch.int64:
            raise ValueError("state rows require one int64 consumed endpoint")
        self.pool, self.commands, self.tables, self.pages = pool, commands, tables, tuple(pages)
        groups, page_count = defaultdict(list), None
        names = set()
        for name, cache in self.pages:
            region = pool.layout.region(name)
            if name in names or region.dtype != torch.uint8 or cache.ndim != 2 or cache.shape[1] != region.nbytes:
                raise ValueError("partial-page regions must be distinct uint8 fields matching their cache records")
            names.add(name)
            _check_tensor(cache, cache.shape, torch.uint8, device, name)
            if cache.shape[0] <= 0 or (page_count is not None and page_count != cache.shape[0]):
                raise ValueError("partial-page caches must share positive physical page capacity")
            page_count = cache.shape[0]
            groups[region.nbytes].append((cache.data_ptr(), region.offset))
        if page_count is None:
            raise ValueError("state transfer requires physical KV page bindings")
        self.pointers = {size: torch.tensor(values, dtype=torch.int64, device=device) for size, values in groups.items()}
        c = commands
        self.validate_args = (c.source_slots, c.destination_slots, c.operation, c.expected_consumed, c.num_ops,
            c.errors, consumed, tables, pool.backing.shape[0], consumed.stride(0), tables.shape[1], page_count, page_tokens)
        self.row_args = (pool.backing, c.source_slots, c.destination_slots, c.operation, c.num_ops, c.errors,
                         pool.layout.row_bytes, 4096)
        self.page_args = [(pointer, pool.backing, tables, c.source_slots, c.destination_slots, c.operation,
            c.expected_consumed, c.num_ops, c.errors, pool.layout.row_bytes, size, tables.shape[1], page_tokens, 1024)
            for size, pointer in self.pointers.items()]
        self._validate, self._copies = None, []

    def prepare(self):
        if self.commands.num_ops.item() != 0:
            raise ValueError("state transfer preparation requires zero live commands")
        self._validate = _compile(_validate, self.validate_args, (4, 1, 1))
        self._copies = [_compile(_copy_rows, self.row_args, (4, triton.cdiv(self.pool.layout.row_bytes, 4096), 1))]
        for args in self.page_args:
            self._copies.append(_compile(_copy_pages, args, (4, args[0].shape[0], triton.cdiv(args[10], 1024))))

    def validate(self):
        if self._validate is None:
            raise RuntimeError("state transfer prepare() must complete before capture")
        self._validate()

    def apply(self):
        if self._validate is None:
            raise RuntimeError("state transfer prepare() must complete before capture")
        for operation in self._copies:
            operation()
