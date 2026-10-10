"""Qwen request defaults and PLE history alongside shared device sampling state."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

from flare.components.sampling import (
    SLOTS,
    _check_tensor,
    _compile,
)
from flare.components.sampling import (
    DeviceRequestState as SamplingState,
)
from flare.components.sampling import (
    RequestAdmission as SamplingAdmission,
)

VOCAB_SIZE = 248320


@dataclass(frozen=True)
class RequestAdmission(SamplingAdmission):
    eos_token_ids: tuple[int, ...] = (248046, 248044)

    def validate(self, vocab_size: int = VOCAB_SIZE) -> None:
        super().validate(vocab_size)


@dataclass
class DeviceRequestState(SamplingState):
    ple_history: torch.Tensor

    @classmethod
    def allocate(cls, device, *, vocab_size: int = VOCAB_SIZE,
                 consumed_tokens: torch.Tensor | None = None,
                 ple_history: torch.Tensor | None = None) -> DeviceRequestState:
        if type(vocab_size) is not int or not 1 <= vocab_size <= VOCAB_SIZE:
            raise ValueError("vocab_size must be in [1, 248320]")
        state = SamplingState.allocate(device, vocab_size=vocab_size, consumed_tokens=consumed_tokens)
        history = torch.full((SLOTS, 2), 248044, dtype=torch.int64, device=device) if ple_history is None else ple_history
        _check_tensor(history, (SLOTS, 2), torch.int64, state.generation.device, "ple_history", contiguous=False)
        if history.stride(0) <= 0 or history.stride(1) != 1:
            raise ValueError("ple_history requires positive row stride and contiguous history columns")
        return cls(**{field.name: getattr(state, field.name) for field in fields(SamplingState)}, ple_history=history)

    def admit(self, slot: int, admission: RequestAdmission, prompt_ids, *, consumed_tokens: int = 0,
              ple_history: tuple[int, int] = (248044, 248044)) -> None:
        if len(ple_history) != 2 or any(type(x) is not int or not 0 <= x < VOCAB_SIZE for x in ple_history):
            raise ValueError("PLE history requires two token IDs")
        super().admit(slot, admission, prompt_ids, consumed_tokens=consumed_tokens)
        self.ple_history[slot].copy_(torch.tensor(ple_history))


@triton.jit
def _gather(Slots, Epochs, Live, Generation, Status, Pending, Consumed, History,
            PackedPending, PackedPositions, PackedHistory, PackedActive, ROWS: tl.constexpr, CS: tl.constexpr, HS: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < 4)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    tl.store(PackedPending + row, tl.load(Pending + slot, valid, other=-1))
    tl.store(PackedPositions + row, tl.load(Consumed + slot * CS, valid, other=0))
    tl.store(PackedActive + row, valid.to(tl.int32))
    history_index = tl.arange(0, 2)
    history = tl.load(History + slot * HS + history_index, valid, other=248044)
    tl.store(PackedHistory + row * 2 + history_index, history)


@triton.jit
def _consume(Tokens, CuSeq, Slots, Epochs, Live, Generation, Status, Error, Pending, Consumed,
             History, ROWS: tl.constexpr, CAPACITY: tl.constexpr, CS: tl.constexpr, HS: tl.constexpr):
    slot = tl.program_id(0)
    rows = tl.arange(0, 4)
    matches = (rows < ROWS) & (rows < tl.load(Live))
    matches &= tl.load(Slots + rows, rows < ROWS, other=-1) == slot
    matches &= tl.load(Epochs + rows, rows < ROWS, other=-1) == tl.load(Generation + slot)
    count = tl.sum(matches.to(tl.int32), 0)
    if (count > 0) & (tl.load(Status + slot) == 1):
        if count > 1:
            tl.store(Status + slot, 3)
            tl.store(Error + slot, 4)
        else:
            row = tl.max(tl.where(matches, rows, 0), 0)
            start, end = tl.load(CuSeq + row), tl.load(CuSeq + row + 1)
            length = end - start
            pending = tl.load(Pending + slot)
            first = tl.load(Tokens + start, (start >= 0) & (start < CAPACITY), other=-1)
            valid = (start >= 0) & (end <= CAPACITY) & (end >= start)
            valid &= (length == 0) | (pending < 0) | ((length == 1) & (first == pending))
            if not valid:
                tl.store(Status + slot, 3)
                tl.store(Error + slot, 3)
            elif length > 0:
                consumed = tl.load(Consumed + slot * CS)
                old_last = tl.load(History + slot * HS + 1)
                last = tl.load(Tokens + end - 1)
                previous = tl.load(Tokens + end - 2, length >= 2, other=0)
                previous = tl.where(length >= 2, previous, old_last)
                final = consumed + length
                tl.store(History + slot * HS, previous)
                tl.store(History + slot * HS + 1, last)
                tl.store(Consumed + slot * CS, final)
                tl.store(Pending + slot, -1)


class PreparedTokenState:
    """Gather packed decode inputs and commit consumed prefill/decode spans.

    Call ``gather`` before forward and ``consume`` after forward. ``token_ids``
    is packed by ``cu_seqlens``; each live row must refer to a unique physical
    slot and its epoch. A pending sample can only be consumed as a one-token
    span containing that sample. Zero-token spans do not change state. Neither
    operation changes penalty counts or RNG state.
    """

    def __init__(self, state: DeviceRequestState, token_ids, cu_seqlens, row_slot_ids, row_generations, live_rows):
        rows, device = row_slot_ids.numel(), state.generation.device
        if device.type != "cuda" or not 1 <= rows <= SLOTS or token_ids.ndim != 1 or not token_ids.numel():
            raise ValueError("token state requires CUDA storage, 1..4 rows and a nonempty packed token capacity")
        for tensor, shape, dtype, name in ((row_slot_ids, (rows,), torch.int32, "row_slot_ids"),
                (row_generations, (rows,), torch.int64, "row_generations"), (live_rows, (1,), torch.int32, "live_rows"),
                (token_ids, (token_ids.numel(),), torch.int64, "token_ids"),
                (cu_seqlens, (rows + 1,), torch.int32, "cu_seqlens")):
            _check_tensor(tensor, shape, dtype, device, name)
        self.state = state
        self.pending_tokens = torch.empty(rows, dtype=torch.int64, device=device)
        self.positions = torch.empty(rows, dtype=torch.int64, device=device)
        self.ple_history = torch.empty((rows, 2), dtype=torch.int64, device=device)
        self.active = torch.empty(rows, dtype=torch.int32, device=device)
        self._gather_args = (row_slot_ids, row_generations, live_rows, state.generation, state.status,
            state.pending_token, state.consumed_tokens, state.ple_history,
            self.pending_tokens, self.positions, self.ple_history, self.active, rows,
            state.consumed_tokens.stride(0), state.ple_history.stride(0))
        self._consume_args = (token_ids, cu_seqlens, row_slot_ids, row_generations, live_rows, state.generation,
            state.status, state.error_code, state.pending_token, state.consumed_tokens,
            state.ple_history, rows, token_ids.numel(),
            state.consumed_tokens.stride(0), state.ple_history.stride(0))
        self._rows = rows
        self._prepared = False

    def prepare(self) -> None:
        self._gather = _compile(_gather, self._gather_args, (self._rows, 1, 1))
        self._consume = _compile(_consume, self._consume_args, (SLOTS, 1, 1))
        self._prepared = True

    def gather(self) -> None:
        if not self._prepared:
            raise RuntimeError("token state prepare() must complete before capture")
        self._gather()

    def consume(self) -> None:
        if not self._prepared:
            raise RuntimeError("token state prepare() must complete before capture")
        self._consume()
