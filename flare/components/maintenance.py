"""Request epochs and rank agreement around immutable continuation transfers."""

import torch
import triton
import triton.language as tl

from .sampling import (
    PreparedAdmission,
    _check_tensor,
    _compile,
    _completion_fields,
    _write_completion,
)


@triton.jit
def _stage(Slots, Epochs, Count, Sources, Destinations, Operations, StateCount, TransferErrors,
           AdmissionErrors, Generation, Status, Validation, MODE: tl.constexpr, CAPACITY: tl.constexpr):
    row = tl.program_id(0)
    count, state_count = tl.load(Count), tl.load(StateCount)
    source, destination, operation = tl.load(Sources + row), tl.load(Destinations + row), tl.load(Operations + row)
    slot, epoch = tl.load(Slots + row), tl.load(Epochs + row)
    present = (row < count) | (row < state_count)
    valid_slot = (slot >= 0) & (slot < 4)
    valid = (count >= 0) & (count <= CAPACITY) & (count == state_count) & valid_slot & (epoch > 0)
    if MODE == 2:
        valid &= (operation == 0) & (source == slot + 1) & (destination >= 5)
        valid &= epoch == tl.load(Generation + slot, valid_slot, other=0)
        status = tl.load(Status + slot, valid_slot, other=0)
        valid &= (status == 1) | (status == 2)
        others = tl.arange(0, 4)
        duplicates = (others < count) & (tl.load(Slots + others) == slot)
        valid &= tl.sum(duplicates.to(tl.int32), 0) == 1
    else:
        valid &= (destination == slot + 1) & (operation == (1 if MODE == 0 else 0))
        valid &= (source == -1) if MODE == 0 else (source >= 5)
        valid &= tl.load(AdmissionErrors + row) == 0
    bad = (tl.load(TransferErrors + row) != 0) | (present & ~valid)
    tl.store(Validation + row, bad.to(tl.bfloat16))
    tl.store(Validation + 4 + row, present.to(tl.bfloat16))


@triton.jit
def _agree(Totals, Errors, AdmissionErrors, WORLD: tl.constexpr, ADMIT: tl.constexpr):
    row = tl.arange(0, 4)
    bad, present = tl.load(Totals + row), tl.load(Totals + 4 + row)
    bad = (bad > 0) | ((present > 0) & (present != WORLD))
    error = tl.where(bad, 8, 0)
    tl.store(Errors + row, error)
    if ADMIT:
        tl.store(AdmissionErrors + row, error << 8)


@triton.jit
def _complete(Slots, Epochs, Count, Errors, Generation, Consumed, Generated, Status, StateError,
                       Records, Frame, Sequence, CAPACITY: tl.constexpr, CS: tl.constexpr, RECORD_FIELDS: tl.constexpr):
    slot = tl.program_id(0)
    rows = tl.arange(0, 4)
    matches = (rows < CAPACITY) & (rows < tl.load(Count)) & (tl.load(Slots + rows) == slot)
    active = tl.sum(matches.to(tl.int32), 0) > 0
    row = tl.min(tl.where(matches, rows, 4), 0)
    epoch, consumed, generated = tl.full((), 0, tl.int64), tl.full((), 0, tl.int64), tl.full((), 0, tl.int64)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        epoch = tl.load(Epochs + row)
        error = tl.load(Errors + row)
        if epoch == tl.load(Generation + slot):
            consumed, generated = tl.load(Consumed + slot * CS), tl.load(Generated + slot)
            status, code = tl.load(Status + slot), tl.load(StateError + slot)
        if error != 0:
            status, code = 3, 64 + error
            consumed, generated = tl.full((), 0, tl.int64), tl.full((), 0, tl.int64)
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      -1, status, code, active, RECORD_FIELDS)


class PreparedMaintenance:
    """Compose reset/restore admission or snapshot publication with row transfers.

    validate() returns eight BF16 flags: errors and participation per command.
    The caller reduces them across all TP ranks and writes validation_sum before
    apply(). Rejection or disagreement in row participation suppresses that
    command on every rank.
    Failed commands return code 72 with zero completion counters and preserve
    model and request state. Native ownership remains responsible for snapshot
    reservation and full-page sharing.
    """

    def __init__(self, mode, requests, transfer, slots, epochs, count, records, frame, sequence, *,
                 world_size, capacity=4, admission_buffers=None):
        if mode not in ("init", "restore", "snapshot") or world_size not in (1, 2, 4) or capacity not in (1, 2, 4):
            raise ValueError("maintenance requires a supported transition, rank count, and row capacity")
        device = transfer.pool.backing.device
        consumed = transfer.pool.region("consumed_tokens")[1:5, 0]
        if requests.consumed_tokens.data_ptr() != consumed.data_ptr() or requests.consumed_tokens.stride() != consumed.stride():
            raise ValueError("request counts must alias continuation storage live rows")
        for value, shape, dtype, name in ((slots, (4,), torch.int32, "slots"),
                (epochs, (4,), torch.int64, "epochs"), (count, (1,), torch.int32, "count"),
                (frame, (1,), torch.int32, "frame"), (sequence, (1,), torch.int64, "sequence")):
            _check_tensor(value, shape, dtype, device, name)
        fields = _completion_fields(records, device)
        self.transfer, self.requests, self.admission = transfer, requests, None
        self.validation_input = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
        self.validation_sum = torch.zeros_like(self.validation_input)
        if mode != "snapshot":
            if admission_buffers is None:
                raise ValueError("reset and restore require request admission buffers")
            for value, expected in ((slots, admission_buffers.admission_slots),
                    (epochs, admission_buffers.admission_generations), (count, admission_buffers.admission_count)):
                if value.data_ptr() != expected.data_ptr():
                    raise ValueError("maintenance and admission must share packed slot commands")
            self.admission = PreparedAdmission(requests, admission_buffers, records, frame, sequence, capacity=capacity)
        self.admission_errors = self.admission.errors if self.admission is not None else torch.zeros(4, dtype=torch.int32, device=device)
        c, r = transfer.commands, requests
        self.stage_args = (slots, epochs, count, c.source_slots, c.destination_slots, c.operation, c.num_ops, c.errors,
            self.admission_errors, r.generation, r.status, self.validation_input, ("init", "restore", "snapshot").index(mode), capacity)
        self.agree_args = (self.validation_sum, c.errors, self.admission_errors, world_size, self.admission is not None)
        self.complete_args = (slots, epochs, count, c.errors, r.generation, r.consumed_tokens, r.generated_tokens,
            r.status, r.error_code, records, frame, sequence, capacity, r.consumed_tokens.stride(0), fields)
        self._stage = None

    def prepare(self):
        self.transfer.prepare()
        if self.admission is not None:
            self.admission.prepare()
        self._stage = _compile(_stage, self.stage_args, (4, 1, 1))
        self._agree = _compile(_agree, self.agree_args, (1, 1, 1))
        self._complete = _compile(_complete, self.complete_args, (4, 1, 1))

    def validate(self):
        if self._stage is None:
            raise RuntimeError("maintenance prepare() must complete before capture")
        self.transfer.validate()
        if self.admission is not None:
            self.admission.validate()
        self._stage()
        return self.validation_input

    def apply(self):
        if self._stage is None:
            raise RuntimeError("maintenance prepare() must complete before capture")
        self._agree()
        self.transfer.apply()
        if self.admission is not None:
            self.admission.apply()
        self._complete()
