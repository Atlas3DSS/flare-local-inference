"""Captured admission, immutable-prefix restore, and candidate publication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .sampling import AdmissionBuffers, DeviceRequestState, PreparedAdmission, _check_tensor, _compile, _completion_fields, _write_completion
from .state import HybridStatePool, PreparedStateTransfers, StateTransferMetadata

TRANSFER_ERROR_BASE = 64
INVALID_COMMAND = 8
STALE_CANDIDATE = 9


@triton.jit
def _stage(Slots, Epochs, Count, Sources, Destinations, Operations, RawSlots, Accepted, StateCount,
           Generation, Status, Consumed, CandidateConsumed, CandidateGeneration, AdmissionErrors,
           FilteredSources, FilteredDestinations, FilteredOperations, FilteredRawSlots, FilteredAccepted,
           FilteredCount, Errors, MODE: tl.constexpr, CAPACITY: tl.constexpr, POOL_SLOTS: tl.constexpr,
           CS: tl.constexpr, BCS: tl.constexpr):
    row = tl.program_id(0)
    count, state_count = tl.load(Count), tl.load(StateCount)
    if row == 0:
        tl.store(FilteredCount, tl.minimum(tl.maximum(count, 0), CAPACITY))
    slot, epoch = tl.load(Slots + row), tl.load(Epochs + row)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    operation, raw_slot, accepted = tl.load(Operations + row), tl.load(RawSlots + row), tl.load(Accepted + row)
    active = row < count
    slot_valid = (slot >= 0) & (slot < 4)
    valid = (count >= 0) & (count <= CAPACITY) & (state_count == count) & slot_valid & (epoch > 0)
    if MODE == 0:
        valid &= (operation == 1) & (source == slot + 1) & (destination == slot + 1)
        valid &= (raw_slot == slot + 1) & (accepted > 0)
    elif MODE == 1:
        valid &= (operation == 0) & (source >= 9) & (source < POOL_SLOTS) & (destination == slot + 1)
        valid &= (raw_slot == slot + 1) & (accepted > 0)
    else:
        valid &= (operation == 0) & (source == slot + 5) & (destination >= 9) & (destination < POOL_SLOTS)
        valid &= raw_slot == -1
        valid &= epoch == tl.load(Generation + slot, slot_valid, other=0)
        status = tl.load(Status + slot, slot_valid, other=0)
        valid &= (status == 1) | (status == 2)
        other_rows = tl.arange(0, 4)
        duplicates = (other_rows < count) & (other_rows < CAPACITY) & (tl.load(Slots + other_rows) == slot)
        valid &= tl.sum(duplicates.to(tl.int32), 0) == 1
    error = tl.where(active & ~valid, 8, 0)
    if MODE == 2:
        candidate = tl.load(CandidateConsumed + slot * BCS, slot_valid, other=-1)
        consumed = tl.load(Consumed + slot * CS, slot_valid, other=-1)
        candidate_epoch = tl.load(CandidateGeneration + slot, slot_valid, other=0)
        candidate_valid = (candidate_epoch == epoch) & (candidate > 0) & (candidate % 64 == 0) & (candidate <= consumed)
        error = tl.where(active & valid & ~candidate_valid, 9, error)
        valid &= candidate_valid
    else:
        admission_error = tl.load(AdmissionErrors + row)
        valid &= admission_error == 0
    tl.store(Errors + row, error)
    enabled = active & valid
    tl.store(FilteredSources + row, tl.where(enabled, source, 0))
    tl.store(FilteredDestinations + row, tl.where(enabled, destination, 0))
    tl.store(FilteredOperations + row, operation)
    tl.store(FilteredRawSlots + row, tl.where(enabled, raw_slot, -1))
    tl.store(FilteredAccepted + row, tl.where(enabled, accepted, 0))


@triton.jit
def _merge_errors(Count, StageErrors, TransferErrors, AdmissionErrors, PublicErrors, CAPACITY: tl.constexpr, ADMIT: tl.constexpr):
    rows = tl.arange(0, 4)
    active = (rows < CAPACITY) & (rows < tl.load(Count))
    staged = tl.load(StageErrors + rows)
    transferred = tl.load(TransferErrors + rows)
    error = tl.where(staged != 0, staged, transferred)
    error = tl.where(active, error, 0)
    tl.store(PublicErrors + rows, error)
    if ADMIT:
        old = tl.load(AdmissionErrors + rows)
        tl.store(AdmissionErrors + rows, old | (error << 8))


@triton.jit
def _snapshot_complete(Slots, Epochs, Count, Errors, Generation, Consumed, Generated, Status, StateError,
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
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      -1, status, code, active, -1, RECORD_FIELDS)


class PreparedMaintenance:
    """Compose validated slot transitions with the completion-frame ABI.

    Modes are ``init`` (reset live hybrid state), ``restore`` (copy immutable
    state to live), and ``snapshot`` (publish a consumed boundary candidate).
    Admission and transfer rows use identical packed order. Native request
    allocation reserves the live slot before init/restore and the immutable
    destination before publication; published immutable destinations must not
    be reused while referenced by the prefix cache.

    Prevalidation gates each transfer before any model-state write. Transfer
    errors suppress admission and appear as error_code=64+transfer_error. A
    failed operation never changes another active generation's session state.
    """

    def __init__(self, mode: str, requests: DeviceRequestState, pool: HybridStatePool,
                 metadata: StateTransferMetadata, completion_records, frame_index, submission_sequence,
                 *, admission_buffers: AdmissionBuffers | None = None, forward_inputs=None,
                 qsa_states=(), capacity: int = 4):
        if mode not in ("init", "restore", "snapshot") or capacity not in (1, 2, 4):
            raise ValueError("maintenance requires init/restore/snapshot and capacity1,2 or4")
        if pool.backing.shape[0] < 10:
            raise ValueError("maintenance requires null/live/candidate slots and an immutable snapshot slot")
        device = pool.backing.device
        if device.type != "cuda" or requests.generation.device != device:
            raise ValueError("maintenance pool and request state must share a CUDA device")
        for tensor, expected in ((requests.consumed_tokens, pool.consumed_tokens[1:5, 0]),
                (requests.ple_history, pool.ple_history[1:5]),
                (requests.boundary_consumed_tokens, pool.consumed_tokens[5:9, 0]),
                (requests.boundary_history, pool.ple_history[5:9])):
            if tensor.data_ptr() != expected.data_ptr() or tensor.stride() != expected.stride():
                raise ValueError("maintenance request history/counts must alias the hybrid pool live/candidate rows")
        for name, dtype in (("source_slots", torch.int64), ("destination_slots", torch.int64),
                ("operation", torch.int32), ("qsa_slots", torch.int64), ("accepted_tokens", torch.int32),
                ("num_ops", torch.int32), ("errors", torch.int32)):
            _check_tensor(getattr(metadata, name), (1,) if name == "num_ops" else (4,), dtype, device, name)
        record_fields = _completion_fields(completion_records, device)
        for tensor, shape, dtype, name in ((frame_index, (1,), torch.int32, "frame_index"), (submission_sequence, (1,), torch.int64, "submission_sequence")):
            _check_tensor(tensor, shape, dtype, device, name)
        self.mode, self.requests, self.pool, self.metadata = mode, requests, pool, metadata
        self.admission = None
        if mode == "snapshot":
            if forward_inputs is None:
                raise ValueError("snapshot maintenance requires packed decoder slot/epoch/count inputs")
            slots, epochs, count = forward_inputs.row_slot_ids, forward_inputs.row_generations, forward_inputs.live_rows
            for tensor, shape, dtype, name in ((slots, (4,), torch.int32, "row_slot_ids"),
                    (epochs, (4,), torch.int64, "row_generations"), (count, (1,), torch.int32, "live_rows")):
                _check_tensor(tensor, shape, dtype, device, name)
            self.admission_errors = torch.zeros(4, dtype=torch.int32, device=device)
        else:
            if admission_buffers is None:
                raise ValueError("init and restore maintenance require admission staging buffers")
            self.admission = PreparedAdmission(requests, admission_buffers, completion_records, frame_index,
                                              submission_sequence, capacity=capacity)
            slots, epochs, count = (admission_buffers.admission_slots, admission_buffers.admission_generations,
                                    admission_buffers.admission_count)
            self.admission_errors = self.admission.errors
        self.stage_errors = torch.zeros(4, dtype=torch.int32, device=device)
        self.filtered = StateTransferMetadata.allocate(4, device)
        self.transfers = PreparedStateTransfers(pool, self.filtered, qsa_states=qsa_states)
        m, f, r = metadata, self.filtered, requests
        self._stage_args = (slots, epochs, count, m.source_slots, m.destination_slots, m.operation, m.qsa_slots,
            m.accepted_tokens, m.num_ops, r.generation, r.status, r.consumed_tokens, r.boundary_consumed_tokens,
            r.boundary_generation, self.admission_errors, f.source_slots, f.destination_slots, f.operation,
            f.qsa_slots, f.accepted_tokens, f.num_ops, self.stage_errors,
            ("init", "restore", "snapshot").index(mode), capacity, pool.backing.shape[0],
            r.consumed_tokens.stride(0), r.boundary_consumed_tokens.stride(0))
        self._merge_args = (count, self.stage_errors, f.errors, self.admission_errors, m.errors, capacity, mode != "snapshot")
        self._complete_args = (slots, epochs, count, m.errors, r.generation, r.consumed_tokens, r.generated_tokens,
            r.status, r.error_code, completion_records, frame_index, submission_sequence, capacity, r.consumed_tokens.stride(0), record_fields)
        self._prepared = False

    def staging_buffers(self) -> dict[str, torch.Tensor]:
        """Exact names registered by the native manifest for state commands."""
        return {f"state_{field}": getattr(self.metadata, field) for field in self.metadata.__dataclass_fields__}

    def prepare(self) -> None:
        if self.admission is not None:
            self.admission.prepare()
        self.transfers.prepare()
        self._stage = _compile(_stage, self._stage_args, (4, 1, 1))
        self._merge = _compile(_merge_errors, self._merge_args, (1, 1, 1))
        if self.mode == "snapshot":
            self._complete = _compile(_snapshot_complete, self._complete_args, (4, 1, 1))
        self._prepared = True

    def __call__(self) -> None:
        if not self._prepared:
            raise RuntimeError("maintenance prepare() must complete before capture")
        if self.admission is not None:
            self.admission.validate()
        self._stage()
        self.transfers()
        self._merge()
        if self.admission is not None:
            self.admission.apply()
        else:
            self._complete()
