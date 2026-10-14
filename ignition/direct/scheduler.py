"""A small fixed scheduler and the graph inputs it packs, mirroring Helios for paged text models.

This is the CPU half of Ignition's diagnostic direct execution. It owns request
slots, state rows, KV pages and an exact-prompt prefix cache, chooses each step,
packs that step's uploads byte for byte as Helios's `paged_text` contract packs
them, and interprets the completion records the graph writes. It replicates
only what reproducing model-level faults needs:

- FIFO admission into free slots. Each request reserves every page its prompt,
  output limit and widest verification round can reach, plus the draft page.
- Initialization, or restoration of a cached checkpoint whose consumed prefix
  is exactly the prompt, batched across the requests admitted together, then
  resume sampling after a restore.
- One prefill chunk per step, padded to the smallest prepared capacity.
- Decode rounds chosen as Helios chooses them (`policy.py`): every prepared
  batch capacity and verification width competes, carrying its highest-valued
  decoding requests, and the graph with the best expected value per graph time
  wins, from each request's observed acceptance, Helios's service prices and
  output ages, and Helios's expiring estimates of each graph's time. Decode
  rounds and prefill chunks alternate while both have work.
- A snapshot at the prompt endpoint of every prompt not yet cached, published as
  a checkpoint that later identical prompts restore, sharing its complete pages.

Partial-prefix reuse, terminal checkpoints, grammar masks, copied proposals and
pipelined submission are not reproduced: each step completes before the next.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .policy import Accounting, Policy

INACTIVE, RUNNING, FINISHED, ERROR = 0, 1, 2, 3
_MAINTENANCE = ("init", "restore", "snapshot")


class AdmissionError(ValueError):
    """A request the prepared capacities cannot serve."""


class CompletionError(RuntimeError):
    """A completion record that contradicts the submitted step; execution state is no longer trusted."""


@dataclass(frozen=True)
class Layout:
    """Prepared capacities and state geometry the scheduler packs against."""

    capacity: int
    max_active: int
    context: int
    page_tokens: int
    table_width: int
    pages: int
    snapshots: int
    prefill_chunk: int
    vocabulary: int
    eos: tuple[int, ...]
    graphs: MappingProxyType
    widths: tuple[int, ...] = ()
    fixed_width: int | None = None
    state_rows_per_request: int | None = None
    draft_pages: int = 0
    record_fields: int = 8

    def __post_init__(self):
        if not 1 <= self.max_active <= self.capacity or self.pages <= 0 or self.page_tokens <= 0:
            raise ValueError("direct execution requires positive pages and slots within request capacity")
        if self.widths and self.record_fields != 24:
            raise ValueError("speculative completions require token-vector records")
        object.__setattr__(self, "graphs", MappingProxyType(dict(self.graphs)))

    @classmethod
    def of(cls, model):
        """Read a prepared model's serving configuration and graph catalog."""
        config, buffers = model.server_config, model.buffers
        speculative = any(signature.kind == "speculative" for signature in model.signatures.values())
        graphs = {(signature.kind, signature.batch_capacity, signature.token_capacity): name
                  for name, signature in model.signatures.items()
                  if signature.speculation_enabled in (None, speculative)}
        return cls(capacity=buffers["row_slot_ids"].numel(), max_active=model.max_active,
                   context=config["max_sequence_tokens"], page_tokens=config["page_tokens"],
                   table_width=buffers["page_tables"].shape[1], pages=config["pages"],
                   snapshots=config["snapshots"], prefill_chunk=config["prefill_chunk_size"],
                   vocabulary=config["model_contract"]["vocabulary"],
                   eos=tuple(config["frontend"]["eos_token_ids"]), graphs=graphs,
                   widths=tuple(config.get("speculative_widths") or ()) if speculative else (),
                   fixed_width=config.get("speculative_fixed_width") if speculative else None,
                   state_rows_per_request=config.get("state_rows_per_request"),
                   draft_pages=int("draft_page_tables" in buffers),
                   record_fields=buffers["completion_records"].shape[-1])

    @property
    def speculative(self):
        return bool(self.widths)

    @property
    def max_width(self):
        return max(self.widths, default=1)

    @property
    def execution_width(self):
        """Helios's widest loaded verification width, which numbers each slot's state-row columns."""
        if not self.widths:
            return 1
        return self.fixed_width or self.max_width

    @property
    def execution_rows(self):
        return self.state_rows_per_request or self.execution_width

    def state_row(self, slot):
        return 1 + slot * self.execution_rows

    def snapshot_row(self, snapshot):
        return 1 + self.max_active * self.execution_rows + snapshot

    def capacities(self, kind, *, width=None):
        """Prepared capacities of one graph kind: token capacities for prefill, batch capacities otherwise."""
        if kind == "prefill":
            return sorted(tokens for (k, _, tokens) in self.graphs if k == kind)
        return sorted(batch for (k, batch, tokens) in self.graphs
                      if k == kind and (width is None or tokens == width))

    def graph(self, kind, size, width=1):
        """Name and capacities of the smallest prepared graph holding `size` rows, or prefill tokens."""
        for capacity in self.capacities(kind, width=None if kind == "prefill" else width):
            if capacity >= size:
                key = (kind, 1, capacity) if kind == "prefill" else (kind, capacity, width)
                return self.graphs[key], key[1], key[2]
        raise AdmissionError(f"no prepared {kind} graph holds {size}" + (f" at width {width}" if width > 1 else ""))


@dataclass(frozen=True)
class SamplingParameters:
    seed: int = 0
    temperature: float = 1.
    top_p: float = 1.
    min_p: float = 0.
    repetition_penalty: float = 1.
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    top_k: int = -1

    def __post_init__(self):
        values = (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                  self.presence_penalty, self.frequency_penalty)
        if any(not math.isfinite(value) for value in values):
            raise AdmissionError("sampling parameters must be finite")
        if self.temperature <= 0:
            raise AdmissionError("temperature must be positive; greedy decoding is not supported")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1 or self.repetition_penalty <= 0:
            raise AdmissionError("require 0 < top_p <= 1, 0 <= min_p <= 1 and repetition_penalty > 0")
        if type(self.seed) is not int or not 0 <= self.seed < 1 << 64 or type(self.top_k) is not int:
            raise AdmissionError("seed must be a uint64 and top_k an integer")

    def values(self):
        """The six FP32 admission parameters, in Helios's order."""
        return (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                self.presence_penalty, self.frequency_penalty)


class Acceptance:
    """Helios's per-request acceptance of each draft depth: a 0.5 prior worth two observations."""

    MEMORY = 128.

    def __init__(self):
        self.depths = [[.5, 2.] for _ in range(7)]

    def observe(self, offered, accepted, rejected):
        if offered == 0:
            return
        for depth in range(accepted):
            self._observe(depth, True)
        if rejected and accepted < offered:
            self._observe(accepted, False)

    def _observe(self, depth, accepted):
        entry = self.depths[depth]
        entry[1] = min(entry[1] + 1., self.MEMORY)
        entry[0] += (float(accepted) - entry[0]) / entry[1]

    def expected(self, width):
        """Expected tokens one round of `width` emits: the pending token, then each accepted draft."""
        chain, total = 1., float(width != 0)
        for probability, _ in self.depths[:min(max(width - 1, 0), 7)]:
            chain *= probability
            total += chain
        return total

    def expectations(self, maximum):
        """`expected(width)` for every width through `maximum`, computed in one pass."""
        chain, total, values = 1., 1., [0., 1.]
        for probability, _ in self.depths[:min(max(maximum - 1, 0), 7)]:
            chain *= probability
            total += chain
            values.append(total)
        return values


@dataclass
class Checkpoint:
    """An immutable snapshot row holding the state after consuming exactly `key`, and its complete pages."""

    key: tuple
    snapshot: int
    pages: tuple[int, ...]
    pins: int = 0
    used: int = 0

    @property
    def consumed(self):
        return len(self.key)


class Request:
    def __init__(self, ident, prompt, max_new_tokens, sampling, *, ignore_eos=False):
        self.ident, self.prompt = ident, list(prompt)
        self.max_new_tokens, self.sampling, self.ignore_eos = max_new_tokens, sampling, ignore_eos
        self.slot = self.prefix = self.capture = self.error = self.finish_reason = None
        self.generation = self.consumed = self.reused = self.shared = self.rounds = 0
        self.pages, self.generated = [], []
        self.phase = "queued"
        self.finished_early = False
        self.acceptance = Acceptance()
        self.accounting = Accounting()
        # Sequence of the last step that carried the request; Helios's `last_use` fence.
        self.last_use = 0

    @property
    def pending_token(self):
        return (self.prompt + self.generated)[self.consumed]

    @property
    def done(self):
        return self.phase == "done"


@dataclass
class Lane:
    request: Request
    start: int
    tokens: int


@dataclass
class Step:
    kind: str
    graph: str
    batch: int
    width: int
    lanes: list
    sequence: int
    participants: object = None

    @property
    def frame(self):
        return self.sequence % 2


class PagePool:
    """Reference-counted KV pages; the smallest free page IDs go first, as in Helios."""

    def __init__(self, count):
        self.free = list(range(count))
        self.references = [0] * count

    @property
    def available(self):
        return len(self.free)

    def take(self, count):
        if count > len(self.free):
            raise AdmissionError("KV pages are exhausted")
        pages = [heapq.heappop(self.free) for _ in range(count)]
        for page in pages:
            self.references[page] = 1
        return pages

    def retain(self, pages):
        for page in pages:
            if self.references[page] <= 0:
                raise RuntimeError("retaining a free KV page")
            self.references[page] += 1

    def release(self, pages):
        for page in pages:
            if self.references[page] <= 0:
                raise RuntimeError("releasing a free KV page")
            self.references[page] -= 1
            if self.references[page] == 0:
                heapq.heappush(self.free, page)


class Uploads:
    """Partial input-buffer writes for one step, as (buffer, byte offset, bytes)."""

    def __init__(self):
        self.items = []

    def put(self, name, offset, values, dtype):
        data = np.asarray(values, dtype=dtype).tobytes()
        if data:
            self.items.append((name, offset * np.dtype(dtype).itemsize, data))

    def i32(self, name, offset, values):
        self.put(name, offset, values, np.int32)

    def i64(self, name, offset, values):
        self.put(name, offset, values, np.int64)


def _padded(values, length, fill):
    return list(values)[:length] + [fill] * max(0, length - len(values))


class Scheduler:
    def __init__(self, layout, *, prefix_cache=True):
        self.layout = layout
        self.prefix_cache = prefix_cache and layout.snapshots > 0
        self.queue = deque()
        self.slots = [None] * layout.max_active
        self.generations = [0] * layout.max_active
        self.page_pool = PagePool(layout.pages)
        self.free_snapshots = list(range(layout.snapshots))
        self.cache = {}
        self.clock = 0
        self.sequence = 0
        self.last_kind = None
        self.policy = Policy(layout)
        self.rounds = Counter()
        self.steps = Counter()
        self.device_errors = Counter()
        self.stats = Counter()

    # Admission and retirement.

    def validate(self, request):
        prompt, layout = request.prompt, self.layout
        if not prompt or any(type(t) is not int or not 0 <= t < layout.vocabulary for t in prompt):
            raise AdmissionError("the prompt must be a nonempty list of vocabulary token IDs")
        if type(request.max_new_tokens) is not int or not 1 <= request.max_new_tokens < 1 << 31:
            raise AdmissionError("the output limit must be between 1 and 2147483647 tokens")
        if len(prompt) + request.max_new_tokens > layout.context:
            raise AdmissionError(f"prompt of {len(prompt)} tokens plus an output limit of {request.max_new_tokens}"
                                 f" tokens exceeds this server's context of {layout.context} tokens")
        if self._pages_needed(request, 0) > layout.pages:
            raise AdmissionError("the request needs more KV pages than the prepared model holds")

    def submit(self, request):
        self.validate(request)
        request.accounting = Accounting(arrival_ns=self.now_ns, last_progress_ns=self.now_ns)
        self.queue.append(request)

    @property
    def now_ns(self):
        """The scheduling clock, which the caller advances; see `advance_to`."""
        return self.policy.now_ns

    def advance_to(self, now_ns):
        """Set the scheduling clock, in nanoseconds; it must not go back.

        Helios's clock is wall time. The direct server advances this one by each
        step's graph time, and by wall time only while it waits for requests,
        so Python's host work between steps ages no request and expires no
        graph estimate, and an eager step counts as its graph time.
        """
        self.policy.advance_to(now_ns)

    @property
    def active(self):
        return [request for request in self.slots if request is not None]

    @property
    def idle(self):
        return not self.queue and not self.active

    def _limit(self, request):
        return min(len(request.prompt) + request.max_new_tokens + self.layout.max_width - 1, self.layout.context)

    def _pages_needed(self, request, shared):
        return -(-self._limit(request) // self.layout.page_tokens) - shared + self.layout.draft_pages

    def _evict(self, pages=0, snapshots=0):
        """Drop least recently used unpinned checkpoints until the requested resources are free."""
        while self.page_pool.available < pages or len(self.free_snapshots) < snapshots:
            candidates = [c for c in self.cache.values() if c.pins == 0]
            if not candidates:
                return False
            victim = min(candidates, key=lambda c: c.used)
            del self.cache[victim.key]
            self.page_pool.release(victim.pages)
            heapq.heappush(self.free_snapshots, victim.snapshot)
            self.stats["checkpoint_evictions"] += 1
        return True

    def _admit(self):
        while self.queue:
            request = self.queue[0]
            slot = next((index for index, occupant in enumerate(self.slots) if occupant is None), None)
            if slot is None:
                return
            hit = self.cache.get(tuple(request.prompt)) if self.prefix_cache else None
            shared = hit.pages if hit is not None else ()
            needed = self._pages_needed(request, len(shared))
            if hit is not None:
                hit.pins += 1
            if not self._evict(pages=needed):
                if hit is not None:
                    hit.pins -= 1
                if not self.active:
                    self.queue.popleft()
                    self._retire(request, error="KV pages are exhausted by cached checkpoints")
                    continue
                return
            self.queue.popleft()
            self.generations[slot] += 1
            self.slots[slot] = request
            request.slot, request.generation = slot, self.generations[slot]
            self.page_pool.retain(shared)
            request.pages = [*shared, *self.page_pool.take(needed)]
            request.shared = len(shared)
            if hit is not None:
                self.clock += 1
                hit.used = self.clock
                request.prefix, request.consumed = hit, hit.consumed
                request.reused = hit.consumed
                request.phase = "restore"
                self.stats["prefix_hits"] += 1
            else:
                request.phase = "init"

    def _retire(self, request, *, error=None):
        if request.slot is not None:
            self.slots[request.slot] = None
            self.page_pool.release(request.pages)
            request.pages = []
        if request.prefix is not None and request.phase == "restore":
            request.prefix.pins -= 1
        if request.capture is not None:
            heapq.heappush(self.free_snapshots, request.capture)
            request.capture = None
        if error is not None:
            request.error = error
            request.finish_reason = "error"
            self.stats["failed"] += 1
        else:
            stopped = request.generated and request.generated[-1] in self.layout.eos and not request.ignore_eos
            request.finish_reason = "stop" if stopped else "length"
            self.stats["completed"] += 1
        request.phase = "done"

    def fail_all(self, error):
        retired = []
        for request in [*self.active, *self.queue]:
            self._retire(request, error=error)
            retired.append(request)
        self.queue.clear()
        return retired

    # Step selection.

    def _step(self, kind, size, lanes, width=1):
        graph, batch, tokens = self.layout.graph(kind, size, width)
        return self._commit(Step(kind, graph, batch, tokens, lanes, 0))

    def _commit(self, step, rates=None):
        """Number the step and record what Helios's `commit` records: last use and service participants."""
        active = self.active
        if rates is None:
            rates = self.policy.rates(active)[0]
        step.participants = self.policy.participants(active, rates)
        self.sequence += 1
        step.sequence = self.sequence
        for lane in step.lanes:
            lane.request.last_use = self.sequence
        return step

    def _phase(self, phase):
        return [request for request in self.active if request.phase == phase]

    def plan(self):
        """The next step, or None when nothing can run."""
        captures = self._phase("snapshot")
        if captures:
            return self._step("snapshot", len(captures), [Lane(r, r.consumed, 1) for r in captures])
        self._admit()
        for phase in ("restore", "init"):
            admitted = self._phase(phase)
            if admitted:
                return self._step(phase, len(admitted), [Lane(r, r.consumed, 1) for r in admitted])
        resuming = self._phase("resume")
        if resuming:
            # Resume samples from the saved hidden state and consumes nothing.
            return self._step("resume", len(resuming), [Lane(r, r.consumed, 0) for r in resuming])
        decoding, prefilling = self._phase("decode"), self._phase("prefill")
        if decoding and (not prefilling or self.last_kind not in ("decode", "speculative")):
            rates, expected = self.policy.rates(self.active)
            chosen = self.policy.decode_round(decoding, self.active, rates, expected)
            if chosen is not None:
                kind = "speculative" if self.layout.speculative else "decode"
                lanes = [Lane(r, r.consumed, self._round_tokens(r, chosen.width)) for r in chosen.requests]
                step = Step(kind, self.layout.graphs[(kind, chosen.batch, chosen.width)], chosen.batch,
                            chosen.width, lanes, 0)
                return self._commit(step, rates)
        if prefilling:
            request = min(prefilling, key=lambda r: r.ident)
            tokens = min(len(request.prompt) - request.consumed, self.layout.prefill_chunk)
            return self._step("prefill", tokens, [Lane(request, request.consumed, tokens)])
        return None

    def _round_tokens(self, request, width):
        return max(1, min(width, self.layout.context - request.consumed,
                          request.max_new_tokens - len(request.generated)))

    # Input packing, as Helios's paged_text contract packs each graph kind.

    def pack(self, step):
        layout, uploads = self.layout, Uploads()
        capacity, lanes, kind = layout.capacity, step.lanes, step.kind
        slots = [lane.request.slot for lane in lanes]
        generations = [lane.request.generation for lane in lanes]
        for lane in lanes:
            row = layout.state_row(lane.request.slot)
            uploads.i64("request_state_rows", lane.request.slot * 8,
                        [row + column if column < layout.execution_width else 0 for column in range(8)])
        admit = kind in ("init", "restore")
        if not admit:
            uploads.i32("row_slot_ids", 0, _padded(slots, capacity, -1))
            uploads.i64("row_generations", 0, _padded(generations, capacity, 0))
            uploads.i32("live_rows", 0, [len(lanes)])
        if kind != "resume":
            for lane in lanes:
                pages = lane.request.pages[:layout.table_width]
                uploads.i32("page_tables", lane.request.slot * layout.table_width, pages)
                if layout.draft_pages:
                    uploads.i32("draft_page_tables", lane.request.slot * layout.table_width,
                                lane.request.pages[1:1 + min(len(lane.request.pages) - 1, layout.table_width)])
        if kind in ("prefill", "decode", "speculative"):
            samples = [int(kind != "prefill" or lane.start + lane.tokens == len(lane.request.prompt)) for lane in lanes]
            uploads.i32("sample_enabled", 0, _padded(samples, capacity, 0))
            uploads.i32("query_lengths", 0, _padded([lane.tokens for lane in lanes], capacity, 0))
        if kind == "prefill":
            lane = lanes[0]
            uploads.i64("token_ids", 0, lane.request.prompt[lane.start:lane.start + lane.tokens])
        elif kind in ("decode", "speculative"):
            uploads.i64("token_ids", 0, _padded([lane.request.pending_token for lane in lanes], step.batch, 0))
        if admit or kind == "snapshot":
            uploads.i64("state_request_slots", 0, _padded(slots, capacity, -1))
            source, destination, operation, expected = [], [], [], []
            for lane in lanes:
                request = lane.request
                if kind == "init":
                    source.append(-1), destination.append(layout.state_row(request.slot))
                    operation.append(1), expected.append(0)
                elif kind == "restore":
                    source.append(layout.snapshot_row(request.prefix.snapshot))
                    destination.append(layout.state_row(request.slot))
                    operation.append(0), expected.append(request.consumed)
                else:
                    source.append(layout.state_row(request.slot))
                    destination.append(layout.snapshot_row(request.capture))
                    operation.append(0), expected.append(request.consumed)
            uploads.i64("state_source_slots", 0, _padded(source, capacity, -1))
            uploads.i64("state_destination_slots", 0, _padded(destination, capacity, -1))
            uploads.i64("state_expected_consumed", 0, _padded(expected, capacity, -1))
            uploads.i32("state_operation", 0, _padded(operation, capacity, 0))
            uploads.i32("state_num_ops", 0, [len(lanes)])
        if admit:
            vocabulary, words = layout.vocabulary, -(-layout.vocabulary // 32)
            uploads.i32("admission_count", 0, [len(lanes)])
            uploads.i32("admission_slots", 0, _padded(slots, capacity, -1))
            uploads.i64("admission_generations", 0, _padded(generations, capacity, 0))
            for row, lane in enumerate(lanes):
                request, sampling = lane.request, lane.request.sampling
                uploads.put("admission_seed_bits", row, [sampling.seed], np.uint64)
                uploads.put("admission_parameters", row * 6, sampling.values(), np.float32)
                uploads.i32("admission_top_k", row, [sampling.top_k])
                uploads.i64("admission_max_new_tokens", row, [request.max_new_tokens])
                uploads.i64("admission_eos_token_ids", row * 8, _padded(list(layout.eos), 8, -1))
                uploads.i32("admission_eos_count", row, [0 if request.ignore_eos else len(layout.eos)])
                uploads.i32("admission_histogram", row * vocabulary,
                            np.bincount(np.asarray(request.prompt, dtype=np.int64), minlength=vocabulary))
                uploads.i32("admission_grammar_mask", row * words, np.full(words, -1, dtype=np.int32))
                uploads.i32("admission_grammar_enabled", row, [0])
        return uploads.items

    # Completion.

    def complete(self, step, records, graph_ns=None):
        """Apply one step's completion records; returns the requests it retired.

        `graph_ns` is the step's graph time, which Helios's estimates and service
        prices observe; None observes nothing, as Helios does without GPU timing.
        """
        layout = self.layout
        records = np.asarray(records)
        if records.shape != (layout.capacity, layout.record_fields):
            raise CompletionError(f"completion records have shape {records.shape}")
        lanes = {lane.request.slot: lane for lane in step.lanes}
        for slot, record in enumerate(records):
            if record[5] != INACTIVE and slot not in lanes:
                raise CompletionError(f"slot {slot} reported status {record[5]} outside step {step.sequence}")
        self.policy.costs.observe((step.kind, step.batch, step.width), graph_ns)
        self.steps[step.kind] += 1
        if step.kind == "speculative":
            self.rounds[(step.batch, step.width)] += 1
        self.last_kind = step.kind
        retired, useful, published = [], {}, []
        live = list(self.active)
        for lane in step.lanes:
            request = lane.request
            record = records[request.slot]
            status, code = int(record[5]), int(record[6])
            if status == INACTIVE or record[0] != request.generation or record[1] != step.sequence:
                raise CompletionError(f"{step.graph} completion for request {request.ident} is inactive or stale:"
                                      f" {record[:8].tolist()}")
            if status == ERROR:
                self.device_errors[code] += 1
                self._retire(request, error=code)
                retired.append(request)
                continue
            if step.kind in _MAINTENANCE:
                if record[2] != request.consumed or record[3] != len(request.generated) or record[4] != -1:
                    raise CompletionError(f"{step.graph} counters disagree for request {request.ident}:"
                                          f" {record[:8].tolist()}")
                if step.kind == "snapshot":
                    self._publish(request)
                    if request.finished_early:
                        self._retire(request)
                        retired.append(request)
                    else:
                        request.phase = "decode"
                elif step.kind == "restore":
                    request.prefix.pins -= 1
                    request.phase = "resume" if request.consumed == len(request.prompt) else "prefill"
                else:
                    request.phase = "prefill"
                continue
            emitted = int(record[3]) - len(request.generated)
            width = lane.tokens if step.kind == "speculative" else 1
            consumed = int(record[2])
            if (not 0 <= emitted <= width or consumed < request.consumed or consumed > lane.start + lane.tokens
                    or (step.kind == "speculative" and consumed - request.consumed != emitted)
                    or (step.kind != "speculative" and status == RUNNING and consumed != lane.start + lane.tokens)):
                raise CompletionError(f"{step.graph} progress disagrees for request {request.ident}:"
                                      f" {record[:8].tolist()}")
            if layout.record_fields == 24:
                if int(record[8]) != emitted:
                    raise CompletionError(f"{step.graph} token vector disagrees for request {request.ident}")
                tokens = [int(token) for token in record[16:16 + emitted]]
            else:
                tokens = [int(record[4])] if emitted else []
            if step.kind == "speculative":
                offered = min(int(record[9]), lane.tokens - 1)
                accepted = min(int(record[10]), offered)
                request.acceptance.observe(offered, accepted, int(record[8]) > int(record[10]))
                request.rounds += 1
            if emitted or (step.kind == "prefill" and consumed > request.consumed):
                # Helios's `credit`: committed prompt and output tokens are this step's useful service.
                useful[request.slot] = (consumed - request.consumed if step.kind == "prefill" else 0) + emitted
                request.accounting.last_progress_ns = self.now_ns
            if emitted:
                request.accounting.pending_publication_fence = step.sequence
                published.append(request)
            request.consumed = consumed
            request.generated.extend(tokens)
            finished = status == FINISHED or len(request.generated) >= request.max_new_tokens
            if step.kind == "prefill" and consumed == len(request.prompt) and self._capture(request):
                request.finished_early = finished
                continue
            if finished:
                self._retire(request)
                retired.append(request)
            elif consumed < len(request.prompt):
                request.phase = "prefill"
            else:
                request.phase = "decode" if request.generated else "resume"
        self.policy.observe_service(step.sequence, step.participants, useful, live, graph_ns)
        for request in published:
            self.policy.published(request, step.sequence)
        return retired

    def _capture(self, request):
        """Reserve a snapshot row for the request's prompt endpoint unless that prompt is cached or in capture."""
        if not self.prefix_cache:
            return False
        key = tuple(request.prompt)
        if key in self.cache or any(r.capture is not None and r.prompt == request.prompt for r in self.active):
            return False
        if not self.free_snapshots and not self._evict(snapshots=1):
            return False
        request.capture = heapq.heappop(self.free_snapshots)
        request.phase = "snapshot"
        return True

    def _publish(self, request):
        pages = tuple(request.pages[:request.consumed // self.layout.page_tokens])
        self.page_pool.retain(pages)
        self.clock += 1
        key = tuple(request.prompt[:request.consumed])
        self.cache[key] = Checkpoint(key, request.capture, pages, used=self.clock)
        request.capture = None
        self.stats["checkpoints"] += 1
