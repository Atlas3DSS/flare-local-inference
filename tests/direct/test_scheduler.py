"""Direct execution's scheduling, prefix bookkeeping and Helios-equivalent input packing, on the CPU."""

import numpy as np
import pytest

from ignition.direct.scheduler import (
    Acceptance,
    AdmissionError,
    CompletionError,
    Layout,
    Request,
    SamplingParameters,
    Scheduler,
)

VOCABULARY, EOS = 1000, (7,)


def layout(**overrides):
    graphs = {("prefill", 1, tokens): f"prefill{tokens}" for tokens in (128, 256, 512)}
    for kind in ("init", "restore", "snapshot", "resume"):
        graphs.update({(kind, batch, 1): f"{kind}{batch}" for batch in (1, 2, 4)})
    graphs.update({("speculative", batch, width): f"speculative{batch}x{width}"
                   for batch in (1, 2, 4) for width in (1, 2, 3)})
    values = dict(capacity=4, max_active=4, context=4096, page_tokens=256, table_width=16, pages=64,
                  snapshots=2, prefill_chunk=512, vocabulary=VOCABULARY, eos=EOS, graphs=graphs,
                  widths=(1, 2, 3), state_rows_per_request=1, draft_pages=1, record_fields=24)
    values.update(overrides)
    return Layout(**values)


def request(ident, prompt, limit=8, seed=0):
    return Request(ident, prompt, limit, SamplingParameters(seed=seed, temperature=1., top_p=.95))


def uploads(scheduler, step):
    """Decode a step's uploads into {buffer: {element offset: values}}."""
    types = {"admission_parameters": np.float32}
    result = {}
    for name, offset, data in scheduler.pack(step):
        dtype = types.get(name, np.int64 if name in _INT64 else np.int32)
        values = np.frombuffer(data, dtype=dtype)
        result.setdefault(name, {})[offset // np.dtype(dtype).itemsize] = values.tolist()
    return result


_INT64 = {"request_state_rows", "row_generations", "token_ids", "state_request_slots", "state_source_slots",
          "state_destination_slots", "state_expected_consumed", "admission_generations", "admission_seed_bits",
          "admission_max_new_tokens", "admission_eos_token_ids"}


def records(scheduler, step, results):
    """Completion rows for `results`: {request: (status, consumed, generated tokens, error, offered, accepted)}."""
    rows = np.zeros((scheduler.layout.capacity, scheduler.layout.record_fields), dtype=np.int64)
    rows[:, 4] = -1
    rows[:, 16:] = -1
    for request, (status, consumed, tokens, error, *acceptance) in results.items():
        row = rows[request.slot]
        emitted = len(tokens)
        row[:8] = (request.generation, step.sequence, consumed, len(request.generated) + emitted,
                   tokens[-1] if tokens else -1, status, error, 0)
        row[8] = emitted
        row[9:11] = acceptance or (0, 0)
        row[12], row[13], row[15] = -1, max(consumed - 1, 0), step.width
        row[16:16 + emitted] = tokens
    return rows


def test_a_cold_request_initializes_prefills_in_chunks_snapshots_and_decodes():
    scheduler = Scheduler(layout())
    first = request(1, list(range(10, 710)), limit=4, seed=3)
    scheduler.submit(first)

    step = scheduler.plan()
    assert (step.kind, step.graph, [lane.request for lane in step.lanes]) == ("init", "init1", [first])
    packed = uploads(scheduler, step)
    assert packed["request_state_rows"] == {0: [1, 2, 3, 0, 0, 0, 0, 0]}
    assert packed["state_source_slots"] == {0: [-1, -1, -1, -1]}
    assert packed["state_destination_slots"] == {0: [1, -1, -1, -1]}
    assert packed["state_operation"] == {0: [1, 0, 0, 0]} and packed["state_expected_consumed"] == {0: [0, -1, -1, -1]}
    assert packed["admission_slots"] == {0: [0, -1, -1, -1]} and packed["admission_generations"] == {0: [1, 0, 0, 0]}
    assert packed["admission_seed_bits"] == {0: [3]} and packed["admission_max_new_tokens"] == {0: [4]}
    assert packed["admission_eos_token_ids"] == {0: [7, -1, -1, -1, -1, -1, -1, -1]}
    assert packed["admission_eos_count"] == {0: [1]}
    assert np.allclose(packed["admission_parameters"][0], [1., .95, 0., 1., 0., 0.])
    histogram = packed["admission_histogram"][0]
    assert len(histogram) == VOCABULARY and histogram[10] == 1 and sum(histogram) == 700
    assert packed["admission_grammar_mask"] == {0: [-1] * 32}
    # Pages through prompt, output limit and the widest round, plus the draft page: ceil(706 / 256) + 1.
    assert first.pages == [0, 1, 2, 3]
    assert packed["page_tables"] == {0: [0, 1, 2, 3]} and packed["draft_page_tables"] == {0: [1, 2, 3]}
    assert "row_slot_ids" not in packed
    assert scheduler.complete(step, records(scheduler, step, {first: (1, 0, [], 0)})) == []

    step = scheduler.plan()
    assert (step.graph, step.lanes[0].start, step.lanes[0].tokens) == ("prefill512", 0, 512)
    packed = uploads(scheduler, step)
    assert packed["token_ids"] == {0: list(range(10, 522))}
    assert packed["query_lengths"] == {0: [512, 0, 0, 0]} and packed["sample_enabled"] == {0: [0, 0, 0, 0]}
    assert packed["row_slot_ids"] == {0: [0, -1, -1, -1]} and packed["live_rows"] == {0: [1]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 512, [], 0)}))

    step = scheduler.plan()
    assert (step.graph, step.lanes[0].start, step.lanes[0].tokens) == ("prefill256", 512, 188)
    assert uploads(scheduler, step)["sample_enabled"] == {0: [1, 0, 0, 0]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 700, [42], 0)}))
    assert first.phase == "snapshot" and first.generated == [42]

    step = scheduler.plan()
    assert step.graph == "snapshot1"
    packed = uploads(scheduler, step)
    assert packed["state_source_slots"] == {0: [1, -1, -1, -1]}
    assert packed["state_destination_slots"] == {0: [5, -1, -1, -1]}
    assert packed["state_expected_consumed"] == {0: [700, -1, -1, -1]}
    assert packed["row_slot_ids"] == {0: [0, -1, -1, -1]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 700, [], 0)}))
    checkpoint = scheduler.cache[tuple(first.prompt)]
    assert (checkpoint.snapshot, checkpoint.pages, checkpoint.consumed) == (0, (0, 1), 700)

    step = scheduler.plan()
    assert (step.kind, step.graph, step.lanes[0].tokens) == ("speculative", "speculative1x3", 3)
    packed = uploads(scheduler, step)
    assert packed["token_ids"] == {0: [42]} and packed["query_lengths"] == {0: [3, 0, 0, 0]}
    assert packed["sample_enabled"] == {0: [1, 0, 0, 0]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 702, [43, 44], 0, 2, 1)}))
    assert first.generated == [42, 43, 44] and first.consumed == 702 and first.pending_token == 44

    step = scheduler.plan()
    # One output token remains, so the row offers no drafts though the graph is wider.
    assert step.lanes[0].tokens == 1
    retired = scheduler.complete(step, records(scheduler, step, {first: (2, 703, [7], 0)}))
    assert retired == [first] and first.finish_reason == "stop" and first.slot == 0
    assert scheduler.idle and scheduler.page_pool.available == 64 - 2


def test_an_identical_prompt_restores_its_checkpoint_and_shares_its_complete_pages():
    scheduler = Scheduler(layout())
    prompt = list(range(10, 610))
    first = request(1, prompt, limit=1)
    scheduler.submit(first)
    for result in ((1, 0, []), (1, 512, []), (2, 600, [42])):
        step = scheduler.plan()
        scheduler.complete(step, records(scheduler, step, {first: (*result, 0)}))
    step = scheduler.plan()
    assert step.graph == "snapshot1"
    assert scheduler.complete(step, records(scheduler, step, {first: (2, 600, [], 0)})) == [first]
    shared = scheduler.cache[tuple(prompt)].pages
    assert len(shared) == 2 and scheduler.page_pool.references[shared[0]] == 1

    copies = [request(2 + index, prompt, limit=6, seed=index) for index in range(3)]
    for copy in copies:
        scheduler.submit(copy)
    step = scheduler.plan()
    assert (step.graph, [lane.request for lane in step.lanes]) == ("restore4", copies)
    packed = uploads(scheduler, step)
    assert packed["state_source_slots"] == {0: [5, 5, 5, -1]}
    assert packed["state_destination_slots"] == {0: [1, 2, 3, -1]}
    assert packed["state_expected_consumed"] == {0: [600, 600, 600, -1]}
    assert packed["admission_count"] == {0: [3]} and packed["admission_slots"] == {0: [0, 1, 2, -1]}
    assert packed["admission_generations"] == {0: [2, 1, 1, 0]}
    for copy in copies:
        assert copy.pages[:2] == list(shared) and len(set(copy.pages[2:]) & set(shared)) == 0
        assert copy.reused == 600 and scheduler.page_pool.references[shared[0]] == 4
    tables = packed["page_tables"]
    assert tables[0][:2] == list(shared) and tables[16][:2] == list(shared)
    scheduler.complete(step, records(scheduler, step, {copy: (1, 600, [], 0) for copy in copies}))

    step = scheduler.plan()
    assert (step.graph, [lane.tokens for lane in step.lanes]) == ("resume4", [0, 0, 0])
    packed = uploads(scheduler, step)
    assert "page_tables" not in packed and "query_lengths" not in packed
    assert packed["row_slot_ids"] == {0: [0, 1, 2, -1]}
    scheduler.complete(step, records(scheduler, step, {copy: (1, 600, [50 + i], 0) for i, copy in enumerate(copies)}))

    step = scheduler.plan()
    assert step.graph == "speculative4x3"
    assert uploads(scheduler, step)["token_ids"] == {0: [50, 51, 52, 0]}
    assert scheduler.stats["prefix_hits"] == 3


def respond(scheduler, step):
    """Plausible device results: prefill consumes its chunk, rounds accept every draft."""
    results = {}
    for lane in step.lanes:
        current = lane.request
        if step.kind == "prefill":
            end = lane.start + lane.tokens
            results[current] = (1, end, [9] if end == len(current.prompt) else [], 0)
        elif step.kind in ("speculative", "decode"):
            results[current] = (1, current.consumed + lane.tokens, [8] * lane.tokens, 0, lane.tokens - 1,
                                lane.tokens - 1)
        elif step.kind == "resume":
            results[current] = (1, current.consumed, [9], 0)
        else:
            results[current] = (1, current.consumed, [], 0)
    return records(scheduler, step, results)


def test_device_errors_retire_the_failed_request_and_free_its_pages():
    scheduler = Scheduler(layout())
    failing, healthy = request(1, [1, 2, 3], limit=64), request(2, [4, 5, 6], limit=64)
    scheduler.submit(failing)
    scheduler.submit(healthy)
    while not (failing.phase == healthy.phase == "decode"):
        step = scheduler.plan()
        scheduler.complete(step, respond(scheduler, step))
    step = scheduler.plan()
    assert step.graph == "speculative2x3"
    before = dict(scheduler.rounds)
    retired = scheduler.complete(step, records(scheduler, step, {
        failing: (3, failing.consumed, [], 6), healthy: (1, healthy.consumed + 2, [10, 11], 0, 2, 2)}))
    assert retired == [failing] and failing.error == 6 and failing.finish_reason == "error"
    assert scheduler.device_errors == {6: 1} and scheduler.rounds[(2, 3)] == before.get((2, 3), 0) + 1
    assert failing.pages == [] and scheduler.slots[0] is None
    assert scheduler.plan().graph == "speculative1x3"


def test_completion_records_must_match_the_submitted_step():
    scheduler = Scheduler(layout())
    current = request(1, [1, 2, 3])
    scheduler.submit(current)
    step = scheduler.plan()
    stale = records(scheduler, step, {current: (1, 0, [], 0)})
    stale[current.slot, 1] -= 1
    with pytest.raises(CompletionError, match="inactive or stale"):
        scheduler.complete(step, stale)
    stray = records(scheduler, step, {current: (1, 0, [], 0)})
    stray[3, 5] = 1
    with pytest.raises(CompletionError, match="outside step"):
        scheduler.complete(step, stray)


def test_admission_rejects_what_the_prepared_capacities_cannot_hold():
    scheduler = Scheduler(layout(pages=8))
    with pytest.raises(AdmissionError, match="exceeds this server's context"):
        scheduler.submit(request(1, [1] * 4000, limit=200))
    with pytest.raises(AdmissionError, match="more KV pages"):
        scheduler.submit(request(2, [1] * 3000, limit=8))
    with pytest.raises(AdmissionError, match="vocabulary"):
        scheduler.submit(request(3, [VOCABULARY]))
    with pytest.raises(AdmissionError, match="greedy"):
        SamplingParameters(temperature=0.)


def test_requests_wait_for_slots_and_pages_and_rounds_alternate_with_prefill():
    scheduler = Scheduler(layout(max_active=2, pages=12))
    requests = [request(index, list(range(1, 600)), limit=4) for index in range(3)]
    for current in requests:
        scheduler.submit(current)
    step = scheduler.plan()
    # Each request reserves ceil(605 / 256) + 1 pages; the third waits for a slot.
    assert step.graph == "init2" and len(scheduler.queue) == 1
    scheduler.complete(step, records(scheduler, step, {r: (1, 0, [], 0) for r in requests[:2]}))
    step = scheduler.plan()
    assert step.lanes[0].request is requests[0]
    scheduler.complete(step, records(scheduler, step, {requests[0]: (1, 512, [], 0)}))
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {requests[0]: (1, 599, [5], 0)}))
    step = scheduler.plan()
    assert step.graph == "snapshot1"
    scheduler.complete(step, records(scheduler, step, {requests[0]: (1, 599, [], 0)}))
    kinds = []
    for _ in range(4):
        step = scheduler.plan()
        kinds.append(step.kind)
        lanes = {lane.request: lane for lane in step.lanes}
        if step.kind == "prefill":
            results = {requests[1]: (1, lanes[requests[1]].start + lanes[requests[1]].tokens, [], 0)}
            if results[requests[1]][1] == 599:
                results[requests[1]] = (1, 599, [6], 0)
        else:
            results = {r: (1, r.consumed + 1, [8], 0, 2, 0) for r in lanes}
        scheduler.complete(step, records(scheduler, step, results))
    assert kinds == ["speculative", "prefill", "speculative", "prefill"]
    # The second prompt is cached already, so it decodes without a snapshot.
    assert requests[1].phase == "decode"


def decoding(scheduler, count, limit=1000):
    """`count` requests decoding from the first round, as Helios's policy tests set them up."""
    requests = [request(index, [1, 2, 3], limit=limit) for index in range(count)]
    for current in requests:
        scheduler.submit(current)
    scheduler._admit()
    for current in requests:
        current.phase, current.consumed, current.generated = "decode", 3, [5]
    return requests


def decode_round(scheduler, nanos, accepted):
    """Plan, time and complete one round; each row accepts up to `accepted` drafts."""
    step = scheduler.plan()
    scheduler.advance_to(scheduler.now_ns + nanos(step))
    results = {}
    for lane in step.lanes:
        offered = lane.tokens - 1
        emitted = min(accepted, offered) + 1
        results[lane.request] = (1, lane.request.consumed + emitted, [8] * emitted, 0, offered, emitted - 1)
    scheduler.complete(step, records(scheduler, step, results), nanos(step))
    return step.batch, step.width, len(step.lanes)


def test_decode_rounds_explore_unmeasured_graphs_then_follow_graph_time_and_acceptance():
    scheduler = Scheduler(layout())
    decoding(scheduler, 2)
    cost = lambda step: (2 + step.width + step.batch) * 1_000_000
    # Every unmeasured graph is explored once before any measured one runs, the most valuable first:
    # both rows at the widest verification in the smallest batch that holds them.
    explored = [decode_round(scheduler, cost, 2) for _ in range(9)]
    assert explored[0] == (2, 3, 2)
    assert sorted(shape[:2] for shape in explored) == [(batch, width) for batch in (1, 2, 4) for width in (1, 2, 3)]
    # Then the measured graph with the best expected value per graph time serves both rows.
    assert decode_round(scheduler, cost, 2) == (2, 3, 2)
    # Rejected drafts lower the expected tokens until the narrowest round is worth the most per time.
    assert [decode_round(scheduler, cost, 0) for _ in range(40)][-1] == (2, 1, 2)
    # Estimates expire after 30 s without an observation, and expired graphs are explored again.
    scheduler.advance_to(scheduler.now_ns + 31_000_000_000)
    assert decode_round(scheduler, cost, 0) == (2, 3, 2)


def test_a_fixed_width_still_chooses_batch_capacity_and_rows_by_value():
    scheduler = Scheduler(layout(fixed_width=2))
    requests = decoding(scheduler, 3)
    # Three rows need batch four; a two-row graph costing a quarter as much carries the two most valuable rows.
    cost = lambda step: {1: 10, 2: 10, 4: 40}[step.batch] * 1_000_000
    shapes = [decode_round(scheduler, cost, 1) for _ in range(12)]
    assert {width for _, width, _ in shapes} == {2} and scheduler.rounds[(4, 2)] == 1
    assert shapes[3:] == [(2, 2, 2)] * 9
    # Fairness prices and output age rotate the rows, so no request waits indefinitely.
    assert all(len(current.generated) > 4 for current in requests)


def test_acceptance_follows_helios_depth_estimates():
    acceptance = Acceptance()
    assert acceptance.expected(1) == 1 and acceptance.expected(3) == 1.75
    acceptance.observe(2, 0, True)
    assert acceptance.depths[0][0] == pytest.approx(.5 - .5 / 3)
    acceptance.observe(0, 0, True)
    assert acceptance.depths[0][1] == 3


def test_layout_selects_the_serving_variant_of_each_graph():
    class Signature:
        def __init__(self, kind, batch, tokens, enabled):
            self.kind, self.batch_capacity, self.token_capacity, self.speculation_enabled = kind, batch, tokens, enabled

    class Tensor:
        def __init__(self, *shape):
            self.shape = shape

        def numel(self):
            return int(np.prod(self.shape))

    class Model:
        max_active = 3
        signatures = {"prefill128": Signature("prefill", 1, 128, True),
                      "prefill128-no-mtp": Signature("prefill", 1, 128, False),
                      "decode4": Signature("decode", 4, 1, False), "init4": Signature("init", 4, 1, True),
                      "speculative4x2": Signature("speculative", 4, 2, True)}
        buffers = {"row_slot_ids": Tensor(4), "page_tables": Tensor(4, 32), "draft_page_tables": Tensor(4, 32),
                   "completion_records": Tensor(2, 4, 24)}
        server_config = {"max_sequence_tokens": 8192, "page_tokens": 256, "pages": 100, "snapshots": 5,
                         "prefill_chunk_size": 128, "model_contract": {"vocabulary": VOCABULARY},
                         "frontend": {"eos_token_ids": [7, 8]}, "speculative_widths": [1, 2],
                         "speculative_fixed_width": None, "state_rows_per_request": 1}

    result = Layout.of(Model())
    assert result.graphs == {("prefill", 1, 128): "prefill128", ("init", 4, 1): "init4",
                             ("speculative", 4, 2): "speculative4x2"}
    assert (result.capacity, result.max_active, result.table_width, result.draft_pages) == (4, 3, 32, 1)
    assert (result.execution_width, result.state_row(2), result.snapshot_row(1)) == (2, 3, 5)
