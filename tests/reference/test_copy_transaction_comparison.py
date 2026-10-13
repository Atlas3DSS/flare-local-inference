"""CPU contracts for model-state comparisons; these do not replace GPU evidence."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from qualify.copy_transaction_comparison import (
    byte_equal, case_plan, committed_state, compare_states, logical_pages, parse_shape, state_column,
)
from qualify.speculative_state_parity import LEGAL_SHAPES


@pytest.mark.parametrize("batch,width", LEGAL_SHAPES)
def test_equal_consumed_positions_snapshot_restore_and_width_transition(batch, width):
    plan = case_plan(batch, width)
    arms = {}
    for arm in ("w1", "wide"):
        requests, snapshots, observations = {}, {}, {}
        sequence = [(arm, plan["common"] + plan[arm])]
        if arm == "wide":
            sequence += [("restored", plan["restore"] + plan["wide"])]
        for label, commands in sequence:
            for command in commands:
                kind = command["kind"]
                for slot in command["slots"]:
                    if kind == "init":
                        requests[slot] = {"consumed": 0, "generated": 0, "pending": False, "last_width": 1}
                    elif kind == "prefill":
                        requests[slot].update(consumed=command["length"], generated=1, pending=True)
                    elif kind == "speculative":
                        request = requests[slot]
                        assert request["pending"]
                        request["consumed"] += command["width"]
                        request["generated"] += command["width"]
                        request["last_width"] = command["width"]
                    elif kind == "snapshot":
                        assert requests[slot]["consumed"] == 64
                        snapshots[command["snapshot"]] = requests[slot]["consumed"]
                    elif kind == "restore":
                        requests[slot] = {"consumed": snapshots[command["snapshot"]],
                                          "generated": 0, "pending": False, "last_width": 1}
                    elif kind == "resume":
                        assert not requests[slot]["pending"]
                        requests[slot].update(generated=1, pending=True)
                if "observe" in command:
                    stage = command["observe"]
                    observations[label, stage] = [dict(requests[slot]) for slot in plan["slots"]]
        arms[arm] = observations
    for stage, position in plan["positions"].items():
        baseline = arms["w1"]["w1", stage]
        wide = arms["wide"]["wide", stage]
        restored = arms["wide"]["restored", stage]
        assert all(a["consumed"] == b["consumed"] == c["consumed"] == position
                   for a, b, c in zip(baseline, wide, restored))
        assert [a["generated"] for a in baseline] == [b["generated"] for b in wide]
        assert all(b["generated"] == c["generated"] + 1 for b, c in zip(wide, restored))
    assert all(x["last_width"] == width for x in arms["wide"]["wide", "transaction"])
    assert all(x["last_width"] == 1 for x in arms["wide"]["wide", "following_w1"])
    assert plan["accepted_drafts_per_lane"] == width - 1
    assert plan["unexercised_rejection_prefixes"] == list(range(width - 1))


def test_shape_contract_has_all_fourteen_without_duplicates():
    assert len(LEGAL_SHAPES) == len(set(LEGAL_SHAPES)) == 14
    assert {parse_shape(f"{b}x{w}") for b, w in LEGAL_SHAPES} == set(LEGAL_SHAPES)
    for value in ("2x8", "3x2", "1x9", "0x1", "1", "1xfoo"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_shape(value)
    with pytest.raises(ValueError):
        case_plan(2, 8)


@pytest.mark.parametrize("slot", range(4))
def test_every_accepted_count_resumes_from_the_live_recurrent_row(slot):
    assert [state_column(slot, count) for count in range(1, 9)] == [slot + 1] * 8
    for accepted in (0, 9, True):
        with pytest.raises(ValueError):
            state_column(slot, accepted)


def test_cross_width_errors_are_measured_and_integer_errors_still_fail():
    reference = {"recurrent": torch.tensor([1., 2.]), "consumed": torch.tensor([72])}
    changed = {"recurrent": torch.tensor([3., 4.]), "consumed": torch.tensor([72])}
    cross = compare_states(reference, changed, exact=False)
    assert cross["passed"]
    assert cross["measurements"]["recurrent"]["relative_l2"] > 1
    assert "recurrent.bitwise" not in cross["checks"]
    assert not compare_states(reference, changed, exact=True)["passed"]
    changed["consumed"] += 1
    assert not compare_states(reference, changed, exact=False)["passed"]


def test_byte_identity_keeps_signed_zero_and_nonfinite_are_failures():
    assert not byte_equal(torch.tensor([0.]), torch.tensor([-0.]))
    result = compare_states({"state": torch.tensor([0.])}, {"state": torch.tensor([float("nan")])}, exact=False)
    assert not result["passed"]
    assert "relative_l2" not in result["measurements"]["state"]
    empty = compare_states({"raw": torch.empty(0, 4)}, {"raw": torch.empty(0, 4)}, exact=True)
    assert empty["passed"] and empty["measurements"]["raw"]["max_abs"] == 0


def test_logical_pages_never_use_physical_allocation_order_or_uncommitted_tail():
    cache = torch.arange(8 * 4 * 2).reshape(8, 4, 2)
    actual = logical_pages(cache, [5, 2, 7], 6)
    assert torch.equal(actual, torch.cat((cache[5], cache[2, :2])))
    assert actual.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr()
    assert logical_pages(cache, [5], 0).shape == (0, 2)
    with pytest.raises(ValueError):
        logical_pages(cache, [5], 5)


def test_committed_state_reads_the_live_row_and_normalizes_cache_on_cpu():
    slot, position, count = 3, 75, 8
    recurrent = torch.arange(40 * 4, dtype=torch.float32).reshape(40, 1, 2, 2)
    convolution = torch.arange(40 * 2 * 10, dtype=torch.bfloat16).reshape(40, 2, 10)
    ple = torch.arange(40 * 2 * 16, dtype=torch.bfloat16).reshape(40, 2, 16)
    regions = {"last_hidden": torch.zeros(40, 3, dtype=torch.bfloat16),
               "consumed_tokens": torch.zeros(40, 1, dtype=torch.int64),
               "ple_history": torch.zeros(40, 2, dtype=torch.int64),
               "rope_delta": torch.zeros(40, 1, dtype=torch.int64),
               "target_residual": torch.zeros(40, 5, dtype=torch.bfloat16),
               "target_residual_position": torch.zeros(40, 3, dtype=torch.int64)}
    regions["consumed_tokens"][slot + 1] = position
    pool = SimpleNamespace(gdn={0: SimpleNamespace(recurrent=recurrent, convolution=convolution)},
                           ple_convolution=ple, region=regions.__getitem__)
    cache = SimpleNamespace(main_k_cache=torch.arange(8 * 64 * 2, dtype=torch.float32).reshape(8, 64, 1, 2),
        main_v_cache=torch.full((8, 64, 1, 2), .25),
        compressed_k_cache=torch.arange(8 * 16 * 4, dtype=torch.float32).reshape(8, 16, 4),
        raw_k_ring=torch.arange(5 * 16 * 4, dtype=torch.float32).reshape(5, 16, 4),
        raw_logical_positions=torch.full((5, 16), -1, dtype=torch.int64),
        raw_rope_positions=torch.full((5, 16, 3), -1, dtype=torch.int64),
        raw_interval_start_positions=torch.full((5,), 67, dtype=torch.int64))
    for p in range(72, 75):
        cache.raw_logical_positions[slot + 1, p % 16] = p
        cache.raw_rope_positions[slot + 1, p % 16] = p
    requests = SimpleNamespace(pending_token=torch.ones(4, dtype=torch.int64), histogram=torch.ones(4, 12, dtype=torch.int32),
                               status=torch.ones(4, dtype=torch.int32), error_code=torch.zeros(4, dtype=torch.int32))
    owner = SimpleNamespace(pool=pool, previous_accepted=torch.tensor([1, 1, 1, count]),
                            requests=requests, cache={3: cache})
    builder = SimpleNamespace(requests={slot: {"consumed": position, "pages": [5, 2]}})
    before_cuda = torch.cuda.is_initialized()
    tensors, metadata, checks = committed_state(owner, builder, [slot])
    assert all(checks.values())
    assert torch.equal(tensors["slot3.gdn.0.recurrent"], recurrent[slot + 1])
    assert torch.equal(tensors["slot3.gdn.0.convolution"], convolution[4, :, 7:10])
    assert torch.equal(tensors["slot3.ple_convolution"], ple[4, :, 7:16])
    assert torch.equal(tensors["slot3.qsa.3.main_k_cache"], torch.cat((cache.main_k_cache[5], cache.main_k_cache[2, :11])))
    assert tensors["slot3.qsa.3.compressed_k_cache"].shape == (18, 4)
    assert torch.equal(tensors["slot3.qsa.3.raw_unfinished_keys"], cache.raw_k_ring[4, 8:11])
    assert metadata["3"]["raw_anchors"] == {"3": 67}
    cache.raw_logical_positions[4, 8] = 56
    assert not all(committed_state(owner, builder, [slot])[2].values())
    assert torch.cuda.is_initialized() == before_cuda


@pytest.mark.parametrize("batch,width", LEGAL_SHAPES)
def test_selected_snapshot_preserves_wide_column_before_private_restore(tmp_path, batch, width):
    from tests.reference.test_speculative_state_parity import make_builder
    from qualify.copy_transaction_comparison import private_restore_checks

    builder = make_builder(tmp_path)
    plan = case_plan(batch, width)
    selected = plan["selected_restore"]
    assert selected["prepare"] == plan["common"] + [plan["wide"][0]]
    assert selected["snapshot_ids"] == list(range(batch, 2 * batch))
    assert not set(selected["snapshot_ids"]) & {operation["snapshot"] for operation in plan["common"] if operation["kind"] == "snapshot"}
    assert all(a != b for a, b in zip(plan["slots"], selected["slots"]))
    source_pages = None
    stage_positions = {}
    before_cuda = torch.cuda.is_initialized()
    for operation in (selected["prepare"] + selected["snapshots"]
                      + selected["restore"] + selected["continuation"]):
        command = builder.prepare(operation)
        kind = operation["kind"]
        if operation in selected["snapshots"]:
            assert builder.previous[operation["slots"][0]] == width
            assert builder.host["state_expected_consumed"].tolist() == [64 + width, -1, -1, -1]
            assert builder.host["state_accepted_tokens"].eq(0).all()
            assert builder.host["state_destination_slots"][0] == 5 + operation["snapshot"]
            source_pages = source_pages or [list(builder.requests[s]["pages"]) for s in plan["slots"]]
        if kind == "restore":
            assert len([s for s in selected["snapshot_ids"] if s in builder.snapshots]) == batch
            assert builder.host["state_expected_consumed"][0] == 64 + width
            assert builder.host["state_source_slots"][0] == 5 + operation["snapshot"]
            assert builder.host["state_destination_slots"][0] == operation["slots"][0] + 1
            assert builder.host["state_accepted_tokens"].tolist() == [1, 0, 0, 0]
        records = []
        for slot in range(4):
            record = [0] * 24
            record[4] = record[12] = -1
            record[16:] = [-1] * 8
            if slot in operation["slots"]:
                request = builder.requests[slot]
                count = operation["width"] if kind == "speculative" else int(kind in ("prefill", "resume"))
                consumed = request["consumed"] + (count if kind == "speculative" else operation.get("length", 0))
                record[:8] = [request["generation"], command["sequence"], consumed,
                              request["generated"] + count, builder.token if count else -1, 1, 0, 0]
                record[8], record[15] = count, operation.get("width", 1)
                if kind == "speculative":
                    record[9:12] = [count - 1, count - 1, (1 << (count - 1)) - 1]
                    record[13] = -1
                record[16:16 + count] = [builder.token] * count
            records.append(record)
        checks, _ = builder.complete(command, records)
        assert all(checks.values()), (command, checks)
        if operation in selected["continuation"]:
            stage_positions[operation["observe"]] = [builder.requests[s]["consumed"] for s in selected["slots"]]
            assert all(builder.previous[s] == 1 for s in selected["slots"])
    assert stage_positions == {"transaction": [64 + width] * batch, "following_w1": [65 + width] * batch}
    restored_pages = [builder.requests[slot]["pages"] for slot in selected["slots"]]
    assert all(private_restore_checks(source_pages, restored_pages, 64 + width).values())
    assert all(builder.snapshots[index]["tokens"] == builder.fixture["prompt"][:63] + [builder.token] * (1 + width)
               for index in selected["snapshot_ids"])
    assert torch.cuda.is_initialized() == before_cuda


def test_physical_slot_remapping_keeps_all_regions_and_rejects_ambiguous_mapping():
    from qualify.copy_transaction_comparison import remap_state_slots

    tensors = {"slot0.gdn.7.recurrent": torch.tensor([1.]), "slot2.request.histogram": torch.tensor([2])}
    actual = remap_state_slots(tensors, [0, 2], [3, 0])
    assert actual.keys() == {"slot3.gdn.7.recurrent", "slot0.request.histogram"}
    assert actual["slot3.gdn.7.recurrent"] is tensors["slot0.gdn.7.recurrent"]
    for slots, reference in (([0, 0], [2, 3]), ([0, 2], [3, 3]), ([0], [3])):
        with pytest.raises(ValueError):
            remap_state_slots(tensors, slots, reference)


def test_private_restore_gate_rejects_source_tail_and_cross_request_aliases():
    from qualify.copy_transaction_comparison import private_restore_checks

    source = [[3, 4, 5], [10, 11, 12]]
    assert all(private_restore_checks(source, [[3, 20, 21], [10, 22, 23]], 72).values())
    assert not private_restore_checks(source, [[3, 4, 21], [10, 22, 23]], 72)["private_pages_distinct_from_sources"]
    assert not private_restore_checks(source, [[3, 20, 21], [10, 20, 23]], 72)["private_pages_distinct_between_requests"]
    assert not private_restore_checks(source, [[10, 20, 21], [3, 22, 23]], 72)["full_pages_shared"]
    assert not private_restore_checks(source, [[3], [10, 22]], 72)["private_tail_present"]


def test_source_fingerprints_detect_hybrid_shared_partial_and_raw_corruption():
    from qualify.copy_transaction_comparison import storage_fingerprints

    cache = SimpleNamespace(**{name: torch.zeros((8, 4, 2), dtype=torch.bfloat16)
        for name in ("main_k_cache", "main_v_cache", "compressed_k_cache", "raw_k_ring")},
        raw_logical_positions=torch.zeros((8, 4), dtype=torch.int64),
        raw_rope_positions=torch.zeros((8, 4, 3), dtype=torch.int64),
        raw_interval_start_positions=torch.zeros(8, dtype=torch.int64))
    owner = SimpleNamespace(pool=SimpleNamespace(backing=torch.zeros((10, 11), dtype=torch.uint8)), cache={3: cache})
    kwargs = {"rows": [6, 9], "pages": [3, 4], "raw_slots": [1]}
    baseline = storage_fingerprints(owner, **kwargs)
    owner.pool.backing[9, -1] = 1
    cache.main_k_cache[3, -1, -1] = 1
    cache.main_v_cache[4, 0, 0] = 1
    cache.raw_logical_positions[1, -1] = 7
    changed = storage_fingerprints(owner, **kwargs)
    assert {key for key in baseline if baseline[key] != changed[key]} == {
        "hybrid.9", "qsa.3.main_k_cache.page3", "qsa.3.main_v_cache.page4", "qsa.3.raw_logical_positions.slot1"}
    cache.main_k_cache[7].fill_(2)
    assert storage_fingerprints(owner, **kwargs) == changed
