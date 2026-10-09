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
def test_all_eight_recurrent_columns_remain_addressable(slot):
    assert [state_column(slot, count, 44) for count in range(1, 9)] == [slot + 1, *range(44 + slot * 7, 51 + slot * 7)]
    for accepted in (0, 9, True):
        with pytest.raises(ValueError):
            state_column(slot, accepted, 44)


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


def test_committed_state_selects_column_eight_and_normalizes_cache_on_cpu():
    slot, position, count, extra = 3, 75, 8, 9
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
                            extra_base=extra, requests=requests, cache={3: cache})
    builder = SimpleNamespace(requests={slot: {"consumed": position, "pages": [5, 2]}})
    before_cuda = torch.cuda.is_initialized()
    tensors, metadata, checks = committed_state(owner, builder, [slot])
    assert all(checks.values())
    assert torch.equal(tensors["slot3.gdn.0.recurrent"], recurrent[36])
    assert torch.equal(tensors["slot3.gdn.0.convolution"], convolution[4, :, 7:10])
    assert torch.equal(tensors["slot3.ple_convolution"], ple[4, :, 7:16])
    assert torch.equal(tensors["slot3.qsa.3.main_k_cache"], torch.cat((cache.main_k_cache[5], cache.main_k_cache[2, :11])))
    assert tensors["slot3.qsa.3.compressed_k_cache"].shape == (18, 4)
    assert torch.equal(tensors["slot3.qsa.3.raw_unfinished_keys"], cache.raw_k_ring[4, 8:11])
    assert metadata["3"]["raw_anchors"] == {"3": 67}
    cache.raw_logical_positions[4, 8] = 56
    assert not all(committed_state(owner, builder, [slot])[2].values())
    assert torch.cuda.is_initialized() == before_cuda
