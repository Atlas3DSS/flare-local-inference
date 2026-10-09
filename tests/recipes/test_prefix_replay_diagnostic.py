"""CPU acceptance checks for the complete-model prefix diagnostic receipt."""

import json

import pytest

from scripts.diagnose_prefix_replay import equality_summary, write_equality_receipt


def test_nested_attention_and_same_argmax_failures_are_reported(tmp_path):
    comparisons = {
        "cold_vs_tail": {
            "gdn.0.recurrent": {"equal": True, "relative_l2": 0.0},
            "logits": {"equal": False, "argmax_left": 9419, "argmax_right": 9419},
            "qsa": {
                "logical_rows": [[0, 448], [0, 449]],
                "tensors": {
                    "query": {"equal": True, "different_elements": 0},
                    "attended": {"equal": False, "different_elements": 7201},
                },
            },
            "metadata": {"enabled": False, "same_argmax": True, "count": 0},
        },
    }
    receipt = {"comparisons": comparisons, "tensor_sha256": "retained tensor digest"}
    output = tmp_path / "receipt.json"

    assert write_equality_receipt(receipt, output, require_bitwise=True) == 1
    saved = json.loads(output.read_text())
    assert saved["comparisons"] == comparisons
    assert saved["tensor_sha256"] == "retained tensor digest"
    assert saved["require_bitwise"] is True
    assert saved["equality_summary"] == {
        "comparison_count": 1,
        "equal_predicates": 4,
        "passed_predicates": 2,
        "failed_predicates": 2,
        "failed_paths": [
            "comparisons.cold_vs_tail.logits.equal",
            "comparisons.cold_vs_tail.qsa.tensors.attended.equal",
        ],
        "passed": False,
    }


def test_default_diagnostic_preserves_success_exit_with_failed_comparison(tmp_path):
    receipt = {"comparisons": {"cold_vs_tail": {"logits": {"equal": False}}}}
    output = tmp_path / "receipt.json"
    assert write_equality_receipt(receipt, output) == 0
    saved = json.loads(output.read_text())
    assert saved["require_bitwise"] is False
    assert saved["equality_summary"]["passed"] is False


@pytest.mark.parametrize("comparisons", [{}, {"metadata": {"same_argmax": True, "rows": []}}])
def test_no_equality_predicates_cannot_pass_gate(tmp_path, comparisons):
    output = tmp_path / "receipt.json"
    assert write_equality_receipt({"comparisons": comparisons}, output, require_bitwise=True) == 1
    summary = json.loads(output.read_text())["equality_summary"]
    assert summary["equal_predicates"] == 0
    assert summary["failed_paths"] == []
    assert summary["passed"] is False


def test_all_nested_equalities_pass_gate(tmp_path):
    comparisons = {"cold_vs_tail": {
        "logits": {"equal": True},
        "qsa": {"tensors": {"attended": {"equal": True}}},
    }}
    assert equality_summary(comparisons)["equal_predicates"] == 2
    assert write_equality_receipt({"comparisons": comparisons}, tmp_path / "receipt.json",
                                  require_bitwise=True) == 0
