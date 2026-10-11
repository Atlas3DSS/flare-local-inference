"""Prefill capacity and speculative catalog configuration."""

import pytest

from flare.recipes.qwen38.bootstrap import PreparationConfig


@pytest.mark.parametrize("size, expected", [
    (64, (64,)),
    (512, (128, 256, 512)),
    (3072, (128, 256, 512, 1024, 2048, 3072)),
    (4096, (128, 256, 512, 1024, 2048, 4096)),
    (8192, (128, 256, 512, 1024, 2048, 4096, 8192)),
])
def test_prefill_catalog_covers_aligned_limit_and_partial_tails(size, expected):
    config = PreparationConfig(prefill_chunk_size=size)
    assert config.prefill_capacities == expected
    for tokens in range(1, size + 1):
        assert next(capacity for capacity in config.prefill_capacities if capacity >= tokens) <= size


@pytest.mark.parametrize("size", [True, 4096.0, "4096"])
def test_prefill_chunk_requires_integer(size):
    with pytest.raises(TypeError, match="prefill_chunk_size"):
        PreparationConfig(prefill_chunk_size=size)


@pytest.mark.parametrize("size", [0, 16, 65, 4097, 262208])
def test_prefill_chunk_requires_page_alignment_and_context_bound(size):
    with pytest.raises(ValueError, match="prefill chunk size"):
        PreparationConfig(prefill_chunk_size=size)


def test_prefill_chunk_must_fit_explicit_context():
    with pytest.raises(ValueError, match="prefill chunk size"):
        PreparationConfig(context=2048, prefill_chunk_size=4096)
    assert PreparationConfig(context=64, prefill_chunk_size=64).prefill_capacities == (64,)


@pytest.mark.parametrize("widths,fixed", [((1, 9), None), ((2, 4), None), ((1, 4, 2), None),
                                         ((1, 2, 2), None), ((1, 2), 8), ((1, True), None)])
def test_speculative_policy_requires_a_prepared_width_catalog(widths, fixed):
    with pytest.raises(ValueError):
        PreparationConfig(mtp=True, speculative_widths=widths, speculative_fixed_width=fixed)


@pytest.mark.parametrize("widths,budget,expected", [
    ((1, 2, 3, 4), 4, ((1, 4), (1, 3), (2, 1), (4, 1))),
    ((1, 2, 3, 4), 8, ((1, 4), (1, 3), (1, 2), (1, 1), (2, 4), (2, 3), (2, 2), (4, 1))),
    ((1, 2, 3, 4), 12, tuple((b, w) for b in (1, 2, 4) for w in (4, 3, 2, 1))),
    (tuple(range(1, 9)), 24, tuple((b, w) for b in (1, 2, 4) for w in range(8, 0, -1))),
])
def test_speculative_graph_budget_prioritizes_small_batches_and_largest_widths(widths, budget, expected):
    config = PreparationConfig(mtp=True, speculative_widths=widths, speculative_graph_budget=budget)
    assert config.speculative_shapes == expected
    assert len(config.speculative_shapes) <= budget


def test_copy_only_uses_the_verification_catalog_without_mtp_proposals():
    config = PreparationConfig(mtp=True, speculative_proposal_source="copy")
    assert config.speculative_shapes == PreparationConfig(mtp=True).speculative_shapes
    with pytest.raises(ValueError, match="proposal source"):
        PreparationConfig(mtp=True, speculative_proposal_source="unknown")


@pytest.mark.parametrize("drafts", range(8))
def test_fixed_drafting_captures_exactly_one_graph_per_batch(drafts):
    from flare.preparation.execution import GraphSignature

    config = PreparationConfig(mtp=True, speculative_widths=tuple(range(1, 9)),
                               speculative_fixed_width=drafts + 1)
    assert config.speculative_shapes == ((1, drafts + 1), (2, drafts + 1), (4, drafts + 1))
    for batch, width in config.speculative_shapes:
        GraphSignature("speculative", batch, width)


@pytest.mark.parametrize("budget", [0, 2, 57, True, 4.0])
def test_speculative_graph_budget_requires_a_capture_for_each_batch(budget):
    with pytest.raises(ValueError, match="graph budget"):
        PreparationConfig(mtp=True, speculative_graph_budget=budget)


def test_fixed_three_with_four_graph_budget_keeps_three_drafts_at_every_batch():
    config = PreparationConfig(mtp=True, speculative_widths=(1, 2, 3, 4, 5),
        speculative_fixed_width=4, speculative_graph_budget=4)
    assert config.speculative_shapes == ((1, 4), (2, 4), (4, 4))
