"""Prefill capacity configuration and native CLI handoff."""

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


@pytest.mark.parametrize("arguments, expected", [([], 4096), (["--prefill-chunk-size", "3072"], 3072)])
@pytest.mark.parametrize("no_compile", [False, True])
def test_cli_passes_prefill_chunk_to_preparation(monkeypatch, arguments, expected, no_compile):
    from flare.__main__ import main

    configurations = []

    class Model:
        memory_report = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def manifest(self):
            return {}

    def prepare(snapshot, config, *, progress, compile_decoder):
        assert compile_decoder is not no_compile
        configurations.append(config)
        return Model()

    monkeypatch.setattr("flare.recipes.qwen38.bootstrap.prepare_model", prepare)
    assert main(["--prepare-only", *arguments, *(["--no-compile"] if no_compile else [])]) == 0
    assert len(configurations) == 1
    assert configurations[0].prefill_chunk_size == expected


@pytest.mark.parametrize("widths,fixed", [((1, 9), None), ((2, 4), None), ((1, 4, 2), None),
                                         ((1, 2, 2), None), ((1, 2), 8), ((1, True), None)])
def test_speculative_policy_requires_a_prepared_width_catalog(widths, fixed):
    with pytest.raises(ValueError):
        PreparationConfig(mtp=True, speculative_widths=widths, speculative_fixed_width=fixed)


def test_speculative_catalog_covers_each_width_inside_micro_band():
    shapes = PreparationConfig(mtp=True).speculative_shapes
    assert len(shapes) == 14
    assert {width for batch, width in shapes if batch == 1} == set(range(1, 9))
    assert {width for batch, width in shapes if batch == 2} == set(range(1, 5))
    assert {width for batch, width in shapes if batch == 4} == {1, 2}


def test_copy_only_uses_the_verification_catalog_without_mtp_proposals():
    config = PreparationConfig(mtp=True, speculative_proposal_source="copy")
    assert config.speculative_shapes == PreparationConfig(mtp=True).speculative_shapes
    with pytest.raises(ValueError, match="proposal source"):
        PreparationConfig(mtp=True, speculative_proposal_source="unknown")


@pytest.mark.parametrize("source", ["mtp", "copy"])
@pytest.mark.parametrize("history_drafts", [False, True])
def test_cli_preserves_explicit_mtp_measurement_policy(monkeypatch, source, history_drafts):
    from flare.__main__ import main

    captured = []

    class Model:
        memory_report = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def manifest(self):
            return {}

    def prepare(snapshot, config, *, progress, compile_decoder):
        assert compile_decoder
        captured.append(config)
        return Model()

    monkeypatch.setattr("flare.recipes.qwen38.bootstrap.prepare_model", prepare)
    arguments = ["--prepare-only", "--mtp", "--mtp-proposal-source", source, "--mtp-widths", "1", "2", "4",
                 "--mtp-fixed-width", "4"]
    if history_drafts:
        arguments.append("--history-drafts")
    assert main(arguments) == 0
    assert captured[0].mtp
    assert captured[0].speculative_proposal_source == source
    assert captured[0].speculative_widths == (1, 2, 4)
    assert captured[0].speculative_fixed_width == 4
    assert captured[0].history_drafts is history_drafts
