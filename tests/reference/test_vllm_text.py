"""CPU reconstruction of vLLM full-vocabulary flat raw-logit records."""

from types import SimpleNamespace

import pytest
import torch

from qualify.vllm_text import reconstruct_logits


VOCABULARY = 10


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("logit reconstruction must remain on CPU")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def flat_scores(*, order=tuple(range(VOCABULARY)), steps=2):
    ids, scores, starts, ends, expected = [], [], [], [], []
    for step in range(steps):
        row = [token * token / 8 - 3 + 10 * step for token in range(VOCABULARY)]
        sampled = (3 + step) % VOCABULARY
        step_ids = [sampled, *order]
        starts.append(len(ids))
        ids.extend(step_ids)
        scores.extend(row[token] for token in step_ids)
        ends.append(len(ids))
        expected.append(torch.tensor(row, dtype=torch.float32))
    return SimpleNamespace(token_ids=ids, logprobs=scores,
                           start_indices=starts, end_indices=ends), expected


@pytest.mark.parametrize("order", [tuple(range(VOCABULARY)), (7, 0, 3, 9, 1, 8, 2, 6, 5, 4)])
def test_full_vocabulary_and_sample_duplicate_reconstruct_exactly(order):
    flat, expected = flat_scores(order=order)
    actual = reconstruct_logits(flat, steps=2, vocabulary=VOCABULARY)
    assert len(actual) == 2
    for row, reference in zip(actual, expected, strict=True):
        assert row.device.type == "cpu"
        assert row.dtype == torch.float32
        assert torch.equal(row, reference)


def test_scored_offsets_select_only_the_declared_rows():
    flat, expected = flat_scores(order=(9, 1, 5, 0, 8, 6, 2, 4, 7, 3))
    flat.token_ids = [-999, -999, *flat.token_ids, -999]
    flat.logprobs = [float("nan"), float("nan"), *flat.logprobs, float("nan")]
    flat.start_indices = [value + 2 for value in flat.start_indices]
    flat.end_indices = [value + 2 for value in flat.end_indices]
    actual = reconstruct_logits(flat, steps=2, vocabulary=VOCABULARY)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("count", [VOCABULARY - 1, VOCABULARY, VOCABULARY + 2])
def test_missing_or_excess_record_count_is_rejected(count):
    flat, _ = flat_scores(steps=1)
    if count > len(flat.token_ids):
        flat.token_ids.append(0)
        flat.logprobs.append(-3.0)
    flat.end_indices[0] = count
    with pytest.raises(ValueError, match="full-vocabulary logits plus the sampled token"):
        reconstruct_logits(flat, steps=1, vocabulary=VOCABULARY)


def test_missing_coverage_rejected_even_with_expected_record_count():
    flat, _ = flat_scores(steps=1)
    flat.token_ids[-1] = 0
    flat.logprobs[-1] = -3.0
    assert len(flat.token_ids) == VOCABULARY + 1
    with pytest.raises(ValueError, match="cover the exact vocabulary"):
        reconstruct_logits(flat, steps=1, vocabulary=VOCABULARY)


@pytest.mark.parametrize("token", [-1, VOCABULARY, 1 << 30])
def test_out_of_range_token_id_is_rejected(token):
    flat, _ = flat_scores(steps=1)
    flat.token_ids[-1] = token
    with pytest.raises(ValueError, match="cover the exact vocabulary"):
        reconstruct_logits(flat, steps=1, vocabulary=VOCABULARY)


def test_unequal_duplicate_cannot_silently_overwrite_a_full_vocabulary_score():
    flat, _ = flat_scores(steps=1)
    assert flat.token_ids[0] == flat.token_ids[4] == 3
    flat.logprobs[0] += 0.125
    with pytest.raises(ValueError, match="inconsistent scores for the duplicated sampled token"):
        reconstruct_logits(flat, steps=1, vocabulary=VOCABULARY)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_score_is_rejected_in_any_step(score):
    flat, _ = flat_scores()
    flat.logprobs[-1] = score
    with pytest.raises(ValueError, match="finite full-vocabulary logits"):
        reconstruct_logits(flat, steps=2, vocabulary=VOCABULARY)


@pytest.mark.parametrize("start_count,end_count,expected_steps", [(2, 2, 1), (1, 2, 2), (2, 1, 2)])
def test_wrong_step_count_or_unpaired_offsets_are_rejected(start_count, end_count, expected_steps):
    flat, _ = flat_scores()
    flat.start_indices = flat.start_indices[:start_count]
    flat.end_indices = flat.end_indices[:end_count]
    with pytest.raises(ValueError, match="different number of scored positions"):
        reconstruct_logits(flat, steps=expected_steps, vocabulary=VOCABULARY)


def test_two_completed_captures_are_retained_before_numerical_failures_are_raised(tmp_path):
    from qualify.vllm_text import capture_cold_repetitions
    import json

    calls = []

    class Engine:
        def reset_prefix_cache(self, *, reset_running_requests, reset_connector):
            assert calls == ['capture0']
            assert not reset_running_requests and not reset_connector
            calls.append('reset')
            return True

    def capture(index, destination):
        assert calls == ([] if index == 0 else ['capture0', 'reset'])
        calls.append(f'capture{index}')
        (destination / 'retained-logits').write_bytes(bytes([index]))
        return {'repetition': index, 'layer_trace_passed': False}

    reports = capture_cold_repetitions(Engine(), count=2, output=tmp_path, capture=capture)
    assert calls == ['capture0', 'reset', 'capture1']
    assert [entry['layer_trace_passed'] for entry in reports] == [False, False]
    assert (tmp_path / 'retained-logits').read_bytes() == b'\x00'
    assert (tmp_path / 'repetition-01/retained-logits').read_bytes() == b'\x01'
    reset = json.loads((tmp_path / 'repetition-01/prefix-cache-reset.json').read_text())
    assert reset['success'] is True and reset['completed_requests'] == 1


def test_failed_prefix_reset_is_retained_and_prevents_the_second_request(tmp_path):
    from qualify.vllm_text import capture_cold_repetitions
    import json

    captures = []
    engine = SimpleNamespace(reset_prefix_cache=lambda **kwargs: False)
    with pytest.raises(RuntimeError, match='prefix-cache reset failed'):
        capture_cold_repetitions(engine, count=2, output=tmp_path,
            capture=lambda index, destination: captures.append(index))
    assert captures == [0]
    reset = json.loads((tmp_path / 'repetition-01/prefix-cache-reset.json').read_text())
    assert reset['success'] is False


def test_single_capture_keeps_original_path_without_prefix_reset(tmp_path):
    from qualify.vllm_text import capture_cold_repetitions

    def forbidden(**kwargs):
        raise AssertionError('the fresh-engine single-request path must stay unchanged')

    reports = capture_cold_repetitions(SimpleNamespace(reset_prefix_cache=forbidden), count=1,
        output=tmp_path, capture=lambda index, path: (index, path))
    assert reports == [(0, tmp_path)]
