"""Tests for serve CLI parsing helpers."""

from serve.cli import _parse_graph_batch_sizes


def test_parse_graph_batch_sizes_defaults_when_enabled():
    assert _parse_graph_batch_sizes(None, enabled=True) == [1, 2, 4, 8]


def test_parse_graph_batch_sizes_dedupes_and_sorts():
    assert _parse_graph_batch_sizes("8,2,2,4", enabled=True) == [2, 4, 8]


def test_parse_graph_batch_sizes_disabled_returns_empty():
    assert _parse_graph_batch_sizes("1,2,4", enabled=False) == []
