"""Collective preparation cannot launch ahead of a peer or mismatched declaration."""

from types import SimpleNamespace

import pytest

from flare.preparation.collectives import CollectivePreparation, dist


def coordinator(monkeypatch):
    group = object()
    monkeypatch.setattr(dist, "get_backend", lambda actual: "gloo")
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda actual: [4, 7])
    return CollectivePreparation(group)


def progress(key="reduce", ranks=(4, 7)):
    return SimpleNamespace(ready_cache=None, ready_tuning=(),
        ready_collectives=(SimpleNamespace(key=key, ranks=ranks),))


def test_waits_for_compiling_peer_before_authorizing_kernel(monkeypatch):
    owner = coordinator(monkeypatch)
    peers = [(), (("reduce", (4, 7)),)]

    def gather(states, pending, *, group):
        assert group is owner.group
        states[:] = [pending, peers.pop(0)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert owner(progress()) is None
    assert owner(progress()) == "reduce"


def test_mismatched_rank_operations_fail_before_kernel_entry(monkeypatch):
    owner = coordinator(monkeypatch)

    def gather(states, pending, *, group):
        states[:] = [pending, (("another-reduction", (4, 7)),)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="declarations differ across ranks"):
        owner(progress())


def test_entry_barrier_rejects_incomplete_group(monkeypatch):
    owner = coordinator(monkeypatch)
    with pytest.raises(ValueError, match="does not span its control group"):
        owner.barrier("reduce", (4,))


def test_entry_barrier_preserves_timeout_and_waits_for_every_rank(monkeypatch):
    owner = coordinator(monkeypatch)
    calls = []
    monkeypatch.setattr(dist, "monitored_barrier", lambda **kwargs: calls.append(kwargs))
    owner.barrier("reduce", (4, 7))
    assert calls == [dict(group=owner.group, timeout=owner.timeout, wait_all_ranks=True)]
