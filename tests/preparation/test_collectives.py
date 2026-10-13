"""Collective preparation cannot launch ahead of a peer or mismatched declaration."""

from types import SimpleNamespace

import pytest

from ignition.preparation.collectives import CollectivePreparation, dist


def coordinator(monkeypatch):
    group = object()
    monkeypatch.setattr(dist, "get_backend", lambda actual: "gloo")
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda actual: [4, 7])
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda states, value, *, group: states.__setitem__(slice(None), [value] * len(states)))
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
    assert calls == [{"group": owner.group, "timeout": owner.timeout, "wait_all_ranks": True}]


def test_finished_rank_keeps_participating_until_peers_finish(monkeypatch):
    owner = coordinator(monkeypatch)
    result = object()
    advances, exchanges = [], []
    done = SimpleNamespace(done=True, ready_cache=None, ready_tuning=(), ready_collectives=())
    job = SimpleNamespace(advance=lambda **kwargs: advances.append(kwargs) or done, result=lambda: result)
    session = SimpleNamespace(begin=lambda requests, **kwargs: job)
    peers = [(False, (), None), (False, (), None), (True, (), None)]

    def gather(states, local, *, group):
        exchanges.append(local)
        states[:] = [local, peers.pop(0)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert owner.prepare(session, ()) is result
    assert len(advances) == 1 and len(exchanges) == 3


def test_mixed_job_authorizes_collective_only_after_every_rank_is_ready(monkeypatch):
    owner = coordinator(monkeypatch)
    ready = progress()
    ready.done = False
    done = SimpleNamespace(done=True, ready_cache=None, ready_tuning=(), ready_collectives=())
    local_states = [ready, ready, done]
    peers = [(False, (), None), (False, (("reduce", (4, 7)),), None),
             (False, (), None), (True, (), None)]
    keys = []
    job = SimpleNamespace(advance=lambda collective_key: keys.append(collective_key) or local_states.pop(0),
                          result=lambda: "prepared")
    monkeypatch.setattr(dist, "all_gather_object", lambda states, local, **kwargs: states.__setitem__(slice(None), [local, peers.pop(0)]))
    assert owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ()) == "prepared"
    assert keys == [None, None, "reduce"]


@pytest.mark.parametrize("peer", [(False, (("reduce", (4, 7)),), None),
                                  (False, (), "ValueError: bad weight geometry")])
def test_finished_rank_releases_trial_resources_on_peer_failure(monkeypatch, peer):
    owner = coordinator(monkeypatch)
    closed = []
    result = SimpleNamespace(close=lambda: closed.append(True))
    done = SimpleNamespace(done=True, ready_cache=None, ready_tuning=(), ready_collectives=())
    job = SimpleNamespace(advance=lambda **kwargs: done, result=lambda: result)
    monkeypatch.setattr(dist, "all_gather_object", lambda states, local, **kwargs: states.__setitem__(slice(None), [local, peer]))
    with pytest.raises(RuntimeError, match="rank (finished|preparation failed)"):
        owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ())
    assert closed == [True]


def test_local_preparation_failure_is_exchanged_before_cleanup(monkeypatch):
    owner = coordinator(monkeypatch)
    closed, exchanged = [], []

    def advance(**kwargs):
        raise ValueError("bad weight geometry")

    def gather(states, local, **kwargs):
        exchanged.append(local)
        states[:] = [local, (False, (), None)]

    job = SimpleNamespace(advance=advance, close=lambda: closed.append(True))
    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="bad weight geometry") as failure:
        owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ())
    assert isinstance(failure.value.__cause__, ValueError)
    assert exchanged == [(False, (), "ValueError: bad weight geometry")]
    assert closed == [True]


def test_reports_ranks_on_this_host_and_whether_the_group_spans_hosts(monkeypatch):
    owner = coordinator(monkeypatch)
    assert owner.local_ranks == 2 and not owner.spans_hosts
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda states, value, *, group: states.__setitem__(slice(None), [value, "another-boot"]))
    owner = CollectivePreparation(object())
    assert owner.local_ranks == 1 and owner.spans_hosts


def test_peers_adopt_rank_zero_selections_and_disagreement_fails(monkeypatch):
    owner = coordinator(monkeypatch)
    cache = SimpleNamespace(records={"key": "peer"}, _agreed_records=None)
    monkeypatch.setattr(dist, "get_rank", lambda: 7)
    monkeypatch.setattr(dist, "broadcast_object_list",
                        lambda values, *, src, group: values.__setitem__(0, {"key": "zero"}))
    owner.share_selections(cache)
    assert cache._agreed_records == {"key": "zero"} and cache.records == {"key": "peer"}

    def selection(config):
        return [{"name": "layer0.mlp", "component": "gemm", "config": config}]

    monkeypatch.setattr(dist, "all_gather_object", lambda states, value, *, group: states.__setitem__(
        slice(None), [{"layer0.mlp": ("gemm", '{"tile": 64}')}, value]))
    owner.agree_selections(selection({"tile": 64}))
    with pytest.raises(RuntimeError, match="1 kernel selections differ"):
        owner.agree_selections(selection({"tile": 128}))
