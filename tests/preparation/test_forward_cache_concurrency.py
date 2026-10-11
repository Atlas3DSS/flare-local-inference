"""Concurrent model launches retain their compiled variant until binding finishes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from flare.preparation.aot import ForwardCache


def test_catalog_publication_waits_for_all_bindings(tmp_path, monkeypatch):
    published, release, attempted, second_entered = (Event() for _ in range(4))
    first, second = object.__new__(ForwardCache), object.__new__(ForwardCache)
    first.root = second.root = tmp_path
    artifact = tmp_path / 'forward.bin'

    def prepare(cache, functions, signatures, **kwargs):
        value = functions['pages']
        if value == 64:
            second_entered.set()
        artifact.write_text(str(value))
        if value == 32:
            published.set()
            assert release.wait(5)
        return int(artifact.read_text())

    def competing_launch():
        attempted.set()
        return second.prepare_many({'pages': 64}, {})

    monkeypatch.setattr(ForwardCache, '_prepare_many', prepare)
    with ThreadPoolExecutor(2) as pool:
        left = pool.submit(first.prepare_many, {'pages': 32}, {})
        try:
            assert published.wait(5)
            right = pool.submit(competing_launch)
            assert attempted.wait(5)
            assert not second_entered.wait(.1)
        finally:
            release.set()
        assert left.result(timeout=5) == 32
        assert right.result(timeout=5) == 64


def test_peer_binding_does_not_wait_for_its_root_publication_lock(tmp_path, monkeypatch):
    import torch.distributed as dist

    cache = object.__new__(ForwardCache)
    cache.root = tmp_path
    monkeypatch.setattr(dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(ForwardCache, '_prepare_many', lambda *args, **kwargs: 'bound')
    with cache._publication_lock():
        assert cache.prepare_many({}, {}, coordinator=SimpleNamespace(ranks=(0, 1))) == 'bound'
