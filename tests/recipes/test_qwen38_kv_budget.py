"""KV budgeting preserves headroom using measured allocation consumption."""

import weakref

import pytest

from flare.recipes.qwen38.bootstrap import _allocate_page_pools


class Pool:
    pass


class Cache:
    def __init__(self):
        self.capacities = []
        self.owners = []

    def resize_pages_for_preparation(self, capacity):
        pool = Pool()
        self.capacities.append(capacity)
        self.owners.append(weakref.ref(pool))
        return pool


def accounting(monkeypatch, free, caches):
    events = []
    available = iter(free)
    monkeypatch.setattr('torch.cuda.synchronize', lambda device: events.append('fence'))
    monkeypatch.setattr('torch.cuda.mem_get_info', lambda device: (next(available), 102014189568))
    monkeypatch.setattr('torch.cuda.memory_allocated', lambda device: 95545701376)
    monkeypatch.setattr('torch.cuda.memory_reserved', lambda device: 96515129344)

    def release():
        assert events[-1] == 'fence'
        assert all(owner() is None for cache in caches.values() for owner in cache.owners)
        events.append('release')

    monkeypatch.setattr('torch.cuda.empty_cache', release)
    return events


def test_auto_capacity_releases_rejected_pools_before_reallocation(monkeypatch):
    caches = {index: Cache() for index in range(3, 48, 4)}
    events = accounting(monkeypatch, [4293984256, (4 << 30) + (65 << 20)], caches)
    messages = []
    pools, capacity = _allocate_page_pools(caches, 8196, page_bytes=1622016,
        device='cuda:0', headroom_bytes=4 << 30, automatic=True, report=messages.append)
    assert capacity == 8154
    assert events == ['fence', 'release', 'fence']
    assert all(cache.capacities == [8196, 8154] for cache in caches.values())
    assert all(cache.owners[0]() is None and cache.owners[1]() is pools[index]
               for index, cache in caches.items())
    assert messages[-1] == '12.32 GiB per GPU · 521,856 tokens'


def test_explicit_capacity_fails_without_resizing(monkeypatch):
    caches = {3: Cache()}
    events = accounting(monkeypatch, [4293984256], caches)
    with pytest.raises(MemoryError, match=r'requested KV cache \(524,544 tokens, 12.38 GiB per GPU\)'):
        _allocate_page_pools(caches, 8196, page_bytes=1622016,
            device='cuda:0', headroom_bytes=4 << 30, automatic=False, report=lambda _: None)
    assert caches[3].capacities == [8196]
    assert events == ['fence', 'release']


@pytest.mark.parametrize('automatic', [True, False])
def test_sufficient_headroom_keeps_capacity_and_live_pools(monkeypatch, automatic):
    caches = {3: Cache()}
    events = accounting(monkeypatch, [4 << 30], caches)
    pools, capacity = _allocate_page_pools(caches, 8196, page_bytes=1622016,
        device='cuda:0', headroom_bytes=4 << 30, automatic=automatic, report=lambda _: None)
    assert capacity == 8196 and pools[3] is caches[3].owners[0]()
    assert events == ['fence']


def test_insufficient_room_for_one_page_fails_after_release(monkeypatch):
    caches = {3: Cache()}
    events = accounting(monkeypatch, [0], caches)
    with pytest.raises(MemoryError, match='insufficient memory'):
        _allocate_page_pools(caches, 1, page_bytes=1622016,
            device='cuda:0', headroom_bytes=4 << 30, automatic=True, report=lambda _: None)
    assert caches[3].capacities == [1]
    assert events == ['fence', 'release']
