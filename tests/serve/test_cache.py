"""Tests for serve/cache/ — page pool and KV cache manager."""

import pytest
import torch

from serve.cache.page_pool import PagePool
from serve.cache.kv_cache import KVCacheManager, RequestKVState


# -- PagePool --------------------------------------------------------------


def test_page_pool_alloc_and_free():
    pool = PagePool(
        num_pages=16, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    assert pool.num_free == 16

    pages = pool.alloc(4)
    assert len(pages) == 4
    assert pool.num_free == 12

    pool.free(pages)
    assert pool.num_free == 16


def test_page_pool_oom():
    pool = PagePool(
        num_pages=4, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    pool.alloc(4)
    with pytest.raises(RuntimeError, match="OOM"):
        pool.alloc(1)


def test_page_pool_utilization():
    pool = PagePool(
        num_pages=10, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    assert pool.utilization == 0.0
    pool.alloc(5)
    assert pool.utilization == pytest.approx(0.5)
    pool.alloc(5)
    assert pool.utilization == pytest.approx(1.0)


def test_page_pool_multi_layer():
    pool = PagePool(
        num_pages=8, num_layers=3, kv_heads=4, head_dim=128, device="cpu"
    )
    assert len(pool.k_cache) == 3
    assert len(pool.v_cache) == 3
    assert pool.k_cache[0].shape == (8, 64, 4, 128)


def test_page_pool_estimate():
    # FP8: 1 byte per element.
    # Per page per layer: 64 * 4 * 128 * 1 * 2 (K+V) = 65536 bytes.
    # 2 layers: 131072 bytes per page.
    n = PagePool.estimate_num_pages(
        memory_bytes=131072 * 10,
        num_layers=2,
        kv_heads=4,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
    )
    assert n == 10


# -- RequestKVState --------------------------------------------------------


def test_request_kv_state_pages_needed():
    state = RequestKVState(request_id=0)
    # Empty state, need 1 page for 1 token.
    assert state.pages_needed(1) == 1
    # Need 1 page for up to 64 tokens.
    assert state.pages_needed(64) == 1
    # Need 2 pages for 65 tokens.
    assert state.pages_needed(65) == 2


def test_request_kv_state_pages_needed_with_existing():
    state = RequestKVState(request_id=0, page_ids=[0], cache_len=32)
    # 32 tokens in 1 page (capacity 64). 32 more fit in existing page.
    assert state.pages_needed(32) == 0
    # 33 more needs a new page.
    assert state.pages_needed(33) == 1


# -- KVCacheManager --------------------------------------------------------


def test_kv_cache_manager_lifecycle():
    pool = PagePool(
        num_pages=100, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    mgr.allocate_request(1)
    mgr.extend_request(1, new_tokens=100)
    state = mgr.get_state(1)
    assert state.cache_len == 100
    assert state.num_pages == 2  # ceil(100/64) = 2.

    mgr.extend_request(1, new_tokens=28)
    assert state.cache_len == 128
    assert state.num_pages == 2  # 128 fits in 2 pages exactly.

    mgr.extend_request(1, new_tokens=1)
    assert state.cache_len == 129
    assert state.num_pages == 3  # Needs a third page.

    mgr.free_request(1)
    assert mgr.num_active == 0
    assert pool.num_free == 100


def test_kv_cache_manager_evict_lru():
    pool = PagePool(
        num_pages=4, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    mgr.allocate_request(1)
    mgr.extend_request(1, new_tokens=64)  # 1 page.
    mgr.allocate_request(2)
    mgr.extend_request(2, new_tokens=64)  # 1 page.
    mgr.allocate_request(3)
    mgr.extend_request(3, new_tokens=64)  # 1 page.
    assert pool.num_free == 1

    # Evict oldest (request 1).
    evicted_id = mgr.evict_lru()
    assert evicted_id == 1
    assert pool.num_free == 2
    assert mgr.num_active == 2


def test_kv_cache_manager_lru_order_updated_on_extend():
    pool = PagePool(
        num_pages=10, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    mgr.allocate_request(1)
    mgr.extend_request(1, new_tokens=64)
    mgr.allocate_request(2)
    mgr.extend_request(2, new_tokens=64)

    # Touch request 1 again — it moves to end of LRU.
    mgr.extend_request(1, new_tokens=1)

    # Evict should now evict request 2 (oldest).
    evicted_id = mgr.evict_lru()
    assert evicted_id == 2


def test_kv_cache_manager_build_page_table():
    pool = PagePool(
        num_pages=100, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    mgr.allocate_request(1)
    mgr.extend_request(1, new_tokens=128)  # 2 pages.
    mgr.allocate_request(2)
    mgr.extend_request(2, new_tokens=64)  # 1 page.

    table = mgr.build_page_table([1, 2], device="cpu")
    assert table.shape == (2, 2)  # max_pages = 2.
    assert table.dtype == torch.int32
    # Request 2 only has 1 page, so table[1, 1] should be 0 (padding).
    assert table[1, 1].item() == 0


def test_kv_cache_manager_build_cache_seqlens():
    pool = PagePool(
        num_pages=100, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    mgr.allocate_request(1)
    mgr.extend_request(1, new_tokens=100)
    mgr.allocate_request(2)
    mgr.extend_request(2, new_tokens=50)

    seqlens = mgr.build_cache_seqlens([1, 2], device="cpu")
    assert seqlens.tolist() == [100, 50]


def test_kv_cache_manager_build_cu_seqlens_q():
    pool = PagePool(
        num_pages=100, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    # Decode: all q_seqlens = 1.
    cu = mgr.build_cu_seqlens_q([1, 1, 1], device="cpu")
    assert cu.tolist() == [0, 1, 2, 3]

    # Extend: variable q_seqlens.
    cu = mgr.build_cu_seqlens_q([10, 5, 20], device="cpu")
    assert cu.tolist() == [0, 10, 15, 35]


def test_kv_cache_manager_try_alloc_or_evict():
    pool = PagePool(
        num_pages=4, num_layers=1, kv_heads=4, head_dim=128, device="cpu"
    )
    mgr = KVCacheManager(pool)

    mgr.allocate_request(1)
    mgr.extend_request(1, new_tokens=64)  # 1 page.
    mgr.allocate_request(2)
    mgr.extend_request(2, new_tokens=64)  # 1 page.
    mgr.allocate_request(3)
    mgr.extend_request(3, new_tokens=64)  # 1 page.
    mgr.allocate_request(4)
    mgr.extend_request(4, new_tokens=64)  # 1 page.
    assert pool.num_free == 0

    # Need 2 pages — must evict 2 requests.
    evicted = mgr.try_alloc_or_evict(pages_needed=2)
    assert len(evicted) == 2
    assert evicted == [1, 2]
    assert pool.num_free == 2
