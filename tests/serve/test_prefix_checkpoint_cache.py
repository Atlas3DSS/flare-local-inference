"""Tests for PrefixCheckpointCache."""

from serve.cache.prefix_checkpoint_cache import PrefixCheckpointCache


class MockPagePool:
    def __init__(self, num_pages=32):
        self._free = list(range(num_pages))

    def alloc(self, n):
        if n > len(self._free):
            raise RuntimeError("OOM")
        result = self._free[-n:]
        del self._free[-n:]
        return result

    def free(self, page_ids):
        self._free.extend(page_ids)

    @property
    def num_free(self):
        return len(self._free)


def _page_tokens(page_idx: int) -> list[int]:
    start = page_idx * 1000
    return list(range(start, start + 64))


def _materialize(cache, pages):
    token_ids: list[int] = []
    for page_idx in range(len(pages)):
        token_ids.extend(_page_tokens(page_idx))
    checkpoint, created = cache.get_or_create_checkpoint(
        cache.root,
        token_ids,
        pages,
    )
    assert created
    assert checkpoint is not None
    return token_ids, checkpoint


def test_lookup_empty_cache_returns_root():
    pool = MockPagePool()
    cache = PrefixCheckpointCache(pool)

    result = cache.lookup(_page_tokens(0))
    assert result.checkpoint_len == 0
    assert result.page_indices == []
    assert result.checkpoint is cache.root


def test_lookup_returns_deepest_aligned_checkpoint():
    pool = MockPagePool()
    cache = PrefixCheckpointCache(pool)
    pages = pool.alloc(2)
    token_ids, checkpoint = _materialize(cache, pages)

    result = cache.lookup(token_ids + _page_tokens(2))
    assert result.checkpoint_len == 128
    assert result.page_indices == pages
    assert result.checkpoint is checkpoint


def test_lookup_does_not_reuse_partial_page():
    pool = MockPagePool()
    cache = PrefixCheckpointCache(pool)
    pages = pool.alloc(1)
    block = _page_tokens(0)
    cache.get_or_create_checkpoint(cache.root, block, pages)

    query = block[:-1] + [999999]
    result = cache.lookup(query)
    assert result.checkpoint_len == 0
    assert result.page_indices == []


def test_get_or_create_checkpoint_dedupes_existing_page():
    pool = MockPagePool()
    cache = PrefixCheckpointCache(pool)
    first_page, second_page = pool.alloc(2)
    block = _page_tokens(0)

    created_checkpoint, created = cache.get_or_create_checkpoint(cache.root, block, [first_page])
    assert created

    deduped_checkpoint, created = cache.get_or_create_checkpoint(cache.root, block, [second_page])
    assert not created
    assert deduped_checkpoint is created_checkpoint
    assert deduped_checkpoint.tail_page_ids == (first_page,)
    assert cache.total_cached_pages == 1


def test_evict_removes_only_leaf_checkpoints_and_cascades_upward():
    pool = MockPagePool()
    cache = PrefixCheckpointCache(pool)
    pages = pool.alloc(2)
    page0 = _page_tokens(0)
    page1 = _page_tokens(1)
    checkpoint_64, created = cache.get_or_create_checkpoint(cache.root, page0, [pages[0]])
    assert created
    token_ids = page0 + page1
    checkpoint_128, created = cache.get_or_create_checkpoint(checkpoint_64, page1, [pages[1]])
    assert created
    assert checkpoint_128 is not None

    assert cache.total_cached_pages == 2
    assert cache.num_evictable_pages == 1

    freed = cache.evict(2)
    assert freed == 2
    assert cache.total_cached_pages == 0
    assert pool.num_free == 32

    result = cache.lookup(token_ids)
    assert result.checkpoint_len == 0


def test_inc_ref_blocks_leaf_eviction():
    pool = MockPagePool()
    cache = PrefixCheckpointCache(pool)
    pages = pool.alloc(2)
    page0 = _page_tokens(0)
    page1 = _page_tokens(1)
    checkpoint_64, _ = cache.get_or_create_checkpoint(cache.root, page0, [pages[0]])
    token_ids = page0 + page1
    checkpoint, _ = cache.get_or_create_checkpoint(checkpoint_64, page1, [pages[1]])

    cache.inc_ref(checkpoint)
    assert cache.evict(2) == 0

    cache.dec_ref(checkpoint)
    assert cache.evict(2) == 2
    assert cache.lookup(token_ids).checkpoint_len == 0
