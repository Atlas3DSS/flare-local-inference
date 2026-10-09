"""Aligned terminal prefix checkpoints for paged KV reuse.

Reuses only previously materialized 64-token terminal checkpoints.
Lookup probes every aligned boundary, but only boundaries that were
explicitly checkpointed are reusable.
"""

from __future__ import annotations

import hashlib
import heapq
import struct
import time
from dataclasses import dataclass, field

from serve.cache.page_pool import PagePool, _PAGE_SIZE


_DIGEST_SIZE = 16
_PERSON = b"b12x-pfx-v1"
ROOT_DIGEST = hashlib.blake2b(
    b"b12x-prefix-checkpoint-root-v1",
    digest_size=_DIGEST_SIZE,
    person=_PERSON,
).digest()


@dataclass(slots=True)
class PrefixCheckpoint:
    """One immutable terminal prefix checkpoint."""

    parent: PrefixCheckpoint | None
    tail_page_ids: tuple[int, ...]
    prefix_len: int
    prefix_digest: bytes
    state_snapshot_slot: int = -1
    child_count: int = 0
    lock_ref: int = 0
    last_access_time: float = field(default_factory=time.monotonic)

    @property
    def is_leaf(self) -> bool:
        return self.child_count == 0

    @property
    def num_tail_pages(self) -> int:
        return len(self.tail_page_ids)

    def touch(self) -> None:
        self.last_access_time = time.monotonic()


@dataclass(slots=True)
class LookupResult:
    """Result of an aligned checkpoint lookup."""

    page_indices: list[int]
    checkpoint_len: int
    checkpoint: PrefixCheckpoint


class PrefixCheckpointCache:
    """Exact 64-token terminal prefix checkpoints with leaf-only eviction."""

    def __init__(self, pool: PagePool, state_arena=None):
        self.pool = pool
        self.state_arena = state_arena
        self.root = PrefixCheckpoint(
            parent=None,
            tail_page_ids=(),
            prefix_len=0,
            prefix_digest=ROOT_DIGEST,
        )
        self.by_prefix: dict[tuple[int, bytes], PrefixCheckpoint] = {}
        self._total_cached_pages = 0

    def lookup(self, token_ids: list[int], *, require_state_snapshot: bool = False) -> LookupResult:
        """Return the deepest cached aligned terminal checkpoint for *token_ids*."""
        checkpoint = self.root
        prefix_digest = self.root.prefix_digest

        for prefix_len in range(_PAGE_SIZE, len(token_ids) + 1, _PAGE_SIZE):
            block_tokens = token_ids[prefix_len - _PAGE_SIZE:prefix_len]
            prefix_digest = self._next_digest(prefix_digest, block_tokens, prefix_len)
            hit = self.by_prefix.get((prefix_len, prefix_digest))
            if hit is None:
                continue
            if require_state_snapshot and hit.state_snapshot_slot < 0:
                continue
            hit.touch()
            checkpoint = hit

        return LookupResult(
            page_indices=self.page_indices(checkpoint),
            checkpoint_len=checkpoint.prefix_len,
            checkpoint=checkpoint,
        )

    def page_indices(self, checkpoint: PrefixCheckpoint) -> list[int]:
        """Return the ordered KV page chain for *checkpoint*."""
        page_chunks: list[tuple[int, ...]] = []
        while checkpoint is not self.root:
            page_chunks.append(checkpoint.tail_page_ids)
            assert checkpoint.parent is not None
            checkpoint = checkpoint.parent
        page_indices: list[int] = []
        for chunk in reversed(page_chunks):
            page_indices.extend(chunk)
        return page_indices

    def get_or_create_checkpoint(
        self,
        parent: PrefixCheckpoint,
        tail_tokens: list[int],
        tail_page_ids: list[int],
        *,
        live_state_slot: int | None = None,
        require_state_snapshot: bool = False,
    ) -> tuple[PrefixCheckpoint | None, bool]:
        """Get or create an aligned terminal checkpoint extending *parent*."""
        if not tail_tokens and not tail_page_ids:
            return parent, False

        if len(tail_tokens) % _PAGE_SIZE != 0:
            raise ValueError(
                f"checkpoint tokens must be {_PAGE_SIZE}-aligned, got {len(tail_tokens)}"
            )
        if len(tail_tokens) != len(tail_page_ids) * _PAGE_SIZE:
            raise ValueError(
                "tail token/page mismatch: "
                f"{len(tail_tokens)} tokens for {len(tail_page_ids)} pages"
            )

        prefix_len = parent.prefix_len + len(tail_tokens)
        prefix_digest = parent.prefix_digest
        for block_start in range(0, len(tail_tokens), _PAGE_SIZE):
            block_end = block_start + _PAGE_SIZE
            block_prefix_len = parent.prefix_len + block_end
            prefix_digest = self._next_digest(
                prefix_digest,
                tail_tokens[block_start:block_end],
                block_prefix_len,
            )

        key = (prefix_len, prefix_digest)
        checkpoint = self.by_prefix.get(key)
        if checkpoint is not None:
            checkpoint.touch()
            return checkpoint, False

        snapshot_slot = -1
        if require_state_snapshot:
            if self.state_arena is None or live_state_slot is None:
                return None, False
            try:
                snapshot_slot = self.state_arena.capture_snapshot(live_state_slot)
            except RuntimeError:
                return None, False

        checkpoint = PrefixCheckpoint(
            parent=parent,
            tail_page_ids=tuple(tail_page_ids),
            prefix_len=prefix_len,
            prefix_digest=prefix_digest,
            state_snapshot_slot=snapshot_slot,
        )
        checkpoint.touch()
        self.by_prefix[key] = checkpoint
        parent.child_count += 1
        self._total_cached_pages += len(tail_page_ids)
        return checkpoint, True

    def restore_state_snapshot(self, checkpoint: PrefixCheckpoint, live_slot: int) -> bool:
        """Copy a cached state snapshot into a live request slot."""
        if checkpoint is self.root or checkpoint.state_snapshot_slot < 0 or self.state_arena is None:
            return False
        self.state_arena.restore_snapshot(checkpoint.state_snapshot_slot, live_slot)
        return True

    def evict(self, num_pages: int, *, on_evict=None) -> int:
        """Evict unlocked leaf checkpoints until *num_pages* pages are freed."""
        freed = 0
        heap = [
            (checkpoint.last_access_time, id(checkpoint), checkpoint)
            for checkpoint in self.by_prefix.values()
            if checkpoint.lock_ref == 0 and checkpoint.is_leaf
        ]
        heapq.heapify(heap)

        while freed < num_pages and heap:
            _priority, _checkpoint_id, checkpoint = heapq.heappop(heap)
            if checkpoint.lock_ref > 0 or not checkpoint.is_leaf:
                continue
            if (checkpoint.prefix_len, checkpoint.prefix_digest) not in self.by_prefix:
                continue

            parent = checkpoint.parent
            if checkpoint.tail_page_ids:
                self.pool.free(list(checkpoint.tail_page_ids))
                freed += len(checkpoint.tail_page_ids)
                self._total_cached_pages -= len(checkpoint.tail_page_ids)
            if checkpoint.state_snapshot_slot >= 0 and self.state_arena is not None:
                self.state_arena.free_snapshot(checkpoint.state_snapshot_slot)
            if on_evict is not None:
                on_evict(checkpoint)

            del self.by_prefix[(checkpoint.prefix_len, checkpoint.prefix_digest)]
            checkpoint.tail_page_ids = ()
            checkpoint.state_snapshot_slot = -1

            if parent is not None and parent is not self.root:
                parent.child_count = max(0, parent.child_count - 1)
                if parent.lock_ref == 0 and parent.is_leaf:
                    heapq.heappush(heap, (parent.last_access_time, id(parent), parent))

        return freed

    def inc_ref(self, checkpoint: PrefixCheckpoint) -> None:
        """Pin a checkpoint and all ancestors from eviction."""
        while checkpoint is not None and checkpoint is not self.root:
            checkpoint.lock_ref += 1
            checkpoint = checkpoint.parent

    def dec_ref(self, checkpoint: PrefixCheckpoint) -> None:
        """Unpin a checkpoint and all ancestors."""
        while checkpoint is not None and checkpoint is not self.root:
            checkpoint.lock_ref = max(0, checkpoint.lock_ref - 1)
            checkpoint = checkpoint.parent

    @property
    def total_cached_pages(self) -> int:
        return self._total_cached_pages

    @property
    def num_evictable_pages(self) -> int:
        return sum(
            checkpoint.num_tail_pages
            for checkpoint in self.by_prefix.values()
            if checkpoint.lock_ref == 0 and checkpoint.is_leaf
        )

    def _next_digest(
        self,
        parent_digest: bytes,
        block_tokens: list[int],
        prefix_len: int,
    ) -> bytes:
        block_bytes = struct.pack(f"<{_PAGE_SIZE}I", *block_tokens)
        return hashlib.blake2b(
            parent_digest + block_bytes + prefix_len.to_bytes(4, "little"),
            digest_size=_DIGEST_SIZE,
            person=_PERSON,
        ).digest()
