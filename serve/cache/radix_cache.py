"""Compatibility shim for the old radix-cache import path."""

from __future__ import annotations

from dataclasses import dataclass

from serve.cache.page_pool import _PAGE_SIZE
from serve.cache.prefix_checkpoint_cache import PrefixCheckpoint, PrefixCheckpointCache


TreeNode = PrefixCheckpoint


@dataclass(slots=True)
class MatchResult:
    page_indices: list[int]
    prefix_len: int
    last_node: PrefixCheckpoint


class RadixCache(PrefixCheckpointCache):
    """Backward-compatible wrapper around PrefixCheckpointCache."""

    def match_prefix(self, token_ids: list[int]) -> MatchResult:
        result = self.lookup(token_ids)
        return MatchResult(
            page_indices=result.page_indices,
            prefix_len=result.checkpoint_len,
            last_node=result.checkpoint,
        )

    def insert(self, token_ids: list[int], page_indices: list[int]) -> None:
        full_pages = min(len(page_indices), len(token_ids) // _PAGE_SIZE)
        if full_pages == 0:
            return
        aligned_tokens = token_ids[:full_pages * _PAGE_SIZE]
        aligned_pages = page_indices[:full_pages]
        checkpoint, created = self.get_or_create_checkpoint(
            self.root,
            aligned_tokens,
            aligned_pages,
        )
        if checkpoint is None:
            return
        if not created and checkpoint.tail_page_ids != tuple(aligned_pages):
            self.pool.free(aligned_pages)
