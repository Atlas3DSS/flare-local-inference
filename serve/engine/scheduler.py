"""Continuous batching scheduler with chunked prefill and graph-aware decode.

Manages the waiting → prefilling → running (decode) → finished lifecycle.
Long prompts are split into fixed-size chunks, interleaved with decode
steps so running requests don't stall.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from serve.cache.page_pool import PagePool, _PAGE_SIZE
from serve.cache.prefix_checkpoint_cache import PrefixCheckpoint, PrefixCheckpointCache
from serve.engine.request import Request


@dataclass
class BatchStep:
    """One batch ready for the model runner."""

    mode: str                      # "prefill" or "decode".
    requests: list[Request]        # Requests in this batch.
    token_ids: torch.Tensor        # [total_tokens].
    q_seqlens: list[int]           # Tokens per request in this forward.
    q_seqlens_t: torch.Tensor      # [batch] int32 tokens per request.
    page_table: torch.Tensor       # [batch, max_pages].
    cache_seqlens: torch.Tensor    # [batch].
    graph_bs: Optional[int] = None # Which graph to use.
    is_last_chunk: bool = True     # For prefill: is this the final chunk?


class BatchScheduler:
    """Continuous batching with chunked prefill and graph-aware decode."""

    SSM_ALLOC_LIVE = 1
    SSM_RESTORE_SNAPSHOT = 2
    SSM_CAPTURE_SNAPSHOT = 3
    SSM_FREE_LIVE = 4
    SSM_FREE_SNAPSHOT = 5

    def __init__(
        self,
        cache: PrefixCheckpointCache,
        pool: PagePool,
        ssm_pool=None,
        enable_prefix_cache: bool = True,
        captured_bs: list[int] | None = None,
        max_running: int = 8,
        max_waiting: int = 256,
        max_prefill_tokens: int = 4096,
        chunk_size: int = 512,
        device: torch.device | str = "cuda",
    ):
        self.cache = cache
        self.pool = pool
        self.ssm_pool = ssm_pool  # Optional LinearStateArena for hybrid models.
        self.enable_prefix_cache = enable_prefix_cache
        self.captured_bs = sorted(captured_bs or [])
        self.max_running = max_running
        self.max_waiting = max_waiting
        self.max_prefill_tokens = max_prefill_tokens
        self.chunk_size = chunk_size
        self.device = torch.device(device)

        self.waiting: deque[Request] = deque()
        self.prefilling: list[Request] = []  # Mid-chunked-prefill.
        self.running: list[Request] = []
        self.finished: list[Request] = []

        # Interleave flag: after a prefill chunk, prefer decode next.
        self._last_was_prefill = False

        # Stats.
        self.stats_prefill_steps = 0
        self.stats_decode_steps = 0
        self.stats_preemptions = 0
        self.stats_evictions = 0
        self.stats_total_finished = 0
        self._ssm_control_ops: list[tuple[int, int, int, int]] = []

    def add_request(self, req: Request) -> None:
        """Submit a new request to the waiting queue.

        Raises RuntimeError if the queue is full (max_waiting exceeded).
        """
        if len(self.waiting) >= self.max_waiting:
            raise RuntimeError(
                f"request queue full ({self.max_waiting} waiting). "
                f"Try again later."
            )
        self.waiting.append(req)

    def step(self) -> Optional[BatchStep]:
        """One scheduling step with interleaving.

        Priority:
        1. Continue prefilling requests (next chunk).
        2. Admit new waiting requests (first chunk).
        3. Decode all running requests.

        Interleaving: after a prefill step, prefer decode if possible.
        """
        self._prune_finished_requests()

        # Interleave: if last step was prefill, do decode first if possible.
        if self._last_was_prefill and self.running:
            self._last_was_prefill = False
            return self._schedule_decode()

        # Continue mid-prefill requests.
        if self.prefilling:
            batch = self._schedule_prefill_chunk()
            if batch is not None:
                self._last_was_prefill = True
                return batch

        # Admit new requests.
        if self.waiting:
            batch = self._try_admit()
            if batch is not None:
                self._last_was_prefill = True
                return batch

        # Decode all running.
        if self.running:
            self._last_was_prefill = False
            return self._schedule_decode()

        return None

    # -- prefill -----------------------------------------------------------

    def _try_admit(self) -> Optional[BatchStep]:
        """Try to admit waiting requests and schedule their first chunk.

        Short prompts (fitting in one chunk) are batched together for
        better throughput. Long prompts go one at a time for chunking.
        """
        to_admit: list[Request] = []
        total_tokens = 0

        while self.waiting:
            if len(self.running) + len(self.prefilling) + len(to_admit) >= self.max_running:
                break

            req = self.waiting[0]

            # Match prefix.
            if self.enable_prefix_cache:
                match = self.cache.lookup(
                    req.prompt_ids,
                    require_state_snapshot=self.ssm_pool is not None,
                )
            else:
                match = self.cache.lookup([])
            req.checkpoint_len = match.checkpoint_len
            req.checkpoint = match.checkpoint
            req.prefill_progress = match.checkpoint_len

            extend_len = req.extend_len
            if extend_len <= 0:
                extend_len = 1

            if self._request_exceeds_capacity(req):
                self.waiting.popleft()
                req._mark_finished("context_too_long")
                self._record_finished([req])
                continue

            # Long prompt — needs chunking. Can't batch with others.
            if extend_len > self.chunk_size:
                if to_admit:
                    break  # Flush short requests first.
                if not self._try_alloc_pages(req, match):
                    break
                self.waiting.popleft()
                self.prefilling.append(req)
                return self._build_prefill_step(req, is_last=False)

            # Short prompt — try to batch.
            if total_tokens + extend_len > self.max_prefill_tokens:
                break

            if not self._try_alloc_pages(req, match):
                break

            self.waiting.popleft()
            to_admit.append(req)
            total_tokens += extend_len

        if not to_admit:
            return None

        if len(to_admit) == 1:
            return self._build_prefill_step(to_admit[0], is_last=True)

        return self._build_batched_prefill(to_admit)

    def _try_alloc_pages(self, req: Request, match) -> bool:
        """Try to allocate pages and a live linear-state slot for a request."""
        pages_needed = self._pages_needed(req)
        if pages_needed > self.pool.num_pages:
            return False
        if pages_needed > self.pool.num_free:
            evictable = self.cache.num_evictable_pages
            if pages_needed <= self.pool.num_free + evictable:
                self.cache.evict(
                    pages_needed - self.pool.num_free,
                    on_evict=self._on_evict_checkpoint,
                )

            if self.pool.num_free < pages_needed and self.running:
                self._preempt_oldest()

            if self.pool.num_free < pages_needed:
                return False

        # Allocate a live linear-state slot if the model has hybrid state.
        if self.ssm_pool is not None:
            if self.ssm_pool.num_free == 0:
                return False
            req.ssm_slot = self.ssm_pool.alloc(1)[0].item()
            self._queue_ssm_op(self.SSM_ALLOC_LIVE, req.ssm_slot)
            self.cache.restore_state_snapshot(match.checkpoint, req.ssm_slot)
            if match.checkpoint is not self.cache.root and match.checkpoint.state_snapshot_slot >= 0:
                self._queue_ssm_op(
                    self.SSM_RESTORE_SNAPSHOT,
                    match.checkpoint.state_snapshot_slot,
                    req.ssm_slot,
                )

        pages = self.pool.alloc(pages_needed) if pages_needed > 0 else []
        req.page_ids = list(match.page_indices) + pages

        if match.checkpoint is not self.cache.root:
            self.cache.inc_ref(match.checkpoint)

        return True

    def _request_exceeds_capacity(self, req: Request) -> bool:
        """Return whether *req* can never fit in the configured page budget."""
        total_pages = (len(req.prompt_ids) + _PAGE_SIZE - 1) // _PAGE_SIZE
        return total_pages > self.pool.num_pages

    def _build_batched_prefill(self, reqs: list[Request]) -> BatchStep:
        """Build a prefill step for multiple short requests at once."""
        self.stats_prefill_steps += 1
        token_ids_list = []
        q_seqlens = []
        for req in reqs:
            chunk_tokens = req.prompt_ids[req.prefill_progress:]
            if not chunk_tokens:
                chunk_tokens = [req.prompt_ids[-1]]
            token_ids_list.extend(chunk_tokens)
            q_seqlens.append(len(chunk_tokens))
            req.cache_len = len(req.prompt_ids)
            req.prefill_progress = len(req.prompt_ids)
            self.running.append(req)

        page_table, cache_seqlens = self._build_batch_tensors(reqs)

        return BatchStep(
            mode="prefill",
            requests=reqs,
            token_ids=torch.tensor(token_ids_list, dtype=torch.long, device=self.device),
            q_seqlens=q_seqlens,
            q_seqlens_t=torch.tensor(q_seqlens, dtype=torch.int32, device=self.device),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            is_last_chunk=True,
        )

    def _schedule_prefill_chunk(self) -> Optional[BatchStep]:
        """Schedule the next chunk for the first prefilling request."""
        if not self.prefilling:
            return None

        req = self.prefilling[0]
        remaining = req.prefill_remaining
        is_last = remaining <= self.chunk_size

        if is_last:
            self.prefilling.pop(0)

        return self._build_prefill_step(req, is_last=is_last)

    def _build_prefill_step(self, req: Request, is_last: bool) -> BatchStep:
        """Build a BatchStep for one prefill chunk."""
        self.stats_prefill_steps += 1
        chunk_start = req.prefill_progress
        chunk_end = min(chunk_start + self.chunk_size, len(req.prompt_ids))
        chunk_tokens = req.prompt_ids[chunk_start:chunk_end]
        if not chunk_tokens:
            # Fully cached — extend by at least one token.
            chunk_tokens = [req.prompt_ids[-1]]
            chunk_end = len(req.prompt_ids)
        chunk_len = len(chunk_tokens)

        # Update cache_len to reflect state after this chunk's KV write.
        req.cache_len = chunk_end

        # Advance prefill progress.
        req.prefill_progress = chunk_end

        if is_last:
            self.running.append(req)

        page_table, cache_seqlens = self._build_batch_tensors([req])

        # Pick graph for prefill chunk (bs=1, q_len=chunk_size).
        graph_bs = None
        if chunk_len == self.chunk_size:
            graph_bs = self._pick_graph_bs(1)

        return BatchStep(
            mode="prefill",
            requests=[req],
            token_ids=torch.tensor(chunk_tokens, dtype=torch.long, device=self.device),
            q_seqlens=[chunk_len],
            q_seqlens_t=torch.tensor([chunk_len], dtype=torch.int32, device=self.device),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            graph_bs=graph_bs,
            is_last_chunk=is_last,
        )

    # -- decode ------------------------------------------------------------

    def _schedule_decode(self) -> BatchStep:
        """Schedule one decode step for all running requests."""
        self.stats_decode_steps += 1
        bs = len(self.running)
        graph_bs = self._pick_graph_bs(bs)

        token_ids = []
        for req in self.running:
            if req.output_ids:
                token_ids.append(req.output_ids[-1])
            else:
                token_ids.append(req.prompt_ids[-1])

        page_table, cache_seqlens = self._build_batch_tensors(self.running)

        return BatchStep(
            mode="decode",
            requests=list(self.running),
            token_ids=torch.tensor(token_ids, dtype=torch.long, device=self.device),
            q_seqlens=[1] * bs,
            q_seqlens_t=torch.ones(bs, dtype=torch.int32, device=self.device),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            graph_bs=graph_bs,
        )

    # -- output processing -------------------------------------------------

    def process_prefill_chunk(
        self,
        next_token_ids: list[int] | None = None,
        requests: list[Request] | None = None,
    ) -> list[Request]:
        """Process a prefill chunk result.

        For non-final chunks: no token output, just advances KV state.
        For the final chunk: appends the sampled token to the specified
        requests (not all running — other running requests are in decode).
        """
        if next_token_ids is None:
            return []

        # Apply tokens only to the prefilled requests.
        targets = requests or self.running[-len(next_token_ids):]
        for i, req in enumerate(targets):
            if i < len(next_token_ids):
                req.append_token(next_token_ids[i])
                req.record_first_token()
                req.cache_len += 1
                self._ensure_page_capacity(req)
                req.check_finished()

        # Collect finished from all running.
        finished = []
        still_running = []
        for req in self.running:
            if req.is_finished:
                self._finish_request(
                    req,
                    cache_terminal=req.finished_reason in {"length", "stop"},
                )
                finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        self._record_finished(finished)
        return finished

    def process_decode_output(self, next_token_ids: list[int]) -> list[Request]:
        """Process decode results. Returns newly finished requests."""
        finished = []
        for i, req in enumerate(self.running):
            if i < len(next_token_ids):
                req.append_token(next_token_ids[i])
                req.cache_len += 1
                self._ensure_page_capacity(req)
                req.check_finished()

        # Collect finished.
        still_running = []
        for req in self.running:
            if req.is_finished:
                self._finish_request(
                    req,
                    cache_terminal=req.finished_reason in {"length", "stop"},
                )
                finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        self._record_finished(finished)

        return finished

    def _finish_request(self, req: Request, *, cache_terminal: bool = True) -> None:
        """Clean up a finished request and release uncached request-local state."""
        self.stats_total_finished += 1
        pinned_checkpoint = req.checkpoint
        if cache_terminal:
            self._materialize_terminal_checkpoint(req)
        self._unpin_checkpoint(pinned_checkpoint)
        uncached_pages = req.page_ids[req.checkpoint_len // _PAGE_SIZE:]
        if uncached_pages:
            self.pool.free(uncached_pages)
        req.page_ids = []
        req.checkpoint = None
        req.checkpoint_len = 0

        # Free the live linear-state slot.
        if self.ssm_pool is not None and req.ssm_slot >= 0:
            self._free_live_ssm_slot(req)

    # -- helpers -----------------------------------------------------------

    def _build_batch_tensors(self, reqs: list[Request]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build page_table and cache_seqlens from request state."""
        bs = len(reqs)
        max_pages = max((len(r.page_ids) for r in reqs), default=1)
        max_pages = max(max_pages, 1)

        pt = torch.zeros(bs, max_pages, dtype=torch.int32)
        cs = torch.zeros(bs, dtype=torch.int32)
        for i, req in enumerate(reqs):
            if req.page_ids:
                pt[i, :len(req.page_ids)] = torch.tensor(req.page_ids, dtype=torch.int32)
            cs[i] = req.cache_len

        return pt.to(self.device, non_blocking=True), cs.to(self.device, non_blocking=True)

    def _prune_finished_requests(self) -> None:
        """Remove cancelled/timed-out requests before spending more GPU work on them."""
        self.waiting = self._collect_finished_requests(self.waiting)
        self.prefilling = self._collect_finished_requests(self.prefilling)
        self.running = self._collect_finished_requests(self.running)

    def _collect_finished_requests(self, reqs) -> list[Request] | deque[Request]:
        kept = []
        finished = []
        for req in reqs:
            req.check_finished()
            if req.is_finished:
                self._finish_request(req, cache_terminal=False)
                finished.append(req)
            else:
                kept.append(req)
        self._record_finished(finished)
        if isinstance(reqs, deque):
            return deque(kept)
        return kept

    def _record_finished(self, reqs: list[Request]) -> None:
        if not reqs:
            return
        self.finished.extend(reqs)
        if len(self.finished) > 1000:
            self.finished = self.finished[-100:]

    def fail_all(self, reason: str) -> None:
        """Fail and release all in-flight requests."""
        finished = []
        for req in list(self.waiting) + self.prefilling + self.running:
            if not req.is_finished:
                req._mark_finished(reason)
            self._finish_request(req, cache_terminal=False)
            finished.append(req)

        self.waiting = deque()
        self.prefilling = []
        self.running = []
        self._record_finished(finished)

    def _preempt_oldest(self) -> bool:
        """Preempt the oldest safe-to-rewind running request to free pages."""
        if not self.running:
            return False

        victim_idx = next(
            (i for i, req in enumerate(self.running) if not req.output_ids),
            None,
        )
        if victim_idx is None:
            return False

        self.stats_preemptions += 1
        victim = self.running.pop(victim_idx)

        pinned_checkpoint = victim.checkpoint
        self._materialize_terminal_checkpoint(victim)
        self._unpin_checkpoint(pinned_checkpoint)

        uncached_pages = victim.page_ids[victim.checkpoint_len // _PAGE_SIZE:]
        if uncached_pages:
            self.pool.free(uncached_pages)
        if self.ssm_pool is not None and victim.ssm_slot >= 0:
            self._free_live_ssm_slot(victim)

        victim.page_ids = []
        victim.checkpoint_len = 0
        victim.prefill_progress = 0
        victim.cache_len = 0
        victim.checkpoint = None
        victim.output_ids = []
        victim.first_token_at = None
        victim.finished_reason = None
        victim.finished_at = None
        victim._token_event.clear()

        self.waiting.append(victim)
        return True

    def _pages_needed(self, req: Request) -> int:
        """Pages needed for this request's non-cached tokens."""
        total_tokens = len(req.prompt_ids)
        cached_pages = req.checkpoint_len // _PAGE_SIZE
        total_pages = (total_tokens + _PAGE_SIZE - 1) // _PAGE_SIZE
        return max(0, total_pages - cached_pages)

    def _materialize_terminal_checkpoint(self, req: Request) -> None:
        """Cache the deepest aligned terminal endpoint reached by *req*."""
        if not self.enable_prefix_cache or not req.page_ids:
            return

        parent = req.checkpoint or self.cache.root
        target_len = (req.cache_len // _PAGE_SIZE) * _PAGE_SIZE
        if target_len <= parent.prefix_len:
            return

        start_page = parent.prefix_len // _PAGE_SIZE
        end_page = target_len // _PAGE_SIZE
        full_tokens = req.prompt_ids + req.output_ids
        tail_tokens = full_tokens[parent.prefix_len:target_len]
        tail_page_ids = req.page_ids[start_page:end_page]

        checkpoint, created = self.cache.get_or_create_checkpoint(
            parent,
            tail_tokens,
            tail_page_ids,
            live_state_slot=req.ssm_slot if self.ssm_pool is not None else None,
            require_state_snapshot=self.ssm_pool is not None,
        )
        if checkpoint is None:
            return
        if not created:
            self.pool.free(list(tail_page_ids))
        elif self.ssm_pool is not None and checkpoint.state_snapshot_slot >= 0:
            self._queue_ssm_op(
                self.SSM_CAPTURE_SNAPSHOT,
                req.ssm_slot,
                checkpoint.state_snapshot_slot,
            )

        req.checkpoint = checkpoint
        req.checkpoint_len = checkpoint.prefix_len

    def _unpin_checkpoint(self, checkpoint: PrefixCheckpoint | None) -> None:
        if self.enable_prefix_cache and checkpoint is not None and checkpoint is not self.cache.root:
            self.cache.dec_ref(checkpoint)

    def _ensure_page_capacity(self, req: Request) -> None:
        """Ensure the request has a page allocated for its next token."""
        pages_for_token = (req.cache_len + _PAGE_SIZE - 1) // _PAGE_SIZE
        if pages_for_token <= len(req.page_ids):
            return

        try:
            req.page_ids.extend(self.pool.alloc(1))
            return
        except RuntimeError:
            pass

        self.cache.evict(1, on_evict=self._on_evict_checkpoint)
        try:
            req.page_ids.extend(self.pool.alloc(1))
        except RuntimeError:
            req._mark_finished("oom")

    def _queue_ssm_op(self, op: int, a: int = 0, b: int = 0, c: int = 0) -> None:
        if self.ssm_pool is None:
            return
        self._ssm_control_ops.append((op, a, b, c))

    def _free_live_ssm_slot(self, req: Request) -> None:
        if self.ssm_pool is None or req.ssm_slot < 0:
            return
        slot = req.ssm_slot
        self.ssm_pool.free(torch.tensor([slot], device=self.device, dtype=torch.int32))
        self._queue_ssm_op(self.SSM_FREE_LIVE, slot)
        req.ssm_slot = -1

    def _on_evict_checkpoint(self, checkpoint: PrefixCheckpoint) -> None:
        if checkpoint.state_snapshot_slot >= 0:
            self._queue_ssm_op(self.SSM_FREE_SNAPSHOT, checkpoint.state_snapshot_slot)

    def drain_ssm_control_ops(self) -> list[tuple[int, int, int, int]]:
        ops = self._ssm_control_ops
        self._ssm_control_ops = []
        return ops

    def _pick_graph_bs(self, bs: int) -> Optional[int]:
        """Pick smallest captured graph size >= bs."""
        for size in self.captured_bs:
            if size >= bs:
                return size
        return None

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_prefilling(self) -> int:
        return len(self.prefilling)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.prefilling) or bool(self.running)

    @property
    def stats(self) -> dict:
        return {
            "waiting": self.num_waiting,
            "prefilling": self.num_prefilling,
            "running": self.num_running,
            "finished": self.stats_total_finished,
            "prefill_steps": self.stats_prefill_steps,
            "decode_steps": self.stats_decode_steps,
            "preemptions": self.stats_preemptions,
            "cache_pages": self.cache.total_cached_pages,
            "free_pages": self.pool.num_free,
        }
