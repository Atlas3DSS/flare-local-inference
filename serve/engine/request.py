"""Request lifecycle tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from serve.cache.prefix_checkpoint_cache import PrefixCheckpoint
from serve.engine.sampling import SamplingParams


@dataclass
class Request:
    """One generation request through its full lifecycle."""

    rid: int
    prompt_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # Generated output.
    output_ids: list[int] = field(default_factory=list)

    # KV cache state (self-attention layers).
    checkpoint_len: int = 0         # Tokens covered by cached checkpoints.
    prefill_progress: int = 0       # Prompt tokens with KV computed so far.
    cache_len: int = 0              # Total KV tokens written so far.
    page_ids: list[int] = field(default_factory=list)
    checkpoint: Optional[PrefixCheckpoint] = None  # Deepest reusable checkpoint.

    # Linear-attention state.
    ssm_slot: int = -1              # Live slot index in the LinearStateArena (-1 = none).

    # Lifecycle timing.
    created_at: float = field(default_factory=time.monotonic)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    finished_reason: Optional[str] = None
    timeout_s: Optional[float] = None  # Per-request timeout in seconds.

    # Completion signal for async callers.
    _done_event: threading.Event = field(default_factory=threading.Event)
    # Pulsed on each new token (for streaming). Cleared by the reader.
    _token_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_finished(self) -> bool:
        return self.finished_reason is not None

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def extend_len(self) -> int:
        """Tokens to compute on initial prefill (prompt minus cached prefix)."""
        return len(self.prompt_ids) - self.checkpoint_len

    @property
    def prefill_remaining(self) -> int:
        """Prompt tokens still needing KV computation."""
        return len(self.prompt_ids) - self.prefill_progress

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.created_at) * 1000

    @property
    def total_time_ms(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.created_at) * 1000

    def check_finished(self) -> None:
        """Update finished_reason based on output state."""
        if self.finished_reason is not None:
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self._mark_finished("length")
            return

        if self.output_ids and self.sampling_params.stop_sequences:
            for stop_seq in self.sampling_params.stop_sequences:
                if not stop_seq or len(self.output_ids) < len(stop_seq):
                    continue
                if self.output_ids[-len(stop_seq):] == stop_seq:
                    del self.output_ids[-len(stop_seq):]
                    self._mark_finished("stop")
                    return

        if self.output_ids and self.sampling_params.stop_token_ids:
            if self.output_ids[-1] in self.sampling_params.stop_token_ids:
                self._mark_finished("stop")
                return

        if self.timeout_s is not None:
            if time.monotonic() - self.created_at > self.timeout_s:
                self._mark_finished("timeout")

    def append_token(self, token_id: int) -> None:
        """Append a generated token and notify streaming waiters."""
        self.output_ids.append(token_id)
        self._token_event.set()

    def cancel(self) -> None:
        """Cancel this request. Safe to call from any thread."""
        if not self.is_finished:
            self._mark_finished("cancelled")

    def _mark_finished(self, reason: str) -> None:
        self.finished_reason = reason
        self.finished_at = time.monotonic()
        self._token_event.set()
        self._done_event.set()

    def record_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()
