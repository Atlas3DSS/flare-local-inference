# Serve Audit TODO

Goal: move `serve/` from lab-quality serving code toward a production-adjacent stack for multi-GPU operators who want less complexity than vLLM or SGLang.

## Priority Order

1. Hybrid TP prefix-cache SSM snapshot consistency
2. Preemption semantics preserve emitted output and request state
3. Full stop-string support in the OpenAI-compatible API
4. Timeout and cancel enforcement across the full scheduler lifecycle
5. Server-loop failure visibility and health semantics
6. Sampling fast-path respects all sampling parameters

## TODOs

### 1. Fix hybrid TP prefix-cache SSM snapshot consistency

Problem:
- Rank 0 restores hybrid linear-attention snapshots when reusing a cached prefix.
- TP follower ranks only receive `ssm_slot` indices and never restore the corresponding snapshot state.
- This can make hybrid TP runs incorrect once prefix reuse is active.

Tasks:
- Add an explicit TP control-path for hybrid snapshot restore events.
- Make follower ranks mirror snapshot restore and slot cleanup operations.
- Verify live-slot allocation, snapshot capture, restore, preemption, and finish paths stay rank-consistent.
- Add regression coverage for TP hybrid prefix reuse.

Acceptance criteria:
- Reused hybrid prefixes produce identical state on all TP ranks.
- Prefix reuse for hybrid models is either correct or explicitly disabled in TP mode.

### 2. Fix preemption semantics so resumed requests preserve emitted output and state

Problem:
- `_preempt_oldest()` clears `output_ids`, timing fields, and effective generation history.
- A streamed request can emit tokens to a client and later resume from a different internal state.

Tasks:
- Define the intended preemption contract for partially generated requests.
- Preserve generated-token history and the correct decode context across preemption.
- Keep TTFT and total timing meaningful after requeue.
- Update tests that currently encode the reset-on-preemption behavior.

Acceptance criteria:
- A preempted request resumes as if generation had been paused, not rewound.
- Streamed tokens are never invalidated by internal memory pressure.

### 3. Fix stop-string handling to support full multi-token stop sequences

Problem:
- The API converts each stop string into only its final token ID.
- Multi-token stops can fail to trigger or can stop on unrelated suffix tokens.

Tasks:
- Represent stop conditions as token sequences, not single IDs.
- Check generated output for full stop-sequence matches without false positives.
- Define whether stop text is included or stripped from the returned output.
- Add API and engine tests for single-token and multi-token stop strings.

Acceptance criteria:
- OpenAI-compatible stop strings work for both single-token and multi-token cases.
- No early stop is triggered by a partial suffix match.

### 4. Enforce timeout and cancel handling across waiting, prefilling, and running states

Problem:
- Requests only check timeout after token append.
- Cancelled or timed-out requests can still sit in queues, consume prefill work, or get another decode step.

Tasks:
- Add lifecycle checks before scheduling waiting, prefilling, and running requests.
- Remove finished requests from the scheduler before more GPU work is spent on them.
- Ensure timeouts during long prefills unblock callers and clean up resources.
- Add regression tests for queue-wait timeout, prefill timeout, and cancel-before-decode.

Acceptance criteria:
- `timeout` bounds total request lifetime, not just decode time.
- `cancel()` prevents further model work once observed by the scheduler.

### 5. Harden server-loop failure visibility and health semantics

Problem:
- If the background step loop crashes, requests can hang and `/health` can still report `"ok"`.

Tasks:
- Catch and record fatal exceptions in the background loop.
- Fail fast for new requests when the loop is unhealthy.
- Expose loop health, last error, and startup/readiness state in health endpoints.
- Ensure shutdown stops the loop cleanly.

Acceptance criteria:
- A dead scheduler loop is visible immediately through health/readiness and request handling.
- Outstanding requests are unblocked or failed deterministically after a fatal loop error.

### 6. Fix sampling fast-path parameter comparison

Problem:
- `sample_batch()` only compares a subset of sampling fields before taking the shared fast path.
- Mixed requests can silently use the wrong sampling policy.

Tasks:
- Compare all behavior-affecting sampling fields before using the fast path.
- Add tests covering `min_p`, `presence_penalty`, `frequency_penalty`, and stop-related behavior where relevant.

Acceptance criteria:
- Batched mixed-parameter sampling matches per-request semantics.
- The fast path is only used when requests are genuinely compatible.
