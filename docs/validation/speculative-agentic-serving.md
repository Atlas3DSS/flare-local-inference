# Copy speculation on agentic requests

One concurrency-1 execution of the frozen 24-context workload returns **23 HTTP
200 responses and one HTTP 400**. Automatic output checks pass **18/24**;
independent manual review passes **16/24**, with eight failures and no inconclusive
cases. The aggregate benchmark fails. The driver records no execution errors;
all 23 admitted requests complete without a native runtime failure or cancellation.

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`
at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The Rust HTTP server
runs copy proposals with adaptive verification width; MTP weights are not loaded.
The request policy explicitly matches the stochastic defaults: temperature 1,
top-k 20, top-p 0.95, min-p 0, repetition penalty 1, and zero frequency/presence
penalties. Recorded per-request seeds and normal EOS termination remain enabled.

The workload contains six recorded tool-turn contexts, four code edits, four
chat questions, four structured JSON requests, two long-context requests, and
four repeated/branched-prefix requests. Output budgets are 512 tokens for tools,
long contexts and prefixes, 1,024 for code, 256 for JSON, and **384 for every chat
question**. Tool results are fixed input records. Generated tool actions and code
were not executed; code review uses AST and unified-diff structure only.

## Output failures

| Context | Observed result |
|---|---|
| `code-timeout-range` | Correct bounds change, but the unified diff declares eight lines per hunk side and contains seven. Its source start also differs from the matching input location. The patch is malformed. |
| `code-docstrings-stop` | HTTP 400 rejects the unchanged string `stop` option. No model output is produced. |
| `chat-capacity` | Reaches 384 tokens during the first remedy; the second remedy and throughput/latency distinction are absent. The proposed capacity merely equals the arrival rate. |
| `chat-tradeoff` | Reaches 384 tokens during the early-flight advantages; disadvantages and a balanced recommendation are absent. |
| `chat-deduction` | Reaches 384 tokens during the blue-draw branch. The answer also incorrectly treats drawing red from a wrongly red-labeled box as ambiguous. |
| `chat-review` | Gives a useful cache-miss counterexample and metrics, then reaches 384 tokens mid-table. The delivered answer is unfinished. |
| `json-literal-tags` | Returns valid JSON but repeats `<tool_call>` and omits the required literal `<think>`. |
| `prefix-branch-retry` | Extracts limits 3/1/1 correctly, but says “Increase … from 3 to 0” and later suggests 1. The requested single change is inconsistent. |

Automatic checks accept the malformed diff and inconsistent retry recommendation;
manual review rejects them. The string-stop rejection also causes that group's
admission, completion, reuse and emission-conservation assertions to fail. These
are consequences of the same rejected request, not additional model-worker errors.

The successful code edit preserves all eight validators except the requested
boolean rejection. Generated tests cover their boundaries and out-of-range values
under static inspection. Both long-context answers extract the requested source
records correctly from prompts of 12,845 and 51,172 tokens. The repeated prefix
request produces identical raw output IDs and text. **All four prefix cases report
zero reused prompt tokens**, so this repeat does not establish cache-restore
behavior. The two observed cache hits are the second tool turns, reusing 384
tokens each.

## Rolling terminal-checkpoint limitation

Each request retains one rolling aligned candidate and publishes its latest
endpoint at completion. Generation replaces the 6,400-token prompt candidate
before it becomes an immutable checkpoint. The resulting endpoints exceed the
subsequent prompt lengths or their common prefix:

| Request | Prompt | Emitted, including pending EOS | Consumed | Latest aligned endpoint | Added cached pages |
|---|---:|---:|---:|---:|---:|
| `prefix-seed` | 6,459 | 144 | 6,602 | 6,592 | 103 |
| `prefix-repeat` | 6,459 | 144 | 6,602 | 6,592 | 0, duplicate endpoint |
| `prefix-branch-owner` | 6,465 | 66 | 6,530 | 6,528 | 102 |
| `prefix-branch-retry` | 6,469 | 292 | 6,760 | 6,720 | 105 |

Consumed lengths follow `prompt + emitted − 1`; the final EOS remains pending.
The aligned endpoints are reconstructed from that contract and agree with the
observed 64-token page increments. No checkpoint-state readback was taken.
The seed/repeat requests have identical bytes and token IDs; both branches share
6,400 aligned tokens with the seed. Each lookup misses, each request runs one
snapshot graph, and no eviction occurs. Free snapshots decrease from 18 to 15,
so snapshot exhaustion does not explain the misses.

[Candidate recording](../../native/src/scheduler.rs#L750) replaces the request's
candidate; [completion](../../native/src/scheduler.rs#L940) freezes it only when
the request finishes; [publication](../../native/src/scheduler.rs#L1018) retains
that single endpoint. [Lookup](../../native/src/cache.rs#L328) matches only ready,
published prefix keys. Sampling seeds do not enter the
[prefix digest](../../native/src/cache/hash.rs#L94), and these text requests have
no conditioning events. The observations follow the specified terminal-checkpoint
policy, with no evidence of an identity or lookup defect.

A continuation containing the previous output can reuse that terminal checkpoint.
A retry or branch before its endpoint needs a separately retained complete model
state; the shorter KV page prefix alone cannot restore GDN and PLE state. The
[source and metric analysis](/data/sparkgap-artifacts/mtp-implementation/copy-corrected-agentic/prefix-publication/analysis.json)
preserves the exact hashes and per-request counters.

## Execution and evidence

The run spans `2026-10-09T21:16:51Z`–`21:18:06Z`, with 4,096-token prefill,
a 262,144-token context limit, 39 captured graphs, and 14 prepared speculative
shapes satisfying batch × width ≤ 8. Concurrency 1 exercises batch-one widths
1–8. The memory report records 6,485 KV pages, 35 snapshots, a 297,959,168-byte
operator scratch arena, and 4 GiB configured headroom.

All 5,026 emitted token IDs are retained alongside SSE bytes, usage, finish reasons
and metrics. The same server process remains running and idle after the run;
all four request slots are free, with KV pages and snapshots conserved. No
shutdown measurement is included.

The native extension SHA-256 is
`a81601daee2c5a8cedc7b7d7d59b0ace4f17e83170ff537e4661fa597a1a1f91`.
All 611 recorded launch source/build hashes match at publication. Verification
reparses 10,012 SSE events, recomputes every observation/group/aggregate, checks
all 24 requests against their frozen bytes, and inventories 158 raw run files.
Both mapped runtime binaries are tied to the before/after process-map device and
inode records. The workload SHA-256 is
`45dda5a1f9d69ef7c1c3edf63c6ee65f3f89d3dae64083d0f5f8d505f4b9ba13`.

The [compact receipt](speculative-agentic-serving.json) links the
[raw HTTP receipt](/data/sparkgap-artifacts/mtp-implementation/copy-corrected-agentic/concurrency-1/receipt.json),
[manual review](/data/sparkgap-artifacts/mtp-implementation/copy-corrected-agentic/manual-review/review.md),
[static review script](/data/sparkgap-artifacts/mtp-implementation/copy-corrected-agentic/manual-review/review.py),
and [verified artifact manifest](/data/sparkgap-artifacts/mtp-implementation/copy-corrected-agentic/manual-review/manifest.json).

This single arm and repetition provides no paired speedup or accuracy attribution.
Learned MTP proposals, concurrent workload repetitions and an equivalent ordinary
decoding arm are not exercised. The four explicit 384-token chat budgets remain
part of the failed result.
