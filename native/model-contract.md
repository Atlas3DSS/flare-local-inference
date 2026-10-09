# Qwen4Exp native model contract

The Rust server consumes the transport manifest in [manifest.md](manifest.md)
and the Qwen4Exp semantics below. Python prepares graphs and persistent storage,
synchronizes preparation, then transfers exclusive mutation access to Rust.
Every tensor listed here is contiguous and remains allocated through confirmed
drain of the model and completion-copy streams.

## Bootstrap

Build with `server,extension-module` and construct:

```python
server = sparkgap_native.Server(owner, manifest, config_json)
print(server.address)
# The interpreter may wait for process signals. Requests execute entirely in Rust.
server.close()
assert server.drained
```

`owner` retains the complete model, graphs, CUDA context, mapped allocations,
weights, KV pages, recurrent state, feature arena and workspaces. The object
exposes `address`, `healthy`, `drained`, `wait(timeout_ms=1000)` and idempotent
`close()`. Construction,
asset loading and shutdown release the GIL. Failed CUDA drain retains the owner.
A failed startup releases it only if its drain receipt permits release. Its
Python exception carries `resources_retained=true` when drain failed; bootstrap
must prohibit explicit graph, session and mapped-allocation teardown in that case.

`config_json` is an object with these fields:

| Field | Meaning |
|---|---|
| `model_id` | Served model identifier; requests must match it |
| `identity` | 64 hexadecimal characters identifying model weights and recipe |
| `max_sequence_tokens` | Positive context capacity divisible by 64 |
| `prefill_chunk_size` | Maximum prompt tokens per prefill graph; 4096 if omitted; divisible by 64, at least 64, and no greater than context or 262144 |
| `pages` | Prepared KV page capacity |
| `snapshots` | Prepared immutable state-snapshot capacity |
| `eos_token_ids` | Up to eight vocabulary token IDs |
| `vision_features_per_request` | 16384 |
| `assets` | Verified asset descriptors, each `{path, sha256}` |
| `template_defaults` | Optional Jinja defaults |
| `generation_defaults` | Optional overrides of checkpoint sampling defaults |
| `max_new_tokens` | Default output limit, 256 if omitted |
| `speculative_widths` | Optional sorted unique subset of integers 1 through 8 including 1; empty or omitted disables speculative scheduling |
| `speculative_fixed_width` | Optional width from that catalog; null or omitted selects adaptive scheduling |
| `speculative_copy` | Boolean, true if omitted; enables history-copy proposals; false forces width one with the copy-only source |
| `speculative_proposal_source` | `mtp` by default; `copy` verifies host-supplied continuations without MTP weights or cache |
| `bind` | Socket address, `127.0.0.1:8000` if omitted |
| `queue_capacity` | Native ingress queue capacity, 16 if omitted |
| `event_capacity` | Buffered token/completion events per request, 128 if omitted |
| `frontend_workers` | Concurrent CPU request-preparation jobs, 2 if omitted |
| `max_body_bytes` | HTTP body bound, 32 MiB if omitted |

Asset keys are `tokenizer`, `tokenizer_config`, `chat_template`,
`preprocessor_config`, `model_config` and `generation_config`. Paths must be
absolute. Each asset is hashed and parsed from the same bytes. Generation
defaults come from the checkpoint; configured defaults override them; explicit
request sampling values override configured defaults. The HTTP default sets
`add_generation_prompt=true` unless configured otherwise. Explicit continuation
selects `add_generation_prompt=false` unless the request sets that flag itself.

The listener binds only after semantic validation and native model-worker
readiness. Missing graph variants or required inputs fail startup. `/health`,
`/v1/models`, `/v1/chat/completions`, and `/metrics` are implemented in Rust. Chat supports
streaming SSE and complete JSON responses, exact checkpoint templates, native
tokenization, sampling controls and PNG data-URL images. At most four images
are admitted, each at most 16384 patches, with 65536 total patches per request.
Remote image fetching, JPEG/WebP/video, stop strings, logprobs, forced tool
selection and `parallel_tool_calls=false` are rejected. Tool choices `auto` and
`none` are supported. The checkpoint's `<think>` output is presented as
`reasoning_content`; complete Qwen Coder function blocks become `tool_calls`
with schema-directed JSON argument types. Partial markers and UTF-8 sequences
remain buffered across token boundaries. Malformed or unknown tool calls are
errors, and incomplete tool blocks cannot produce successful terminal output.

`output::ReasoningParser` and `output::ToolCallParser` consume presented token
IDs and stable decoded spans and emit transport-neutral events. The
`model::qwen4exp::output::QwenOutput` composition owns Qwen delimiters,
reasoning/tool transitions and schema-directed parameter conversion. Its
initial reasoning state comes from the open assistant prefix, excluding
preceding user and tool messages. Model selection is static outside the token
loop. `server::output::Presentation` and its accumulator serve both JSON and
SSE, preserving append-only tool IDs and argument fragments without knowledge
of model syntax. Qwen calls remain buffered until their closing delimiter and
are published atomically.

Parser termination distinguishes EOS, output limits, user stops, cancellation
and execution errors. Cancellation and errors notify the protocol without
publishing buffered output. Presented token IDs include tokens that emit no
text and EOS; custom token stops filter their matching IDs before presentation.
Raw generation observations retain those omitted IDs separately.

`response_format` supports `text`, `json_object` and `json_schema`; `grammar`
accepts a Lark grammar. Native llguidance compilation is bounded and rejects
unsupported schema assertions, external references and compiler warnings.
Structured requests disable thinking and reject explicit thinking, assistant
continuation, tool generation and custom token stops. The parser initializes
the admission mask, advances exactly once per generated token, and uploads
the next physical-slot mask before another constrained sampling step. The
scheduler admits only one in-flight step while a constrained request is batched.
Output-length termination can return a valid but incomplete grammar prefix.
Structured output uses verbatim presentation, so literal reasoning and tool
tags inside generated JSON or grammar text remain part of the content.

### Token and timing observation

Chat requests may set `return_token_ids=true`. JSON responses include
`prompt_token_ids` at the top level and raw generated IDs in
`choices[0].token_ids`. SSE includes prompt IDs in its first chunk and emits
one metadata-only chunk per generated ID, with `delta={}` and
`choices[0].token_ids=[id]`. Text, reasoning and tool deltas carry no duplicate
IDs. Observations include EOS and tokens omitted by stop-token presentation.
They retain normal EOS termination, prefix lookup, output limits and scheduling.

`return_token_timestamps=true` requires `return_token_ids=true` and adds an
aligned `choices[0].token_timestamps_ns` array. These host timestamps mark
validated active-request completions immediately after `scheduler.complete`,
before grammar advancement, stop handling or output parsing. They do not mark
GPU kernel completion or socket delivery. Nanoseconds are relative to a
process-local monotonic origin initialized at native runtime startup. Unobserved
requests read no per-token clock and emit no observation events.

Observed terminal responses include `reused_prompt_tokens` and ordinary token
usage. Timed responses also include one top-level `performance` object with
`ingress_ns`, `enqueue_ns` and `reused_prompt_tokens`. Ingress is chat-handler
entry after HTTP JSON decoding, before waiting for frontend preparation;
enqueue is immediately before submitting the prepared request to the model
queue. Both timestamps use the token timestamp origin.

`stream_options.include_usage=true` emits a final usage-only SSE chunk with
`choices=[]` before `[DONE]`; preceding chunks have `usage=null`. Otherwise an
observed stream includes usage on its terminal choice chunk. Observation flags
and `include_usage` accept only booleans or null; null is equivalent to false.
The existing bounded event queue also bounds observation events. A slow reader
can cancel an observed request through the same backpressure policy as normal
generation. Observation changes no CUDA graph, device allocation or replay ABI.

`stop_token_ids` and `stop_token_sequences` accept explicit vocabulary IDs.
At most sixteen sequences of at most 64 tokens are supported. Matching tokens
are omitted from presentation; pending prefixes are buffered until resolved.
A host stop cancels remaining scheduling and discards the private unpublished
candidate. It does not rewrite consumed model history or free in-flight storage.

HTTP admission allows at most `queue_capacity + max_active` requests, including
CPU preparation, queued generation, and streamed responses awaiting delivery. Admitted
requests wait asynchronously for one of `frontend_workers` CPU preparation
slots. The native ingress channel and pending admission deque each hold at most
`queue_capacity` requests. A dropped response cancels its request. A slow reader that fills its
bounded event queue cancels generation and receives an error if still
connected. Neither case blocks graph scheduling. HTTP shutdown allows five
seconds for connection drain, then drops remaining connections; CUDA ownership
remains governed by its separate stream-drain receipt.

Full HTTP ingress returns 429; invalid request preparation returns 400.
Scheduler admission rejection, including exhausted KV pages or snapshots awaiting
GPU retirement, returns 503 for nonstream requests. It does not cancel other
requests or mark the model unhealthy. Model execution, device and output errors
return 500. After SSE headers have been sent, either admission rejection or
execution failure emits one terminal `error` event under HTTP 200, without
`[DONE]`. Admission rejection uses a distinct native event; error text does not
determine the status code.

`/metrics` exposes native Prometheus counters and gauges for request lifecycle,
graph launches, prefix reuse, snapshot publication and eviction, and resource
capacity. Metric labels contain no request identifiers, prompt contents, or token IDs.
`sparkgap_mixed_grammar_decode_steps_total` counts successful decode submissions
whose actual request lanes include both constrained and unconstrained requests.
`sparkgap_constrained_max_inflight` records the maximum outstanding executor
frames immediately after a successful constrained submission. It is zero before
any constrained submission and must remain one afterward. Both measurements
cover the lifetime of the native model worker.

## Graph families

There is exactly one graph for each required `(kind, batch_capacity,
token_capacity, sampling_mode)`:

| Kind | Capacities | Semantics |
|---|---|---|
| `init` | Batch 1, 2, 4; token 1 | Validate, reset live hybrid/QSA state, initialize request sampler, finalize |
| `restore` | Batch 1, 2, 4; token 1 | Validate, copy immutable state, invalidate QSA raw history, initialize sampler, finalize |
| `snapshot` | Batch 1, 2, 4; token 1 | Copy valid terminal candidate to immutable state, finalize |
| `prefill` | Batch 1; tokens 1, 16, powers of two from 64 through `prefill_chunk_size`, and `prefill_chunk_size` itself | Consume prompt chunk; sample only on its final chunk |
| `decode` | Batch 1, 2, 4; token 1 | Consume GPU-resident pending token and sample its successor |
| `speculative` | Batch 1, 2, 4; configured widths with batch times width at most eight | Verify pending input and proposals; commit and emit the accepted prefix |
| `resume` | Batch 1, 2, 4; token 1 | Sample saved hidden state after a complete prompt cache hit |
| `vision` | Batch 1; patches 256, 1024, 4096, 16384 | Encode one image into its request's feature arena |

Only batches up to `max_active.next_power_of_two()` are required. Padded rows
and tokens never mutate state or produce active completion records. Graphs
share persistent input backing and execute on one model stream. Every variant
finalizes all four physical completion lanes for its submission sequence.

Every graph uses sampling mode `general`. Temperature zero remains a valid
request value and produces a one-hot target distribution through the same
stochastic sampling path. Positive-temperature and mixed-temperature batches
use that path too; there is no separate greedy executable. Slot generations
must match their admissions. Missing or duplicate graphs reject startup.

Capacities are sorted and deduplicated; a chunk size of 3072 requires 1, 16, 64,
128, 256, 512, 1024, 2048 and 3072. With four slots and chunk size 4096, the
catalog contains 28 graphs without speculation; a 512-token configuration
contains 25. When `speculative_widths` is nonempty, ABI3 and the configured
speculative variants replace ordinary decode. All eight widths give 39 graphs
at chunk size 4096, or 36 at chunk size 512: widths 1–8 at batch one, 1–4 at
batch two, and 1–2 at batch four. Graph names use the kind and capacity.
The fixed metric `sparkgap_sampling_graph_launches_total` exposes successful
sampling graph submissions with the `general` mode label, including
consumption-only prefill.

### Speculative ownership and scheduling

Speculative catalogs register `copy_token_ids:int64[4,7]`,
`copy_token_count:int32[4]`, `copy_expected_consumed:int64[4]` and
`copy_generation:int64[4]`.
Rows are physical request slots. The `mtp` proposal source additionally requires
`draft_page_tables:int32[4,context/64]` on prefill, resume and speculative graphs.
The native worker uploads appended target and draft page entries before use.

For MTP, target logical page `j` uses request page ID `j`; draft logical page
`j` uses request page ID `j+1`. Each live MTP request reserves one extra physical
page. At a target prefix of `64N` tokens, cached IDs `0..N-1` own the target
prefix and complete draft pages through ID `N-1`. The partial draft page uses
private ID `N`. The hybrid checkpoint retains that partial page's contents and
selector continuation state, and the recipe restores them before using the
private draft page. Publication transfers exactly `N` target-page IDs; private
suffix and extra pages retire behind the last referencing fence. Branches never
share a writable partial draft page. Copy-only requests require neither draft
page tables nor the extra page; speculative completion field 13 is `-1`.

Admission reserves target verification space through the output budget plus
maximum width minus one, capped at the model context. Width selection is bounded
by remaining context rather than remaining output budget; device commit clips
at EOS or the budget. Every speculative round runs alone through its completion
transfer. Ordinary decode retains two in-flight submissions. Device counters
remain authoritative, and completions validate transactionally before any lane
advances. Custom token stops require width one. Grammar requests with the MTP
proposal source also require width one.

Copy proposals search at most 4096 previous positions for an exact suffix of
eight or four tokens and return up to seven observed continuation IDs. MTP uses
copies for unconstrained lanes at every temperature. Copied IDs use their actual
one-hot proposal distributions for stochastic verification. The copy-only source
bounds width by the shortest valid continuation among the
active lanes plus one, the context limit and `8 / batch_capacity`. No available
copy, or `speculative_copy=false`, selects width one. Stochastic copied IDs have
a one-hot proposal distribution; target filtering and rejection sampling retain
the configured temperature, penalties, top-k, top-p and min-p.

Copy-only graphs additionally register `copy_grammar_mask:int32[4,8,7760]`
and `copy_grammar_mask_count:int32[4]`.
The worker clones the llguidance parser and lexer caches before advancing the
copy chain. Mask row zero is the committed output prefix, which already contains
the incoming pending token. Row `j` follows the first `j` copied tokens; the
last mask covers the target correction or bonus. Forbidden copied tokens, EOS,
or a successful transition with no available next-token mask truncate the copy
before that token. Parser errors fail the request. The target can still emit
EOS from its allowed distribution. Only emitted completion tokens advance the
committed parser. A constrained width-one round carries exactly its committed
mask when no copy is available. Unconstrained mask count is zero.

The worker uploads the legal copy count and corresponding mask count, which may
exceed the selected graph width. Graphs must validate the generation, consumed
count, used token IDs and sufficient mask rows before any target mutation.
Inactive metadata uses generation zero, consumed count `-1` and zero counts.
The ordinary committed mask remains the input for prefill and resume sampling.

Adaptive scheduling measures emitted tokens per full-graph CUDA nanosecond by
batch bucket and context bins up to 2048, up to 32768, and above 32768. It samples
unmeasured prepared widths, then uses an exponential moving average with bounded
exploration every 64 completed nonempty rounds. A configured fixed width selects
the largest prepared width no greater than that width, remaining context, and
eight divided by the batch capacity. Adaptive selection uses the same bounds.
No expert-reuse or confidence predictor is implied by these measurements.

Metrics `sparkgap_speculative_{rounds,gpu_nanoseconds,emitted_tokens,offered_drafts,
accepted_drafts}_total` use fixed batch and width labels. Graph time excludes
uploads and D2H completion transfer. `sparkgap_speculative_drafts_total` records
offered and accepted counts by one-based depth and actual `mtp`/`copy` source;
accepted counts are clipped at EOS and output budget. Completed records can
include a lane cancelled after its graph submission.

Initialization and restoration prevalidate both transfer and admission before
mutating the paired packed row. Transfer failures suppress admission and reach
completion status 3. Snapshot failures prevent cache publication. Vision runs
after initialization/restoration and before prefill or resume.

## Forward inputs

| Buffer | Type and full backing shape | Meaning |
|---|---|---|
| `row_slot_ids` | int32[4] | Packed rows → physical request slots 0..3; padding -1 |
| `row_generations` | int64[4] | Positive request epochs |
| `live_rows` | int32[1] | Logical packed row count |
| `query_lengths` | int32[4] | Logical prefill lengths |
| `token_ids` | int64[prefill_chunk_size] | Prefill token IDs |
| `prefill_positions` | int64[prefill_chunk_size,3] | Interleaved temporal, height, width mRoPE positions |
| `vision_token_indices` | int32[prefill_chunk_size] | -1 for text; relative request-arena feature index otherwise |
| `page_tables` | int32[4,context/64] | Physical-slot rows of KV page IDs |
| `sample_enabled` | int32[4] | Per-packed-row sampling gate |
| `sampling_grammar_mask` | int32[4,7760] | Physical-slot persistent sampling masks |

Prefill starts at consumed position 0 or a 64-token boundary. Full chunks
consume `prefill_chunk_size` tokens. The final tail uses the smallest captured
capacity that fits its logical length and transitions directly to decode.
Only logical token, position and image-index spans are uploaded; graph padding
is masked. Configured buffer extents and every required graph variant are
validated before worker startup. Decode
derives pending tokens, consumed positions, history and mRoPE delta on the
device. Rust uploads row metadata and only newly assigned page-table entries.
Prompt tokens, histograms and unchanged prefix page tables are not retransmitted
per decode step.

## Admission and state transfers

All leading-four admission arrays use the same packed order as state transfers.
The admission field prefix is `admission_`:

| Suffix | Type and shape |
|---|---|
| `count` | int32[1] |
| `slots` | int32[4] |
| `generations` | int64[4] |
| `seed_bits` | int64[4], bit-preserving views of uint64 seeds |
| `parameters` | float32[4,6]: temperature, top-p, min-p, repetition, presence, frequency |
| `top_k` | int32[4] |
| `max_new_tokens` | int64[4] |
| `eos_token_ids` | int64[4,8] |
| `eos_count` | int32[4] |
| `histogram` | int32[4,248320], full prompt counts including cached tokens |
| `grammar_mask` | int32[4,7760] |
| `grammar_enabled` | int32[4] |

State staging names are `state_source_slots` int64[4],
`state_destination_slots` int64[4], `state_operation` int32[4],
`state_qsa_slots` int64[4], `state_accepted_tokens` int32[4], and
`state_num_ops` int32[1]. COPY is 0 and RESET is 1. Null state row 0 is padding;
live rows are `1 + physical_slot`, fixed candidate rows `5 + physical_slot`,
and immutable rows `9 + SnapshotId`. Initialization/restoration uses QSA row
`1 + physical_slot` and accepted count 1, giving anchor `consumed - 1`.
Snapshot copies omit QSA invalidation with row -1 and accepted count 0.
Inactive transfer rows have source/destination 0.

## Vision inputs

| Buffer | Type and full backing shape |
|---|---|
| `vision_pixel_values` | float32[16384,1536] |
| `vision_interpolation_indices` | int64[16384,4] |
| `vision_interpolation_weights` | float32[16384,4] |
| `vision_position_ids` | int64[16384,2] |
| `vision_cu_seqlens` | int32[2], `[0, num_patches]` |
| `vision_num_patches` | int32[1] |
| `vision_slot` | int32[1], physical request slot |
| `vision_generation` | int64[1] |
| `vision_feature_offset` | int32[1] |

Each graph writes `num_patches/4` BF16[2560] features into persistent arena
`[4,16384,2560]` at the given offset after checking the request generation.
Vision completion preserves consumed/generated counters and emits no token.
Prefill validates feature ownership and bounds before any model mutation.

## Candidate and completion invariants

The decoder exports the actual hybrid state at every crossed 64-token boundary,
including boundaries inside a prefill chunk. The fixed candidate therefore
contains the greatest completed boundary strictly beyond the restored prefix.
Its exact state includes recurrent values, convolution windows, hidden state,
history, consumed count and boundary-correct mRoPE delta. Rust records the
corresponding producing fence; it never labels a live unaligned state as a
rounded checkpoint. A request with no newly crossed boundary reuses its pinned
prefix and publishes no candidate.

After terminal generation and all in-flight writes, a snapshot graph copies
the fixed candidate into a privately reserved immutable row. Publication waits
for its successful completion. Cancellation drops scheduling immediately but
retains all referenced slots, pages, snapshots and generations through their
last GPU fence. Completion status 3 carries device validation failures; it
cannot produce a successful checkpoint or response.

Cache eviction reclaims unpinned leaves first. If snapshot pressure remains and
all leaves are pinned, an ancestor with one child and no direct request pin may
be removed. Its ordered tail pages transfer to that child, whose checkpoint ID,
snapshot, and complete page chain remain unchanged. The removed prefix stops
matching lookups. Its snapshot retires behind its last GPU fence; transferred
pages retain that fence on the child. This permits appended conversations to
outlive the snapshot count while preserving their newest checkpoint. Branching
ancestors and directly pinned checkpoints cannot be compacted. One snapshot
slot cannot simultaneously retain a matched checkpoint and reserve another
request's publication slot.

## Raw token qualification

Bootstrap configuration `qualification_token_trace=true` enables
`POST /__qualify/generate` on a loopback listener. The default is false, and
the route is absent when disabled. The handler uses the ordinary Rust chat
preparation, request queue, scheduler, prepared graph catalog, and completion
records. It bypasses text decoding and tool/reasoning presentation.

The JSON envelope contains `schema_version: 1`, `id`, `ignore_eos: true`,
`expected_prompt_tokens`, and `request` containing an ordinary chat request.
The request must be nonstreaming text with temperature 0, a 64-token output
budget, neutral repetition/presence/frequency penalties, and no grammar,
custom token stops, or images. Tool-bearing templates and assistant
continuations retain their normal preparation semantics. Rendered token IDs
must equal `expected_prompt_tokens` before model admission.

Qualification admission skips prefix lookup and starts at consumed position 0.
Allocation, eviction, generation stamps, terminal snapshot publication, and
GPU retirement use the same scheduler as ordinary requests. Only that
admission uploads `eos_count=0`; EOS tokens are consumed as ordinary subsequent
model inputs until the output budget is reached. Ordinary chat retains its
configured EOS behavior. Each validated completion token is emitted directly
before token-stop processing, including EOS IDs.

The JSON response contains `schema_version`, `id`, `model_id`, the 32-byte
model/recipe identity as hexadecimal `model_identity`, `rendered_prompt`,
`prompt_token_ids`, `generated_token_ids`, `ignore_eos: true`, `cold: true`,
`reused_prompt_tokens`, `finish_reason`, and `usage`. Success requires exactly
64 observed IDs, a length finish, matching prompt counts, and zero reused
tokens reported by the admitted request. A qualification collector records
the model revision, tokenizer/template fingerprints, prepared manifest,
native binary, library sources, and request/response bytes alongside these
observations.
