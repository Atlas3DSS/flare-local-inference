# Native inference APIs

Helios serves the loaded runtime package through four generation protocols.
Every protocol uses the same tokenizer, prepared chat template, image processor,
sampling defaults, scheduler, prefix cache, and CUDA graphs. Clients execute tools
and return their results in subsequent requests.

| Route | Input | Output |
| --- | --- | --- |
| `POST /v1/chat/completions` | Chat messages | Assistant message or chat chunks |
| `POST /v1/completions` | Raw text or token IDs, optionally batched | Text choices or completion chunks |
| `POST /v1/messages` | Anthropic messages and system blocks | Message content blocks or named SSE events |
| `POST /v1/messages/count_tokens` | Anthropic message request without generation | Exact prepared input token count |
| `POST /v1/responses` | Responses input items and instructions | Ordered output items or named SSE events |

`GET /v1/models` returns one card for the loaded model. Its `id` is the first
nonempty configured alias, or the checkpoint identifier when no alias is set.
`GET /v1/models/{model}` returns that same card for the checkpoint identifier or
any configured alias; repository IDs containing slashes are supported. Unknown
names return HTTP 404 with an OpenAI-style `model_not_found` error. Both routes
use the configured bearer authentication.

Model cards include the vLLM-compatible fields `id`, `object`, `created`,
`owned_by`, `root`, `parent`, `max_model_len`, and `permission`. `max_model_len`
is the effective serving context limit in tokens, including prompt and output.
`root` is the loaded model's identifier, `parent` is null, and `owned_by` remains
`local`. `created` is the serving instance's discovery initialization time in
Unix seconds; cards and permission IDs stay stable until the server restarts.
Permissions advertise sampling and viewing, with unsupported log probabilities,
fine-tuning, engine creation, and search indices disabled.

The additional `flare` object reports prepared and selected serving metadata:

| Field | Meaning |
| --- | --- |
| `vocab_size` | Prepared output vocabulary size |
| `max_concurrency` | Selected maximum active requests |
| `prefill_chunk_size` | Maximum tokens in a prefill chunk, not the context limit |
| `default_max_tokens` | Default output-token budget when a request omits it; remaining context still bounds output |
| `sampling_defaults` | Packaged seed, temperature, filters, and penalties used for omitted request fields |
| `max_images_per_prompt` | Prepared image count limit; zero for text-only packages |
| `input_modalities` | `text`, plus `image` when vision is prepared |
| `output_modalities` | `text` |

Generation requests always use the loaded model, regardless of the request's
`model` string. Returned generation objects identify the loaded model. `/health`
and `/metrics` retain their existing behavior.

## Transport and admission

Generation uses HTTP JSON or `stream: true` with SSE. WebSockets, background
execution, and hosted tools are unsupported. Anthropic requests require
`anthropic-version: 2023-06-01`.

When a serving key is configured, OpenAI routes use `Authorization: Bearer KEY`.
Anthropic routes also accept `x-api-key: KEY`; supplying both headers requires
both to match. `/health` remains unauthenticated. Flare reads its serving key
from `--api-key-file`, independently of client configuration.

Request bodies support `identity`, `gzip`, zlib `deflate`, `br`, and `zstd` content
encodings. Both encoded and decoded bodies are bounded by `max_body_bytes`
(32 MiB by default). Decompression and frontend processing share the bounded
frontend worker pool. Invalid requests receive protocol-specific JSON errors;
failures after streaming begins produce protocol-specific terminal error events.

Each completion batch reserves one admission slot per generated choice. Every
child is prepared and validated before the group is submitted to the native
worker. An oversized group is rejected; temporary admission exhaustion returns
429. Disconnecting a stream cancels all unfinished children, and native work
drains before its resources are reclaimed. Output queues remain bounded.

## Conversation and tool semantics

The neutral request representation in `helios/src/server/inference.rs` separates
conversation messages from raw prompts. Conversation parts represent text,
images, reasoning, tool calls, and tool results. Protocol decoders populate these
types directly. `server/request.rs` lowers them into the packaged chat template
and model frontend; no endpoint translates through another endpoint's JSON.
When developer instructions are present, system and developer instructions become
one leading system message. Instruction sections retain their role labels and
request order; ordinary conversation messages retain their relative order.
Conversations without developer messages use the template's native system-role
rules where supported. The shared frontend probes the prepared template once for
inline system-message support. If the template rejects or omits inline system
text, noninitial system messages become user messages at the same positions,
with their text wrapped in `<system>...</system>`. Text blocks are joined with
blank lines. The leading system message and preceding conversation remain
unchanged. This conversion supplies a textual role cue; it does not provide a
native system role to models whose templates cannot represent one inline.

Model output parsers emit neutral semantic events. The ordered accumulator in
`server/output.rs` retains text, reasoning, and tool blocks. Protocol encoders use
the same accumulated output for buffered responses and SSE. Tool arguments are
JSON objects, bounded to 8 MiB. Existing model parsers buffer tool envelopes until
they are complete, so argument events do not imply incremental generation of an
unfinished call. Truncated tool envelopes are omitted and the response reports
its output limit; malformed envelopes at natural completion are errors.

Qwen treats `<tool_call>` as an implicit end to reasoning. Function-tag calls
complete at `</function>` and accept omitted outer tool tags in answer content.
A missing `</parameter>` is closed by the next parameter or function boundary.
Parameter schemas guide
best-effort type conversion; non-strict schemas do not reject invalid values.
Qwen removes one leading and one trailing wrapping newline from parameter values;
MiMo's compact tags preserve their values verbatim, including indentation and
newlines.

Supported tools are client-executed JSON functions with `tool_choice` set to
`"auto"`, `"none"`, or `"required"`. Anthropic's `{"type":"any"}` selects the same
required mode. Required mode needs at least one declared tool, adds an instruction
to call a tool to the prompt, and validates that generation produced at least one
completed call to a declared tool. Sampling is unchanged; no calls are fabricated
and no retries are attempted. Missing calls, including output truncated before a
complete call, produce HTTP 500 for buffered requests or a terminal error event
for streams. A stream may deliver partial text before that error, but does not
emit a success terminator. Responses also supports text custom tools by presenting a
JSON `{ "input": string }` function to the model and restoring the custom call on
output. Anthropic's `type: "custom"` uses its ordinary `input_schema` contract.
Responses namespaces become unique `namespace__name` names in the model template
and regain their namespace on output; ambiguous names are rejected.

Responses custom tools may specify Lark or regex grammars. Helios compiles them
before admission, includes their contract in the model's tool description, and
validates completed input against the exact bytes before emitting a completed
call. Invalid output fails the response; generation is not constrained to the
grammar and no automatic repair is attempted. Grammar compilation uses the same
bounded llguidance compiler as structured output.

`parallel_tool_calls: false` permits at most one tool call per response.
Anthropic's `tool_choice.disable_parallel_tool_use: true` applies the same limit.
The output validator fails the response when a second call begins, before emitting
that call. Buffered requests receive HTTP 500; streams receive a terminal error
and may already have delivered the first call. Combined with required mode, a
successful response contains exactly one call. Generation uses ordinary sampling.

Named tool selection and strict function schemas are rejected. Hosted search,
computer execution, file stores, audio, and remote image downloads are unsupported. Images must be
base64 PNG/JPEG data and fit the loaded package's prepared image capacities.

Anthropic histories accept inline system text and preserve text, thinking, tool
use/results, errors, and image blocks. Message-level control metadata, including
`output_config.effort`, is ignored; top-level generation controls apply to the
request. Empty system messages produce no prompt text. Assistant-prefill messages
use the packaged template's continuation support. `context_management` may
request preservation of all thinking blocks
(`clear_thinking_20251015` with `keep: "all"`); active context edits are rejected.
Cache-control hints do not alter the native prefix-cache policy.

## Stateless Responses

Responses requests contain the complete conversation. `store` defaults to false;
`store: true`, `previous_response_id`, `conversation`, background requests, and
automatic truncation are rejected. No response retrieval or deletion service is
provided. Clients append returned output items and tool-result items to their
next request.

Output items retain their order and have stable IDs throughout a response.
SSE includes response, output-item, content-part, text/reasoning, and tool-call
lifecycle events with increasing sequence numbers. Success ends in
`response.completed`; exhausted output budgets end in `response.incomplete` with
`max_output_tokens`; generation failures end in `response.failed`. Responses SSE
does not use an OpenAI Chat `[DONE]` sentinel.

Readable reasoning is returned as `reasoning_text` content with an empty summary.
Opaque encrypted-only histories are rejected. Helios does not fabricate hosted
provider reasoning signatures or encrypted continuation state.

## Reasoning, sampling, and usage

Prepared model defaults fill omitted sampling fields; explicit request values
win. Temperature must be positive. Explicit `temperature: 0` (including negative
zero) returns HTTP 400 with `invalid_request_error` and the message:
“temperature=0 is unsupported because greedy decoding incorrectly collapses the
model’s probability distribution, and is not appropriate for most use cases”.
Omit temperature to use the model default, or provide a positive value. The request is never silently clamped or replaced.
Native extensions such as `top_k`, `min_p`, repetition penalties, token stops,
`grammar`, and `chat_template_kwargs` remain available. JSON output formats and
Lark output grammars constrain the final answer while permitting the model's
reasoning and tool-call envelopes. Tool arguments do not use the final-answer
schema. Active tools and thinking remain available; assistant continuation and
custom stops cannot be combined with output constraints.
For conversational JSON output, the schema and a JSON-only formatting instruction
are included in the system prompt as well as enforced by token masks.

Anthropic thinking budgets are advisory effort hints, not independent token
budgets. Budgets up to 5,000 map to low effort, up to 20,000 to high effort, and
larger budgets to maximum effort. Responses and Anthropic named effort settings
map into the same neutral levels. The frontend resolves those levels against the
effort vocabulary accepted by the packaged template: for example Qwen uses
`low/medium/xhigh`, while DeepSeek uses `low/high/max`. Explicit native
`reasoning_effort` or `chat_template_kwargs` values pass through unchanged.
The generation output limit includes reasoning and tool syntax.

Anthropic thinking blocks carry a deterministic `flare-thinking-v1:` marker
derived from their visible text. Replayed local markers are checked for matching
text. This marker is a local round-trip consistency check, not provider
authentication; readable thinking from other providers can be replayed.

Usage counts native prompt and generated tokens, including model control tokens.
Chat and Completions always include `prompt_tokens_details.cached_tokens`,
equivalent to enabling vLLM's `--enable-prompt-tokens-details`. This counts the
prompt tokens actually reused at admission, including restored prefixes; it is
zero when no prefix is reused. Cached tokens remain part of `prompt_tokens`
and `total_tokens`. Image requests also include
`prompt_tokens_details.multimodal_tokens.image`, the image feature tokens already
included in the prompt total. Responses reports reuse through
`input_tokens_details.cached_tokens`.

Anthropic reports reused tokens in `cache_read_input_tokens` and the remaining
prompt tokens in `input_tokens`; their sum is the token-count endpoint's result.
`cache_creation_input_tokens` remains zero because Flare does not separately
account for cache writes. These are native reuse counts, independent of
Anthropic cache-control billing policies. In streams, `message_start` initially
reports the full prompt with zero cache reads; `message_delta` supplies the final
input, output, and cache counts. Chat and Completions include final usage when
`stream_options.include_usage` is true. Chat token observation extensions
(`return_token_ids` and `return_token_timestamps`) preserve usage details,
including the legacy terminal usage when token IDs are requested without a
separate usage chunk. Stop sequences report the exact matched string through
Anthropic `stop_sequence`.

## Raw completions

`prompt` accepts a string, an array of strings, a token-ID array, or an array of
token-ID arrays. Text uses the prepared tokenizer's normal special-token policy;
explicit IDs are preserved. Empty text requires a packaged BOS token; empty token
arrays are rejected. Raw prompts bypass chat templates and reasoning/tool parsers.

`n` generates independent choices for each prompt. Choice indices are
`prompt_index * n + choice_index`; buffered results use that order and streams may
interleave indices. The configured or explicit seed advances by choice index. `echo` prepends the
original text or decoded token prompt. `max_tokens: 0` performs tokenization and
optional echo without GPU submission. Prompt usage counts each input once;
cached usage uses the first choice's admission count for each input, independent
of completion order. Zero-token requests report zero cached tokens because they
do not perform a cache lookup. Completion usage sums all generated choices.
`best_of` must equal `n`; ranked
sampling, suffix insertion, prompt scoring, and log probabilities are rejected.

## Clients and tests

The OpenAI SDK uses `base_url="http://127.0.0.1:8000/v1"`. The Anthropic SDK uses
`base_url="http://127.0.0.1:8000"`. Supply the configured serving key as `api_key`;
an arbitrary nonempty key satisfies SDK construction when server authentication
is disabled. The SDK tests use OpenAI 2.54.0 and Anthropic 0.125.0.

[Claude Code settings](clients/claude-code.json) configure its base URL, model,
and API key. Client settings may use environment variables; Helios behavior is
configured through its CLI, configuration file, and prepared package.

[Codex provider settings](clients/codex.toml) and its
[model catalog](clients/codex-models.json) select HTTP Responses, disable hosted
search, and describe local model capabilities. Set the catalog's context limit
to the loaded package's capacity. The catalog uses Codex's freeform `apply_patch`
tool with completed-input grammar validation.
The coding-client tests exercise Codex 0.162.1 and Claude Code 2.1.296 in normal
and bare modes with actual file inspection, editing, and test execution in
temporary repositories.

Install SDK test dependencies and run against an idle native service:

```sh
uv sync --locked --extra api-test
FLARE_API_TEST_URL=http://127.0.0.1:8000 \
  .venv/bin/python -m pytest tests/server/test_protocols.py -q
FLARE_API_TEST_URL=http://127.0.0.1:8000 FLARE_RUN_CLIENT_TESTS=1 \
  .venv/bin/python -m pytest tests/server/test_coding_clients.py -q
cargo test --manifest-path helios/Cargo.toml --features server,runtime-package --lib
```

`FLARE_API_TEST_KEY` supplies an authenticated test endpoint's key. Coding-client
tests use temporary repositories and check a real read/edit/test task. Scheduler
regressions in `tests/server/test_scheduler.py` run against ordinary, fixed-width,
and adaptive speculative execution. These environment variables configure tests,
not the native server.
