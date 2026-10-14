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

`GET /v1/models` lists the loaded model and configured aliases. As with Chat
Completions, generation requests always use the loaded model, regardless of the
request's `model` string. Returned objects identify the loaded model. `/health`
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
rules.

Model output parsers emit neutral semantic events. The ordered accumulator in
`server/output.rs` retains text, reasoning, and tool blocks. Protocol encoders use
the same accumulated output for buffered responses and SSE. Tool arguments are
JSON objects, bounded to 8 MiB. Existing model parsers buffer tool envelopes until
they are complete, so argument events do not imply incremental generation of an
unfinished call. Truncated tool envelopes are omitted and the response reports
its output limit; malformed envelopes at natural completion are errors.

Supported tools are client-executed JSON functions with automatic selection or
`tool_choice: "none"`. Responses also supports text custom tools by presenting a
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

Forced, required, or named selection, strict function schemas, and disabling
parallel calls are rejected. Hosted search, computer execution,
file stores, audio, and remote image downloads are unsupported. Images must be
base64 PNG/JPEG data and fit the loaded package's prepared image capacities.

Anthropic histories preserve text, thinking, tool use/results, errors, and image
blocks. Assistant-prefill messages use the packaged template's continuation
support. `context_management` may request preservation of all thinking blocks
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
Lark output grammars use the existing constrained decoder and cannot be combined
with active tools, thinking, assistant continuation, or custom stops.

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
Anthropic cache creation/read counters are zero because its billing cache
contract differs from Helios prefix reuse. Responses exposes native reused
prompt tokens through `input_tokens_details.cached_tokens`. Chat observation
extensions (`return_token_ids`, `return_token_timestamps`, and prefix reuse)
retain their existing behavior. Stop sequences report the exact matched string
through Anthropic `stop_sequence`.

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
completion usage sums all generated choices. `best_of` must equal `n`; ranked
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
The coding-client tests exercise Codex 0.162.1 and Claude Code 2.1.296 with actual
file inspection, editing, and test execution in temporary repositories.

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
