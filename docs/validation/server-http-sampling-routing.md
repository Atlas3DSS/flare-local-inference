# HTTP sampling graph selection

The exact checkpoint server completed 11 model requests with **118 of 120 checks
passing**. Both failed checks require the constrained stream to finish its entire
JSON literal; it exhausted the unchanged 1,024-token output budget with a valid
prefix. The [receipt](/data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing.json) preserves both failures,
all checks, requests, metrics and source/build provenance. The
[artifact archive](/data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-artifacts.tar.gz) contains raw
JSON/SSE responses, fixed fixtures, memory configuration, the executed scripts
and `grammar-budget-diagnostic.json`.

## Configuration

- Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
  revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`.
- Native ABI 2 library SHA256:
  `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
- Actual CLI process: PID 839497, `http://127.0.0.1:38761`, GPU
  `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`.
- Prepared storage: 37 graph variants, 9,768 KV pages / 625,152 cached tokens,
  37 snapshots, four active request slots and 4 GiB headroom.
- Fixture SHA256:
  `b8f529f1522343d88e3b97874748f1a1c300ca08e0e37c3f30dd0b4853a56f6e`.

The client had exclusive model-request access. Every isolated counter window
began and ended quiescent and checked exact admissions/completions with zero
failures/cancellations. Process identity, installed library, source hashes and
fixed fixtures remained unchanged. All 127 HTTP requests are preserved.

## Routing and request-state evidence

| Case | Observed result |
|---|---|
| Temperature zero, cold 128-token prompt | Greedy prefill/decode |
| Same complete aligned prefix hit | Greedy resume/decode; 128 tokens reused |
| Explicit temperature 0.7 with top-k 1 | General resume/decode |
| Two seeded positive-temperature requests with identical full-hit geometry | Identical messages, finish reasons and usage |
| Concurrent temperature-zero / positive-temperature pair | Both output 64 tokens; 65 decode launches, including 64 general and one greedy |
| Concurrent greedy grammar / positive-temperature pair | 63 mixed-grammar decode steps through general mode; greedy prefill and 960 additional greedy decode launches |
| Constrained-request in-flight bound | Maximum one |
| Final resource accounting | Four free request slots; 9,745 free + 23 cached KV pages; 33 free + four cached snapshots |

The mixed-temperature pair uses the same two 128-token prompts and 64-token
budgets as its separate baseline requests. Two complete prefix hits reuse 256
tokens and execute two resumes without prefill. Exactly two requests are
admitted/completed; both API usage objects report 64 generated tokens. Sixty-five
decode launches are fewer than the 126 minimum for separate execution, proving
shared decode execution under the exclusive-client conditions. The native
selector uses general mode whenever a selected lane has positive temperature.
The mixed-grammar counter separately records 63 shared constrained/unconstrained
decode submissions.

## Preserved grammar completion failures

`grammar-mixed.stream_complete` and `grammar-mixed.exact_literal` fail. The
stream returns HTTP 200, one `[DONE]`, no error event, and finish reason `length`.
Its output is the exact first 3,066 bytes of the required 3,074-byte JSON string;
the missing suffix is ` alpha "` (eight ASCII bytes). There are 1,024 nonempty
content deltas. The SSE response provides no completion-usage object.

The complete literal's canonical tokenizer encoding has 514 tokens, but that
count does not bound every legal generated tokenization. Each recorded delta
independently encodes to one token, and concatenating those encodings decodes
exactly to the observed prefix. After the initial quote, `a`, `lp`, `ha` pieces,
the stream repeatedly emits ` alp`, `ha`. This supplies a legal longer
tokenization that exhausts the fixed budget. SSE does not expose sampled GPU
token IDs, so their unique reconstruction is unavailable. The fixed input,
budget, expected literal and both failed checks remain unchanged.

No cold/cache or batch-shape numerical equivalence is established. Numerical
model acceptance gates and the unchanged HTTP regression suite are separate
receipts. Request timings are diagnostic observations.

The [server shutdown receipt](server-sampling-shutdown.json) records exit status
zero, listener removal and GPU release after both HTTP suites completed.

## Reproduction

Use an exclusively assigned running server with the recorded catalog/configuration.
Set `SERVER_PID` to that server's actual listening process ID and extract the
saved fixture into a fresh artifact directory so the exact nonce and token IDs
remain fixed. `MEMORY_REPORT` must identify that server's preparation report.

```sh
ROUTING_ARTIFACTS=$(mktemp -d /tmp/sparkgap-sampling-routing-reproduction.XXXXXX)
tar -xOf /data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-artifacts.tar.gz \
  sampling-routing/fixtures.json > "$ROUTING_ARTIFACTS/fixtures.json"
.venv/bin/python scripts/qualify_sampling_routing.py \
  --artifacts "$ROUTING_ARTIFACTS" \
  --base-url http://127.0.0.1:38761 \
  --memory-report "${MEMORY_REPORT:?Set MEMORY_REPORT to the server preparation report}" \
  --server-pid "${SERVER_PID:?Set SERVER_PID to the listening server process ID}"
```
