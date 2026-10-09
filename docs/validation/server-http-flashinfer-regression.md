# HTTP regression with prepared FlashInfer sampling

The unchanged `scripts/qualify_server.py` default suite plus `--mixed-grammar`
passes **150 of 152 checks** across 39 model requests and 215 total HTTP requests.
It exits with status 1. Both failures compare the first JSON response with the
subsequent streamed response for the same prompt:

- `canonical-tool-results.greedy_stream_equivalence`: both finish with `stop`,
  but the first response says `Hello! 👋` and the repeated response incorporates
  the two tool results.
- `png-image.greedy_stream_equivalence`: both consume their 32-token output
  budgets, but the generated image descriptions differ.

The [receipt](/data/sparkgap-artifacts/validation-receipts/server-http-flashinfer-regression.json) preserves the failed
aggregate and all original checks. The [artifact archive](/data/sparkgap-artifacts/validation-receipts/server-http-flashinfer-regression-artifacts.tar.gz)
contains exact request bodies, raw JSON/SSE responses, fixture image, executed
script, memory report and unmodified raw receipt. No retries or input/oracle
changes were applied.

## Execution contract

The client used the actual 37-graph ABI 2 server at `http://127.0.0.1:38761`,
process 839497, with exclusive model-request access and BF16 GDN intermediate
arithmetic. The checkpoint is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Installed native SHA256 is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
The executed script SHA256 is
`dacdfec000f3b74d3179cd21f54da922a039d36cbedb535aa27aed3614cf8466`.
The receipt records checkpoint frontend asset hashes, pinned Transformers
revision, memory configuration, source hashes and unchanged-source check.

The suite retains text/chat, thinking, tools and tool results, four simultaneous
requests, complete/aligned prefix reuse and branches, structured literal tags,
PNG image input, grammar coexistence, cancellation, request-slot reuse and
health checks. There are zero case exceptions. The explicit cancellation is
observed in native counters and subsequent requests reclaim all four slots.

## Historical mixed-grammar fixture

The unchanged historical prompt requires the same 512-word `alpha ` JSON string
and retains its 1,024-token output budget. It completes the exact 3,074-byte
literal with finish reason `stop` and 516 nonempty SSE content deltas.
The stream exposes no completion-usage object. The unconstrained simultaneous
request answers `Paris`. Native counters record one shared constrained/plain
decode submission, 516 total decode launches, two completed requests, zero
failures/cancellations, and maximum constrained in-flight count one. All
mixed-grammar checks pass.

The [sampling-routing fixture](server-http-sampling-routing.md) has a different
fixed prompt and preserves its separate 118/120 failure result: that stream
uses a longer legal tokenization and reaches its budget before closing the
literal. Both receipts retain their exact inputs, budgets and completion
oracles. Neither HTTP suite supplies an independent numerical model reference.

The [server shutdown receipt](server-sampling-shutdown.json) records exit status
zero, listener removal and GPU release after both HTTP suites completed.

## Command

```sh
.venv/bin/python scripts/qualify_server.py \
  --base-url http://127.0.0.1:38761 \
  --gdn-intermediate-dtype bfloat16 \
  --memory-report /tmp/sparkgap-sampling-routing-memory.json \
  --require-metrics --mixed-grammar \
  --artifacts /tmp/sparkgap-http-flashinfer-regression \
  --output /data/sparkgap-artifacts/validation-receipts/server-http-flashinfer-regression.json
```
