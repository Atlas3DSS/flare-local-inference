# HTTP qualification with fixed QSA partitions

The unchanged default HTTP suite and its original mixed-grammar case pass
**152/152 checks**, with zero case errors, across 39 model requests and 214 total
HTTP requests. The server uses checkpoint
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, BF16 GDN intermediates, FlashInfer
sampling, and fixed 16-way QSA attention at every graph capacity.

The [receipt](/data/sparkgap-artifacts/validation-receipts/server-http-fixed16.json) retains every check and metric delta.
The [evidence archive](/data/sparkgap-artifacts/validation-receipts/server-http-fixed16-artifacts.tar.gz) contains the original
receipt, exact requests, JSON/SSE responses, scripts, server/client logs, memory
report and shutdown evidence. Its SHA256 is
`c8b75ccccfd308db21733ba64b991f4675ad1a049ad55a279aa2d634a1082d32`.

| Contract | Result |
| --- | --- |
| Canonical tool-result prompt, cold JSON versus cached SSE | Identical complete message and finish reason; 448-token prefix reused |
| PNG image prompt, cold JSON versus cached SSE | Identical complete message and finish reason; 64-token prefix reused |
| Text, canonical chat, thinking, tool calls and structured literal tags | All existing completion, token-count and JSON/SSE equivalence checks pass |
| Mixed constrained/unconstrained generation | Exact constrained literal, shared decode observed, maximum one in-flight constrained submission |
| Concurrent requests, cancellation and slot reuse | Existing acceptance checks and resource accounting pass |
| Final health and native source identity | Pass |

The [row-dependent QSA baseline](server-http-flashinfer-regression.md) passed
150/152 checks; its failures were the tool-result and PNG cold/cache comparisons.
The saved fixtures, budgets and assertions are unchanged. The separate
[595-predicate prefix gate](prefix-replay-fixed16.md) establishes bitwise state,
logit and layer-3 QSA intermediate equality for the 459-token tool-result prompt.
Neither receipt establishes independent full-model numerical accuracy.

The installed [patch receipt](qsa-partitions-install.json) identifies fixed QSA
query partitions and unchanged FlashInfer sampling code. Native extension SHA256
is `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
The embedded sampler hash identifies an installation snapshot; the original
HTTP receipt does not pin the loaded sampler binary. See
[sampling library provenance](sampling-binary-provenance.md).
The catalog has 37 graphs, 9,768 KV pages, 625,152 cached-token capacity,
37 checkpoint slots and four active request slots. QSA workspace remains
166,879,232 bytes, including 24.094 MiB of attention partials and LSE.

The actual CLI process was PID 1052913, launched with `--bind 127.0.0.1:0` and
resolved to `127.0.0.1:43073`. SIGTERM after qualification exited zero, released
the GPU allocation and closed the listener. `PreparedModel.close()` rejects
unconfirmed native drain; a separate drain flag was not exported by this CLI.

```bash
.venv/bin/python scripts/qualify_server.py \
  --base-url http://127.0.0.1:43073 \
  --gdn-intermediate-dtype bfloat16 \
  --memory-report /tmp/sparkgap-fixed16-memory.json \
  --require-metrics --mixed-grammar \
  --artifacts /tmp/sparkgap-http-fixed16 \
  --output /data/sparkgap-artifacts/validation-receipts/server-http-fixed16.json
```

Reproduction requires a running server with the recorded configuration and
fresh output destinations. Request durations are diagnostic observations;
this suite does not measure serving throughput or the native replay floor.
