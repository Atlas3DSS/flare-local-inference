# Autotuned HTTP vision, cache identity and mixed grammar

The fixed vision/grammar fixture passes **44/44 checks** across **eight model
requests** and 23 total HTTP requests, with zero case errors. The server runs
from the main Sparkgap source with the exact Qwen checkpoint at revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, sequential single-key-split BF16 GDN,
16-way QSA, FlashInfer sampling and all 37 captured graph variants.

All eight request files are byte-identical to the
[fixed-QSA fixture](server-http-vision-grammar-fixed16.md). All 44 check names
and their order are unchanged. The [receipt](/data/sparkgap-artifacts/validation-receipts/server-http-vision-grammar-autotuned.json)
preserves the assertions, raw JSON/SSE responses and every metric delta.

| Contract | Result |
| --- | --- |
| Full-size image | Both encodes execute at 16,384 patches and 4,096 merged features; canonical prompt usage is 4,116 tokens |
| Full-size image JSON versus SSE | Complete output and finish reason agree; cached execution reuses the recorded prefix |
| Identical text with distinct RGB images | Both cold requests reuse zero tokens; each repeat restores its own 64-token checkpoint |
| Image-conditioned cache output | Both cold/repeat message and finish-reason comparisons agree |
| Grammar and plain generation | Exact constrained literal and Paris answer; shared decode observed with one constrained submission in flight |
| Completion, resource accounting and health | Every original check passes |

The full-size source PNG SHA-256 is
`93a5fe9a906cd3081f52d12144db9cfee9e1c7632fac48e743b8da889ad1cde6`.
Its processor grid is `[1,128,128]`; the receipt retains canonical pixel,
position, interpolation and mRoPE hashes. The two cache-identity images retain
the original RGB/greyscale fixture hashes.

The unchanged qualifier has SHA-256
`dacdfec000f3b74d3179cd21f54da922a039d36cbedb535aa27aed3614cf8466`.
The external fixture wrapper sets the request-construction timestamp to
`2026-10-09T10:37:32.143040+00:00`, preserving nonce `708c656965cc`.
Actual HTTP execution began at **2026-10-09T11:58:57.993155+00:00** and finished
at **2026-10-09T11:59:12.692669+00:00**. The original timestamp remains in
`started_at` solely as an input to fixture construction; `execution_started_at`
records this execution. The wrapper validates all eight payloads before sending
any corresponding request. Its SHA-256 is
`4bce009149be54570272306ad516874e1ff7f7f9169193890d301b6776fe8203`.

Two harness failures occurred before HTTP execution and remain archived. The
isolated-root wrapper could not find `native/Cargo.toml` during provenance
construction and exited 1. Pointing the wrapper at the main repository preserved
the qualifier bytes; a subsequent invocation exited 2 because the prior artifact
directory contained a copied `qualify_server.py`. That directory and both failure
logs/receipts are retained. Execution using a fresh artifact destination completed
with exit 0. No model request, fixture, budget or acceptance assertion changed.

The [evidence archive](/data/sparkgap-artifacts/validation-receipts/server-http-vision-grammar-autotuned.tar.gz) contains raw
requests/responses, source snapshots, the fixture PNG, original fixture receipt,
both wrapper versions and preflight failures, memory report, native binary and
actual sampler mapping/binary. All 155 main-source hashes and 21 installed
runtime-file hashes match the recorded qualified source and
[30-file recurrence-policy installation](recurrence-policy-install.json).
The native extension is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`;
the mapped sampler binary is
`1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`.
The same process and memory configuration serve the
[152-check default HTTP suite](server-http-autotuned.md).

Archive SHA-256:
`7bf53515dab6bf9bc533de2b575d086661840d484cd1f81f038ef82ae94336ce`.
All 799 payload members were re-read and verified against `MANIFEST.json`.
The final server log, shutdown receipt and process mapping are included.
SIGTERM produces exit 0, closes the listener and releases the GPU allocation;
source hashes and the mapped sampler remain unchanged. These checks establish
the recorded capacity/cache/API behavior; independent encoder accuracy and
serving performance acceptance remain unresolved.

Reproduction requires the archived wrapper, original fixture receipt and request
files, and full-size PNG at their recorded paths, plus the unchanged qualifier.
Restore those local artifacts from the manifest before running; use fresh output
destinations and the memory report from the actual server process:

```bash
.venv/bin/python /tmp/sparkgap-autotune-replay-vision-fixture.py \
  --base-url http://127.0.0.1:38761 \
  --memory-report /tmp/sparkgap-autotuned-server-memory.json \
  --artifacts /tmp/sparkgap-http-vision-grammar-autotuned-reproduction \
  --output /tmp/sparkgap-http-vision-grammar-autotuned-reproduction.json
```
