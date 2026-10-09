# Image cache identity with fixed QSA partitions

The unchanged full-size image, image-cache identity and mixed-grammar suite
passes **44/44 checks across 23 HTTP requests**, with zero case errors. The
[receipt](server-http-vision-grammar-fixed16.json) retains every check, response
and metric delta. Checkpoint revision is
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` of
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`.

Both different-image cold requests reuse zero prefix tokens, and each identical
image repeat restores its 64-token checkpoint. Both cold/repeat greedy output
comparisons pass. These are the two assertions that failed in the
[row-dependent QSA receipt](server-http-vision-grammar.json).

The full-size PNG exercises 16,384 patches, 4,096 merged vision features and a
canonically counted 4,116-token prompt. Its cold JSON and repeated SSE responses
match. The original mixed-grammar literal completes exactly, shares a decode
submission with the unconstrained request, and respects the one-in-flight bound.
All resource-accounting and final-health checks pass.

The embedded sampler hash references the installation snapshot, as detailed in
[sampling library provenance](sampling-binary-provenance.md); the original HTTP
receipt does not pin the loaded sampler binary.

The server uses BF16 GDN intermediates, FlashInfer sampling and fixed 16-way QSA
attention, with the installed build and memory configuration recorded in
[default HTTP qualification](server-http-fixed16.md). CLI PID 1078008 binds
`127.0.0.1:46279`; it exits zero on SIGTERM, closes the listener and releases GPU
storage after qualification. Independent vision/model numerical accuracy and
serving performance are not measured by this suite.

The [evidence archive](/data/sparkgap-artifacts/validation-receipts/server-http-vision-grammar-fixed16-artifacts.tar.gz) includes
exact PNG-bearing requests, JSON/SSE responses, scripts, raw receipt, logs,
memory report and shutdown evidence. Its SHA256 is
`2b9f2501d2da6648e002e2d7fba60ea650025c50b5a5d79fb3e876f5e0e3ab3b`.
The complete reproduction command is preserved in the receipt; it selects
`initial_health models png_full_capacity image_cache_identity mixed_grammar final_health`
with the original `bucket_16384.png` fixture, a 300-second timeout, metrics,
image-identity and mixed-grammar checks enabled. Fresh output destinations and
an exclusively assigned matching server are required.
