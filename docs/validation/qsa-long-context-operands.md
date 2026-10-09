# QSA operands at the context boundary

One unchanged context-boundary request completed through the Rust HTTP server
using the pinned Qwen3.8 checkpoint and the 37-graph general/greedy catalog.
The response reports 262,136 prompt tokens, eight generated tokens, total usage
262,144 and `finish_reason="length"`. Native metrics record 512 greedy prefill
launches, seven greedy decode launches, one admission, one completion and one
snapshot. The [receipt](qsa-long-context-operands.json) preserves the exact
request, response, token-ID and source hashes.

Layer 3 selected-attention operands are saved in
`/tmp/sparkgap-qsa-long-context/attention-operands.pt`, SHA256
`d988ee3ec3c2b3be0fc0afc3e780178045b8bcf9535267eba7df0e81d8fc4e2c`
(86,858,821 bytes).

| Record | Captured capacity | Active rows | Query positions | Selected keys per row | Rows with 2,051 keys | Referenced KV pages |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `last_prefill512` | 512 | 504 | 261632–262135 | 2048–2051 | 126 | 436 |
| `last_decode1` | 1 | 1 | 262142 | 2051 | 1 | 98 |

Both records populate all 65 selection tiles of width 32. Unlike the
[459-token operand matrix](qsa-fixed-partitions.md), these inputs exercise
multiple nonempty tiles per split for partition counts 16 and 32.

Each record contains the exact BF16 query and attended output, selected
positions, request IDs, query positions, main block table, sequence lengths and
query-start offsets. Main K/V tensors include every physical page referenced
by that record's selected positions. The receipt identifies every tensor's
shape, dtype and byte hash, complete physical-page set and per-row support
counts. Position capacity is 262,144, page size is 64, selection capacity is
2,051, head geometry is 24 query heads / two KV heads / 256 dimensions, and
softmax scale is 1/16.

`Layer3QSARecorder` was installed only during Python preparation. Its hooks were
restored before starting the native server at an ephemeral loopback address.
Captured copies retain per-operator buffers outside shared graph scratch;
general and greedy variants share each operator's recording buffers. The last
invocation of each operator therefore remains available after request completion.
Recording adds 84,673,469 bytes of diagnostic tensors and extra graph copy nodes.

The HTTP request is the exact saved file
`/tmp/sparkgap-context-limit-262144/005-context-boundary.request.json`, SHA256
`599de46324e483a02b3265133692b7424cfb64cf65b70ddd436cb1a907931dcc`.
Canonical tokenization reproduces its saved 262,136 token IDs. No prompt,
generation budget, precision, model kernel or production dispatch was changed.
Native extension SHA256 is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
The checkpoint is revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` of
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`.

After the sole request reached native idle, `Server.close()` confirmed both
streams drained before any recorded GPU tensor was copied to the CPU. Main KV
pages are explicitly labeled `cache_after_request`: later decode appends
positions beyond the recorded prefill queries. All selected positions are
checked causal, so selected prefill values remain unchanged. Unselected page
tails may contain later decode writes. Raw-ring and compressed selector fields
are omitted because they would represent another observation time.

PreparedModel and recorder references retain GPU tensor storage throughout
readback. Logical pages can already be cached or reclaimed at idle, but no
second request is admitted, so no physical page is reused. Model cleanup follows
the copies; the process exits 0 and GPU4 returns to 2 MiB with no compute process.

Preparation took approximately 95 seconds and the instrumented HTTP request
took 55.52 seconds. These durations include diagnostic capture overhead and are
not a serving performance measurement. This artifact supplies actual long-context
operands for partition arithmetic and replay qualification; it does not establish
independent attention or full-model numerical accuracy.

An initial capture harness failed before constructing its first HTTP connection:
an `http()` helper shadowed the imported module. Model cleanup completed and no
request was sent. The helper rename and unchanged rerun are preserved in the
[failed-attempt receipt](qsa-long-context-operands-failure-before-http.json).
The [evidence archive](/data/sparkgap-artifacts/validation-receipts/qsa-long-context-operands-evidence.tar.gz) contains both
scripts/logs, the baseline recorder source, request, response, canonical token
IDs, metrics and complete receipt. Tensor storage remains in the artifact path
above. Reproduction uses the archived `capture.py` on the explicitly assigned
idle GPU with `CUDA_HOME=/opt/cuda`, after choosing absent output paths.
