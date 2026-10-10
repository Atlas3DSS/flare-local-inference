# Full-context execution qualification

The native Flare server passed **18/18 checks** at its configured **262,144-token context limit** using the exact Qwen3.8 checkpoint. The valid request consumed a 262,136-token prompt and returned eight generated tokens with `finish_reason="length"`; HTTP usage reported 262,144 total tokens. The returned content was `x x x x x x x x`.

The prompt token IDs were generated with the pinned checkpoint tokenizer and chat template, serialized as little-endian uint32, and preserved with the rendered text and request bodies. Their SHA-256 is `baa27f12f1a737bbf0e9a954f538556c05192e001a057704c0be9db8cf7715ac`. The explicit Lark literal requires at least 129 ordinary checkpoint tokens under an exhaustive shortest-segmentation check, preventing EOS before the requested eight-token budget. This tests memory, indexing, and graph execution; repeated-marker input is not a long-context quality test.

| Observation | Result |
| --- | --- |
| Same prompt with output budget 9, total 262145 | HTTP 400 before model admission; no graph submissions. |
| Valid prompt/output usage | 262136 + 8 = 262144. |
| Native submissions for valid request | 512 prefill, 7 decode, 1 init, 1 snapshot; no restore, resume, or vision. |
| Cache reuse for valid request | Zero hits and zero reused tokens. |
| Full-context reservation | 4096 pages observed with one active request. |
| Completed snapshot | 4095 cached pages added; one reserved tail page returned. |
| Valid-request errors/cancellations | Zero. |
| Subsequent short request | 42 prompt tokens + 8 generated; successful length termination. |
| Resource accounting | Page, snapshot, and request-slot totals reconcile after both completions. |
| Source/native identity | 112 source fingerprints and installed native hash unchanged. |

Twelve native metric samples recorded prefill progress, from zero through 509 submitted graphs before completion; final counters report 512. Submitted counts may include an in-flight graph. The valid request completed in 55.535 seconds under a fixed 3,600-second deadline. That elapsed time is a diagnostic observation, not performance acceptance.

The consumed length **262,143** is derived from exact HTTP usage and seven successful decode completions: the first output is sampled after prefill, seven outputs are consumed, and the eighth remains pending. The last consumed token has zero-based position 262142; the pending final sample has position 262143. This receipt does not directly read back GPU state.

The server used model `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, and native extension SHA-256 `735bf767999b26affeacc5ae574b6e52c6a5e1dc18f42cc208ee8b9f0b2a18ab`. The memory report contains 9,795 KV pages, 37 snapshot slots, and 25 captured graph variants. No input or capacity reductions were made.

- [Execution receipt, checks, raw response records and metric samples](server-http-context-limit.json)
- [Requests, raw responses, exact prompt IDs, rendered text and client sources](/data/sparkgap-artifacts/validation-receipts/server-http-context-limit-artifacts.tar.gz)
- Local artifacts: `/tmp/sparkgap-context-limit-262144`.
- Client: `scripts/qualify_context_limit.py`.
