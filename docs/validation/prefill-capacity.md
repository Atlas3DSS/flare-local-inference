# Configurable prefill HTTP execution

The main CLI prepared and served `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, with an explicit 512-token prefill limit and with its 4,096-token default. Both processes used GPU `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`, context 262,144, four request slots, the installed FlashInfer kernels, and the native HTTP server.

The 512-token and 4,096-token text runs each pass **98/98 checks**. The default-4,096 image/grammar run passes **44/44 checks**. All 12 paired cold/restored text responses have identical content, finish reason and usage across configurations. Sources and installed binaries remain unchanged within each run; both servers drain on SIGTERM, exit zero, and leave no listener or GPU compute process. The shutdown observations report 2 MiB GPU usage.

[Summary and provenance](prefill-capacity.json) · [Complete evidence archive](/data/sparkgap-artifacts/validation-receipts/prefill-capacity.tar.gz)

The archive contains 592 payload files and a SHA-256/length manifest. Independent readback verifies every member, with no missing or extra members. Archive SHA-256: `dc3397ec17f84e39e18961a124ddb9c0d9afb8ffde796a202b53af587bf7deaf`.

## Request and graph coverage

Six retained fixtures contain exactly 1,024, 2,048, 4,043, 4,096, 4,113 and 8,209 canonical prompt tokens. Each request uses greedy sampling and permits one output token. Token bytes, request JSON and their hashes are archived. The HTTP client adds `stream: false`; the emitted request bytes match exactly between configurations. Independent prompt prefixes differ before the first 64-token cache boundary.

| Prompt tokens | Cold prefill launches, limit 512 | Cold prefill launches, limit 4,096 | Reused prefix tokens | Restored continuation |
|---:|---:|---:|---:|---|
| 1,024 | 2 | 1 | 1,024 | One resume graph |
| 2,048 | 4 | 1 | 2,048 | One resume graph |
| 4,043 | 8 | 1 | 4,032 | One prefill graph |
| 4,096 | 8 | 1 | 4,096 | One resume graph |
| 4,113 | 9 | 2 | 4,096 | One prefill graph |
| 8,209 | 17 | 3 | 8,192 | One prefill graph |

Each restored request executes one restore graph. Every text request generates exactly one token without a decode launch. Native counters verify admission, completion, cache reuse, zero failure/cancellation and reconciled slot/page/snapshot resources. Both text runs contain 31 HTTP requests, including metrics and health requests, and 12 model requests.

The image/grammar run replays all eight model-request payloads from [the autotuned image/grammar fixture](server-http-vision-grammar-autotuned.md) byte-for-byte, preserving the 44 assertion names and order. It contains 23 HTTP requests. Coverage includes full-size 4,116-token image prompts crossing the 4,096-token chunk boundary, JSON/SSE output agreement, distinct-image cache identities, repeated-image reuse and concurrent constrained/unconstrained decode.

## Preparation and memory

| Recorded quantity | Explicit 512-token limit | Default 4,096-token limit |
|---|---:|---:|
| Captured graph variants | 37 | 43 |
| KV pages | 9,708 | 8,046 |
| Aggregate cached token capacity | 621,312 | 514,944 |
| KV allocation, bytes | 15,746,531,328 | 13,050,740,736 |
| Recurrence workspace, bytes | 67,108,864 | 347,490,304 |
| Expert workspace, bytes | 125,436,208 | 664,804,772 |
| QSA workspace, bytes | 166,879,232 | 214,990,848 |
| Free GPU bytes after preparation | 4,329,635,840 | 4,394,647,552 |
| Required GPU headroom, bytes | 4,294,967,296 | 4,294,967,296 |

The 4,096-token preparation records 14,983 declarations, 43 selection-cache hits and 208 measured candidates. Its shared benchmark operands own 546,758,908 bytes and borrow the 5,319,819,776-byte hybrid-state pool. Trial cleanup releases the temporary 368,868,096-byte MoE workspace, restores borrowed model state behind successful device fences, and precedes graph capture and final KV sizing. The measured allocation reduction during preparation cleanup is 17,193,574,912 bytes.

Three failed preparation attempts remain archived with their logs, source manifests and process-exit observations:

1. A vision-injection guard rejected capacities above 512 before graph capture.
2. A duplicate 4.96 GiB GDN benchmark-state allocation exceeded available memory.
3. A GDN candidate requested 332 MiB with 93.06 MiB free while candidate workspaces remained retained.

The successful 4,096-token run differs from the 512-token run in seven runtime files: preparation coordination, bootstrap, decoder workspace registration, GDN, MoE, hybrid-state reset and vision injection. The archive preserves each source snapshot and every file hash. Consequently, the recorded HTTP durations do not isolate chunk-size performance.

The installed native extension is `0dcc9ca245ed83bcbf7c0ea8147640a09c3ea2f7de6a8e6bc0559826058a0dca`; all 34 native build inputs match its retained build snapshot. Both processes map sampler binary `1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`, confirmed from `/proc` device/inode mappings and retained binary copies before and after the requests. The 4,096-token supervisor records the exact main CLI command without `--prefill-chunk-size`, unsets `PYTHONPATH`, and retains start/exit receipts.

## Supporting checks and remaining gates

Native server tests pass 91 cases with one ignored case; core tests pass 58. Formatting and Clippy pass. The complete preparation/recipe CPU suite passes 255 cases with 54 CUDA cases skipped. Constructor validation passes 21 cases. Focused vision-injection CUDA tests pass six cases, including 512/3,072/4,096/8,192 capacities. Logs, test sources and commands are retained; overlapping suites are not summed.

[CPU workspace planning](prefill-capacity-planning.md) covers non-power-of-two capacities and reports structural kernel limits. These metadata checks do not establish that every accepted configuration fits GPU memory.

HTTP verifies configured chunking, prefix reuse, returned outputs and resource conservation. Complete 43-graph Python/Rust device-state parity, replay-allocation stability and strict cross-capacity prefix-state comparison remain separate gates. The independent model numerical failures documented for the [512-token catalog](decoder-autotuned.md) remain unresolved; matching one greedy token does not measure logit agreement or vision arithmetic accuracy.
