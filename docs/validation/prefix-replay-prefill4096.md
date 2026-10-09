# 4,096-token prefill prefix qualification

The checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` records **595/595 passing predicates** and **0 failures** under the `--require-bitwise` gate. Execution exits **0**. The [complete receipt](/data/sparkgap-artifacts/validation-receipts/prefix-replay-prefill4096.json) and [archive](/data/sparkgap-artifacts/validation-receipts/prefix-replay-prefill4096.tar.gz) preserve every comparison, request, source hash, selected configuration and completion record.

The fixture `/tmp/sparkgap-prefill-capacity-fixtures/4043.request.json` contains **4,043 tokens**: cold execution uses capacity 4,096, the snapshot boundary is 4,032, and its 11-token tail executes at capacities 16 and 4,096. The prepared catalog contains 43 graphs. QSA fixes 16 partitions; all 324 GDN prefill declarations use sequential recurrence with one key split. Vision graphs are prepared but not submitted.

Preparation reused **51 selection-cache entries** and measured **0 candidates**. The recorded tuning interval is 3.720226 seconds. The configured prefill capacities are `1, 16, 64, 128, 256, 512, 1024, 2048, 4096`; decode capacities are `1, 2, 4`. General and greedy sampling variants are captured, while prefix comparisons suppress sampling.

| Prepared resource | Bytes or count |
|---|---:|
| Diagnostic CUDA allocation after capture | 95,105,565,696 |
| KV pages | 7,489 |
| Cached-token capacity | 479,296 |
| Hybrid-state backing borrowed during preparation | 5,319,819,776 bytes |
| Temporary MoE workspace released before capture | 368,868,096 bytes |
| Preparation allocation reduction after trial cleanup | 16,642,603,008 bytes |

| Comparison | Passed | Failed |
|---|---:|---:|
| `cold_vs_cold_repeat` | 91 | 0 |
| `cold_candidate_vs_immutable` | 77 | 0 |
| `immutable_vs_restored_before_tail` | 77 | 0 |
| `cold_candidate_vs_split_prefix` | 77 | 0 |
| `cold_vs_restored_small_tail` | 91 | 0 |
| `cold_vs_restored_full_capacity_tail` | 91 | 0 |
| `restored_small_tail_vs_split_small_tail` | 91 | 0 |

Independent CPU recomputation reproduces all 595 recorded equality predicates and numerical metrics exactly. The bytewise comparison count is 595/595; every tensor is hashed. Canonical rendering reproduces the fixture's 4,043 token IDs, SHA-256 `ace647f765c84046a4a9e0b01ad7daec0c8df4d37bae440d0f902d32a6434f17`. All 12 expected completion records are verified. The executed diagnostic rejects CUDA-allocation growth within each replay interval and requires native stream drain before writing its result.

The archive includes 25 frozen runtime files, 533 installed b12x Python files, 34 companion native source/build files, the actual native binary, executed diagnostic, raw log, exact request, CPU verifier and tensor inventory. The native source companion is archived separately from the diagnostic's runtime-source hashes. The sampler mapping was captured from PID 1696564 while alive: inode 33069572, SHA-256 `1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`. Its copied binary, maps and build metadata are retained. The execution tool observed exit 0 for session 40332; the process is absent at collection.

The native extension SHA-256 is `0dcc9ca245ed83bcbf7c0ea8147640a09c3ea2f7de6a8e6bc0559826058a0dca`. Supporting CPU checks retain **255 passes / 54 CUDA skips**, **54 focused passes**, and independent MoE candidate-memory accounting. All 25 runtime files match the frozen cohort snapshot and remained unchanged through process exit.

The external tensor artifact `/tmp/sparkgap-prefix-prefill4096.pt` is 2,506,761,913 bytes, SHA-256 `9e933a37cdb1b5c738b4fc1614bbaafb311d0f105910405b2d3c091bf224e17f`. The archive is 9,293,067 bytes, SHA-256 `aa5201a8055408091db89617d5b2ee163d5d8f75f51aa82333a9dab6b18c88ee`; every member hash is verified. Tensor contents remain outside the archive; their complete inventory is included.

Canonical model accuracy and 43-graph executor parity remain unqualified. Serving performance was not measured.
