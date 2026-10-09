# Sequential-key GDN prefix-replay qualification

The exact checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` passes **595 of 595 bitwise prefix-equivalence checks** with QSA `attention_splits=16` and GDN `recurrence_policy="sequential_k1"`. The diagnostic ran with `--require-bitwise` and exited **0**. The [complete receipt](/data/sparkgap-artifacts/validation-receipts/prefix-replay-sequential-k1.json) preserves the request, token IDs, comparisons, completion records, selected configurations and source hashes.

The unchanged tool-results request contains **459 tokens**. A capacity-512 cold prefill is compared with a snapshot at token **448**, followed by the remaining **11 tokens** at capacity **16**, and separately at capacity **512**. All model mutations execute through native graph submissions. CPU observations follow completed fences. The 12 submissions completed without device-reported errors or increases in the cumulative CUDA allocation counter; the native executor confirmed drainage before comparison and serialization. The catalog contains 37 graphs, but this diagnostic submits only the initialization, prefill, snapshot and restore graphs required by this sequence. Vision graphs are not submitted.

| Comparison | Passed | Failed |
|---|---:|---:|
| `cold_vs_cold_repeat` | 91 | 0 |
| `cold_candidate_vs_immutable` | 77 | 0 |
| `immutable_vs_restored_before_tail` | 77 | 0 |
| `cold_candidate_vs_split_prefix` | 77 | 0 |
| `cold_vs_restored_small_tail` | 91 | 0 |
| `cold_vs_restored_full_capacity_tail` | 91 | 0 |
| `restored_small_tail_vs_split_small_tail` | 91 | 0 |

An independent CPU verifier recomputed every predicate from the saved tensors, including all 77 hybrid-state regions, logits and logically aligned layer-3 QSA operands. Its comparison, completion-record and QSA-policy verification blocks are byte-identical to the verifier used for the [fixed-16 baseline](prefix-replay-fixed16.json). Canonical tokenization reproduces the exact 459 token IDs, whose little-endian uint32 SHA-256 is `d787632f8fe5fdb40b299e728307651391d5f4fac13bd7d4eeeeea5beadc1cc3`. The archive contains both verifier sources, output, logs and an inventory with individual hashes for all **249 tensors**.

The [unconstrained autotuned catalog](prefix-replay-autotuned.md) failed 83 of these predicates. The passing catalog uses a fresh selection-cache namespace and retunes other operator families as well as GDN. It is therefore a qualification of the policy-constrained catalog, rather than a `k_split`-only ablation. All **216 GDN prefill declarations** select sequential recurrence with `k_split=1`; `v_split`, stages and workspace-window choices remain tunable. The [public policy contract](gdn-recurrence-policy.md) enforces this restriction during candidate selection and explicit-configuration validation.

A separate CPU cross-catalog diagnostic compares the saved tensors against `/tmp/sparkgap-prefix-fixed16.pt`: all **nine complete hybrid-state observations** and all **six logits tensors** are byte-identical. Its inputs and results are preserved in `evidence/baseline-comparison.json`. This comparison is separate from the 595-predicate acceptance gate.

Preparation declared **11,350 requests**, used **8 compiler workers**, measured **1,271 candidates**, and recorded **0 selection-cache hits** in **148.369824 seconds**. Compiled kernel caches were retained. Selection sources comprise 1,749 tuned declarations, 6,093 fixed declarations and 3,508 explicit overrides. The diagnostic's complete elapsed time was **334.952435 seconds**; these durations include preparation and do not measure serving latency or throughput.

| Memory measurement | Bytes |
|---|---:|
| Shared benchmark resources | 5,620,630,048 |
| Allocated before trial release | 87,693,114,880 |
| Allocated after trial release | 80,114,958,848 |
| Preparation allocations released | 7,578,156,032 |
| Expert scratch before cleanup | 130,244,788 |
| Expert scratch retained | 125,436,208 |
| KV storage: 9,598 pages | 15,568,109,568 |
| Final CUDA allocation | 95,692,641,792 |
| Final device memory free | 4,352,704,512 |

The state pool provides 37 immutable snapshot slots. Layer-3 recording reserves an additional 84,673,469 device bytes during preparation. The receipt preserves the full memory report and workspace bounds.

The [evidence archive](/data/sparkgap-artifacts/validation-receipts/prefix-replay-sequential-k1.tar.gz) contains the exact executed script and coordinator, all **155 snapshot files**, **533 installed b12x Python files**, all **30 patched FlashInfer files**, the install log, request, raw receipt, execution log, native extension and mapped sampler binary. Every source hash and every archive-member hash was verified. FlashInfer base revision is `98fb87a461183f98d6a7049ca947fb136cc6fb59`; the declared patch SHA-256 is `4b78c54a1f3078c2e1c67a322ada8ef8eb3223bb6db8770e5b10db7e5ed91d0e`. Native extension SHA-256 is `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.

The sampler binary mapped by PID **1328829** was recorded while the process was alive at `2026-10-09T11:34:42.715695+00:00`: device `103:07`, inode **33069572**, SHA-256 **`1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`**. The archive retains that binary, `/proc` mappings, Ninja build metadata and the mapping receipt. The pre-launch filesystem hash `9dd90584c6792335f0ad45bb6c3c254e0bec70a25cff65ff1a535751f307bc99` predates JIT rebuilding and does not identify the binary used by this run.

The external tensor artifact `/tmp/sparkgap-prefix-sequential-k1.pt` is **1,229,193,399 bytes**, SHA-256 `097ad168fa88c99d390eae1eb8bd43c791deb1566c5be64ce5030d2fb5621ca7`. It remains outside the archive; its complete tensor inventory is archived. The comparison baseline is also identified and hashed in `evidence/external-tensors.json`.

Archive SHA-256: `5c03bc212036dfd389129de632cffcccfdcb03b27c0b0a8bf36767478653f8fc`; **782 members**, **9,924,110 bytes**. The execution tool observed exit 0 for session 41031; no shell exit-marker file was produced. The exact command and environment are recorded in the JSON receipt and `evidence/execution-command.json`.

This result qualifies strict prefix equivalence for the specified request and catalog. Canonical eager-model numerical equivalence, the complete native-versus-Python graph catalog comparison, and serving performance require their separate acceptance checks.
