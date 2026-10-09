# Autotuned prefix-replay qualification

The exact Qwen checkpoint at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` failed the unchanged bitwise prefix gate: **512 of 595 checks passed; 83 failed**. The diagnostic exited **1** with `--require-bitwise`. Every failed check belongs to `cold_vs_restored_small_tail`. The [complete receipt](/data/sparkgap-artifacts/validation-receipts/prefix-replay-autotuned.json) preserves every comparison, request, token ID, completion record, selected configuration and source hash.

The tool-results request contains **459 tokens**. The diagnostic compares a capacity-512 prefill against a snapshot at token **448** followed by the remaining **11 tokens** at capacity **16**, and separately at capacity **512**. All model mutations execute through native graph submissions; CPU observations follow completed fences. The 12 submissions completed without device-reported errors. Layer-3 QSA recording reserves 84,673,469 device bytes during preparation. Vision graphs are present in the catalog but are not submitted.

| Comparison | Passed | Failed |
|---|---:|---:|
| `cold_vs_cold_repeat` | 91 | 0 |
| `cold_candidate_vs_immutable` | 77 | 0 |
| `immutable_vs_restored_before_tail` | 77 | 0 |
| `cold_candidate_vs_split_prefix` | 77 | 0 |
| `cold_vs_restored_small_tail` | 8 | 83 |
| `cold_vs_restored_full_capacity_tail` | 91 | 0 |
| `restored_small_tail_vs_split_small_tail` | 91 | 0 |

The first differing recorded state is GDN layer 1's recurrent state: maximum absolute difference **8.94069671631e-08**, relative L2 **6.2431794845e-08**. Layer 0 recurrent/convolution state and layer 1 convolution state remain bit-identical. The final small-tail logits have relative L2 **0.27349537611**; both argmax IDs are 9419. Equality of argmax IDs does not satisfy the bitwise gate.

Snapshot copy/restore comparisons pass, and direct split execution produces exactly the same result as snapshot restoration followed by the small tail. Restoring and executing the tail at capacity 512 is bit-identical to the cold path. Against the [default-preparation fixed-16 QSA baseline](prefix-replay-fixed16.json), every saved full-state/logit observation is exact except the two capacity-16 tail observations; the CPU comparison is archived as `evidence/baseline-comparison.json`. Its cross-run tail-logit relative L2 is 0.266327708960, distinct from the within-run cold-versus-restored value above.

Cached GDN prefill-16 uses sequential recurrence with `k_split=2`; prefill-512 uses sequential recurrence with `k_split=1`. QSA uses `attention_splits=16` throughout. The selected reduction partition differs at the failing graph capacity, and the first observed discrepancy occurs before QSA layer 3. These observations motivate constraining the GDN reduction policy; they do not establish a passing replacement policy.

The batched preparation reused **42 selection-cache groups**, measured **0 candidates**, and took **1.863675 seconds**. All 1,749 declarations that require searched selections used cached results; 6,093 declarations used fixed configurations and 3,508 used explicit overrides. The graph catalog contains 37 graphs. This is evidence of warm-cache preparation, not empty-cache startup or model inference performance. The [startup receipt](autotune-bootstrap.md) records the preceding partially populated-cache preparation separately.

The [archive](/data/sparkgap-artifacts/validation-receipts/prefix-replay-autotuned.tar.gz) contains the raw failed receipt, log, exact request, baseline comparison, all 155 frozen snapshot files, 533 b12x Python files, and the FlashInfer patch/install metadata. Recorded recipe, script and b12x hashes match the archived bytes. `MANIFEST.json` identifies archive members. Native extension SHA-256 is `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`; sampler SHA-256 `9dd90584c6792335f0ad45bb6c3c254e0bec70a25cff65ff1a535751f307bc99` was checked before and after execution.

The external tensor artifact `/tmp/sparkgap-prefix-autotuned.pt` is **1,229,192,379 bytes**, SHA-256 `dead65400fcfffd68f67e37c3c22b5f611bbdad9c0936a76a2078bc0c54a4bbf`. It is hashed and retained outside the archive. The baseline tensor artifact `/tmp/sparkgap-prefix-fixed16.pt` is also hashed in `evidence/external-tensors.json`. No gate, input, reference or tolerance changed. Prefix-equivalence and numerical acceptance remain failed/unqualified; the isolated implementation is not accepted into the canonical serving source.

Archive SHA-256: `7b9c5b689b0e72401406f9bc139678cbb903d4f0718baf380971e394e082efaf`. The execution tool observed exit 1 for session 46640; no shell exit-marker file was produced.
