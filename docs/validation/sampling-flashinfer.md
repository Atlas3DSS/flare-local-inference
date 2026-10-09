# Captured FlashInfer sampling qualification

The installed sampler passes **37 component tests and 552 native CUDA graph
replay blocks** on GPU4, NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120). Each block
checks exact sampled IDs, completion records, request counters, histogram,
history and terminal state against the FlashInfer-policy CPU oracle. CUDA
allocation counters remain unchanged. The isolated AIR normalization change
also passes **375 deterministic public-API comparisons and 300 fixed-threshold
apply comparisons, all bit-exact**.

The [installed-build receipt](sampling-flashinfer-installed.json) identifies the
18-file [FlashInfer serving patch](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving.patch),
[manifest](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving.json), installed sources, Ninja
rebuild and tests. The base revision is
`98fb87a461183f98d6a7049ca947fb136cc6fb59`. The native extension remains
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.

## Component measurements

The workload uses four fingerprinted BF16 vocabulary-logit vectors from the
specified Qwen checkpoint, verified against
[decoder-execution.json](decoder-execution.json). CUDA events bracket the sampler,
sampled-ID trace copy, and common pending-token gather/copy/consume operations.
Captured resets, readbacks, model forward, vocabulary projection, HTTP and native
scheduling are outside the timed span.

The table gives median microseconds per active batch step for checkpoint defaults
(temperature 1, top-k 20, top-p 0.95, min-p 0). Ranges cover the separately captured
one-sample measurements associated with the 32- and 64-step block fixtures.

| Implementation | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| Preserved sorted-CDF sampler | 173–174 | 782–783 | 812 |
| FlashInfer with serial AIR apply | 204–206 | 227 | 249–250 |
| Installed FlashInfer with parallel AIR apply | 129–130 | 150–151 | 170 |

The initial FlashInfer composition regressed batch-1 component time despite
improving batches 2 and 4. The installed AIR change removes that regression.
The isolated greedy sampler measures approximately 18–20 µs across these batch
sizes. Temperature-zero general and greedy executions have bit-identical
sampled state; their full timing distributions are retained in the receipts.

Each implementation measures 24 captured variants with three warmup and twenty
measured repetitions. Stochastic 32/64-step blocks retain configured EOS and
subsequent terminal no-ops; they are not divided by the block length as though
all iterations remained active. The one-sample cases provide the active-default
measurements above.

Raw event repetitions, quantiles and state hashes are in the
[sorted-CDF baseline](/data/sparkgap-artifacts/validation-receipts/sampling-paths-original.json),
[serial AIR composition](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer.json), and
[installed composition](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer-installed.json).

## Sampling policy and RNG

Penalties, temperature and grammar produce FP32 scores. FlashInfer applies
threshold top-k, softmax and deterministic threshold top-p. Server operations
apply min-p and normalization, then one FlashInfer categorical call samples each
packed row. Top-k and top-p retain cutoff ties; min-p retains equality.
Temperature zero uses `torch.max`, selecting the lowest token ID on ties.

Each stochastic call receives one row and device-resident seed/offset tensors.
Its Philox subsequence is zero and word offset is `4 * request_rng_counter`.
One logical counter increment follows each successful sample, including greedy
samples. Packed-row and physical-slot reassignment do not change the seeded
stream. Inactive, stale, invalid, disabled and terminal rows do not advance it.
No PyTorch generator bookkeeping participates in native replay.

FlashInfer's `(0,1]` CUDA uniform conversion and cutoff-tie policy differ from the
sorted-CDF sampler's high-24-bit `[0,1)` conversion and stable token-ID top-p
prefix. Historical seeded outputs are preserved in the baseline receipt and
[executable-source archive](/data/sparkgap-artifacts/validation-receipts/sampling-original-evidence.tar.gz); they are not
claimed compatible with the FlashInfer policy. The parallel AIR apply preserves
the initial FlashInfer policy and exact tested outputs. Detailed source anchors
and workspace ownership are in
[flashinfer-sampling-contract.md](flashinfer-sampling-contract.md).

## Profile and implementation checks

The captured batch-1 profile attributes about **84 µs** to AIR's serial
full-vocabulary normalization/apply kernel, versus about 21 µs for top-k,
20 µs for softmax reduction and 6 µs for the final categorical draw.
[Profile evidence](sampling-flashinfer-profile.json) includes the CUPTI trace and
five repetitions. The installed AIR implementation preserves each original
logical thread's summation order and the same 1024-thread CUB reduction, then
writes normalized probabilities in parallel. Existing histogram workspace is
reused after radix selection; public API and workspace size are unchanged.
The isolated sum/reduction/apply kernels measured approximately 6.0/1.2/1.7 µs.

A captured `torch.softmax(out=...)` substitution passed all 552 correctness and
allocation checks but increased batch-1 default component time to 272–273 µs.
That substitution was rejected; its
[measurement receipt](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer-torch-softmax.json) remains intact.

The first strict comparison of nondeterministic radix mode failed at vocabulary
8192, batch 2, dense probabilities and top-p values `[0.01, 1]`. Repeated calls to
the unchanged reference also differed in 5 of 20 paired replays, including cutoff
support. No tolerance was relaxed. Applying the original and parallel kernels
to identical thresholds passes all 300 bitwise cases; the production
`is_deterministic=True` API passes all 375 full-operation cases. The
[operator receipt](sampling-air-parallel-qualification.json) and
[source, failure, binary and trace archive](/data/sparkgap-artifacts/validation-receipts/sampling-air-parallel-evidence.tar.gz)
preserve these results.

The installed sampling library was rebuilt by Ninja from header SHA-256
`6753f53c511ad0c64114d4817b50993880804a12664c7d7bfcb8053f3442a48b`;
the resulting sampling binary SHA-256 is
`88017d41217c4428528fb6b198279603d021790152e818a0ccec6f593a1f77ed`.
The combined patch applies to a fresh checkout, and the preparation setup
validator verifies all 18 declared file hashes. The separate user FlashInfer
worktree was not modified.

The [full-model 37-graph parity gate](graph-execution-parity-flashinfer.md) passes
2,216 checks and 207,620 bytewise tensor comparisons with zero mismatches for
this installed sampler build. It covers all greedy variants, mixed-temperature
batches and alternating general/greedy graphs, with unchanged CUDA allocator
counters and confirmed native drain.

[HTTP sampler routing](server-http-sampling-routing.md) passes 118 of 120 checks
across 11 actual-model requests. Native mode selection, seeded repeatability,
mixed-temperature batching and the constrained one-in-flight barrier pass.
The long constrained JSON string exhausts its fixed 1,024-token budget eight
bytes before completion, so its exact-literal and `stop` assertions fail.
The response is an exact prefix of the requested string; health and resource
accounting pass. The original fixture and failed oracles remain unchanged.

The [unchanged HTTP regression suite](server-http-flashinfer-regression.md) passes
150 of 152 checks across 39 model requests. Its original mixed-grammar case
completes the required literal and clean `stop`; the two failures remain the
recorded tool-result and image cold/cache output differences.
[CLI shutdown](server-sampling-shutdown.json) exits zero, closes the listener
and releases all model GPU allocations.
These measurements do not establish model throughput, HTTP latency or independent
model-numerical accuracy.
