# Python/Rust replay parity with FlashInfer sampling

The exact Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD checkpoint at revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` passed **2,216/2,216 checks**
through execution ABI 2. Every one of **207,620 tensor observations** was
bit-identical between Python and Rust replay, with zero mismatches.

The installed FlashInfer sampling policy is `flashinfer_top_k_top_p_min_p_philox_v1`.
Installed FlashInfer uses base revision `98fb87a461183f98d6a7049ca947fb136cc6fb59`
plus the declared 18-file serving patch, including the reduction-preserving parallel
AIR top-p apply kernel.
The [installed sampler qualification](sampling-flashinfer-installed.json) records its component gates
and source/JIT provenance. The native extension SHA256 is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`; the AIR header SHA256 is
`6753f53c511ad0c64114d4817b50993880804a12664c7d7bfcb8053f3442a48b`.

## Execution and coverage

Python constructed **37 complete graphs**: 25 general variants and 12 greedy variants.
Each side replayed **76 commands in 69 groups** with identical graph handles, registered
buffer addresses and uploaded bytes. The same model resources were reset in place before
exclusive Rust ownership. The reset comparison covered 1,145 entries and
23,212,106,730 bytes.

- General and greedy prefill capacities: 1, 16, 64, 128, 256 and 512 tokens.
- General and greedy decode/resume batch capacities: 1, 2 and 4 rows.
- Mixed zero/nonzero temperatures on the general path, plus both general→greedy and
  greedy→general ordering within two-frame submissions.
- Cold/chunked prefills, sample-disabled consumption, snapshot/restore, full-prefix
  resume, row permutations, inactive padding, terminal replay, histograms and RNG state.
- All 77 hybrid state regions, tracked backing bytes, logical KV pages, raw QSA state,
  logits, completion records, feature outputs and unrelated-state canaries.
- All four vision graph capacities execute the pinned 256-patch fixture. This does not
  measure full-capacity image content.

All 138 replay-group allocation checks and
138 unrelated-state checks passed. Native streams drained,
source and extension hashes remained unchanged, and the process exited 0. GPU4
(`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`) returned to 2 MiB with no compute processes before release.

## Memory

The captured catalog used 95,836,020,224 PyTorch-allocated bytes
and left 4,302,372,864 device bytes free against a 4 GiB configured
headroom. The KV pool contains **9,768 pages**, covering
**625,152 tokens**. Packed weights use
72,667,278,336 bytes, hybrid state uses
5,319,819,776 bytes, and mapped-host PLE tables use
28,800,138,240 bytes.

## Evidence

- [Machine-readable summary](graph-execution-parity-flashinfer.json), including full memory,
  command, graph-variant, launch and cleanup metadata.
- [Compressed complete receipt](/data/sparkgap-artifacts/validation-receipts/graph-execution-parity-flashinfer.json.gz), SHA256
  `91bcba24ac673a35f37d3dc03de85f681fe23e522a320f77b8dec5582faf42f0`. The uncompressed receipt SHA256 is
  `ab41568ca71d4684452f62ae8fda8255e41ad34628fd51bd11c1086c11acccc1`.
- Raw receipt, log, executed script and 182 content-addressed upload files:
  `/tmp/sparkgap-graph-execution-parity-flashinfer`. Every upload filename was verified against its content SHA256.

Elapsed wall time was 603.722s, including preparation, reset hashing and
CPU readback. These results establish same-graph replay parity and allocation stability.
Cross-capacity equivalence, independent model numerical accuracy, HTTP scheduler routing
and serving performance require their corresponding qualification results.
