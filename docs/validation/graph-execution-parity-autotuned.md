# Python/Rust replay parity with prepared kernel selections

The exact Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD checkpoint at revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` passes **2,216 of 2,216 checks**
through execution ABI 2. All **207,620 tensor observations** are bit-identical
between Python and Rust replay; there are zero mismatches.

Python constructs 37 complete graphs: 25 general variants and 12 greedy
variants. Each side executes 76 commands in 69 groups using identical graph
handles, registered buffer addresses, and upload bytes. Before exclusive native
handoff, mutable resources are reset in place. All 1,145 reset hash entries,
covering 23,118,450,870 bytes, agree before replay.

Coverage includes general and greedy prefill capacities 1, 16, 64, 128, 256,
and 512; decode/resume batch capacities 1, 2, and 4; cold and chunked prefixes;
partial tails; sample-disabled consumption; immutable snapshots and restoration
into different physical slots; full-prefix resume; packed-row permutations;
terminal no-ops; seeded sampling; histogram and RNG progression. Mixed-temperature
resume/decode commands use the general path. Seven paired submissions exercise
two in-flight frames, including both general/greedy orderings.

Readbacks cover all 77 hybrid regions in each of 19 tracked state rows, complete
backing bytes including padding, request state, logits, touched KV pages, all
raw QSA slots, feature outputs, and unrelated-row/unused-page canaries. Every
one of the 138 replay-group allocation checks and 138 unrelated-state checks
passes. The four vision graph capacities execute the pinned 256-patch image
fixture; this does not exercise full-capacity image content.

The numerical configuration uses fixed 16-way QSA attention partitioning and
the public GDN `recurrence_policy="sequential_k1"` contract. All 216 selected
GDN prefill declarations use sequential recurrence with one key split; value
splitting, stages, and window size remain tuned. FlashInfer uses the
[30-file serving patch](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving-recurrence-policy.patch)
over revision `98fb87a461183f98d6a7049ca947fb136cc6fb59`.

Batched preparation takes 1.827 seconds with **42 selection-cache hits and
zero measured candidates**. Of 11,350 declarations, 1,749 use cached selections,
6,093 use fixed configurations, and 3,508 use explicit overrides. Parent-process
CuTe and Triton compilation counts are both zero. These figures describe
populated-cache preparation. The receipt preserves the numerical namespace,
workspace requirements, search constraints, and all selected configurations.

The catalog uses 95,744,252,416 PyTorch-allocated bytes and leaves
4,329,635,840 device bytes free against a 4 GiB headroom requirement. Its 9,708
KV pages cover 621,312 tokens. Packed weights occupy 72,667,278,336 bytes;
hybrid state occupies 5,319,819,776 bytes; mapped-host PLE tables occupy
28,800,138,240 bytes. Unselected expert scratch ownership releases 4,808,580
bytes before final cache sizing.

The qualifier verifies 568 recipe, preparation, b12x, and FlashInfer source
hashes. Its isolated source directory has no `native/src`, so Rust source
provenance is supplied separately: all 31 Rust files and both Cargo files match
[native-sampling-modes-build.json](native-sampling-modes-build.json), and the
installed extension matches
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
The companion is included in the archive and summary; the raw harness source
list is unchanged.

The FlashInfer sampling library is tied to live PID 1366976 by its preserved
process maps, device `103:07`, and inode `33069572`. The copied binary and
post-run cached file both hash to
`1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`.
Its [component qualification](sampling-recurrence-policy.md) passes separately.
Native streams drain, the process exits zero, and the execution supervisor's
post-run compute-process query is empty. Elapsed wall time is 623.312 seconds,
including preparation, reset hashing, and CPU readback.

The [summary](graph-execution-parity-autotuned.json) records actual counts,
memory, tuning, native-source verification, mapped-binary provenance, and
cleanup. The [compressed complete receipt](/data/sparkgap-artifacts/validation-receipts/graph-execution-parity-autotuned.json.gz)
preserves every check and tensor descriptor. Its SHA-256 is
`55602c4ad0484d957b42c431017650ef5909093e43112df315ae7ec8c75fbe8d`;
the 128,842,445-byte uncompressed receipt hashes to
`8b86424c5579a9a8fa229f3b6dee9aac24eda7d7dce29f57ffb5b0d4d7a2fcf3`.

The [source and upload archive](/data/sparkgap-artifacts/validation-receipts/graph-execution-parity-autotuned.tar.gz) contains
796 verified members, including the source snapshots, native companion,
collector, mapped sampler binary, log, and all 182 upload files totaling
147,186,128 bytes. Every upload filename equals its content SHA-256. Archive
SHA-256: `1f15caaf799b3001de758fe158dc53953bb93906f0a8d57210df36ed7bf1acfb`.

This qualification compares identical captured graphs across execution engines.
Cross-capacity equivalence, independent model numerical accuracy, HTTP scheduler
behavior, and serving performance require their separate gates.
