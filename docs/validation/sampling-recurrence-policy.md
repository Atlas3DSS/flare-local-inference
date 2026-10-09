# Sampler qualification with the GDN recurrence-policy build

The FlashInfer sampler passes **37 component tests and 552 native replay
blocks** on GPU4, NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120). Exact sampled
IDs, completion records, request counters, RNG state, history, histograms, and
terminal state match the unchanged FlashInfer-policy oracle. Every replay
block preserves its CUDA allocation count. The native worker closes cleanly;
the process exits zero.

The input consists of the same four BF16, 248,320-element checkpoint logit
vectors identified by [decoder-execution.json](decoder-execution.json).
The complete input and case records equal those in
[sampling-paths-flashinfer-installed.json](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer-installed.json),
including seeds, EOS behavior, expected state hashes, and sampled IDs.
The checkpoint is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Its saved execution artifact has
SHA-256 `261aac8a7ccf7691501bc75509a0a6e25f69534ce69a162a76d67f02cf447731`.

The harness captures 24 sampler variants and 24 corresponding reset graphs.
It covers batches 1, 2, and 4; temperature-zero general/greedy comparisons;
32/64-step blocks with checkpoint-default sampling and terminal no-ops; and
separate one-sample captures. Each variant runs three warmup blocks and twenty
measured blocks: 72 warmup and 480 measured observations, all checked. Resets
occur outside the timed span. Raw timings remain in the
[receipt](/data/sparkgap-artifacts/validation-receipts/sampling-recurrence-policy.json); the workload excludes model
forward, vocabulary projection, and HTTP scheduling.

The loaded sampling library has SHA-256
`1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`.
Its identity is tied to the live qualification process: PID 1361540's
`/proc/1361540/maps` snapshot at `2026-10-09T11:39:03.378186Z` contains five
mappings for the sampling library, including its executable segment, with
device `103:07` and inode `33069572`. The archived binary matches the mapping
receipt's SHA-256. Parsed address, permissions, offset, device, inode, and path
fields agree between that receipt and the preserved process maps. A readback
after process exit confirms the same library hash, device, and inode.

The live mapping snapshot falls between the harness's recorded start and
finish. The [archive](/data/sparkgap-artifacts/validation-receipts/sampling-recurrence-policy.tar.gz) retains the copied
2,443,272-byte binary, process maps, mapping receipt, Ninja manifest and log,
raw qualification receipt, component-test output, replay log, exact input
tensor artifact and provenance receipt, executable sampler/test sources, and
the [30-file FlashInfer patch](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving-recurrence-policy.patch)
with its manifest. Every member is checked against `MANIFEST.json`.
Archive SHA-256:
`1cc6cb54c8e28361081bde9e4832e498db33b06d01fb351c8b65e3f2b8966e93`.

The native extension remains
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
All source hashes recorded by the harness remain unchanged. Component tests
take 3.14 seconds and report one CuTe DSL `Arch` import deprecation warning;
the complete warning is preserved. The sampling policy remains
`flashinfer_top_k_top_p_min_p_philox_v1`.

This component qualification establishes correctness and allocation stability
for the recorded sampler workload and mapped binary. Full-model recurrence,
prefix-cache equivalence, graph-catalog execution, and serving behavior have
separate gates. No comparative performance conclusion is drawn from this run.
