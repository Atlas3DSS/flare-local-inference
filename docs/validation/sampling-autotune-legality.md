# Sampling after the MoE legality installation

The installed FlashInfer sampler passes 37 component tests and 552 native replay
blocks on GPU4 (SM120). Replay checks exact sampled IDs, completion records,
request counters, history, histograms and terminal state against the unchanged
FlashInfer-policy oracle. CUDA allocation counts remain unchanged.

The qualification uses the original four fingerprinted checkpoint logit vectors
and the existing 32/64-step fixtures, with three warmup and twenty measured
repetitions. [Raw results](/data/sparkgap-artifacts/validation-receipts/sampling-autotune-legality.json) retain all timings
and observations; these are sampler component measurements.

FlashInfer rebuilt its sampling library during initialization. The resulting
`sampling.so` SHA-256 is
`9dd90584c6792335f0ad45bb6c3c254e0bec70a25cff65ff1a535751f307bc99`.
The [archive](/data/sparkgap-artifacts/validation-receipts/sampling-autotune-legality.tar.gz) includes that binary, its Ninja
build manifest/log, test output, executable sampler source and original receipt.
The native executor extension remains
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.

The cached sampler present before initialization had SHA-256
`abf5c8569d4a6b948120d4b2628b7c90e3725b0a86a684d2e283289bb5552451`,
which differs from the binary recorded by the earlier installation receipt.
These binary identities are distinct; identical source or build-command hashes
alone do not establish machine-code equivalence.
