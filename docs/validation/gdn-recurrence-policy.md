# GDN prefill recurrence policy

The b12x public GDN prefill declaration accepts
`gdn_prefill.Caps(recurrence_policy="sequential_k1", ...)`. This constrains
preparation to sequential recurrence with `k_split=1`, while retaining tuning
of `v_split`, `stages`, and `window_tiles`. The default `"auto"` retains every
previously eligible configuration and the same default configuration.

The [30-file serving patch](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving-recurrence-policy.patch)
applies to FlashInfer revision `98fb87a461183f98d6a7049ca947fb136cc6fb59`.
Its [manifest](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving-recurrence-policy.json) preserves
all 25 file hashes from the MoE route-legality patch and adds five files: three
runtime contract modules, the GDN policy documentation, and focused CPU tests.
No recurrence kernel arithmetic changes.

Key splitting changes the association of key-axis reductions. Chunk-parallel
recurrence introduces transfer-matrix composition. Value splitting instead
partitions independent value rows; stages control tile prefetch, and window
boundaries store and reload FP32 recurrent state. The policy excludes the first
two arithmetic choices without fixing the remaining scheduling dimensions.

The [autotuned prefix-replay receipt](prefix-replay-autotuned.md) records the
requirement: a 459-token prefill differs from restoration at token 448 followed
by an 11-token capacity-16 tail. The restored capacity-512 tail is exact.
The selected GDN key split differs between those capacities. That failed
receipt remains unchanged. GPU output/state equivalence for the constrained
policy is not established by the CPU checks below.

`recurrence_policy` propagates through `Caps`, `GdnPrefillQuery`, its serialized
payload, preparation coalescing keys, persisted selection keys, and compiler
worker reconstruction. Query-schema version 23 and candidate-contract version
5 distinguish the declaration from prior GDN selection records. Configuration
schema 22 and its fields are unchanged. Validation rejects incompatible
explicit overrides; always-on correctness predicates enforce the policy during
ordinary and exhaustive search. KDA has no policy field, and its query codecs,
configuration codecs, schema versions, and selection identities are unchanged.

CPU validation uses 16 key heads, 48 value heads, head dimension 128, BF16
activations, FP32 state, one sequence, 46 state slots, null slot zero, and
checkpoint export. Eight token capacities are checked under both ordinary and
exhaustive search. The constrained candidates are exactly the sequential,
single-key-split subset of `auto`:

| Token capacity | Ordinary `auto` | Ordinary constrained | Exhaustive `auto` | Exhaustive constrained |
| --- | ---: | ---: | ---: | ---: |
| 1, 2, 4, 16 | 64 | 24 | 192 | 24 |
| 64 | 96 | 36 | 288 | 60 |
| 128 | 96 | 36 | 416 | 108 |
| 256 | 160 | 36 | 672 | 204 |
| 512 | 192 | 36 | 1,184 | 396 |

The 35 focused tests pass in 1.75 seconds. They check eligible configurations,
native layout metadata, default selection, incompatible overrides, exhaustive
constraints, serialized compiler payloads, selection-cache separation, invalid
values, and KDA isolation. The aggregate run passes 127 tests in 2.23 seconds,
including tuning predicates and MoE deterministic-route legality. CUDA
initialization is prohibited and remains false. Independent source review finds
no blocking defect; its separate focused run reports 35 passes.

The 32-case baseline comparison verifies identical `auto` configuration sets
and defaults for GDN and KDA. All 16 KDA query payloads and persisted selection
keys are also identical. GDN selection identities change because the encoded
policy and contract versions are part of the key. A fresh pinned checkout
passes patch application and verification of all 30 source hashes.

The [JSON receipt](gdn-recurrence-policy.json) records exact commands, raw log
paths, source hashes, and every baseline comparison. The
[archive](/data/sparkgap-artifacts/validation-receipts/gdn-recurrence-policy.tar.gz) contains the five source/test/documentation
files, focused and combined patches, manifests, candidate-audit script, complete
before/after enumeration, comparison receipt, and test logs. Every archive
member is verified against its recorded SHA-256. Archive SHA-256:
`a77ee79862c54cae0839994ff0be3c60d720f2a977a3bfe4ff236914ae19c451`.

These checks compile and execute no GPU kernels. Exact model checkpoint/state
replay, captured graph execution, allocation stability, and serving regressions
remain required for the selected configurations. No inference performance
measurement is included.
