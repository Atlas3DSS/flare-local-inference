# MoE autotuning route legality

The pinned b12x candidate contract admitted dynamic direct routing with
deterministic output, although native compile planning rejects that combination.
For Qwen's 512 experts, hidden width 2560, intermediate width 640 and top-10
routing, capacities 1 and 2 each admitted four unsupported direct configurations.
Deterministic NVFP4 candidates could also name a Triton route planner while
native lowering selected the internal planner.

The [serving patch](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving-autotune-legality.patch)
propagates declared determinism through candidate validation, heuristic routing,
workspace geometry and launch planning. Unsupported explicit direct/Triton
configurations fail validation. Candidate-contract version 24 invalidates affected
selection-cache entries; query and configuration codecs are unchanged.

CPU metadata validation covers eight capacities, both determinism values, and
NVFP4, multi-recipe and automatic NVFP4 declarations. Across 48 queries, accepted
candidates decrease from 1,910 to 1,382 and compile-plan failures decrease from
24 to zero. Nondeterministic candidate counts and this checkpoint's default
routes are unchanged. The 70 focused tests pass. Related tests produce 159 passes
and two failures for trellis table resource limits; both failures reproduce on
the pinned baseline.

The [receipt](moe-autotune-legality.json) and
[source/log archive](/data/sparkgap-artifacts/validation-receipts/moe-autotune-legality.tar.gz) preserve commands, source
hashes, candidate enumeration and baseline failures. These checks prohibit CUDA
initialization and do not qualify kernel numerics. The
[installation receipt](autotune-legality-install.json) verifies 18 packaged
runtime files against the 25-file patch manifest and records the sampler binary
present at installation.
