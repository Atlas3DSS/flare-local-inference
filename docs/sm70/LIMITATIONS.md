# Limitations

What does not work, what was not tested, and how this fork differs from
upstream. Read this before you rely on the port.

## Not supported on SM70

- **No `ignition export` for SM70.** The recipe declares a 262,144-token context,
  which the SM70 attention refuses, and the CLI does not pass a CUTLASS include
  directory. Prepare with the script in [EXPORT.md](EXPORT.md).
- **One recipe.** Only `qwen3.8-27b` (`nvidia/Qwen3.8-27B-NVFP4`, revision
  `482ca0f`) was prepared and served. Other recipes and models were not run on
  SM70.
- **One GPU per process.** The recipe is TP1. Tensor parallelism and multi-host
  serving were not run on SM70. Serve more cards as separate processes
  ([SERVE.md](SERVE.md#several-cards-behind-a-router)).
- **At most 4 concurrent requests and 32,768 tokens per request.** The S1L
  attention kernel is fixed at four agents and admits contexts up to 32,768
  tokens. Preparation refuses more.
- **No added quantization.** By the fork's policy, nothing is quantized beyond
  what the checkpoint ships. NVFP4 projections are weight-only: activations are
  not quantized to FP4. KV pages are BF16. FP8 KV, the upstream recipe default,
  is not used or tested on SM70. Lower-bit weights or KV formats are out of
  scope.
- **No training.** Flare is an inference server.
- **Windows** was not built or run for SM70.

## Not validated, or validated only in part

- **Shared code on SM120/SM121.** These changes apply on every architecture but
  were run only on SM70: the `qwen38_dense` recipe declaration (MTP on by
  default with 3 draft tokens, at most 4, and concurrency capped at 4) and the
  Helios scheduler changes.
- **Real traces.** The release package has no real-trace end-to-end
  measurement. The only real-trace figure in the run log is for a different
  package (STATUS.md line 68).
- **Serving binary.** The published serving numbers were measured with a `flare`
  binary built before this branch's Helios commits
  ([BENCHMARK.md](BENCHMARK.md#what-was-run)). The fresh server binary has SHA-256
  `d4ffb1fa3453bf37f2017a7b58d09559010881311b2d9afcdfb33b4011caa667`, built
  from `71029458`; production source is unchanged through `c6fdc120`. Native
  serving, TTFT/prefill and long-context benchmarks on that binary are running
  now. Until their results land in [BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md),
  no serving number in these documents was measured on it.
- **Release-tree component evidence.** The public GPU runner at `b4987de5`
  (production source unchanged through `c6fdc120`) passed A: 223 tests plus
  17 subtests; B: 218 tests, with zero failures, exemptions or skips
  ([tests-A.log](receipts/tests-A.log), [tests-B.log](receipts/tests-B.log),
  [release-tests-FINAL.json](receipts/release-tests-FINAL.json)).
  Maximum-context attention, dynamic draft metadata and generic GDN admission
  tests passed. The separate CPU package suite passed 88 tests with 9
  intentional GPU skips; the signed shared-memory native fixture separately
  passed on V100. These component tests are separate from the full-model
  export gates and the native serving measurements.
- **Export qualification is complete; serving qualification is not.** The
  release export passed every gate on 2026-10-23: strict 47/47, long-sequence
  invariance, commit-capacity growth, consumer identity, components and
  measurement qualification
  ([release-tests-PREPAREAPI.json](receipts/release-tests-PREPAREAPI.json)). An
  independent rebuild on a separate GPU reproduced its coding-token/tau goldens
  and measurement (`prior_measurement_identity: true`,
  [release-tests-REBUILD.json](receipts/release-tests-REBUILD.json)). These
  gates run retained CUDA graphs, not the native server
  ([validation record](ARCHITECTURE.md#export-qualification-of-the-release-package)).
  The new public inputs do not reproduce or validate the historical results
  ([VALIDATE.md](VALIDATE.md#gate-2-preparation-with-every-export-gate)).
- **SM70 and SM120 outputs differ slightly.** A measured comparison of
  ordinary graphs on 3 public prompts found small, non-zero full-vocabulary KL
  (mean 0.0015 to 0.0024 per prompt), and one of the three independent greedy
  continuations parts after 9 tokens. It is not a hardware-only comparison,
  covers neither MTP nor native serving, and asserts no acceptance threshold
  ([comparison](ARCHITECTURE.md#sm70-and-sm120-comparison-ordinary-graphs)).
- **What is public and what is not.** Public in this tree: the export and gate
  harness (`tools/prepare_replay.py`, `prepare_replay.sh`, `validate_replay.py`,
  `validate_candidate.py`, `tests_replay.sh`), the measurement helpers
  (`measure_mtp.py`, `measure_components.py`, `check_consumers.py`,
  `accuracy.py`), the native benchmark clients (`native.py`,
  `benchmark_prefill.py`, `benchmark_mtp.py`, `benchmark_long.py`), the
  architecture comparison (`compare_architectures.py`), public prompt
  generation (`generate_prompts.py`) and the FP32 reference generators
  (`reference_model.py`, `john_reference.py`), documented in
  [EXPORT.md](EXPORT.md), [VALIDATE.md](VALIDATE.md) and
  [BENCHMARK.md](BENCHMARK.md), plus the run receipts in [receipts/](receipts/).
  References and tau goldens are generated locally by these tools, not
  shipped. Not public: signing keys, raw run logs beyond those receipts, the
  historical lab traces and prompt sets, and the lab's job orchestration.
- **Adaptive verification.** `--mtp-adaptive-verification` makes sampled outputs
  depend on which requests share a batch. Serve with fixed width when outputs must
  not depend on co-scheduled requests ([SERVE.md](SERVE.md#fixed-width-verification)).
- **Long prompts are prefill-bound.** With four prompts of 22.4k to 30.2k tokens
  at 4 agents, TTFT p50 was 90.7 s on the older binary `2c792073` (STATUS.md
  line 65). Release-binary prefill and TTFT are being measured now.
- **Default KV limit.** All measurements used `--kv-cache-limit concurrency`.
  The KV pool at the default `memory` setting was not measured.
- **Flags not measured on the release package:** `--mtp-draft-tokens 3` (the
  production setting; the tables use 4), `--scheduler-policy agent`,
  `--mtp-pipeline`, `--prefix-cache-memory 16G` (the benchmarks used 8G) and the
  host prefix store's disk tier.
- **Review scope.** Independent Astra (`gpt-6-astra`) review on 2026-10-23
  cleared the MTP draft-attention routing `0aea71a` and GDN capacity check
  `7cb1bc7` for GPU testing on unchanged implementation blobs. The public
  [review record](ARCHITECTURE.md#public-release-review-and-current-validation)
  closes their missing independent-review records. It does not approve final
  release, performance claims or cross-architecture exactness. The hang fix
  retains its separately recorded single independent review.
- **Upstream test inherited broken.** The Helios integration test
  `helios/tests/deepseek_requests.rs` does not compile with
  `--features runtime-package,server`. The fork does not touch it; the error is
  in the upstream base `e7b8c01`. Run the Helios tests with `--lib`
  ([BUILD.md](BUILD.md#build-the-native-server)).

## Parked or not in this tree

These exist in the fork's history but are not part of this branch:

- **A fused decode-stack variant.** The fork built SM70 decode and verification
  on fused per-row kernels. It was correct but slower in the fork's measurements,
  because the one-CTA-per-row kernels were latency-bound, and it was parked.
- **Chunk-256 prefill with deterministic chunking** (`7ffb3cf`). Prefill in this
  tree runs 128-token chunks.
- **Two scheduler fixes from another fork branch.** One makes greedy readmission
  re-derive delivered tokens when the preempted checkpoint cannot be kept
  (`b25e756`). The other fixes a host-store admission deadlock in a one-context
  pool (`0f17156`). This tree's `e81534a` separately makes a refused request
  retry when a host-store copy-out pin is released. Under real page pressure a preempted checkpoint cannot always be kept, so
  readmission can re-prefill the request's generated tokens with prefill
  arithmetic, which can change later tokens (STATUS.md lines 25, 32).

## Upstream divergence

- **Base.** The fork uses upstream snapshot `e7b8c01` (2026-10-20), from a
  separate lineage rooted in squash `d88e29bf`. Commit ancestry alone therefore
  overstates the missing upstream content.
- **Distance as of `e19549a` (2026-10-23).** There are 816 upstream commit SHAs
  absent from the fork base's ancestry, including 96 on the first-parent line.
  The history audit classifies 680 of those as content already represented in
  the squashed base, and 136 as subsequent upstream development. The 680 are
  ancestors of full-history merge `863ee3d5`; the 136 come after that merge.
  The content classification used tree comparison and spot checks, not a
  per-commit equivalence proof. These are not 816 missing fixes, nor a claim
  that every change in the 136 should be ported. Reproduce the ancestry split
  with the fixed revisions (after fetching the upstream history):

  ```sh
  git rev-list --count e19549a --not e7b8c01             # 816
  git rev-list --count 863ee3d5 --not e7b8c01           # 680
  git rev-list --count e19549a --not e7b8c01 863ee3d5   # 136
  git rev-list --count --first-parent e19549a --not e7b8c01  # 96
  ```

- **Hardware scope.** Upstream targets SM120/SM121 and asks contributors to
  avoid interchangeable backends and portability layers (`docs/design-review.md`,
  principle 4). An SM70 path is outside that scope. That is why it lives in a
  fork.
- **Review policy.** Upstream `AGENTS.md` and `docs/design-review.md`
  principle 14 say that Claude Opus 5.5 must not design or implement low-level
  CUDA, PTX, CuTe DSL or Triton kernels, or changes to model fusion and GPU
  execution optimizations. They also say not to incorporate such changes through
  another contributor. This fork replaces that rule (commit `fd5109b`): such
  changes may be authored by Claude Opus 5.5, but each must be reviewed and
  cleared by an independent reviewer before it runs on a GPU or is merged, and
  the author never reviews its own change. Several SM70 kernels and kernel integrations in this branch were written by
  Claude Opus, as their headers and commit messages state: `sm70_pgemm.cuh`
  (`8ad725c`), `ca16f00`, `0aea71a`, and the kernels integrated in `6b808ef` and
  `f8af352`. Under upstream's rule they are not contributable as they are.
- **Commit trailers.** Upstream's commit hook requires a
  `Design-Signed-Off-By: ...; design-review=<hex4>` trailer, where hex4 is the
  start of the SHA-256 of the staged `docs/design-review.md`. Because `fd5109b`
  changes that file, commits after it acknowledge `45e8`, not upstream's `6aeb`.
  Nine early commits (`88a888a` to `8fdc604`) carry no trailer.
- **No upstream pull requests** have been opened from this branch.

## Reporting issues

Report problems with the SM70 port on this fork,
[Atlas3DSS/flare-local-inference](https://github.com/Atlas3DSS/flare-local-inference),
not upstream; upstream does not support SM70. Issues are enabled on the fork;
open one there.

Include:

- GPU model and memory, driver version, CUDA toolkit, and `torch.__version__`;
- the branch commit (`git rev-parse HEAD`) and how the `flare` binary was built;
- the preparation command and its JSON report, or the gate output that failed;
- the full `flare run` command and the server log up to `Start server`,
  including the `Plan cache` and `Allocate cache` lines;
- for serving problems, `/metrics?format=json` before and after, and the request
  that failed;
- for performance numbers, the columns listed in
  [BENCHMARK.md](BENCHMARK.md#reporting-your-numbers).

Do not report security vulnerabilities in public. For upstream code, follow
[SECURITY.md](../../SECURITY.md).
