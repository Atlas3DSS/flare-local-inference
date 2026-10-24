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
- **Real traces.** Neither the server binary `48b69ed4` (package `f21b9abb`)
  nor the release binary `d4ffb1fa` (previous package) has a real-trace
  end-to-end measurement. The real-trace figures in the run log (best 47.23 tok/s, run
  `best24`) used the 2026-10-22 release-tree package with a pre-release binary,
  `2f9128e4` ([measurement history](../../BENCHMARKS-SM70.md#measurement-history-2026-10-20-to-2026-10-23)).
- **Serving binary.** The published serving numbers were measured with a `flare`
  binary built before this branch's Helios commits
  ([BENCHMARK.md](BENCHMARK.md#what-was-run)). The current package `f21b9abb`
  was benchmarked on 2026-10-24 with the server binary `48b69ed4`
  (`48b69ed46f5183ff0b62f1906b5e80819e370408f035a3643be4db68c83d7342`, built
  from `4528c220`)
  ([section](../../BENCHMARKS-SM70.md#package-f21b9abb-512-token-prefill-chunks-on-server-binary-48b69ed4)). The previous package
  `fd7c24dc` was benchmarked on 2026-10-23 with the release binary
  `d4ffb1fa3453bf37f2017a7b58d09559010881311b2d9afcdfb33b4011caa667`, built
  from `71029458`, whose production source is unchanged through `c6fdc120`
  ([previous-package section](../../BENCHMARKS-SM70.md#previous-package-fd7c24dc-on-release-binary-d4ffb1fa-128-token-prefill-chunks)). Both ran
  on newly authored public prompts. Figures labelled historical come from older
  binaries and internal prompts.
- **Release-tree component evidence.** The public GPU runner at `b4987de5`
  (production source unchanged through `c6fdc120`) passed A: 223 tests plus
  17 subtests; B: 218 tests, with zero failures, exemptions or skips
  ([tests-A.log](receipts/tests-A.log), [tests-B.log](receipts/tests-B.log),
  [release-tests-FINAL.json](receipts/release-tests-FINAL.json)).
  Maximum-context attention, dynamic draft metadata and generic GDN admission
  tests passed. The separate CPU package suite passed 88 tests with 9
  intentional GPU skips; the signed shared-memory native fixture separately
  passed on V100. For the 512-token prefill change, groups A and B on a tree
  with the same kernel and test changes passed 225 tests with 17 subtests and
  245 tests ([groups-A](receipts/g128/groups-A-release-g128-tree.log),
  [groups-B](receipts/g128/groups-B-release-g128-tree.log)). On the merged commit `4acbecf0`, with the committed group lists, group A passed 225 tests with 17 subtests and group B 218 tests; the committed B list does not include `tests/preparation/test_module_images.py` (27 tests), which the lab run above did, hence 218 rather than 245. The prefill-rows identity tests ran separately on the same commit, 74 passed; that run predates the line that adds them to group A ([groups-A-merged-4acbecf0.log](receipts/g128/groups-A-merged-4acbecf0.log), [groups-B-merged-4acbecf0.log](receipts/g128/groups-B-merged-4acbecf0.log), [groups-A-rows-merged-4acbecf0.log](receipts/g128/groups-A-rows-merged-4acbecf0.log), [tests-replay-merged-4acbecf0.out](receipts/g128/tests-replay-merged-4acbecf0.out)).
  These component tests are separate from the full-model export gates and the
  native serving measurements.
- **Export qualification is complete; serving qualification is not.** The
  current package `f21b9abb` passed every gate on 2026-10-24: strict 47/47,
  long-sequence invariance, commit-capacity growth, consumer identity,
  components, accuracy and measurement qualification, with
  `prior_measurement_identity: true` against the previous package's goldens. An
  independent rebuild on a separate GPU did the same
  ([gate-summary.json](receipts/g128/gate-summary.json),
  [validation record](ARCHITECTURE.md#export-qualification-of-package-f21b9abb)).
  The previous package passed every gate on 2026-10-23
  ([release-tests-PREPAREAPI.json](receipts/release-tests-PREPAREAPI.json),
  [release-tests-REBUILD.json](receipts/release-tests-REBUILD.json),
  [validation record](ARCHITECTURE.md#export-qualification-of-the-previous-package-fd7c24dc)).
  These gates run retained CUDA graphs, not the native server.
  The new public inputs do not reproduce or validate the historical results
  ([VALIDATE.md](VALIDATE.md#gate-2-preparation-with-every-export-gate)).
- **SM70 and SM120 outputs differ slightly.** Measured on the previous
  package's tree and not repeated on the current one, a comparison of ordinary
  graphs on 3 public prompts found small, non-zero full-vocabulary KL
  (mean 0.0015 to 0.0024 per prompt), and one of the three independent greedy
  continuations parts after 9 tokens. Against a shared FP32 reference, 30
  public prompts (3,840 positions) show no measurable difference between the two
  builds in mean or p99 KL at that sample size
  ([distance from the FP32 reference](ARCHITECTURE.md#distance-from-the-fp32-reference)).
  It is not a hardware-only comparison, covers neither MTP nor native serving,
  and asserts no acceptance threshold
  ([comparison](ARCHITECTURE.md#sm70-and-sm120-comparison-ordinary-graphs)).
- **What is public and what is not.** Public in this tree: the export and gate
  harness (`tools/prepare_replay.py`, `prepare_replay.sh`, `validate_replay.py`,
  `validate_candidate.py`, `tests_replay.sh`), the measurement helpers
  (`measure_mtp.py`, `measure_components.py`, `check_consumers.py`,
  `accuracy.py`), the native benchmark clients (`native.py`,
  `benchmark_prefill.py`, `benchmark_mtp.py`, `benchmark_long.py`), the
  architecture comparison (`compare_architectures.py`), public prompt
  generation (`generate_prompts.py`) and the FP32 reference generators
  (`reference_model.py`, `gpu_reference.py`), documented in
  [EXPORT.md](EXPORT.md), [VALIDATE.md](VALIDATE.md) and
  [BENCHMARK.md](BENCHMARK.md), plus the run receipts in [receipts/](receipts/).
  References and tau goldens are generated locally by these tools, not
  shipped. Not public: signing keys, raw run logs beyond those receipts, the
  historical lab traces and prompt sets, and the lab's job orchestration.
- **Adaptive verification.** `--mtp-adaptive-verification` makes sampled outputs
  depend on which requests share a batch. Serve with fixed width when outputs must
  not depend on co-scheduled requests ([SERVE.md](SERVE.md#fixed-width-verification)).
- **Long prompts are prefill-bound.** With package `f21b9abb`, effective
  prefill is 866-994 tok/s at 2,048 to 30,720 tokens (previous package
  719-807), and concurrent prefills do not overlap: a 30,720-token prompt waits
  35.46 s for its first token alone, and 139.7-142.4 s when four arrive
  together (previous package 42.7 s and 170.2-171.0 s;
  [BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md#package-f21b9abb-512-token-prefill-chunks-on-server-binary-48b69ed4)).
  There is no isolated GPU prefill timer, so these are wall-time rates. A
  profile of one prefill of the previous package on the release binary shows
  where the time goes. That package prefills in 128-token chunks, 240 of them
  for a 30,720-token prompt. The projection
  GEMMs take 63-66 % of the time (the NVFP4 MLP alone 44 % at 8,192 tokens), the
  GDN recurrence about 11 % and attention 7-10 %. Attention grows with context,
  to about 12 % extrapolated over a whole 30,720-token prompt. The GPU is idle
  only 1.4-1.9 % of the time, so the cost is in the kernels, not in gaps between
  them ([prefill-profile.json](receipts/measurements/prefill-profile.json)).
  The current package was not profiled.
- **A smaller KV pool with 512-token prefill.** At `--memory-limit 31G` the
  current package's prefill graphs (capacities 128, 256 and 512) leave a
  124,928-token KV pool, limited by memory, 6,656 tokens (5.1 %) below the
  previous package's 131,584-token `--kv-cache-limit concurrency` cap. Four
  30,720-token contexts fit (122,880 tokens); four 32,768-token contexts
  (131,072) do not, and how the server schedules that case was not measured.
  Preparing with `--chunk 128` gives the previous catalog, pool and prefill
  rate ([EXPORT.md](EXPORT.md#3-prepare-and-export)).
- **512-token prefill chunks.** A prefill chunk of the current package holds
  up to 512 tokens, four times the previous package's. While it runs, other
  active requests wait for their next decode step: at the measured 30,720-token
  rate a 512-token chunk takes about 0.59 s on average, against about
  0.18 s for a 128-token chunk of the previous package. The four-agent
  long-context decode rates include that effect
  ([BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md#package-f21b9abb-512-token-prefill-chunks-on-server-binary-48b69ed4)).
- **Opt-in prefill ordering.** The shortest-remaining-first prefill option is
  off by default. It is in the server binary `48b69ed4` and not in the release
  binary `d4ffb1fa`. In a burst of
  equal prompts it lowers the mean first-token time, not the worst case
  ([BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md#opt-in-prefill-ordering-shortest-remaining-prompt-first)).
  With package `f21b9abb` and four cold 28.7k-token agents it brought the
  first-token p50 from 132.1 s to 83.2 s, but it costs decode speed: the early
  finishers decode while the others' 512-token prefill chunks run, so the
  per-agent decode median over the 12 four-agent requests fell from 38.2 to
  21.0 tok/s (previous package 41.5)
  ([BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md#long-context-native-serving-package-f21b9abb)).
  The refinement knobs `--prefill-srpt-short-tokens` and
  `--prefill-srpt-decode-priority-tokens`, in this tree but not in the release
  binaries, removed that decode cost and the short-prompt wait in the lab
  (73.3 tok/s per agent; a 512-token p95 of 1.29 s from send). They make the
  later agents' first tokens later, and under decode-heavy load they make the
  longest prompts' worst case worse: a 30,720-token TTFT of up to 200 s,
  against 86-122 s with the shipped option and 138-173 s by default
  ([refined ordering](../../BENCHMARKS-SM70.md#refined-prefill-ordering-opt-in-not-in-the-release-binaries)). The option's
  aging bound of 256 chunks was sized at 128-token chunks, where it is about
  41 s; at 512-token chunks it is about 133 s at short context, and time a
  prompt spends held behind decoding does not count toward it.
  A soak on the previous package, with a lab build of the option, used
  mixed-length Poisson arrivals: one request every 6 s on average,
  50 % 512, 30 % 2,048, 15 % 8,192 and 5 % 30,720-token prompts, 4 client
  workers and 8 output tokens. With the option on it ran 45 minutes and
  completed 475 of 475 requests; with it off it ran 15 minutes and completed
  157 of 157. Neither run had an error or a stall
  ([soak-on.json](receipts/prefill-srpt/soak-on.json), [soak-off.json](receipts/prefill-srpt/soak-off.json)). On
  the 157 arrivals both runs share, time from send to first token (mean / p95 /
  max) was:
  - 512-token prompts: 1.65 / 2.73 / 3.11 s off, 8.09 / 43.92 / 46.42 s on.
    Short prompts wait behind a long prompt already in prefill, because a
    started prompt is not preempted;
  - 30,720-token prompts: 103.9 / 157.4 / 157.4 s off, 48.7 / 65.7 / 65.7 s on;
  - all requests: a mean of 11.40 s off and 11.99 s on, and a worst case of
    157.4 s off and 65.7 s on.

  Measured from arrival instead, which adds the wait for one of the 4 client
  workers, the mean over all requests is 17.46 s off and 26.15 s on, and the
  worst 161.9 s off and 103.7 s on. The option trades short-prompt tail latency
  under mixed streaming for long-prompt and worst-case latency. It suits bursts
  of agents on long contexts when the first token matters more than decode
  speed, but it is not a free win for mixed traffic or for decode speed, which
  is why it is off by default. The 256-chunk aging bound
  moves a waiting prompt into arrival order; it does not cap the total wait. The
  soak estimates the longest wait at about 353 chunks with the option on and 710
  with it off.
- **KV pressure can preempt a decoding request.** Package `f21b9abb`'s pool
  is 976 pages (124,928 tokens). Four 28.7k-token agents need about 912 pages
  of context, and cached prefixes or long outputs take the rest. When the pool
  runs out, the server preempts a decoding request and later prefills its
  context again (about 33 s for 28.7k tokens), so that request stalls
  mid-decode. With 512 output tokens this happened once in the measured runs,
  with `--prefill-srpt`: a 72.0 s stall after token 480 in a batch whose
  server prefill counter shows one context prefilled twice
  ([g128-long-token-gaps.json](receipts/prefill-srpt-r2/g128-long-token-gaps.json));
  not in the default chain and not with configuration Y. With 2,048 output
  tokens it happened in every policy, the default included, with stalls of up
  to 190 s (`l2k-*` in [receipts/prefill-srpt-r2/](receipts/prefill-srpt-r2/)).
  This is pre-existing behaviour of the shared pool, not of the ordering
  options. Evicting cached prefix pages before preempting a decoder would
  avoid some of it; that is not implemented.
- **Profiling issue under investigation.** Node-level CUDA-graph tracing with
  Nsight Systems (`nsys --cuda-graph-trace=node`) made the server fail with CUDA
  error 719 (`unspecified launch failure`) after about 127 cumulative 128-token
  prefill chunks of the previous package, in 3 of 3 attempts. Unprofiled runs and graph-level tracing
  completed every request. The cause is not established; a compute-sanitizer
  run is the next check ([prefill-profile.json](receipts/measurements/prefill-profile.json)).
- **Default KV limit.** All measurements used `--kv-cache-limit concurrency`.
  The KV pool at the default `memory` setting was not measured.
- **Flags not measured on either package:** `--mtp-draft-tokens 3` (the
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
- **Chunk-256 prefill with deterministic chunking** (`7ffb3cf`). This tree
  prefills in 512-token chunks by a different route: exact fusions and a
  per-128-row attention split plan
  ([ARCHITECTURE.md](ARCHITECTURE.md#5-512-token-prefill-exact-fusions-and-the-g128-split-plan)).
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
