# Qwen4Exp native runtime acceptance coverage

The prepared model is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`
at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. This audit compares
[the core design](../core-design.md) with the implementation and recorded
qualification. Native build
`0dcc9ca245ed83bcbf7c0ea8147640a09c3ea2f7de6a8e6bc0559826058a0dca`
implements execution ABI 2 with configurable prefill capacities, explicit
general/greedy graph variants and fixed sampler-mode metrics. Its 34 build inputs
are retained in [the configurable-prefill receipt](prefill-capacity.json).
The 43-graph FlashInfer catalog with a 4,096-token prefill limit passes
[Python/Rust parity](graph-execution-parity-prefill4096.md): 3,693 checks and
413,708 byte-identical tensor observations with zero mismatches. Both executors
run 124 commands across 117 groups, including all general/greedy variants,
changing slots, mixed temperatures, large prefills and interleaved two-frame
replay. All 234 allocation and 234 unrelated-state checks pass. Native shutdown
drains both streams. The pre-reset check verifies every hybrid-state backing
byte after borrowed tuning-state restoration.

The configurable prefill default is 4,096 tokens, producing 43 graph variants.
[Shared hyperconnection priming and measured KV budgeting](hyperconnection-priming.md)
remove 15.08 GiB of duplicate priming buffers and account for actual KV allocator
consumption before graph rebinding. The actual main CLI passes 98/98 text checks,
with all twelve responses equal to the retained 4,096-token baseline, and keeps
the configured 4 GiB headroom. Its preparation/recipe CPU suite passes 271 tests
with 54 CUDA skips. Independent numerical and performance gates remain open.
[Configured-chunk HTTP qualification](prefill-capacity.md) passes 98/98 text and
44/44 image/grammar checks with all 12 paired cold/restored text outputs equal to
the 512-token configuration. Native counters verify chunking, prefix restoration
and resource conservation. The [4,043-token prefix fixture](prefix-replay-prefill4096.md)
passes all 595 state, QSA-operand and logit equality predicates against a
4,032-token snapshot plus an 11-token tail at capacities 16 and 4,096; its native
replay intervals preserve allocation counters. The separate
[512-token parity receipt](graph-execution-parity-autotuned.md) retains its
2,216 checks and 207,620 tensor observations under
[native build 6048fb01](native-sampling-modes-build.json). Independent model
numerical qualification and sustained-decode performance acceptance remain open.
Structural acceptance of a 64-aligned limit does not establish that its operator
queries and workspaces fit. Those constraints must pass during preparation;
unsupported geometry or insufficient GPU memory must fail startup explicitly.

The 25-graph sorted-CDF [Python/Rust parity receipt](graph-execution-parity.json)
passes 1,495 checks and
126,816 tensor comparisons across all 25 graph variants. The comparison covers
51 commands per executor, changed slots, checkpoint restoration, request state,
all 77 hybrid-state regions, touched KV storage, and allocation stability.
[HTTP admission](server-http-admission.json) and
[appended conversation](server-http-conversation.json) receipts establish
resource reclamation and prefix reuse under their recorded inputs. These
receipts do not establish the numerical or performance acceptance bounds in
the design.

## Required qualification without an observed implementation failure

| Contract | Recorded evidence | Required check |
| --- | --- | --- |
| Constrained and unconstrained requests share decode while constrained execution has one step in flight. | [Autotuned HTTP vision/grammar qualification](server-http-vision-grammar-autotuned.md) observes shared decode, constrained maximum in-flight count one, exact constrained output, correct unconstrained answer, clean terminals and resource conservation. | This case passes. Preserve these assertions when sampling or scheduling changes. |
| Identical text with different images has distinct cache identity; repeated identical images can reuse prefixes. | [Autotuned HTTP vision/grammar qualification](server-http-vision-grammar-autotuned.md) passes 44/44 checks: zero reused tokens for both different-image cold requests, a 64-token checkpoint hit for each identical-image repeat, equal cold/repeat outputs and reconciled resource counts. | Cache identity and replay-output checks pass for the pinned fixtures. Independent vision arithmetic remains unqualified. |
| Each image admits up to 16,384 patches; a request admits four images and 65,536 aggregate patches. | The HTTP vision/grammar receipt completes both full-size single-image encodes with canonical 4,116-token usage and equal JSON/SSE outputs. [Aggregate vision qualification](/data/sparkgap-artifacts/validation-receipts/server-http-vision-aggregate.json) passes 30 checks for four distinct full-size images, cold/repeat JSON and SSE, twelve vision launches and resource conservation. | Admitted capacity execution passes. Independent full-capacity model arithmetic remains unqualified; HTTP does not expose GPU arena buffers directly. |
| The declared context limit is usable through its last legal consumed position. | [Context-limit qualification](server-http-context-limit.json) passes 18 checks: a canonically counted 262,136-token prompt generates eight tokens, launches 512 prefill and seven decode graphs, and reports total usage 262,144. A total of 262,145 is rejected before admission. | Capacity execution and resource conservation pass. Consumed length 262,143 follows from graph counts and pending-token semantics; there is no direct GPU state readback or long-context numerical oracle. |
| Worker failure is reflected in health and responses; startup failure occurs before binding. | [Installed ABI-2 constructor rejection](native-startup-validation-abi2.md) passes 21 malformed manifest/configuration cases without CUDA access, network syscalls or thread creation; all owners are released. Inputs use synthetic undereferenced addresses. | [Fatal-worker qualification](native-worker-failure.md) passes 22 checks through the installed runtime: malformed captured completion → waiting JSON/SSE errors → health and subsequent admission 503 → confirmed stream drain, scheduler reclamation and owner release. These boundary checks use a small deliberately invalid graph fixture, not checkpoint inference or a forced CUDA fault. |

The [row-dependent QSA vision/grammar receipt](server-http-vision-grammar.json)
retains two cold/repeat output failures among 44 checks. With fixed 16-way QSA,
the [default-preparation suite](server-http-vision-grammar-fixed16.md) and
[autotuned main-source suite](server-http-vision-grammar-autotuned.md) each pass
all 44 checks across eight model requests, including both image-identity repeat
comparisons. Their payloads, check names and ordering are identical.
The mixed-batch, image-identity and single-image cases use
[the HTTP qualifier](../../scripts/qualify_server.py). The required
mixed-batch and image-identity cases are specified in
[correctness acceptance](../core-design.md#correctness-acceptance). The full
image limits are declared in [the native model contract](../../native/model-contract.md).

The maximum consumed length of a legal HTTP request with a 262,144-token
context is 262,143. Admission requires `prompt_tokens + max_new_tokens <=
262144` and a positive output budget
([request validation](../../native/src/server/request.rs#L265)). The final
sampled token remains pending; it is not consumed by a subsequent decode.
Near-limit HTTP execution tests memory/indexing behavior. It requires a
separate numerical oracle to establish long-context model accuracy.

For the pinned `bucket_16384` image, the 2,033-by-2,033 source resizes to
2,048-by-2,048: grid `[1,128,128]`, 16,384 patches and 4,096 merged features.
With thinking disabled, its HTTP prompt contains 4,116 tokens. Four full-size
images fill the per-request 16,384-feature arena. The existing single-image
qualifier checks the exact source RGB hash and cold/cache output equality;
the row-dependent QSA receipt retains its output differences.

## API capability boundaries

[Request preparation](../../native/src/server/request.rs#L46) explicitly rejects
`tool_choice=required`, named tool selection and `parallel_tool_calls=false`.
It also rejects nonempty effective tools with a structured output grammar
([composition guard](../../native/src/server/request.rs#L135)). These policies
are documented in the native contract. The design requires tools and structured
output but does not explicitly specify these combinations.

The [69-scenario tool receipt](tool-eval-native-standard69.md) makes their
practical effect concrete: TC-45 was capability-excluded; TC-65, TC-66, TC-67
and TC-69 received HTTP 400 when later tool-bearing turns requested structured
final output. These are missing capabilities, not JSON serialization failures.
`tool_choice=none` already disables effective tools while allowing schema output.

Supporting these combinations requires a request-local policy that distinguishes
Qwen tool envelopes from schema-constrained final content. Required/named tool
selection also needs enforceable tool-name and argument constraints. Removing
the guard, changing benchmark requests or relying only on a prompt instruction
would not provide that policy. No change to model arithmetic is needed to design
the native grammar/presentation behavior.

Configured assistant continuation formerly bypassed the structured-output
conflict check. [The continuation receipt](structured-continuation.json) records
the reproduction and correction. Effective defaults now participate in the
check; explicit false, empty string, null and nested template options retain
precedence. Exact-checkpoint CPU request preparation passes 16 structured cases.
That receipt does not qualify an HTTP deployment configured with continuation.

## Sampling implementation and performance evidence

[The sampling contract](../core-design.md#sampling-and-rng) requires captured
greedy, bounded top-k and general routes. [PreparedSampling](../../sparkgap/recipes/qwen38/sampling.py)
provides general FlashInfer filtering and a greedy maximum reduction. The
[pinned FlashInfer contract](flashinfer-sampling-contract.md) defines threshold
ties, top-p cutoff groups, explicit device RNG, and shared workspace ownership.
Server operations apply penalties, grammar masks, min-p and completion updates.

[Decoder construction](../../sparkgap/recipes/qwen38/bootstrap.py) and
[resume construction](../../sparkgap/recipes/qwen38/decoder_state.py) prepare both
variants. Rust selects greedy for exclusively zero-temperature scheduled lanes
and general for mixed or stochastic batches. CPU catalog, admission-routing and
full-model replay checks pass. [Autotuned HTTP mode-counter qualification](server-http-sampling-routing-autotuned.md)
passes 118 of 120 checks across 11 model requests. Greedy and general routing,
equal-geometry seeded repeats, shared mixed-temperature decode and the grammar
one-in-flight barrier pass. The constrained string reaches its fixed 1,024-token
budget at an exact 3,066-byte prefix of the required 3,074-byte JSON string;
its `stop` and complete-literal assertions fail. Inputs and oracles are unchanged.
Bounded top-k execution remains open.

The [autotuned default HTTP suite](server-http-autotuned.md) passes all 152
checks with its original mixed-grammar case: the constrained literal completes
exactly with `stop`, and tool-result/image cold-cache comparisons agree.
The [row-dependent QSA suite](server-http-flashinfer-regression.md) retains its
two cold-cache failures; the [fixed-partition default-preparation suite](server-http-fixed16.md)
passes 152/152. The autotuned routing fixture retains the same 3,066-byte output
and two failures as its [default-preparation counterpart](server-http-sampling-routing-fixed16.md).
The three autotuned HTTP suites share one main-source process; SIGTERM exits
zero, closes the listener and releases GPU storage.

The [sorted-CDF baseline](/data/sparkgap-artifacts/validation-receipts/sampling-paths-original.json) and
[installed FlashInfer sampler](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer-installed.json) each pass 552 native
replay blocks with exact tokens, request state, histograms and RNG counters
under their respective CPU sampling policies. Every block preserves CUDA
allocation counts. Median times below cover a sampler, trace copy and common
pending-token consumption on recorded checkpoint logits; they exclude model
forward and native request scheduling. Lower is better.

| Batch | Sorted-CDF general, temperature 0 | FlashInfer general, temperature 0 | FlashInfer greedy | Sorted-CDF checkpoint defaults | FlashInfer checkpoint defaults |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 163.0–163.3 µs | 120.9–121.3 µs | 18.0–18.3 µs | 173.3–174.0 µs | 128.7–130.2 µs |
| 2 | 772.9–773.1 µs | 143.3–144.8 µs | 18.5–18.6 µs | 782.3–783.2 µs | 150.4–150.7 µs |
| 4 | 801.6–802.3 µs | 165.0–165.7 µs | 19.6–20.0 µs | 811.6–811.9 µs | 169.6 µs |

Ranges are the medians from the 32- and 64-step configurations. Temperature-zero
blocks remain fully active. Defaults use separately captured one-active-sample
timings because longer stochastic blocks include EOS and terminal no-ops.
Each condition records 20 repetitions. The installed FlashInfer path improves
component latency at all measured batch capacities. Its cutoff-group and cuRAND
policies change seeded stochastic sequences; the sorted-CDF receipt retains
its original results. Greedy tokens and persistent-state semantics agree.

The [unmodified FlashInfer composition](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer.json) regresses
batch-1 general latency to 195 µs. A captured trace attributes approximately
84 µs to serial top-p application. The declared serving patch distributes that
operation while preserving logical-thread sums and the CUB reduction order.
[Operator qualification](sampling-air-parallel-qualification.json) passes
375 bitwise deterministic API comparisons and 300 same-threshold apply cases.
The installed package passes all 37 sampler/admission tests and the complete
552-block component gate. Its public API and workspace size are unchanged.

The three-way comparison of vLLM, minimal graph replay and the complete Rust
runtime remains required by
[performance acceptance](../core-design.md#performance-acceptance), including
the sustained-decode target within 5% of the same graph sequence's replay floor.
Functional HTTP timings, same-graph readback duration and tool-suite scores
cannot establish that bound. Kernel numerical acceptance remains unresolved:
the [main-source autotuned decoder](decoder-autotuned.md) matches four greedy
tokens but retains logit relative L2 of 0.159–0.244 against the unchanged
canonical reference. The [vision comparison](vision-serving-reference.md)
also retains its differences. Fixed QSA partitions and constrained GDN pass the
separate [595-predicate prefix gate](prefix-replay-sequential-k1.md). TP, SM121, MTP and a
second prepared model require their own qualification and are outside this audit.

## Prepared kernel selection

The bootstrap collects default-primed declarations, runs one batched tuning pass
with active benchmark fixtures and checkpoint weights, refreshes bindings,
and releases trial storage before final graph capture and freeze. Serialized GDN
trials borrow the hybrid-state pool and restore it behind successful device fences;
MoE candidates share a temporary arena released before final KV sizing. The target
GPU defaults to eight compiler workers during tuning. Its
[preparation contract](../core-design.md#preparation-lifecycle) defines cache
identity, workspace ownership, and rejection of incomplete or cancelled tuning.
BF16 projection configurations and MXFP8 inner GEMM lowerings remain unsearched.

[Bootstrap qualification](autotune-bootstrap.md) captures all 37 graph variants
with partially populated selection caches. Unconstrained GDN selection then
[fails 83 of 595 prefix predicates](prefix-replay-autotuned.md), preserving its
complete receipt. With [sequential single-key-split GDN](gdn-recurrence-policy.md)
and 16-way QSA, the unchanged
[prefix gate](prefix-replay-sequential-k1.md) passes 595/595 predicates. The run
measures 1,271 candidates with zero selection-cache hits in 148.37 seconds using
eight workers. A fresh selection namespace does not imply an empty executable
cache. [Sampler qualification](sampling-recurrence-policy.md) records the actual
loaded library and passes 37 component tests and 552 native replay blocks.

[Autotuned-catalog Python/Rust parity](graph-execution-parity-autotuned.md)
passes all 2,216 checks and 207,620 bytewise tensor comparisons. The main-source
[default HTTP](server-http-autotuned.md), [vision/grammar](server-http-vision-grammar-autotuned.md)
and [sampling-routing](server-http-sampling-routing-autotuned.md) reruns record
152/152, 44/44 and 118/120 respectively. All 39, eight and 11 model-request
payloads are byte-identical to their fixtures; check names and ordering are
unchanged. These runs use a 512-token prefill limit; their 155 recorded main-source
files match the qualified isolated snapshot and remain unchanged throughout
execution and shutdown. Historical receipts retain
their own preparation provenance. The 4,096-token catalog passes
[configured-chunk HTTP](prefill-capacity.md), the
[4,043-token strict prefix gate](prefix-replay-prefill4096.md), and
[complete-catalog Python/Rust parity](graph-execution-parity-prefill4096.md). These checks do not resolve independent
text/vision accuracy or the sustained-decode replay-floor performance target.

## Sampler measurement method

The component comparison requires no full-model reload.
`/tmp/sparkgap-decoder-execution.pt`, produced by
[decoder qualification](decoder-execution.json), contains four finite
BF16 `[248320]` logits tensors from the exact checkpoint. Its SHA-256 is
`261aac8a7ccf7691501bc75509a0a6e25f69534ce69a162a76d67f02cf447731`.
CPU inspection loaded it with `map_location="cpu", weights_only=True` and
left CUDA uninitialized.

1. Prepare independent `DeviceRequestState` and `PreparedSampling` owners for
   general and greedy modes at row capacities 1, 2 and 4. Use identical saved
   logits, slot maps, epochs, seeds, histograms and temperature-zero parameters.
   Derive prompt counts from the artifact's prompt and preceding generated
   tokens. Test penalized, grammar-masked, tied-score and padded-row correctness
   separately before timing. General and greedy must emit the same tokens and
   update RNG, histogram, pending token and completion state identically.
2. Capture a fixed block of 32 or 64 samples. After each sample, use the existing
   `PreparedTokenState.gather`, pending-token copy and `consume` operations from
   [the full-vocabulary test](../../tests/recipes/test_qwen38_sampling.py#L192).
   Repeating `sampler()` alone fails with `UNCONSUMED_PENDING` after its first
   sample. Give every active row a sufficient output budget and use neutral
   penalties for the fixed-logit timing baseline. Preserve EOS and subsequent
   terminal no-op behavior. Record actual live/sample counts for each step;
   do not normalize a partly inactive block as fully active sampling latency.
3. Record preallocated `torch.cuda.Event(enable_timing=True, external=True)`
   events as explicit graph nodes around
   the block. Register both modes and captured reset-copy graphs with the native
   replay worker. Reset each mode from identical immutable device snapshots
   outside its timed span. After handoff, Python only submits commands and reads
   fenced results; it does not mutate captured storage. Native executor
   completion events disable timing, so they cannot supply elapsed GPU time.
4. Warm and synchronize before measurement. Interleave general/greedy blocks in
   both orders, retain each elapsed sample, and report per-step median and tail
   GPU time for each row capacity. Check unchanged CUDA allocation counters and
   exact final request state after every measured block. A kernel trace should
   identify filtering, categorical draws and any unexpected gaps. Preserve event
   mode, graph size, source/input hashes, clock state and device identity.
5. Report these times as sampling plus identical token-state maintenance on
   recorded logits, excluding vocabulary projection and transformer execution.
   Measure the checkpoint's nonzero-temperature defaults separately in general
   mode; replacing them with greedy would change semantics. A bounded top-k
   implementation has no valid timing comparison until its unchanged top-k tie,
   top-p/min-p and RNG correctness gates pass.

Threshold top-k retains every token equal to the kth score. A bounded route cannot
truncate tied survivors to exactly k IDs; it needs a semantics-preserving
overflow policy within captured execution. With neutral penalties, row 2 of the
recorded logits retains 21 tokens at `top_k=20` (threshold 10.875); the other
three rows retain 20. This tie case already occurs in the model artifact.

FlashInfer's scalar `top_k_first` shortcut selects exactly k entries and uses
rejection-based top-p sampling. It cannot preserve threshold ties or the
qualified one-draw RNG contract. On SM120, graph-safe CUB top-k can produce an
unsorted bounded candidate set, but sorted/deterministic FilteredTopK requires
128 KiB shared memory. A bounded implementation needs deterministic candidate
ordering and an extra selected entry to detect cutoff-tie overflow. Device-side
overflow must select a captured general branch; computing both branches retains
the general path's full-vocabulary cost. Compaction also changes softmax/CDF
reduction trees and selects a different small-vocabulary top-p kernel. Identical
seed/offset inputs alone do not establish identical probabilities, cutoff
decisions or sampled IDs. The pinned source anchors are
[`sampling.py`](https://github.com/flashinfer-ai/flashinfer/blob/98fb87a461183f98d6a7049ca947fb136cc6fb59/flashinfer/sampling.py#L1570),
[`topk.py`](https://github.com/flashinfer-ai/flashinfer/blob/98fb87a461183f98d6a7049ca947fb136cc6fb59/flashinfer/topk.py#L649),
[`sampling.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/98fb87a461183f98d6a7049ca947fb136cc6fb59/include/flashinfer/sampling.cuh#L1021)
and [`renorm.cu`](https://github.com/flashinfer-ai/flashinfer/blob/98fb87a461183f98d6a7049ca947fb136cc6fb59/csrc/renorm.cu#L43).

This design measures the available component paths. Execution ABI 2 selects one graph per
`(kind, batch_capacity, token_capacity, sampling_mode)`. Mixed sampling requests
share a general decode batch; all-greedy batches use the captured maximum
reduction. Component timing does not establish a serving speedup.
