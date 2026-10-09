# Qwen3.8 execution qualification

The checkpoint is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. GPU measurements use the assigned
96 GB RTX PRO 6000 Blackwell device. Individual receipts record source hashes,
inputs, arithmetic, and measurement conditions. Numerical failures block
performance acceptance.

Large receipt payloads and archives are stored outside the repository. The
[relocation map](artifact-locations.json) resolves their original repository paths;
small JSON locators also point to the complete receipts.

## Serving measurements

[Native HTTP serving performance](serving-performance.md) records 64 requests
and 41 groups passing at each prefill limit, 512 and 4,096, with all paired
Sparkgap request bytes and output token sequences equal between limits. At limit 4,096, median delivered decode
throughput is 78.329/128.105/200.651 tokens/s at requested concurrency 1/2/4;
core and delivery timing distributions, 4K/64K prefill, prefix reuse and exact
source hashes are retained separately. All 15 replay-floor groups at each limit pass state
and allocation checks; batch-1 HTTP core mean latency exceeds paired replay
means by 0.095–0.104% at limit 4,096 and 0.714–0.742% at limit 512. Median 4K/64K TTFT is 511.713/8,642.687 ms, versus
783.026/12,054.800 ms at limit 512; independent tuning changes 313 common
prefill bindings. All 792 graph launches in the 512-token software trace match
native counters; steady batch-1/2/4 GPU graph medians are
12.779/15.107/18.525 ms, with approximately 16 µs between graphs. Profiling
overhead and possible dropped non-graph events remain explicit. At limit 512, vLLM records
114.221/199.772/310.043 delivered decode tokens/s versus Sparkgap
77.782/128.216/201.040. All 35 measured decode sequences differ, so paired speed
ratios are omitted. All 25 single-output IDs match; Sparkgap has lower measured
prefill/prefix TTFT. At limit 4,096, vLLM records 111.262/197.400/353.295
tokens/s and lower 4K/64K cold-prompt TTFT (281.197/4,964.189 ms versus
511.713/8,642.687 ms); Sparkgap retains lower short-prompt and prefix-hit TTFT.
The same output-equality limitations apply. All 791 launches in the limit-4,096 trace also match counters, with steady
batch 1/2/4 graph medians 12.761/15.146/18.536 ms and 15–16.5 µs median gaps.
Accepted-socket TCP_NODELAY passes the full unchanged corpus with all raw IDs
and 16,711 configurations equal; normal-client delivery p99 falls to
12.889/15.338/18.832 ms, with similar core timing/throughput. Ordinary-entrypoint
warm readiness takes 156.207 s, with 51 selection hits and zero candidates.
Independent numerical
acceptance remains open. Full raw records are external; the
[compact JSON receipt](serving-performance.json) indexes them.

## Kernel preparation

The serving bootstrap primes default plans while constructing all capacities,
then runs one batched b12x tuning pass, refreshes every selected binding, releases
trial storage, and warms/captures the final catalog before freezing. Searchable
operators use active fixtures with actual checkpoint weights. Serialized GDN
trials borrow and restore hybrid-state storage. One operator scratch arena
covers the legal candidate envelope, then shrinks to the maximum selected-plan
requirement before final binding refresh and graph capture. The target
GPU defaults to eight compiler workers. BF16 projection configurations and
packed MXFP8 inner GEMM lowerings remain fixed; QSA fixes 16 partitions and GDN
prefill permits only sequential recurrence with one key split. See the
[preparation contract](../core-design.md#preparation-lifecycle).

[Shared operator scratch](unified-scratch.md) records the arena's ownership,
memory use, HTTP comparisons, and complete-catalog replay. Its full evidence is
stored under `/data/sparkgap-artifacts/unified-scratch/`; the repository retains
the Markdown receipt and a compact JSON summary.

[Shared hyperconnection priming](hyperconnection-priming.md) removes 15.08 GiB
of duplicate preparation buffers at capacity 4,096. Measured KV budgeting
preserves the 4 GiB headroom despite allocator overhead. The actual main CLI
captures 43 graphs and passes 98/98 HTTP checks, with all twelve responses equal
to the retained prefill-capacity baseline. The CPU suite passes 271 tests with
54 CUDA skips; native hyperconnection qualification passes sixteen comparisons.

The 37-graph autotuned catalog receipts use a 512-token prefill limit, equivalent
to `--prefill-chunk-size 512`. The configurable default prepares 43 graphs with
a 4,096-token limit and passes [98/98 text and 44/44 image/grammar HTTP checks](prefill-capacity.md).
All 12 paired cold/restored text responses match the 512-token configuration;
metrics verify chunk counts, prefix reuse and resource conservation. The
[4,043-token prefix fixture](prefix-replay-prefill4096.md) passes all 595 state,
QSA-operand and logit equality predicates against a 4,032-token snapshot plus an
11-token tail at capacities 16 and 4,096.
[Complete-catalog Python/Rust replay](graph-execution-parity-prefill4096.md) passes
3,693 checks, 413,708 byte-identical tensor observations, and all 234 allocation
and 234 unrelated-state checks across 43 variants. Independent numerical and
serving-performance acceptance remain open. The
[summary and archive](prefill-capacity.json) preserve all three preparation
failures, the seven-file source difference between configurations, native build
inputs, mapped sampler binaries and clean shutdown evidence.

- [Configured 4,096-token Python/Rust replay](graph-execution-parity-prefill4096.md)
  executes 124 commands in 117 groups per executor across all 43 variants. Six
  large-prefill cases cover capacities 1,024/2,048/4,096 in both sampling modes,
  snapshot/restore into different slots, 65-token tails and decode. The 76-command
  512-token fixture retains its semantics and upload prefixes; only inactive
  padding expands. The pre-reset scan verifies all borrowed hybrid-state bytes.
  A failed full-backing scan and its bounded-memory replacement are preserved;
  the replacement retains complete byte coverage and passes six CPU tests.
- [Configured 4,096-token prefix replay](prefix-replay-prefill4096.md) passes
  595/595 exact equality predicates over seven complete-model comparisons, with
  12 verified completions and native replay allocation guards. All 43 variants are
  prepared; prefix submissions suppress sampling and do not submit vision graphs.
  Preparation reuses 51 selection-cache entries with no candidate measurements.
- [Exact-checkpoint bootstrap](autotune-bootstrap.md) captures all 37 variants
  and exits successfully under `--prepare-only`. Selection caches are partially
  populated; its 28.866-second batched preparation is not a cold-start result.
  [Missing-producer](autotune-bootstrap-missing-producer.json) and
  [MoE route-legality](autotune-bootstrap-moe-legality.json) failures remain
  archived with their actual inputs and source hashes.
- [Unconstrained autotuned prefix replay](prefix-replay-autotuned.md) passes
  512/595 predicates and retains all 83 failures. GDN selected different key-axis
  splits at capacities 16 and 512; the first recorded difference precedes QSA.
- [Constrained GDN policy](gdn-recurrence-policy.md) preserves the public `auto`
  candidate set while exposing an explicit sequential/single-key-split constraint.
  CPU tests cover legality, serialized identity, cache separation, and KDA isolation.
- [Constrained autotuned prefix replay](prefix-replay-sequential-k1.md) passes
  all 595 unchanged predicates over seven complete-model comparisons: 77 hybrid
  regions, complete logits, and recorded QSA intermediates. Preparation measures
  1,271 candidates with zero selection-cache hits in 148.37 seconds using eight
  workers and captures 37 graphs. Releasing preparation owners returns
  7,578,156,032 allocated bytes. Executable-cache cold-start cost is not established.
- [Sampler after recurrence-policy installation](sampling-recurrence-policy.md)
  passes all 37 component tests and 552 native replay blocks, retaining the actual
  loaded sampler binary and allocation checks. The
  [preceding sampler receipt](sampling-autotune-legality.md) and
  [binary provenance audit](sampling-binary-provenance.md) retain their separate
  binary identities and measurement scopes.
- [Autotuned-catalog Python/Rust parity](graph-execution-parity-autotuned.md)
  passes 2,216 checks and 207,620 bytewise tensor comparisons across all 37 graph
  variants, including all 138 allocation and 138 unrelated-state checks. Each
  executor runs 76 commands in 69 groups. Preparation reuses 42 cached searches
  with no candidate measurements; native shutdown drains both streams.
- [Main-source decoder execution and comparison](decoder-autotuned.md) produces
  the same four greedy tokens with finite logits and zero replay allocations.
  The unchanged canonical comparison retains logit relative L2 of 0.159–0.244;
  all four logits and hidden tensors match the unconstrained-autotuning run
  byte-for-byte. Numerical equivalence remains unqualified.

The autotuned implementation is integrated into the main repository. Its
512-token HTTP reruns below preserve all 39 + 8 + 11 model-request payloads
byte-for-byte and every check name and ordering. Their 155 recorded main-source
files match the qualified isolated snapshot and remain unchanged through server shutdown.
Independent canonical text/vision accuracy and serving performance acceptance
remain open; historical receipts retain their recorded preparation conditions.

## Execution and ownership

- [Agentic copy-speculation workload](speculative-agentic-serving.md) records
  23/24 HTTP 200 responses, 18/24 automatic output checks and 16/24 manual passes
  across one frozen stochastic concurrency-1 run. Failures retain the unsupported
  string stop, malformed diff, four 384-token truncated chat answers, missing JSON
  literal and inconsistent retry recommendation. All admitted requests complete;
  no paired speedup or accuracy attribution follows from this single arm.
- [Copy speculation through native HTTP serving](speculative-copy-serving.md)
  passes 36 checks across 14 exact-checkpoint stochastic requests, accepting
  300 copied drafts across all 14 speculative shapes. The receipt distinguishes
  forced-output transaction checks from the open agentic, numerical and
  performance requirements, and retains the independently reproduced sampler
  endpoint failure.
- [FlashInfer chain endpoint correction](flashinfer-chain-endpoint.md) fixes
  probability-one acceptance when the Philox uniform draw is exactly one.
  Four paired GPU fixtures preserve replay allocation counts; 72 endpoint,
  seed/offset and sampler graph regressions pass.

- [Autotuned default HTTP suite](server-http-autotuned.md) passes 152/152 checks
  across 39 model requests, including tool-result/image cold-cache equality,
  concurrent requests, cancellation, structured output and its original
  mixed-grammar case. The recorded main CLI prepares 37 graphs with 42 cached searches
  and no candidate measurements. SIGTERM exits zero, closes the listener and
  releases GPU storage after all three suites.
- [Autotuned vision/grammar suite](server-http-vision-grammar-autotuned.md)
  passes 44/44 checks across eight model requests. Full-capacity 16,384-patch
  image execution, JSON/SSE equality, distinct-image cache identity, both
  cold/repeat output comparisons and shared grammar/plain decode pass.
- [Autotuned sampling routing](server-http-sampling-routing-autotuned.md)
  records 118/120 passing checks across 11 model requests. Greedy/general
  routing, seeded repeats, mixed-temperature batching and the constrained
  one-in-flight barrier pass. The unchanged 1,024-token grammar budget produces
  the same 3,066-byte prefix of the required 3,074-byte literal; exact-literal
  and `stop` assertions remain failed.
- [ABI-2 startup rejection](native-startup-validation-abi2.md) passes 21 malformed
  manifest/configuration cases through the installed native constructor. Every
  case releases its resource owner before any CUDA device/library open, network
  syscall or thread creation. Inputs use synthetic undereferenced addresses;
  this qualifies rejection boundaries, not model preparation or inference.
- [Fatal model-worker propagation](native-worker-failure.md) passes 22 checks
  using deliberately malformed captured completion records. Waiting JSON and
  SSE responses report the execution error, health and later admission return
  503, both streams drain, and scheduler resources and graph owners are released.
  The installed runtime and HTTP routes are exercised with a small graph fixture;
  checkpoint inference, panic and failed-drain behavior are outside this test.
- [Installed FlashInfer sampling](sampling-flashinfer.md) uses
  the [pinned sampling contract](flashinfer-sampling-contract.md) with explicit
  device RNG and native graph replay. All 37 focused/admission tests pass.
  [Component measurement](/data/sparkgap-artifacts/validation-receipts/sampling-paths-flashinfer-installed.json) passes 552
  checked blocks with no replay allocation growth. Active default sampling
  takes approximately 129/151/170 µs at batches 1/2/4, versus 174/783/812 µs for
  the [sorted-CDF baseline](/data/sparkgap-artifacts/validation-receipts/sampling-paths-original.json). These measurements
  include common token-state maintenance and exclude model forward/scheduling.
  [The declared top-p patch](sampling-air-parallel-qualification.json) preserves
  bitwise output in 375 deterministic API and 300 same-threshold comparisons.
  ABI 2 routes all-greedy batches to 18–20 µs greedy samplers; full-model replay
  passes the 37-graph parity gate below. Stochastic policies differ from sorted-CDF;
  historical seed results remain attached to their recorded implementation.
- [HTTP sampler routing](server-http-sampling-routing.md) passes 118 of 120
  checks across 11 model requests. All-greedy requests select greedy graphs;
  positive-temperature requests, including top-k 1, select general graphs.
  Equal-full-hit seeded responses repeat exactly. Two mixed-temperature
  requests produce 64 tokens each in 65 decode launches, below the unbatched
  minimum of 126. The grammar/plain window records 63 shared general decode
  submissions with one constrained submission in flight. Its long literal
  reaches the fixed 1,024-token budget before the closing quote, so the exact
  literal and `stop` checks fail. Health and resource conservation pass;
  the failed receipt and original inputs remain intact.
- [HTTP regression with FlashInfer](server-http-flashinfer-regression.md) passes
  150 of 152 checks across 39 model requests and 215 total HTTP requests. The
  default suite and its original mixed-grammar case retain their inputs and
  oracles. Exact constrained output, tools, thinking, concurrent requests,
  cancellation and slot reuse pass. Only the recorded tool-result and image
  cold/cache output comparisons fail. [CLI shutdown](server-sampling-shutdown.json)
  exits zero, closes the listener and returns the assigned GPU to 2 MiB with
  no compute processes.
- [Structured continuation validation](structured-continuation.json) rejects
  inherited assistant continuation when a fresh response grammar is requested.
  Explicit false, null, empty-string and nested request overrides retain their
  precedence. Exact-checkpoint CPU request preparation passes 28 text, two tool
  round-trip, five image and 16 structured cases. The
  [packaged native build](native-grammar-build.json) also exposes mixed-grammar
  decode counts and the maximum constrained submission depth; its Rust suite
  has 83 passing tests and one checkpoint-dependent test omitted.
- [Native host-path audit](native-host-path.md) records bounded scheduler work,
  upload sizes, allocation sites, grammar costs, and output parsing scans. It
  supplies profiling targets without claiming measured latency or speedup.
- [Complete decoder replay](decoder-execution.json): native graph execution
  produced ` Paris. The capital` from `The capital of France is`, with finite
  logits and zero CUDA allocator activity during each replay.
- [FlashInfer catalog Python/Rust parity](graph-execution-parity-flashinfer.md):
  2,216 checks and 207,620 bytewise tensor comparisons pass with zero mismatches.
  Each executor runs 76 commands across 69 groups covering all 37 graph variants,
  mixed-temperature batches and both general/greedy interleaving orders.
  Request state, hybrid-state regions, KV pages and completion frames agree;
  replay preserves CUDA allocator counters and native shutdown drains both streams.
  Preparation takes 92.5 seconds and leaves 9,768 KV pages with 4 GiB headroom.
  This compares execution of identical graphs, without an independent numerical oracle.
- [25-graph prepared model](prepared-model.json): all 25 graph variants prepare and
  capture, including vision, prefill, decode, resume, initialization, restoration,
  and snapshot publication. Cache capacity is calculated from measured free VRAM.
- [Sorted-CDF catalog Python/Rust parity](graph-execution-parity.json): 1,495 checks and
  126,816 bytewise tensor comparisons pass with zero mismatches. Each side runs
  the same 51 commands across all 25 captured graph variants. Checks cover 77
  regions in live, candidate, tested snapshot, and canary rows; request state,
  RNG progression, penalty histograms, touched KV pages, logits, and completion
  frames. All 94 replay groups preserve CUDA allocator counters and unrelated
  state. Reset storage hashes and graph/buffer identities match before the
  exclusive native handoff, and native shutdown drains both streams. Vision
  uses one 256-patch fixture at every graph capacity; full-capacity image content
  and independent model arithmetic are outside this comparison. The summary
  links compressed complete observations and preserved harness failures.
- [HTTP qualification with native metrics](server-http-native-metrics.json):
  132 of 134 checks pass across 81 HTTP requests. Four simultaneous requests
  complete with outputs matching their individual baselines. OpenAI JSON-string
  tool arguments round-trip through checkpoint-template rendering. Text,
  Unicode, thinking, tool-call generation, grammar, cancellation, and subsequent
  admission execute through Rust HTTP/SSE. The two failures are cold/cache greedy
  output differences on a tool-result prompt and PNG prompt.
- [Full-image, conditioned-cache and mixed-grammar qualification](server-http-vision-grammar.json):
  42 of 44 checks pass across 23 HTTP requests, including metrics. An actual
  16,384-patch image produces 4,096 features and a 4,116-token prompt; JSON and
  SSE outputs agree, and the repeat restores 4,096 cached tokens. Identical text
  with different RGB images produces separate cache misses, then each image
  restores its own 64-token checkpoint. Both small-image cold/cache greedy
  comparisons differ. One decode submission contains both constrained and
  unconstrained lanes, the constrained in-flight maximum remains one, and
  both the exact JSON string and plain-text Paris response pass. Every request
  completes and idle resources reconcile. These checks do not establish
  independent encoder or decoder numerical equivalence.
- [Four-image capacity qualification](server-http-vision-aggregate.md): all
  30 checks pass for three actual requests, each encoding four 16,384-patch
  images into 16,384 features and a 16,412-token prompt. Cold JSON, repeated
  JSON and SSE outputs agree; each request launches four vision graphs and
  idle resources reconcile. Canonical CPU processing verifies image grids,
  expansion and positional metadata. This establishes execution at the declared
  aggregate image limit, without an independent model-numerical oracle.
- [Context-limit execution](server-http-context-limit.md): all 18 checks pass.
  A 262,136-token prompt plus eight generated tokens executes 512 prefill and
  seven decode graphs. HTTP usage reaches the 262,144-token limit; consumed
  length 262,143 follows from those counts and the pending-token contract.
  A total of 262,145 is rejected before admission. The final snapshot retains
  4,095 pages and returns the remaining private page; a subsequent short
  request completes. These are capacity and indexing checks, not long-context
  quality measurements. [Shutdown](server-capacity-shutdown.json) exits with
  status zero and leaves the assigned GPU at 2 MiB with no compute processes.
- Cache metrics confirm restoration of 128-token prefixes. A complete prompt
  hit executes one restore and one resume graph with zero prefill graphs;
  appended tails and independent branches execute one restore and one prefill.
  The HTTP receipt preserves every raw scrape and counter delta.
- [Cache pressure and stalled-reader qualification](/data/sparkgap-artifacts/validation-receipts/server-http-pressure.json):
  224 of 224 checks pass across 246 HTTP requests, including metric scrapes.
  Four GPU-active request slots are observed. Publishing 38 distinct prefixes
  across 37 snapshot slots causes eviction; the oldest prefix replays cold with
  identical output. A reader paused for five seconds permits unrelated request
  completion, and its subsequent disconnect cancels once. Idle slot, page, and
  snapshot counts reconcile. The test does not saturate the native output queue.
- [Appended-conversation cache qualification](cache-conversation.json): the CPU
  scheduler completes 12 appended turns with three snapshot slots, retaining
  every prefix page and reclaiming all storage afterward. Single-child ancestor
  compaction reclaims snapshot slots when a pinned descendant prevents leaf
  eviction. Tests cover transferred GPU retirement fences, pending-child
  cancellation, direct pins, and duplicate publication. The native library has
  82 passing tests and one environment-dependent grammar trace omitted from the
  general test command. [Build provenance](native-conversation-build.json)
  identifies the packaged implementation.
- [HTTP appended-conversation qualification](server-http-conversation.json):
  210 of 210 checks pass across 86 HTTP requests, including metrics. Forty model
  requests grow one conversation from 160 to 2,656 prompt tokens across 37
  snapshot slots. All 39 appends reuse the entire available prefix, totaling
  52,416 reused tokens, while three ancestor snapshots are reclaimed. A repeat
  restores the final 2,624-token checkpoint. Idle resource accounting reconciles
  after every request; shutdown exits with status 0 and leaves GPU usage at 2 MiB.
- [KV exhaustion and admission qualification](server-http-admission.json):
  74 of 74 checks pass on the [packaged admission build](native-admission-build.json).
  Two requests reserve 4096 KV pages each. Excess admission returns HTTP 503
  for JSON and a terminal SSE error after HTTP 200 headers. Both admitted
  requests continue generating; cancellation reclaims their resources and a
  subsequent request completes. Four concurrent requests and slot reuse also
  pass. The [exhaustion diagnostic](server-http-kv-pressure.json) records the
  HTTP 500 response before typed admission rejection.
- [Tool workload qualification](tool-eval-native.json): all 15 scenarios pass,
  scoring 30/30, matching the recorded exact-checkpoint vLLM result. The same
  clean benchmark commit, scenario configuration, seed 42, temperature 0, and
  reference date are used. [Raw results](tool-eval-native-raw.json) preserve each
  tool trajectory. The reference uses TP2, FP8 KV and MTP; Sparkgap uses one GPU,
  BF16 KV and target-only decode. These conditions do not establish a matched
  performance comparison or numerical acceptance.
- [Standard tool-suite qualification](tool-eval-native-standard69.md): all 69
  scenarios are recorded, scoring 120/136 (88%) with 54 passes, 12 partials,
  two scored failures, and one capability exclusion. TC-45 is excluded because
  forced tool selection is unsupported. Four tool-plus-schema cases receive
  partial credit after HTTP 400 rejects their final structured-output requests.
  TC-68 returns prose instead of raw JSON. TC-51 triggers the benchmark safety
  gate: its local evaluator permits same-turn calendar/email ordering, but the
  runner overrides that result using a stricter cross-turn dependency rule.
  [Raw results](tool-eval-native-standard69-raw.json) retain every outcome and
  trace. Sources and the native extension remain unchanged across execution.
- [Paired standard tool-suite comparison](tool-eval-standard69-comparison.md):
  the exact-checkpoint TP1 vLLM run records all 69 cases at 119/138 (86%), with
  55 passes, nine partials, five failures, and no exclusions or HTTP errors.
  The benchmark settings and all 225 recorded harness source/fixture hashes
  match Sparkgap. vLLM passes forced tool selection and the four tools-plus-schema
  cases. Both runtimes fail TC-51 under the runner dependency rule and TC-68's
  prompt-only JSON contract. Original denominators and safety warnings are
  retained; 56 point awards agree and 13 differ. One run per runtime does not
  establish statistical quality superiority, numerical equivalence, or measured
  performance acceptance.
- [Pinned TP1 vLLM startup and shutdown](vllm-tp1-startup.md): full-context
  preparation reaches readiness in 298.094 seconds with 18.74 GiB of cache.
  The receipt preserves recovered allocation warnings, two inference-time
  compilation warnings, and selected kernel plans. Controller/API exit is zero
  and GPU memory is released; vLLM internally force-kills its remaining engine
  worker and reports one leaked semaphore.
- [Admission and image diagnostic](server-http-initial.json) and
  [tool/image repetition diagnostic](server-http-supplemental.json) retain the
  failing bounded-admission and tool-template cases alongside cold/cache output
  differences. Their requests and raw responses remain available for comparison.
- [In-flight shutdown](server-shutdown.json): SIGTERM after a visible generated
  token closed the stream with a shutdown error, exited with status 0, and left
  the assigned GPU at 2 MiB usage.
- [Admission-build shutdown](server-admission-shutdown.json): SIGTERM after
  visible output closes the stream with the shutdown error in 0.788 seconds;
  process exit is 0 and the model allocation is released. The
  [readiness observation](native-server-readiness.json) records 70 refused
  connections during preparation before the first healthy HTTP response.

## Prefix-state isolation

[Policy-constrained autotuned replay](prefix-replay-sequential-k1.md) passes
595/595 bitwise predicates on the exact 459-token tool-result prompt, with BF16
GDN intermediates, sequential single-key-split recurrence and an aligned checkpoint
at 448 tokens. Twelve native
submissions preserve CUDA allocation counts and return the expected completion
records. All nine decoder capacities bind 16 QSA partitions.

| Comparison | Result |
|---|---|
| Cold execution repeated | All 77 state regions and logits bit-identical |
| Candidate to immutable snapshot | All 77 regions bit-identical |
| Immutable snapshot to restored live row | All 77 regions bit-identical |
| Candidate versus independently executed 448-token prefix | All 77 regions bit-identical |
| Cold versus restored 11-token tail at capacity 512 | All 77 regions and logits bit-identical |
| Restored versus directly split execution at tail capacity 16 | All 77 regions and logits bit-identical |
| Cold versus restored tail at capacity 16 | All 77 regions and complete logits bit-identical |

Thirteen recorded layer-3 QSA tensors also agree on every common logical row.
CPU readback independently reproduces all 595 predicates. The unchanged
[autotuned default HTTP suite](server-http-autotuned.md) passes 152/152 checks, and
[full-size image/cache-identity qualification](server-http-vision-grammar-autotuned.md)
passes 44/44. Tool-result and image cold/cache output comparisons pass.

The [row-dependent baseline](prefix-replay-bf16.json) has logit relative L2
0.287058 for the capacity-16 restored tail. Its
[recorded operands](prefix-qsa-recording.json) isolate the first difference to
attention after exact projection/selection; [same-input replay](qsa-capacity-isolation.json)
reproduces the direct-versus-split difference. Fixed partitions remove that
capacity-dependent reduction order. [Short-prefix](qsa-fixed-partitions.md) and
[2,051-key attention matrices](qsa-long-partitions.md) cover 4,610 native replays,
bitwise capacity invariance and allocation stability. The long-key matrix also
passes the unchanged sparse-reference gate. Different partition counts still
produce different BF16 values; the recipe fixes the count at 16.

[Installed operator qualification](qsa-fixed-splits-installed.md) passes 19/21
checks. Its two request-ID dtype errors reproduce identically on the historical
build; all nine fixed-partition and three exact-checkpoint tests pass. These
results do not resolve the independent model-reference failures below.

## Numerical qualification

[Exact-checkpoint MTP module comparison](mtp-module.md) establishes bit-exact
embedding and feedback on three identical inputs, with complete-module logit
relative L2 of 0.219–0.302 against pinned vLLM. Recorded key-history differences
explain substantial attention amplification; their initial cause remains open.
The receipt links raw tensors, source snapshots and preserved failures in the
external artifact store. Full speculative serving is not established by this
component measurement.

[Identical-input MTP operator comparisons](mtp-operators.md) isolate prefill
key differences to projection arithmetic. Both backends reproduce their
recorded forward outputs; QK normalization is exact on identical projections.
Sparkgap's tested prefill K projection exactly matches the CPU FP64 dot rounded
to BF16, while the pinned reference differs in 577 of 2,560 values. These
measurements preserve production arithmetic and leave expert and complete
serving qualification separate.

[Teacher-forced text serving logits](text-serving-oracle.md) compare the recorded
Sparkgap decoder against the pinned vLLM engine over four identical token contexts.
Sparkgap/vLLM relative L2 is 0.177–0.224, with all argmax IDs equal and top-10
overlap 9/10/9/6. Two steps exceed the corresponding vLLM/canonical discrepancy.
An uninstrumented vLLM repetition also differs by 0.167–0.209. Two identical
cold requests in one vLLM engine differ by 0.190–0.240, first diverging at the
layer-0 MoE output despite identical MoE inputs. The
[48-layer comparison](text-layer-comparison.md) records that progression beside
Sparkgap's discrepancy, whose first unequal operation is HC normalization.
Component alignment and task-level generation agreement remain open. Raw
tensors, worker observations and launch provenance are stored under
`/data/sparkgap-artifacts/text-serving-oracle/`; the repository retains a compact
JSON summary and Markdown receipt.

[Exact-checkpoint serving provenance](exact-serving-reference.md) identifies a
recorded vLLM run using the same 21 checkpoint shard files and matching metadata
assets. Its logged selections include b12x GDN, QSA, and linear kernels, with
FlashAttention for vision. Its arithmetic differs from the eager Transformers
adapter; the recorded run provides a separate serving reference.

| Component and conditions | Result | Evidence |
|---|---|---|
| Main-source autotuned text model versus canonical checkpoint reference | Four greedy tokens agree; logit relative L2 0.159–0.244 and hidden relative L2 0.159–0.278 remain unqualified | [Autotuned decoder comparison](decoder-autotuned.md), [reference specification](decoder-reference.json), [stage comparison](decoder-stage-comparison.json) |
| BF16-intermediate GDN, 65-token checkpoint case | Pointwise gate fails | [GDN diagnostic](gdn65.json) |
| GDN normalization after tensor-core dot products, same checkpoint case | Relative L2 improves from 0.0143301 to 0.0135014; four unchanged pointwise failures remain | [Isolated precision diagnostic](gdn-normalization-factors.md) |
| GDN V′/U/MQK BF16 residual corrections, same complete checkpoint test | Initial prefill and decode pass; continued prefill fails with relative L2 0.0155813 and three pointwise errors | [Continuation diagnostic](gdn-compensated-bf16.md) |
| Explicit FP32-intermediate GDN, same checkpoint case | Unchanged component gate passes; output relative L2 0.00150533 | [FP32 qualification](gdn-fp32.json) |
| GDN 92-token prefill versus 64-token checkpoint plus 28-token tail | Both arithmetic variants are bit-identical on identical projected operands | [FP32 qualification](gdn-fp32.json) |
| Vision with flash attention arithmetic | Encoder relative L2 0.0780035 exceeds 0.03 limit | [Vision qualification](vision.json) |
| Vision with actual vLLM module and serving preparation policy | Sparkgap/vLLM relative L2 0.132588 / 0.0706193; vLLM/canonical 0.122242 / 0.0641935; normalized BF16 inputs bit-identical | [Paired serving reference](vision-serving-reference.md) |
| Vision with explicit BF16 score and normalized-probability boundaries | Two-image encoder relative L2 0.0933215 / 0.119356 exceeds unchanged limit | [Attention variant qualification](vision-eager-bf16.json), [stage trace](vision-eager-bf16-stages.json) |

The explicit FP32 GDN variant reduces workspace from 44,624,896 to 6,144 bytes
at capacity 512, but increases measured 512-token recurrence latency from 42.74
to 607.59 microseconds. These are isolated operator timings, not complete-model
performance acceptance. Decode uses a separate recurrence kernel.

The vision CPU arithmetic diagnostics isolate attention rounding on identical
checkpoint inputs: [image 0](vision-attention-arithmetic-cpu.json) and
[image 1](vision-attention-arithmetic-cpu-variant1.json). They do not replace the
unchanged CUDA encoder gate.

With exact canonical patch embeddings, the explicit vision attention variant
still produces 0.0808825 / 0.0587151 encoder relative L2 on the two image inputs.
Its same-input raw attention errors remain below 0.0001 in every block; the
complete quantized encoder amplifies small component differences. Serving uses
flash attention arithmetic. The explicit vision variant and FP32 GDN variant
remain source-only kernel diagnostics outside the locked preparation patch.

[Component-family trajectories](vision-component-trajectories.json) replace one
family at a time while retaining canonical execution everywhere else. Patch
embedding, QKV, attention, output projection, and block MLP replacements each
exceed the 0.03 encoder limit on both images, ranging from 0.0493205 to
0.0891624. Merger-only errors are 0.0001004 / 0.0000826. The canonical control
and composed native control are bit-identical to their respective complete
encoders. A single-family arithmetic correction is insufficient for these
inputs.
