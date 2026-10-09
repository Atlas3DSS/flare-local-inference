# Core correction closeout plan

This plan covers every item in [the second correction set](core-design-corrections-2.md),
including the requirements it carries forward from [the first set](core-design-corrections.md).
The audit uses the working trees on 2026-10-09.
The correction documents remain the requirements; this file records implementation
order, remaining work, and evidence required for closure.

## Scope and constraints

- Python loads, plans, tunes, warms, and captures. Rust owns serving after handoff.
  Runtime execution remains exclusively captured graphs, with no Python callbacks.
- Library model operators use public b12x/FlashInfer APIs on NVIDIA SM120/121.
  Captured tensor composition, metadata, and state orchestration remain recipe
  responsibilities. Operator scratch is caller-owned and shared across sequential layers.
- Sampling has one probability-sampling implementation. Temperature zero produces
  a point mass through that implementation; it does not introduce a greedy runtime
  route or a separate speculative verifier.
- Reasoning parsing and tool-call parsing have separate model-independent Rust
  interfaces. Each supported model supplies their implementations and composition.
  The HTTP layer consumes neutral events; Qwen is the first implementation.
- The primary serving measurements use temperature 1, top-k 20, and top-p 0.95.
  The frozen 82-prompt diagnostic retains its exact 64-token, temperature-zero,
  ignore-EOS contract through the common sampler. Its qualification-only behavior
  does not become a public serving feature.
- Qualification preserves request bytes, seeds, budgets, checkpoint identity, and
  reference revisions. Source changes require new provenance and frontend proof;
  identical payload hashes establish continuity between those records.
- The requested default-output-budget change remains deferred. Explicit request
  budgets continue to win. No fixture budget is enlarged to turn a failure into a pass.

The qualification checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, loaded from
`/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`.
The reference is vLLM `64fc1eec593397cbd765cc082e9ec90fbeb5ea66` with FlashInfer
`98fb87a461183f98d6a7049ca947fb136cc6fb59`. GPU work uses
`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`; its SM120 results do not establish SM121 behavior.

## Audited position

Copy speculation is implemented with ABI 3 completions, one round in flight,
all 14 shapes satisfying batch × width ≤ 8, and cloned per-candidate grammar
states. The [native HTTP transaction receipt](validation/speculative-copy-serving.md)
exercises all shapes with forced literal output. The
[24-context agentic run](validation/speculative-agentic-serving.md) records
23 successful HTTP admissions, 18 automatic semantic passes, and 16 manual
passes. Its eight failures remain in the evidence. Learned MTP serving and
paired speculative speedup are unqualified.

The [copy-state replay comparison](validation/copy-state-replay.md) passes all
30 commands through Python and Rust under `qwen4exp.graph-inputs.v2` and typed
native errors: 2,852 checks pass with zero state mismatches and unchanged replay
allocations, including an exact 72-token snapshot and private partial-page restore.
Independent W1/Wn execution and selected-column restored continuations cover
all 14 shapes and produce 44,640 bitwise-equal state comparisons for all-accepted
literal copies. The ordinary-decoder arm, complete accepted-prefix coverage,
and calibrated cross-capacity numerics remain open.

The [native output-protocol HTTP checks](validation/output-protocol-http.md)
pass JSON/SSE literal-marker preservation and string stops. The unchanged
`code-docstrings-stop` workload completes within its original budget with all
eight function bodies preserved. The complete agentic workload rerun remains open.

The [FlashInfer chain acceptance correction](validation/flashinfer-chain-endpoint.md)
passes its endpoint and sampler regressions. The
[identical-input MTP operator measurements](validation/mtp-operators.md) isolate
projection arithmetic differences and show b12x closer to the rounded mathematical
dot on those operands. Neither result closes complete-model numerical acceptance.

The [serving performance receipt](validation/serving-performance.md) establishes
the historical ordinary-decoding host overhead and graph durations. It contains
GPU graph timing, but no named per-kernel GPU breakdown. Those measurements do
not establish the overhead of the speculative sequence or the unified stochastic
serving implementation.

The static native model contract owns Qwen input packing, schema validation,
prompt interpretation, and completion semantics. JPEG/EXIF decoding passes
checkpoint admission and all 26 valid-image HTTP admissions; two constrained
outputs reach their explicit budget in whitespace loops. Typed native errors
preserve source chains, driver codes, and status mapping; all 22 installed fatal
completion HTTP/SSE checks pass. Private worker phases preserve transaction
ordering and pass CPU phase tests and [full-model checkpoint HTTP checks](validation/exact-checkpoint-http.md).
Tool-plus-schema grammar and the upstream AIR submission remain open. Preparation, recipes, the native
runtime, and qualification tooling have separate source commits. The completed items in correction section 0 remain
closed, subject to their original measurement scopes.

## Execution order

1. **Maintain source history and qualify replay.** Extend the passing copy-state
   replay comparisons to changed catalogs and uncovered transaction cases. Restore port 8000
   after exclusive GPU runs. Keep useful debugging payloads outside Git and commit
   validated changes in reviewable groups. Keep the FlashInfer dependency pin
   matched to committed serving changes. This work does not wait for every feature to pass.
2. **Establish numerical and construction gates.** Replace the eager-based limits,
   define calibrated cross-capacity checks, preserve the real-constructor CPU bootstrap
   tests, and run the frozen generation/reference-variance corpus. Keep identical
   graph execution and same-capacity restore comparisons bitwise. Capture the
   unchanged ordinary common-sampler B1/B4 kernel baseline alongside this work;
   that measurement does not depend on learned MTP qualification.
3. **Preserve the native model boundary.** Keep the static model contract, typed
   errors, private worker phases, and Qwen-owned packing/output semantics aligned.
   Preserve upload bytes and transaction ordering.
4. **Complete local serving contracts and copy qualification.** Implement
   tool-plus-schema grammars and rerun the agentic workload with native string stops.
   Preserve JPEG/EXIF admission coverage and its unresolved output failures.
   Qualify the exact checkpoint-retention
   implementation described below. Qualify copied drafts against independent W1
   execution, including actual prefix restores, every accepted count, and failures.
5. **Qualify learned MTP.** Use the qualified transaction with the checkpoint's MTP
   weights, feedback, attention/cache, experts, and head. Complete full startup,
   prefill absorption, accepted/rejected rounds, prefix restore, and HTTP workloads.
6. **Measure and change kernel policies.** Use the ordinary kernel baseline and
   capture speculative timelines, fixed-width costs, state-write costs, and the
   speculative replay floor. Remove
   production tuning pins one at a time under the numerical gates. Measure fixed
   depths before calibrating adaptive width; compare the complete workload arms.
7. **Complete extraction and documentation.** Move remaining qualification
   logic without changing its contracts, remove receipt narrative from the
   specifications, and run the final installed-entrypoint acceptance on pinned
   sources. Commit each coherent change as it is validated.

CPU work on documentation, image fixtures, failure tests, and constructor coverage
can proceed in parallel. GPU qualification uses the assigned device exclusively.
Source files recorded by an active run remain frozen until that run completes;
parallel implementations use isolated source copies until integration.

## Correction ledger

“Implemented; evidence open” means the requested behavior exists but the correction's
acceptance condition is not yet satisfied. A passing component fixture does not
close a serving or performance requirement.

### MTP design and execution

| Item | Status and evidence | Work required to close |
|---|---|---|
| 1.1 One speculative round in flight | Implemented in [scheduler.rs](../native/src/scheduler.rs); speculative submission installs a barrier. | Measure native overhead against a replay of the same useful speculative graph sequence, including required uploads and actual committed outputs. Require ≤5%; retain graph gaps and timing distributions. |
| 1.2 Copy before learned MTP | Copy-only preparation excludes MTP weights/cache. Same-graph Python/native replay and independent W1/Wn comparisons pass the recorded all-accepted literal cases across all 14 shapes. Agentic acceptance remains open. | Complete ordinary-decoder, rejected-prefix, and changed-catalog comparisons, then rerun the unchanged agentic corpus. Separate engine defects, unsupported API behavior, and model-output failures using ordinary/reference arms. Qualify learned MTP only after transaction qualification. |
| 1.3 Widths follow kernel bands | All B1 widths 1–8, B2 widths 1–4, and B4 widths 1–2 exist. | Record repeated fixed-shape round-cost distributions for all 14 shapes. Measure the grouped-path transition with diagnostic B2×4/5 and B4×2/3 comparisons. Shapes beyond eight total rows stay outside the serving catalog until measured and qualified. |
| 1.4 State-write cost | The design records 108 MiB recurrent state per verified row. Measured write cost remains open. | Report logical bytes written per graph and measured GDN/state-selection time. Separate compulsory checkpoint-column writes, commit/copy traffic, and total round duration. Do not present a bandwidth-derived lower bound as measured time. |
| 1.5 Stochastic primary path | Implemented with FlashInfer; no greedy route remains. One copy-adaptive C1 workload repetition exists. | Measure ordinary, copy-only, fixed learned-MTP, adaptive MTP without copying, and adaptive MTP with copying on identical stochastic workloads at C1/2/4. Report acceptance by depth/source and completed-work latency/throughput. Report temperature-zero timings separately through the same FlashInfer path; the 82-prompt correctness run is not a performance measurement. |
| 1.6 Grammar speculation | Cloned llguidance state and candidate/bonus masks are implemented. Independent W1/Wn literal comparisons pass with all copied tokens accepted. | Cover mixed requests, rejection, EOS, empty masks, cancellation, and grammar-state commits beyond the singleton-language cases. Preserve exact accepted output and state comparisons. |
| 1.7 Research outside specification | Closed: [mtp-design.md](mtp-design.md) links [competitive-mechanisms.md](/data/sparkgap-artifacts/mtp-research/competitive-mechanisms.md). | Preserve the external research artifact and its source fingerprints when moving documentation assets. |

The copy transaction gate must cover QSA groups of four and pages of 64; all accepted
prefix lengths; GDN W8→W1 with eight previously accepted inputs; convolution and
PLE history; inactive/permuted lanes; request epochs; penalties and RNG; terminal
clipping; snapshot pressure; cancellation; and image-conditioned prefixes.
The [state qualifier](../qualify/speculative_state_parity.py) compares every hybrid
region and row through the same graph handles. It does not substitute for the
independent W1 or learned-draft-cache comparisons.

### Numerical acceptance and performance

| Item | Status and evidence | Work required to close |
|---|---|---|
| 2.1 Numerical acceptance | Specification amended; implementation incomplete. GDN 0.015 and vision 0.03 eager limits remain. The 82-prompt corpus/tools exist without a completed agreement receipt. | Remove whole-logit/encoder relative-L2 gates; retain diagnostic values. Replace the GDN 0.015 component assertion and associated pointwise bound with independently measured component tolerances. Run Sparkgap and two independent pinned-vLLM continuations on all 82 prompts; publish first divergence, median agreement, full-64 agreement fraction, and the reference repeatability floor. Extend identical-input coverage to all target, vision, and MTP operator families. |
| 2.2 Determinism | Runtime and correctness contracts require identical graph inputs, persistent state, and explicit RNG state to produce bitwise-identical logits and transitions. The validation index records pinned-vLLM variance. Updated speculative-catalog evidence remains open. | Qualify repeated identical inputs across the admitted catalog. Preserve reference-variance evidence and distinguish determinism from differences between capacities, plans, or sampling algorithms. |
| 3.1 Measured position | Historical baseline closed at its recorded revisions and sampling configuration. | Preserve its raw results. Establish the matching baseline for changed stochastic code before using it in any new performance comparison. Do not inherit historical timing claims after kernel or sampling changes. |
| 3.2 Per-kernel GPU timeline | Open. The [graph-node attribution probe](validation/graph-node-attribution.md) maps a two-operation graph through explicit CUPTI/Nsight node relationships. Ordinary Qwen B1/B4 kernel measurements remain pending. | Apply node tracing to the actual ordinary operator sequence. Report kernel names/counts, layer/operator attribution, summed kernel time, graph span, and gaps. Verify outputs and quantify instrumentation overhead against uninstrumented replay. |
| 3.3 Free autotuning and policy A/B | Open. [gdn.py](../sparkgap/recipes/qwen38/gdn.py) pins sequential_k1; [attention.py](../sparkgap/recipes/qwen38/attention.py) pins 16 splits. Prefix qualification lacks the required tolerance gate. | Calibrate and freeze cross-capacity floating-state/component predicates with exact metadata; keep same-capacity checks bitwise. Complete logits remain diagnostic under item 2.1; generation uses the reference floor. Expose pinned choices as explicit debug configuration. A/B GDN splits, QSA partitions, MoE reduction/fast-math policy, and hyperconnection composition separately, each with correctness and before/after timelines. Then allow ordinary preparation to select qualified plans freely. |
| 3.4 Kernel-count workstream | Bandwidth-bound wording is corrected; kernel attribution/fusion work remains open. | Use the timeline to rank adjacent small-operator fusion opportunities, especially hyperconnection normalization/projection. Implement reusable compute changes in main-tree b12x, preserving scratch planning, tuning, capture, and numerical contracts. Report kernel count and full-graph benefit, not only isolated-kernel speed. |

Component tolerances must be frozen before evaluating the candidate being
accepted. Exact token/page indices, generations, histories, logical lengths, and
ownership remain exact at every capacity. Numerical tolerance applies to floating
state and individual operator outputs where arithmetic changes with the selected
plan. Item 2.1's amended acceptance controls complete-model logits: record them
for diagnosis and use the generation/reference floor, without reinstating a
whole-logit relative-L2 gate during cross-capacity work.

The determinism contract constrains policy selection: a nondeterministic MoE
experiment may be measured, but cannot become the default while item 2.2 requires
identical same-graph logits. Separate fast math from reduction-order changes.
FP32 checkpoint normalization weights remain the faithful baseline; measure their
cost before proposing a precision change. Compensated-BF16 and normalization-reorder
work remains stopped.

### Repository, construction, and documentation

| Item | Status and evidence | Work required to close |
|---|---|---|
| 4.1 Commit working tree | Implemented: legacy removal, native runtime, Python preparation, qualification, and documentation use separate commits. | Commit each coherent validated change as it lands. FlashInfer pinning remains item 4.3. |
| 4.2 Externalize archives | Closed: bulk files live under `/data/sparkgap-artifacts/validation-receipts`; repository links and compact locators resolve there. | Keep generated archives out of Git using the archive ignore rules. Retain debugging data only while it remains useful. |
| 4.3 FlashInfer commits and pin | Main checkout is clean at `c691c681ebcbda066faf5bd80fef0d257b256fbc`, with seven commits separating operator contracts, diagnostics, and sampler changes. Source validation and locked dependency checks pass. | Submit the AIR normalization change (`f1cd7bb29`) upstream separately with its correctness and performance tests. |
| 4.4 Specification cleanup | Receipt narratives are removed from core design; sampling, determinism, acceptance, native model packing, worker phases, and preparation fixtures are specified. | Keep section-7 mappings aligned with remaining qualification-tool extraction. |
| 4.5 CPU bootstrap | Closed: [constructor tests](../tests/preparation/test_bootstrap_construction.py) invoke actual `prepare_model`, weight-tree assembly, Prepared constructors, and public b12x plans for ordinary/copy/MTP catalogs (28/39/39 graphs). Five tests pass with CUDA initialization forbidden. | Preserve graph/operator declarations, capacities, dtypes, scratch ownership, and malformed-weight rejection. These tests stop at tuning; compilation, capture, and numerics require GPU checks. |
| 4.6 Receipt prose | Closed: validation index and acceptance coverage state behavior, limitations, and remaining gates; historical measurements stay in their individual reports. | Keep acceptance summaries aligned with implementation and actual test results. |

Debugging payloads remain outside Git while useful. Source commits and dependency
pins identify reproducible code; uncommitted imported-code changes invalidate a
clean-checkout claim.

### Local features

| Item | Status and evidence | Work required to close |
|---|---|---|
| 5.1 JPEG and EXIF | Implemented with native libjpeg-turbo and all eight orientations. All 26 CPU fixtures match Pillow RGB exactly under the unchanged tolerance of 2, and all 26 pass [installed HTTP admission](validation/jpeg-exif-http.md); four invalid image requests are rejected. | Preserve decoder, preprocessing, resource-limit, and HTTP admission coverage. Investigate the two constrained-output whitespace loops separately: PNG orientations 2 and 5 exhaust the unchanged 64-token budget. Semantic vision accuracy remains open. |
| 5.2 Tools plus structured output | Open. [request.rs](../native/src/server/request.rs) rejects tool-plus-grammar and forced selections. | Compose checkpoint tool-call envelopes, admitted per-tool argument schemas, and final-content schema in llguidance. `auto` admits the union, `required` only tool alternatives, a named choice only that tool, and `none` only content. Share this grammar with speculative cloned-prefix masks. Require correct JSON/SSE presentation and no committed tokens outside the selected language. |
| 5.3 Bounded top-k | Closed as an explicit deferral in [core-design.md](core-design.md). | Keep historical 120–170 µs component cost labeled by its measurement scope. No custom bounded-top-k implementation is required for correction closure. |

If a supported JPEG fixture exceeds the required RGB bound, retain that failure
and investigate decoder configuration; do not relax the tolerance or substitute
PNG as evidence. Tool grammars must describe the checkpoint's actual function-tag
serialization rather than an invented model output format. Preserve schema
validation, malformed-tool errors, and streaming/nonstreaming equivalence.

Tool/schema construction needs a CPU proof over Qwen's actual raw parameter
bodies. The pinned llguidance API does not directly provide a schema-to-Qwen
raw-string bridge. Establish schema-preserving required/additional-property,
nested, union, and string-constraint behavior before admitting the combined
language. Specify how reserved parameter delimiters inside values are represented;
do not silently weaken a schema or change the checkpoint's serialization.
The model output protocol must distinguish the tool alternative from final JSON:
literal reasoning/tool markers inside final JSON remain content. Qualify thinking
options, forced/named selections, and subsequent tool-result turns.

### Native and preparation architecture

| Item | Status and evidence | Work required to close |
|---|---|---|
| 6.1 Per-model Rust contract | Implemented: `ModelContract` statically dispatches Qwen admission, packing, prompt/output interpretation, and completion checks. Required preparation metadata declares the schema, vocabulary, and vision capabilities. All 26 packed-upload fixtures match; invalid metadata is rejected before asset loading or CUDA. Rust/Python tests and six installed HTTP parsing/stop checks pass. All image fixtures are admitted, with the structured-output limitation described above. | Preserve generic transport ABI/fence validation and model-owned geometry during checkpoint changes. |
| 6.2 Typed errors | Implemented in [error.rs](../native/src/error.rs): admission, execution, and frontend variants preserve source chains and numeric CUDA codes. CPU cause/status tests, installed startup rejection, all 22 [fatal-completion HTTP/SSE checks](validation/native-worker-failure.md), and full-model JSON/SSE capacity rejection pass. | Preserve variant-based status dispatch, first failure, terminal error events, and owner retention after failed drain. |
| 6.3 Worker phases | Implemented in [runtime.rs](../native/src/server/runtime.rs) as private admit/plan/submit/complete/publish/retire methods. Exact instruction/prompt publication remains atomic in scheduler completion; stopped requests retire before another completion. Five CPU phase tests, installed fatal-completion checks, and [full-model HTTP checks](validation/exact-checkpoint-http.md) pass. | Preserve observation-before-presentation, stop-before-next-submit, reserve-before-pack ordering, and fence-protected release. Raw graph replay passes its unchanged executor path. |
| 6.4 Tuning fixtures | Implemented in [preparation/fixtures](../sparkgap/preparation/fixtures/), with six operator families. CPU comparison preserves complete ordinary/copy/MTP constructor contracts, graph signatures, scratch requirements, and 250 benchmark tensors across 38 producer/reset observations. All 501 CPU tests pass; 103 CUDA tests are skipped. Actual fixture-owner release and failed-finalizer cleanup are covered. | Preserve checkpoint operands, plan keys, scratch envelopes, serialized tuning identity, and cleanup. GPU capture/numerical tests were not rerun for this source move. |
| 6.5 Qualification package | Partial: HTTP and graph-execution implementations live in `qualify.server` and `qualify.graph_execution`; script entrypoints and dependent callers preserve their CLI and request contracts. CPU regression suites pass. | Move remaining diagnosis/trace/qualification scripts and the test-tree qualifier. Share existing HTTP/SSE and fixture helpers where duplication remains; preserve request bytes, assertions, and result semantics. |
| 6.6 Unsafe safety proofs | Closed: unsafe calls outside cuda.rs state allocation, context/thread, and drain/lifetime obligations locally; ownership/failure tests pass. | Preserve the proofs when changing native ownership. |
| 6.6 Resolved upload bindings | Closed: validate_uploads returns reusable resolved descriptors consumed directly by staging and enqueue. CPU range/alias/reuse tests and four real-CUDA transport tests pass. | Preserve exact ranges and pending-frame ownership during model-contract extraction. |
| 6.6 Grammar debug configuration | Closed: the checkpoint fixture takes an explicit optional trace path; no trace environment switch remains. The exact-checkpoint test compares tracing disabled/enabled and passes. | Keep debug output test-only and disabled unless explicitly selected. |
| 6.6 Qualification replay binding | Closed: raw replay bindings require the qualification feature. Production/qualification export tests and installed production export checks pass; GraphExecutor/Server remain available. | Build component replay tests with the qualification feature. |
| 6.6 Dataclass conventions | Closed: the recipe package docstring specifies scratch reuse, fixed storage geometry, metadata mutation, and persistent-state ownership. | Keep model-specific state behavior beside its implementation. |

Four request lanes and 8/24 completion columns must be classified explicitly as
transport/catalog constraints or Qwen semantics during extraction. Moving a
constant without moving its validation is insufficient. The first implementation
of the contract remains Qwen4Exp; no second-model backend is added for this work.

### Per-model output protocol

The native model contract includes a model-owned output protocol alongside its
graph-input contract. Output parsing uses independent `ReasoningParser` and
`ToolCallParser` interfaces. The model supplies concrete associated types and a
composer that determines routing, ordering, and interleaving. Shared syntax may
reuse an implementation; the server never assumes every model uses Qwen markers
or a universal reasoning-then-tools pipeline.

Select the protocol with the supported model/template contract at startup.
The Qwen implementation initializes parser state once per request from resolved
template options, assistant continuation, reasoning settings, admitted tools,
tool choice, and output grammar. Generic request preparation delegates that
interpretation to the model contract. Literal markers in user or tool messages
do not determine the assistant parser's initial state.

| Contract | Responsibility |
|---|---|
| Reasoning parser | Interpret the model's reasoning framing, initial state, delimiters/control tokens, and transitions into or out of answer output. |
| Tool-call parser | Interpret model-specific tool envelopes and parameter serialization; produce stable call identities, names, and correctly typed JSON arguments. |
| Model output composer | Route fragments through those parsers according to the model protocol and selected grammar alternative; preserve literal markers in final structured content. |
| Shared response accumulator/renderer | Consume neutral content, reasoning, and tool-call events; construct OpenAI JSON/SSE fields, call IDs, usage, and finish reasons. |

The incremental contracts accept lossless token-aware pieces: raw token IDs and
stable decoded text spans, including control tokens with no printable text.
They retain bounded state across arbitrary token/UTF-8/marker boundaries.
Termination supplies an explicit cause: EOS, output budget, matched user stop,
cancellation, or execution error.

Neutral events include content/reasoning deltas and tool-call begin, JSON-argument
delta, and end events. A model may buffer a complete validated call, as Qwen does,
or stream append-only argument fragments when its serialization permits it.
The renderer owns API representation; model parsers do not construct `serde_json`
response objects. Accumulated arguments must parse to the admitted object with
correct types; the API's `function.arguments` field remains a JSON-encoded string.
Published deltas cannot be retracted, re-escaped, or silently discarded.

JSON and SSE responses use the same incremental decoding, parsing, and accumulation
path. Their transport encodings differ, but their completed semantic result must
agree. A truncated emitted tool call cannot become a successful complete call;
the explicit user-stop policy may discard only a never-emitted buffered call.

Worker-side stop filtering supplies ordered token/text pieces to this protocol.
Raw token observation and usage remain independent. If a stop occurs inside one
token, presentation parsers receive only the visible text span; the hidden suffix
and its control markers are suppressed.

[Qwen parsers and their composer](../native/src/model/qwen4exp/output.rs) implement
[the neutral interfaces](../native/src/output.rs). The
[shared renderer](../native/src/server/output.rs) handles JSON/SSE presentation.
CPU tests cover UTF-8/marker partitions, continuation state, literal markers,
reasoning and tools, parameter types, and partial-call termination.
[Installed HTTP checks](validation/output-protocol-http.md) cover literal markers,
string stops, a tool call, and the unchanged stopped code-edit request. Combined
tool/final-content grammars and the complete agentic workload remain open.

This extraction changes native presentation, not CUDA graph execution. Parsers
remain Rust-owned and do not call Python. PyTorch replay and state-comparison
harnesses remain under qualification tooling; production Python ends model
execution responsibility at prepared-resource handoff.

### Corresponding document edits

Correction section 7 is an acceptance checklist for the specifications:

- MTP scheduling: one round in flight, 14 legal shapes, copy-first dependencies,
  cloned grammar states, and the common stochastic verifier.
- MTP memory: distinguish retained capacity, per-round recurrent writes, and
  measured write/commit time; keep external research linked from one mechanism paragraph.
- Core runtime: deterministic admitted operators and the per-model native packing contract.
- Core correctness: identical-operand components, the frozen generation/reference
  floor, exact same-graph/same-capacity state, and calibrated cross-capacity numerics.
- Core performance: named per-kernel traces, one-policy A/B, kernel-count work,
  and the weight-traffic figure labeled as a bandwidth bound.
- Core purpose/preparation/acceptance sections: present-state specification with
  one qualification-status link. Sampling retains the bounded-top-k deferral.

## Serving findings that affect copy qualification

**String stops.** Rust performs incremental UTF-8 matching with bounded prefix
holdback before further model submission. JSON and SSE suppress the matched string
and same-token suffix while retaining raw sampled IDs and usage. Installed HTTP
checks pass, including the unchanged `code-docstrings-stop` request: all eight
function bodies remain AST-equivalent, `END_EDIT` is hidden, and generation ends
with `stop` after 541 tokens within its 1,024-token budget. The complete workload
rerun is open. Requests combining string stops with structured grammar or custom
token stops remain rejected.
Explicit user stops discard only buffered, never-emitted incomplete tool calls;
ordinary malformed-tool termination remains an error.

**Exact checkpoint retention.** The implementation retains rendered instruction,
final prompt, and terminal consumed endpoints independently, coalescing equal
endpoints. The Qwen model contract derives the instruction boundary from the
actual template and exact token-prefix verification. Tools and reasoning defaults
can synthesize instructions. Prefill stops at each required endpoint without
16-token or 64-token rounding; the scheduler drains work before freezing live state.

Admission reserves each required destination from the preallocated snapshot pool.
After the copy fence, publication pins the shared ancestor and transfers completed
page ownership without changing physical page order. A duplicate publication
releases its unused reservation while retaining the request's computed pages.
Reuse accounting records only the original admission hit. Cancellation, errors,
and cache eviction remain fence-protected.

Snapshots contain normalized accepted GDN state and convolution windows, PLE
history, hidden/residual/position state, target raw selector state, and any partial
target attention page. Aligned target endpoints leave shared full pages untouched.
Learned MTP retains `max(S−1,0)` pairs and its own raw anchor, plus up to two shifted
private boundary pages. Rejected raw tags are excluded. Pool row 0 is null, rows 1–4
are live, and immutable snapshots begin at row 5; no rolling candidate rows or
aligned checkpoint exports remain.

Native scheduling and checkpoint frontend tests pass; 59 CUDA transfer,
maintenance, metadata, and completion checks pass, including allocation stability
and immutable source protection. Full-model native replay of a fixed 4,043-token
prompt preserves exact transfer and restored/direct-split continuation at
endpoints [4,032](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/prefix4032.json)
and [10](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/prefix10.json).
The aligned split also matches monolithic execution across both tail capacities.
The 10-token split does not: complete-logit relative L2 is 0.167491, with the split
result as denominator; argmax remains unchanged.

The [layer-0 trace](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/gdn-prefix10/analysis.json) reproduces baseline
state and logits exactly. Projection inputs, QKV/Z/A/B projections, convolution
outputs, and carried convolution/recurrent state are exact. The first differing
stage is recurrence, beginning at the first resumed token; output relative L2 is
0.00185507 against the monolithic output and maximum absolute error is 0.001953125.
Splitting inside a 16-token BF16 recurrence chunk changes its arithmetic partition.
Calibrated operator acceptance for arbitrary partitions remains open.

Actual-model selected-column snapshot and continuation pass all 14 supported
batch/width shapes at endpoints 65–72. Restores use different physical slots and
private tail pages; immutable snapshots and replay allocations remain unchanged.
[Installed HTTP checks](validation/exact-checkpoint-http.md) pass short instruction
reuse, changed user/tool identity, exact prompt retries, image conditioning,
snapshot/conversation/KV pressure, and cancellation. The combined run retains one
fixture-construction error; the corrected system/tool case passes separately.
HTTP coverage of every residue 0/1/2/3/63, duplicate publication, failed freeze,
and explicit reuse of consumed generated terminal prefixes remains open.

The malformed diff, literal-tag omission, contradictory retry recommendation,
and four explicit-budget truncations remain failed outputs. Ordinary and pinned
reference runs establish whether these reflect engine changes or model behavior;
the plan does not assign their cause from the copy-adaptive run alone.

## Validation and completion criteria

Each implementation group has targeted CPU checks, followed by the relevant real
path. The final source freeze receives the complete acceptance run:

1. CUDA-hidden constructor/declaration, scheduler, frontend, grammar, image, error,
   lifecycle, and upload-contract tests; Rust format/check/clippy and server-feature tests.
2. Installed constructor rejection before CUDA/network activity; prepared model
   lifetime, shutdown, poisoned-worker, and failed-drain handling.
3. Actual checkpoint startup with configurable prefill through at least 4,096,
   caller-owned scratch, b12x tuning/warmup, final binding refresh, graph capture,
   memory headroom, and zero-allocation replay after handoff.
4. Same-graph Python/Rust and same-capacity restore byte equality; calibrated
   cross-capacity checks; copy and learned-MTP accepted-state/cache transactions.
5. Frozen 82-prompt agreement and reference repeatability, complete identical-input
   operator coverage, and retained diagnostic eager comparisons.
6. Actual Rust HTTP/SSE text, image, tools/schema, stop, cancellation, cache-pressure,
   and agentic cases on the exact checkpoint. Preserve every failed case and raw output.
7. Fixed-width and paired stochastic measurements at C1/2/4; named per-kernel B1/B4
   timelines; useful speculative replay floor; ≤5% native overhead; source/depth
   acceptance, round costs, TTFT, median/p99 token latency, memory, and startup.
8. Clean pinned setup, small tracked documentation, verified external artifacts,
   logical commits, and restored serving on port 8000.

One-paragraph summaries and compact machine-readable receipts enter the validation
index. Raw traces, tensors, HTTP bytes, source/build fingerprints, and failed
attempts remain in the external artifact store. An item closes only when its
specific acceptance evidence exists; related passing counts are not substitutes.
