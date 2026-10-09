# Core correction closeout plan

This plan covers every item in [the second correction set](core-design-corrections-2.md),
including the requirements it carries forward from [the first set](core-design-corrections.md).
The audit uses the working trees on 2026-10-09. The second document's SHA-256 is
`5cb9d05d874683521f4e9fde80758b2280fa73c4a5fc09aff745135d93e498f2`.
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

The [copy-state replay run](/data/sparkgap-artifacts/mtp-implementation/speculative-state-parity/receipt.json)
completed its 30 Python commands and native initialization, then stopped at
`upload.1.identical` before the second native graph. The intended identical-input
contract was not satisfied. Diagnose that upload discrepancy and retain this
failed receipt before rerunning; Python/Rust state parity remains open.

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

The native model-contract extraction, typed errors, worker-loop decomposition,
JPEG/EXIF support, tool-plus-schema grammar, and the upstream AIR submission remain
open. Preparation, recipes, the native runtime, and qualification tooling have
separate source commits. The completed items in correction section 0 remain
closed, subject to their original measurement scopes.

## Execution order

1. **Preserve evidence and establish source history.** Resolve the recorded
   copy-state upload discrepancy and complete qualification. Restore port 8000
   after exclusive GPU runs. Archive large validation
   payloads with verified hashes, repair their links, and commit Sparkgap in
   reviewable groups. Inventory FlashInfer changes before committing the serving
   changes and updating its pin. This work does not wait for every feature to pass.
2. **Establish numerical and construction gates.** Replace the eager-based limits,
   define calibrated cross-capacity checks, add the real-constructor CPU bootstrap
   test, and run the frozen generation/reference-variance corpus. Keep identical
   graph execution and same-capacity restore comparisons bitwise. Capture the
   unchanged ordinary common-sampler B1/B4 kernel baseline alongside this work;
   that measurement does not depend on learned MTP qualification.
3. **Extract the native model boundary.** Introduce the statically dispatched model
   contract, typed errors, and worker phases, in that order. Move existing
   speculative packing, output parsing, and interpretation into the Qwen implementation before
   adding further MTP behavior. Preserve upload bytes and transaction ordering.
4. **Complete local serving contracts and copy qualification.** Finish string stops,
   JPEG/EXIF, and tool-plus-schema grammars. Resolve the prompt-checkpoint retention
   requirement described below. Qualify copied drafts against independent W1
   execution, including actual prefix restores, every accepted count, and failures.
5. **Qualify learned MTP.** Use the qualified transaction with the checkpoint's MTP
   weights, feedback, attention/cache, experts, and head. Complete full startup,
   prefill absorption, accepted/rejected rounds, prefix restore, and HTTP workloads.
6. **Measure and change kernel policies.** Use the ordinary kernel baseline and
   capture speculative timelines, fixed-width costs, state-write costs, and the
   speculative replay floor. Remove
   production tuning pins one at a time under the numerical gates. Measure fixed
   depths before calibrating adaptive width; compare the complete workload arms.
7. **Complete extraction and documentation.** Move tuning fixtures and qualification
   logic without changing their contracts, remove receipt narrative from the
   specifications, and run the final installed-entrypoint acceptance on pinned
   sources. Commit each coherent change as it is validated.

CPU work on documentation, image fixtures, error types, and constructor coverage
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
| 1.2 Copy before learned MTP | Implemented; copy-only preparation excludes MTP weights/cache. Full state parity and agentic acceptance remain open. | Complete same-graph Python/Rust state qualification and independent W1 comparisons, then rerun the unchanged agentic corpus. Separate engine defects, unsupported API behavior, and model-output failures using the ordinary/reference arms. Load learned MTP only after transaction qualification. |
| 1.3 Widths follow kernel bands | All B1 widths 1–8, B2 widths 1–4, and B4 widths 1–2 exist. | Record repeated fixed-shape round-cost distributions for all 14 shapes. Measure the grouped-path transition with diagnostic B2×4/5 and B4×2/3 comparisons. Shapes beyond eight total rows stay outside the serving catalog until measured and qualified. |
| 1.4 State-write cost | The design records 108 MiB recurrent state per verified row. Measured write cost remains open. | Report logical bytes written per graph and measured GDN/state-selection time. Separate compulsory checkpoint-column writes, commit/copy traffic, and total round duration. Do not present a bandwidth-derived lower bound as measured time. |
| 1.5 Stochastic primary path | Implemented with FlashInfer; no greedy route remains. One copy-adaptive C1 workload repetition exists. | Measure ordinary, copy-only, fixed learned-MTP, adaptive MTP without copying, and adaptive MTP with copying on identical stochastic workloads at C1/2/4. Report acceptance by depth/source and completed-work latency/throughput. Report temperature-zero timings separately through the same FlashInfer path; the 82-prompt correctness run is not a performance measurement. |
| 1.6 Grammar speculation | Cloned llguidance state and candidate/bonus masks are implemented; accepted grammar-constrained copied tokens are observed. | Run an independent W1 arm with the same constrained literal and require byte-identical output with at least one accepted speculative draft. Add mixed requests, rejection, EOS, empty masks, cancellation, and grammar-state commit checks. |
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
| 2.2 Determinism | The property is stated under correctness acceptance; the validation index records pinned-vLLM variance. Updated speculative-catalog evidence remains open. | Add the property to the runtime-contract section as requested and qualify repeated identical inputs across the admitted catalog. Preserve the reference-variance evidence. Keep determinism distinct from differences between capacities, plans, or sampling algorithms. |
| 3.1 Measured position | Historical baseline closed at its recorded revisions and sampling configuration. | Preserve its raw results. Establish the matching baseline for changed stochastic code before using it in any new performance comparison. Do not inherit historical timing claims after kernel or sampling changes. |
| 3.2 Per-kernel GPU timeline | Open. Nsight software tracing contains graph GPU activity but no kernel-activity breakdown. | Obtain graph-node GPU activity with CUPTI/Nsight, or an event-instrumented diagnostic capture of the actual operator sequence. For B1 and B4, report kernel names/counts, layer/operator attribution, summed kernel time, graph span, and gaps. Verify outputs and quantify instrumentation overhead against uninstrumented replay. |
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
| 4.4 Specification cleanup | Receipt narratives removed from core design; graph counts, sampling, determinism, acceptance, and README status reflect the implementation. | Complete the native model-packing contract and review the first correction set's section-7 mappings as the remaining architectural changes land. |
| 4.5 CPU bootstrap | Open. Meta model-tree and ownership tests do not invoke all production Prepared constructors. | Drive actual `prepare_model` with a recording session and metadata-only storage, replacing IO/compilation boundaries while retaining constructors. Assert graph/operator declarations, capacities, dtypes, and scratch/state contracts for ordinary, copy, and MTP preparation; assert CUDA remains uninitialized. Do not add a serving fallback. |
| 4.6 Receipt prose | Closed: validation index and acceptance coverage state behavior, limitations, and remaining gates; historical measurements stay in their individual reports. | Keep acceptance summaries aligned with implementation and actual test results. |

Debugging payloads remain outside Git while useful. Source commits and dependency
pins identify reproducible code; uncommitted imported-code changes invalidate a
clean-checkout claim.

### Local features

| Item | Status and evidence | Work required to close |
|---|---|---|
| 5.1 JPEG and EXIF | Open. [native/Cargo.toml](../native/Cargo.toml) enables PNG only; [image admission](../native/src/frontend/images/mod.rs) rejects EXIF. | Add native JPEG decoding with resource limits and all eight EXIF orientation transforms before resize. Compare decoded RGB to fixed Pillow fixtures at per-channel absolute tolerance 2. Given identical decoded pixels, require exact resize/normalization/patch outputs. Exercise the actual HTTP image path, malformed images, and orientation-dependent dimensions. |
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
| 6.1 Per-model Rust contract | Open. [model.rs](../native/src/model.rs) and [inputs.rs](../native/src/model/inputs.rs) contain Qwen vocabulary, vision, position, and speculative-buffer assumptions. | Define a statically dispatched model contract for admission, prefill/decode/speculative packing, state/vision packing, and completion interpretation. Move Qwen details into `qwen4exp`; declare vocabulary/capacities/schema through preparation metadata. Keep transport ABI/fence validation generic. Compare exact packed bytes before/after and reject bad schemas before CUDA. No backend registry or dynamic per-token dispatch. |
| 6.2 Typed errors | Open: [lib.rs](../native/src/lib.rs) aliases Result to String. Admission-rejection events already differ structurally; the correction's substring count is stale. | Introduce typed admission, execution, and frontend errors, preserving underlying causes and numeric driver codes. Map HTTP/SSE behavior by variants and use variant assertions. Preserve first failure and owner retention after failed drain. Eliminate error-text classification, without confusing ordinary token/metrics string checks with error dispatch. |
| 6.3 Worker phases | Open. [runtime.rs](../native/src/server/runtime.rs) still combines admission, planning, completion, grammar, stops, copy proposals, and retirement in run_loop. | After contract/error extraction, introduce a worker struct with admit/plan/submit/complete/publish/retire methods. Preserve candidate-record-before-complete, observation-before-presentation, stop-before-next-submit, and fence-protected release. Test phases against scheduler/executor state, then qualify the actual HTTP path and graph replay. |
| 6.4 Tuning fixtures | Open. GDN, MoE, vision, and hyperconnection benchmark-input classes remain inside recipes. | Move declaration-time benchmark fixtures/arena helpers into operator-keyed preparation modules. Preserve real checkpoint inputs, plan keys, scratch envelope, serialized tuning identity, and cleanup. Verify the same declaration/catalog and no escaping trial storage. |
| 6.5 Qualification package | Partial: `qualify/` exists, but large scripts retain their implementations and duplicate clients. | Move `qualify_server.py` and `qualify_graph_execution.py` logic first, then other diagnosis/trace/qualification scripts and the test-tree qualifier. Share HTTP/SSE, fixture loading, receipt/provenance, and artifact helpers. Keep scripts as thin entrypoints; preserve CLI behavior, request bytes, assertions, and receipt semantics. |
| 6.6 Unsafe safety proofs | Open. | Add local SAFETY comments describing allocation bounds, thread/context ownership, and drain/lifetime requirements for every unsafe block outside cuda.rs. Validate the reasoning against ownership and failure tests. |
| 6.6 Resolved upload bindings | Open. | Return resolved bindings from validate_uploads and consume them directly in submit, using bounded reusable storage rather than adding an allocation per decode. Test missing/duplicate/overlapping/out-of-range uploads and reuse of persistent staging frames; preserve exact staged bytes. |
| 6.6 Grammar debug configuration | Open; SPARKGAP_GRAMMAR_TRACE is confined to a test fixture, not the production server. | Replace the environment switch with an explicit qualification/test debug input, disabled by default. Verify tracing leaves masks and generated output unchanged. No live-server configuration is needed for a test-only trace. |
| 6.6 Qualification replay binding | Open. | Feature-gate the ReplayWorker Python binding as a qualification primitive. Verify the production extension excludes it and the qualification build includes it; keep production GraphExecutor/Server ownership unchanged. |
| 6.6 Dataclass conventions | Open. | Document scratch lifetime/reuse, metadata mutability/shape, and persistent-state ownership once in the recipe package docstring. Keep model-specific state behavior beside its implementation. |

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
Initialize parser state once per request from the resolved template options,
actual assistant continuation prefix, reasoning setting, admitted tools, tool
choice, and output grammar. Remove Qwen marker scans and whole-prompt `rfind`
reasoning inference from generic request preparation. Literal markers in user
or tool messages do not determine the assistant parser's initial state.

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

Stop filtering must preserve this contract. Raw token observation and usage remain
independent. If a stop occurs inside one token, presentation parsers receive only
the visible text span, not the hidden suffix or its control markers. The isolated
string-stop patch's text-only event path must be integrated with this token/span
contract before application.

Implement Qwen reasoning and tool parsers first by extracting
[output.rs](../native/src/server/output.rs), preserving its admitted behavior.
Add interface-level tests for chunk invariance, initial/continued assistant state,
literal markers, tools with reasoning, multiple calls, schema-directed argument
types, partial calls under each termination cause, and JSON/SSE equivalence.
Use distinct synthetic syntax in contract tests to detect hardcoded Qwen assumptions
in generic composition; this does not add a second serving model.

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

**String stops.** The unchanged agentic request `code-docstrings-stop` receives
HTTP 400. A Rust-only patch is prepared separately for worker-side incremental
UTF-8 matching and bounded prefix holdback. Integrate it with the model/error/worker
boundaries above. Match before another graph submission; preserve full sampled IDs
and usage; suppress the stop string and same-token suffix in both JSON and SSE.
Explicit user stops may discard only a buffered, never-emitted incomplete tool
call. Ordinary malformed-tool termination remains an error. Qualify cross-token
and subtoken matches, overlapping prefixes, EOS/budget races, and resource release.

**Prompt checkpoint retention.** The cache publishes only the latest rolling
boundary at request termination. A 6,459-token prompt followed by 144 emitted
tokens has 6,602 consumed tokens and publishes boundary 6,592. An identical
6,459-token retry cannot restore that checkpoint; the common 6,400 boundary has
already been overwritten. The observed misses occur with no evictions and free
snapshot capacity. This is a retention-policy limitation.

Retain the final-prefill boundary alongside the terminal checkpoint: freeze it
before decode, publish/pin it, transfer complete-page ownership into the live
request's shared parent chain, and rotate to another preallocated snapshot.
Keep physical page order unchanged. Separate original admission reuse accounting
from the evolving page-ownership boundary so newly computed tokens are not
reported as reused. Qualify duplicate publication, pressure, cancellation, failed
freeze, exact retries, divergent tails, and appended conversations. Do not replace
the terminal checkpoint, which preserves generated-state reuse for agentic turns.

Reserve the additional snapshot before entering the prompt freeze; a live request
must not wait for the only terminal destination it already occupies. Treat prompt
retention as auxiliary under snapshot pressure: preserve mandatory terminal
capacity and record a retention skip if no auxiliary slot is available. An existing
duplicate checkpoint does not justify replacing live computed KV pages with another
request's pages. Leave that request's physical chain unchanged and release only
unused reservations after their fences. For learned MTP, separately qualify the
P−1 draft length, open raw-selector history, and shifted private boundary page.

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
   cross-capacity checks; copy and learned-MTP state/candidate/cache transactions.
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
