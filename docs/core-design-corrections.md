# Corrections to the core design and its implementation

Open items from this document are carried forward in
[the second corrections set](core-design-corrections-2.md), which also covers
the MTP design; closed items are listed in its section 0.

This document lists required changes to the Flare serving stack and to
[the core design](core-design.md), based on the repository state on
2026-10-09 with the untracked working tree. Each item states the current
state, the defect or risk, the required change, and the condition under which
the item is closed. Items are ordered by priority within each section.

## 1. Numerical acceptance oracle

### 1.1 Compare serving operators on identical inputs

Serving arithmetic dynamically quantizes activations and may retain BF16
recurrence intermediates. A Transformers eager model using dequantized weights
and FP32 intermediates computes different arithmetic. Whole-model logit error
alone cannot distinguish expected precision differences from recipe defects.

Test each prepared operator against its Transformers or b12x reference on
identical operands. Include the checkpoint's normalization weights, activation
policy, cache dtype, and recurrence precision in the comparison. Retain
teacher-forced full-vocabulary logits to locate divergence, then isolate the
first differing component. Relative L2 uses the reference norm as denominator;
logit diagnostics also include top-10 overlap and FP32 softmax KL divergence.

### 1.2 Exercise generation behavior

Use at least 50 prompts spanning chat, tool calls, and long contexts, with fixed
sampling settings and a 64-token output budget. Check request completion and
task behavior. Token agreement is meaningful only with matched sampling and
contexts; interpret differences against measured repeatability on identical
inputs. Store reusable test inputs and expected outputs as ordinary fixtures.

### 1.3 Keep eager comparisons diagnostic

Derive component tolerances from their precision and reduction arithmetic.
For GDN, compare FP32 and BF16 intermediates separately instead of imposing a
whole-model threshold that conflates the two. Preserve the operator's intended
arithmetic; require a localized defect before changing recurrence rounding or
normalization order to improve aggregate logit agreement.

## 2. Determinism constraints

### 2.1 Replace the cross-capacity bitwise predicate with a tolerance

**Current state.** The prefix gate requires a restored prefix plus suffix to
match a cold run bitwise, where the cold run computes the suffix inside a
large-capacity prefill graph and the restored run computes it in a small
capacity tail graph. Under free autotuning, 83 of 595 predicates failed
(prefix-replay-autotuned.md). The
recipe now pins `recurrence_policy="sequential_k1"` for every GDN prefill
capacity (`serve/recipes/qwen38/gdn.py:547`) and `attention_splits=16` for
every QSA capacity (`serve/recipes/qwen38/attention.py`), and the design doc
sections "Sparse attention arithmetic" and "GDN prefill arithmetic" specify
these as part of the recipe.

**Defect.** Bitwise equality across different graph capacities is not a
requirement in the design and no serving stack provides it. Different tilings
round differently. Enforcing it selects the slowest legal kernel
configurations for the largest prefill chunks, and the cost has not been
measured.

**Required change.**

- Keep bitwise equality only where it proves the runtime: Python-versus-Rust
  replay of the same graph with the same inputs, and same-capacity replay
  after restore.
- Make the cross-capacity cold-versus-restored predicate a tolerance on state
  and logits. Set the tolerance from the measured same-input variation
  between capacities under free autotuning.
- Remove the fixed `recurrence_policy` and `attention_splits` from the serving
  recipe's prepared queries so autotuning selects them. Retain both as an
  explicit debug configuration for diagnosing capacity-dependent divergence.
- Rewrite the two design-doc sections to describe the tolerance-based
  acceptance and the debug-only fixed configurations.

**Acceptance.** The prefix gate passes under free autotuning with the
tolerance predicate, and item 3.1 records the prefill throughput difference
between the fixed and autotuned configurations.

## 3. Performance measurement

### 3.1 Produce the minimum measurement set

**Current state.** Section 5 of the design specifies a three-way comparison
and a sustained-decode target within 5 percent of the replay floor. No
measurement exists. The only recorded timing is the sampler component:
120 to 170 µs per step on the general path and 18 µs greedy, which is 3 to 4
percent of the 3.9 ms weight-traffic floor. Cold-cache tuning took 148
seconds.

**Required change.** Record, with the exact-checkpoint server at the default
4,096-token configuration and the 512-token configuration:

| Measurement | Conditions |
|---|---|
| Time to first token | 512-token and 4,096-token prompts, cold cache |
| Inter-token latency, median and p99 | Concurrency 1, 2, 4; 256 generated tokens |
| Decode throughput, tokens per second | Concurrency 1, 2, 4 |
| Prefill throughput, tokens per second | One 4,096-token prompt; one 64K-token prompt |
| Prefix-hit time to first token | 4,096-token cached prefix plus 11-token tail |
| Startup time | Cold selection cache; warm selection and kernel caches |
| Host submission gap per decode step | From the native executor's event timeline |

Report each against the 3.9 ms per-token floor from the design's weight-traffic
table and against the serving baseline on the same GPU with matched settings.
Keep raw measurements and the configuration needed to interpret them.

**Acceptance.** The design's 5 percent sustained-decode overhead target is met,
or profiling attributes the gap to a named component.

## 4. Repository hygiene

### 4.1 Commit the working tree

**Current state.** The repository has one commit. Roughly 34,000 lines of
Python and Rust, the patches, and the documentation are untracked.

**Required change.** Commit in logical groups after applying item 4.2:
packaging and environment (`pyproject.toml`, `uv.lock`, `.gitignore`,
`.python-version`, `scripts/setup_preparation.py`, `patches/`); preparation,
recipes, reference, and their tests; `helios/`; qualification scripts; and
documentation. Imperfect state is acceptable; an untracked tree is not.

### 4.2 Move validation archives out of the tree

**Current state.** `docs/validation` holds 448 MB: tarballs up to 78 MB,
gzipped JSON up to 15 MB, and JSON receipts up to 6 MB. `git ls-files
--others` reports 452 MB that would enter history on the first commit.

**Required change.** Keep Markdown receipts and JSON under 1 MB in the tree.
Move `*.tar.gz`, `*.json.gz`, and larger JSON to an artifact store outside the
repository, for example `/data/sparkgap-artifacts/<receipt-name>/`, and record
the store path in `docs/validation/README.md`. The receipts already carry
SHA-256 hashes, so integrity does not depend on the files being in git. Add
the patterns to `.gitignore`.

### 4.3 Develop FlashInfer changes in the local checkout as an editable install

**Current state.** `scripts/setup_preparation.py` clones the pinned FlashInfer
revision into `.build-sources/flashinfer`, applies
`patches/flashinfer-serving-recurrence-policy.patch`, and builds there;
`pyproject.toml` points `flashinfer-python` at that clone. `patches/` holds
five cumulative patch files with fingerprint manifests, of which only the last
is applied. One hunk changes `include/flashinfer/air_top_p.cuh`, which is
FlashInfer core rather than the b12x experimental tree.

**Defect.** The clone-and-patch flow turns kernel work into a second,
non-version-controlled copy of a repository that already exists at
`~/projects/flashinfer`. Changes land as patch files instead of commits, the
manifests reimplement what git provides, and the installed package diverges
from the checkout where b12x is actually developed.

**Required change.** All FlashInfer and b12x work happens in the local
checkout at `~/projects/flashinfer`, on a branch, as ordinary commits.
Flare consumes it as an editable install:

- In `pyproject.toml`, point `[tool.uv.sources] flashinfer-python` at
  `../flashinfer` with `editable = true`, so kernel edits take effect without
  reinstalling.
- In `[tool.flare.source-revisions]`, pin the branch's commit SHA.
  Preparation and qualification verify that the checkout's HEAD matches the
  pin and that the worktree is clean, and record both in the receipt, rather
  than cloning and patching.
- Remove the `.build-sources/flashinfer` clone, the patch application step,
  the `patches/` directory, and the manifest machinery.
- Submit the `air_top_p` parallelization upstream to FlashInfer separately,
  since it is outside the experimental tree you own; until it merges, it
  lives on the same local branch.

**Acceptance.** `scripts/setup_preparation.py` performs no clone and applies
no patch; a receipt records the FlashInfer checkout path, HEAD, and clean
status; `patches/` no longer exists.

### 4.4 Restore the design document to a specification

**Current state.** `docs/core-design.md` grew from 754 to 1,051 lines, mostly
pass counts and receipt narrative. `README.md` has the same pattern.

**Required change.** Remove from the design doc:

- In section 1, the two paragraphs beginning "The recorded 37-graph prefix and
  Python/Rust qualifications" through "Independent numerical and
  serving-performance acceptance remain open."
- In section 2, the receipt sentences in "Sparse attention arithmetic" and
  "GDN prefill arithmetic", and the final paragraph of "Preparation lifecycle"
  beginning "The main-source HTTP qualification".
- In the viability assessment, the sentence "Required additions are a prepared
  GDN causal convolution factored from PLE state machinery, an independent
  preparation entry for the existing KV writer", which describes work that the
  patch already contains; state the present operator set instead.
- Any other sentence whose content is a check count or a receipt link.

Replace them with one "Qualification status" subsection in section 5 that
points to `docs/validation/README.md`. Reduce the README status paragraph to a
table with one row per acceptance area: text numerics, vision numerics,
runtime parity, HTTP suites, performance, each with a state and a link.

### 4.5 Add a CPU bootstrap smoke test

**Current state.** `tests/recipes/test_qwen38_bootstrap.py` runs only under
CUDA. A constructor-signature mismatch between `decoder.py` and `ple.py`
passes the CPU suite and fails at server start.

**Required change.** Add a test that drives the real `prepare_model` call path
on the meta device with a preparation session that records declarations and
skips priming, so that every `Prepared*` constructor is invoked with its
production arguments without a GPU. Assert the declaration catalog against a
stored expectation.

## 5. Feature gaps for the local target

### 5.1 JPEG and EXIF orientation

**Current state.** Native image admission decodes PNG only and rejects EXIF.
JPEG is the dominant format for locally captured images.

**Required change.** Add a JPEG decoder in Rust. Bit-exact agreement with
Pillow is not achievable across decoders because IDCT and chroma upsampling
differ, so define the JPEG acceptance as decoded RGB within a per-channel
tolerance of 2 against the Pillow fixture, with the resize, normalization,
and patch pipeline still bitwise against fixtures given the decoded pixels.
Implement the eight EXIF orientations as a pre-resize transform.

### 5.2 Tool calls combined with structured output

**Current state.** Request preparation rejects nonempty tools with a structured
output grammar, `tool_choice=required`, and named tool selection
(`helios/src/server/request.rs`). Four of the 69 workload scenarios fail on
this.

**Required change.** Build the constraint as a union grammar in llguidance:
the checkpoint's tool-call envelope with each admitted tool's argument schema
as a JSON sub-grammar, or the final-content schema. `tool_choice=required`
restricts the union to the envelope alternatives; a named tool restricts it to
that tool's schema. `tool_choice=none` keeps the current behavior.

### 5.3 Bounded top-k

**Current state.** Open. The general FlashInfer path costs 120 to 170 µs per
step.

**Required change.** None for acceptance. Record the item as deferred in the
design's sampling section, with the measured general-path cost as the reason.

## 6. Architecture and code hygiene

### 6.1 Delete the superseded Python serving stack

**Current state.** `serve/engine`, `serve/model`, `serve/cache`,
`serve/kernels`, `serve/api`, `serve/tp`, `serve/cli.py`, `serve/logging.py`,
and `tests/serve` total about 18,000 lines. No module under
`serve/preparation`, `serve/recipes`, `serve/reference`, `scripts/`, or
`helios/` imports any of them.

**Required change.** Delete them. Where the design doc cites the old
scheduler, graph, and prefix-cache modules as policy and test references,
cite the commit `9d70e7f25310df4957a953f6000b089c7a4c60c2` that contains them.

**Acceptance.** `serve/` contains only `preparation`, `recipes`, `reference`,
and `__main__.py`; `tests/serve` is gone except for the packaging test, which
moves to `tests/preparation`.

### 6.2 Make the Rust model layer an explicit per-model contract

**Current state.** `helios/src/model.rs` declares `VOCABULARY = 248_320`,
requires `vision_features_per_request == 16384`, and `helios/src/model/inputs.rs`
packs about 38 named buffers with `admission_`, `state_`, and `vision_`
semantics specific to Qwen4Exp. The design states that Rust sees only page
IDs, state-slot IDs, capacities, and graph entries, and that the bundle
contract is model-independent.

**Defect.** The runtime contains a Qwen4Exp input-packing layer. That is a
defensible placement, since input packing must happen off the Python path,
but the design's model-independence claim and the plan to qualify the
contract with a second prepared model are not true as written.

**Required change.** Introduce a model-contract trait in `helios/src/model.rs`
covering admission packing, prefill and decode lane packing, state-operation
packing, vision packing, and completion interpretation. Move the current
implementation to a `qwen4exp` module implementing it. Vocabulary size,
vision feature capacity, and buffer schema come from the manifest and model
configuration rather than constants. Update the design doc's runtime section
to state that the native core includes a per-model input-packing contract
declared by preparation.

**Acceptance.** No model-specific constant or buffer name appears outside the
`qwen4exp` module; the server is generic over the trait.

### 6.3 Replace string errors with typed errors

**Current state.** `helios/src/lib.rs` defines
`pub type Result<T> = std::result::Result<T, String>`. The HTTP layer and the
runtime tests classify failures by substring, for example
`.contains("matching admission")` in `helios/src/server/runtime.rs`.

**Defect.** The design requires admission to fail with a concrete capacity or
feature error. A string cannot be matched by kind, so status-code mapping and
tests depend on message text.

**Required change.** Add `thiserror` and define enums for admission
(capacity, feature, grammar, image), execution (poisoned, drain, manifest),
and frontend (template, tokenizer, image decode) errors. Map HTTP status by
variant. Keep the string form for display only.

**Acceptance.** No `.contains(` or `starts_with(` on error text in
production or test code.

### 6.4 Split the server loop

**Current state.** `helios/src/server/runtime.rs` is 1,030 lines, and
`run_loop` is one 307-line function covering admission, planning,
submission, completion handling, checkpoint publication, and metrics. Its
tests assert against rendered Prometheus text.

**Required change.** Make the loop a struct with one method per phase:
admit, plan, submit, complete, publish, retire. Test each phase against
scheduler and executor state rather than metrics output.

### 6.5 Move tuning fixtures out of the recipe modules

**Current state.** `serve/recipes/qwen38/gdn.py`, `moe.py`, `vision.py`, and
`hyperconnection.py` each contain private benchmark-input classes and arena
helpers (`_GDNBenchmarkInputs`, `_BenchmarkStateArena`,
`register_gdn_benchmark_state`, `_ExpertBenchmarkInputs`,
`_AttentionBenchmarkInputs`, `_HyperconnectionBenchmarkInputs`). They are
about a quarter of `gdn.py`.

**Required change.** Move them to a fixtures module under
`serve/preparation`, keyed by operator, so recipe modules contain only the
forward composition and its state.

### 6.6 Consolidate qualification scripts into a package

**Current state.** `scripts/` holds 18 qualification, diagnosis, and trace
scripts; `qualify_server.py` is 1,060 lines and `qualify_graph_execution.py`
is 906. Five scripts implement their own HTTP client. Receipt writing, file
hashing, and provenance capture are reimplemented per script.
`tests/recipes/qualify_hyperconnection.py` is a qualifier living in the test
tree.

**Required change.** Create a `qualify` package with a shared HTTP and SSE
client, a receipt writer that records source hashes, environment, GPU
identity, and artifact digests, and a fixture loader. Reduce each script to
an entrypoint over that package. Move the test-tree qualifier into it.

### 6.7 Rename the Python package

**Current state.** The package is `serve`, the project is `flare`, and the
package no longer serves requests.

**Required change.** Rename `serve` to `flare` before the first commit of
the new code, updating the entrypoints in `pyproject.toml`.

### 6.8 Remove meta-commentary from validation receipts

**Current state.** Receipts under `docs/validation` consist largely of
statements about what was not done or not established, for example
"inspection read source without launching" and "does not establish."

**Required change.** Restate each receipt as conditions, measurement, result,
and conclusion. State a limitation once as a fact of the measurement rather
than as a disclaimer on each sentence.

### 6.9 Smaller items

- Add a `SAFETY` comment to each `unsafe` block outside `helios/src/cuda.rs`:
  two each in `server.rs` and `server/runtime.rs`, two in
  `execution/worker.rs`, four in `execution/transport.rs`, three in
  `python.rs`.
- In `execution/transport.rs`, return resolved bindings from
  `validate_uploads` instead of repeating `.find(...).unwrap()` in `submit`.
- Replace the `FLARE_GRAMMAR_TRACE` environment switch in `grammar.rs` with
  an explicit debug configuration field.
- Feature-gate the `ReplayWorker` Python binding as a qualification primitive,
  since production uses `GraphExecutor` through the server.
- Change the README test command to
  `cargo test --manifest-path helios/Cargo.toml --features server --offline`;
  without the feature it runs 58 of 91 tests.
- Record the scratch, metadata, and state dataclass convention used across the
  recipe modules once, in the `serve/recipes/qwen38/__init__.py` docstring.

## 7. Corresponding edits to the core design

| Section | Edit |
|---|---|
| 5, Correctness acceptance | Use identical-input component tests with measured precision tolerances, generation tests, and eager whole-model diagnostics as specified in items 1.1 to 1.3. |
| 5, Correctness acceptance | Add the tolerance-based cross-capacity predicate and the bitwise same-capacity predicate from item 2.1. |
| 2, Sparse attention arithmetic; 2, GDN prefill arithmetic | Rewrite per item 2.1. |
| 5, Performance acceptance | Add the measurement table from item 3.1 as the required minimum set. |
| 1, 2, viability assessment | Remove receipt narrative per item 4.4. |
| 3, Sampling and RNG | Record bounded top-k as deferred per item 5.3. |
| 3, Public interface | Document JPEG support and the tool-plus-schema grammar policy once items 5.1 and 5.2 land. |
| 3, Prepared graph bundle; 3, Native replay loop | State that the native core includes a per-model input-packing contract declared by preparation, per item 6.2, and replace the sentence that Rust sees only page IDs, slot IDs, capacities, and graph entries. |
| 5, Implementation sequence | Replace references to the old scheduler, graph, and prefix-cache modules with the commit SHA, per item 6.1. |
