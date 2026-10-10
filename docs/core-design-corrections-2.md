# Corrections, second set: MTP design and remaining items

This document lists required changes to the Flare serving stack,
[the core design](core-design.md), and [the MTP design](mtp-design.md), based
on the repository state on 2026-10-09 after the serving-path oracle, the
performance measurements, the FlashInfer editable migration, the package
rename, and the removal of the superseded serving stack. It supersedes the open
items of [the first corrections document](core-design-corrections.md); closed
items from that document are listed in section 0 and not repeated. Each item
states the current state, the defect or risk, the required change, and the
condition under which the item is closed.

## 0. Items closed since the first set

| Item | Closed by |
|---|---|
| 1.1 Serving-path oracle | text-serving-oracle.md, text-layer-comparison.md; the result amends the acceptance criteria in section 2 below |
| 3.1 Performance measurement set | serving-performance.md; the result amends section 3 below |
| 4.3 FlashInfer as editable install | `pyproject.toml` points at `../flashinfer` with `editable = true`; `patches/` and `.build-sources` removed; the clone and patches are archived under `/data/sparkgap-artifacts/flashinfer-source-migration`. One step remains, item 4.3 below |
| 6.1 Superseded serving stack | `serve/engine`, `serve/model`, `serve/cache`, `serve/kernels`, `serve/api`, `serve/tp`, `serve/cli.py`, `serve/logging.py`, `tests/serve` removed |
| 6.7 Package rename | The Python package is `flare` |
| 6.9 README test command | `cargo test --features server --offline` |

## 1. MTP design

The MTP design's core is correct and is not changed by this section: a
speculative round as a state transaction; GDN verification through the decode
kernel's checkpoint columns; convolution and PLE histories retained per
candidate until acceptance; QSA logical lengths and tags excluding rejected
rows; commit clipping with pending-token semantics; the MTP cache built during
prefill-plus-absorb and checkpointed with the prefix including its open group;
one-row chain-sampler invocations with explicit RNG offset accounting; and
fixed depths measured before adaptive policy.

### 1.1 Run speculative rounds one in flight

**Current state.** The design retains at most two speculative rounds in flight
for unconstrained requests, with width chosen from older statistics and
device results from one round feeding the next.

**Defect.** Two in flight exists to hide host latency behind a 12.8 ms
single-token step. A speculative round is 15 to 25 ms and the host gap is on
the order of 100 µs, under 1 percent and inside the 5 percent target. The
second in-flight round costs stale width selection, doubled private-state
reservation, terminal-flag no-op rounds, and a device-side n-gram lookup for
copy proposals, and it prevents per-candidate grammar masks and host-evaluated
custom stops in speculative rounds.

**Required change.** Speculative rounds run one in flight, using the grammar
barrier mechanism that already exists. Width selection, copy proposals,
grammar masks, and custom stops use the just-completed round's result. Two in
flight remains a measured option, not the default.

**Acceptance.** Native overhead for the speculative graph sequence is within
5 percent of its measured replay floor with one round in flight.

### 1.2 Build copy proposals before MTP drafting

**Current state.** The dependency list loads MTP weights at step 2 and adds
copy proposals at step 4. `helios/src/scheduler/speculation.rs` already
contains `copy_proposal`.

**Required change.** Swap the order. Implement the chain transaction, the
verification-row logits, state selection, bounded vector completions, and the
ABI extension with host-supplied n-gram drafts first. This exercises every
part of the round except the MTP model and pays off immediately on tool-result
and code-edit workloads. MTP drafting lands afterward as a second proposal
source into a qualified transaction.

**Acceptance.** The chain transaction passes its correctness cases with copy
drafts on the agent workload before any MTP weight is loaded.

### 1.3 Catalog widths follow the kernel bands

**Current state.** The design proposes widths `W ∈ {1, 2, 4, 8}` at request
capacities `B ∈ {1, 2, 4}`, twelve round graphs per sampling mode.

**Evidence.** b12x's MoE direct-micro path owns a tiny-decode band of 1 to 8
tokens per launch (`_MICRO_MAX_TOKENS = 8` in `moe/fused_moe/_impl.py`),
streams each routed expert once for all rows in the launch, and selects tuned
configurations per exact shape, so odd widths cost no more than even ones. GDN
decode allows at most eight state columns. The measured batch-1 step is 12.8 ms,
about 550 GB/s effective on a 1.79 TB/s part, so the step is bound by per-kernel
latency rather than bytes, and additional rows inside the micro band are close
to free in wall time:

| Batch 1 width | Expert-union bytes per round | Bytes the existing 12.8 ms can move |
|---:|---:|---:|
| 1 | 1.3 GB | 23 GB |
| 3 | 3.4 GB | 23 GB |
| 5 | 5.7 GB | 23 GB |
| 7 | 7.8 GB | 23 GB |

The band limit is total rows per launch. Batch 4 at width 8 is 32 tokens and
320 routed rows, which leaves the micro path for the dynamic grouped GEMM. That
planner's speculative-verify tile is keyed to a 256-expert, n=1024 model, so
Qwen's 512 experts at n=640 fall through to default tile logic that has not
been measured for this model.

**Required change.** At batch 1, capture every width from 1 through 8 and let
the controller select among them. At batch 2 and 4, keep total rows at or
below 8 by default, widths up to 4 and up to 2 respectively. Any shape beyond
the micro band enters the catalog only after its round cost is measured on
the grouped path.

**Acceptance.** Per-width round cost is recorded for every captured shape, and
the band cliff is located by measurement.

### 1.4 Price state writes, not only state capacity

**Current state.** The memory section counts 108 MiB per saved verification
position and derives capacity totals.

**Required change.** Add the write cost: the GDN decode kernel writes one full
column per verified row, so a round writes `rows × 108 MiB` of recurrent state,
about 0.5 ms at batch 1 width 8 and about 2 ms at batch 4 width 8 at this
GPU's bandwidth. Record it per round once rounds exist. This is the stronger
motivation for compact recurrence than capacity is.

### 1.5 The stochastic chain is the primary path

**Current state.** The checkpoint's default sampling is temperature 1.0,
top-k 20, top-p 0.95, so most traffic is stochastic. The design treats greedy
and stochastic acceptance symmetrically and notes that the fixed-K chain
sampler forces stochastic requests onto the captured width.

**Required change.** Treat the stochastic chain as the primary path. Measure
acceptance and round cost at the checkpoint defaults, not at greedy; report
greedy separately. The adaptive controller's width choice for stochastic
requests is a choice among captured graphs, which is why item 1.3 captures
every batch-1 width.

### 1.6 Grammar-constrained speculation

**Current state.** Grammar-constrained requests use the one-token variant.
Multi-token grammar speculation is deferred pending cloned tentative grammar
states and per-candidate masks.

**Required change.** With one round in flight (item 1.1), the host advances a
cloned llguidance state along the draft chain before submission and uploads
one mask per candidate row; verification applies mask `j` to row `j`, and only
accepted tokens advance the committed state. Implement this with the copy
proposal path of item 1.2, since tool-call JSON is where copy drafts and
grammars coincide.

**Acceptance.** A grammar-constrained request completes a speculative round
with at least one accepted draft and byte-identical output to its
one-token execution.

### 1.7 Move the research comparison out of the specification

**Current state.** The MTP design contains a fourteen-row comparison of
external systems and papers.

**Required change.** Move the table and its discussion to a research document
under `/data/sparkgap-artifacts/mtp-research/`, where the source fingerprints
already live. The design keeps one paragraph naming the mechanisms it adopts:
bounded adaptive width from measured round cost and delayed acceptance, copy
proposals under the same verification budget, and compact recurrence as a
later state-representation extension.

## 2. Numerical acceptance, amended

### 2.1 Replace logit relative L2 as an acceptance metric

BF16 reduction order and normalization precision can introduce small component
perturbations that grow across 48 layers and four residual streams. Whole-model
logit relative L2 does not isolate a recipe defect.

Acceptance requires component comparisons on identical operands, with tolerances
derived from FP32-versus-BF16 arithmetic and the selected kernel's repeatability.
Generation tests cover chat, tool calls, and long contexts with fixed sampling
settings. Teacher-forced logits remain useful for locating the first differing
layer; investigate its operators before changing model arithmetic.

The eager comparison remains diagnostic. Replace arbitrary whole-model logit
and encoder relative-L2 gates with component-specific bounds. Comparison
utilities accept explicit saved reference tensors or token sequences.

### 2.2 State determinism as a design property

**Required change.** State the reduction semantics in the runtime contract.
Plans with deterministic reductions must produce identical logits for identical
inputs. Plans using atomic reductions require measured numerical tolerances on
identical inputs; state copies and metadata retain exact equality requirements.

## 3. Performance, amended

### 3.1 Measurement conditions

Measure complete GPU steps, native scheduling, and HTTP delivery separately.
Use the same checkpoint, GPU assignment, activation precision, graph capacity,
prompt lengths, sampling settings, generated-token counts, and cache conditions
for comparisons. Include vocabulary projection, collectives, and sampling in
GPU step timing.

Measure GDN recurrence splits, QSA partitions, MoE reduction policy, and
hyperconnection projection fusion independently. A small native replay overhead
does not establish that the captured model is efficient; attribute remaining
GPU time to operators and dependencies with a kernel timeline.

### 3.2 Produce a per-kernel decode timeline

**Current state.** The hardware-tracing export in the performance receipt
contains host graph-launch calls but no GPU kernel activity, so the 12.8 ms
batch-1 step has no breakdown. At 550 GB/s effective the step is not
bandwidth-bound.

**Required change.** Capture a per-kernel GPU timeline of one batch-1 decode
graph and one batch-4 decode graph, with CUPTI graph-node tracing or an
event-instrumented capture. Report per-operator and per-layer time, kernel
count per layer, and the sum of kernel durations against the step time.

**Acceptance.** A receipt attributes the batch-1 step to named operators, and
states the kernel count per decode graph.

### 3.3 Lift the determinism pins against the timeline

**Current state.** `flare/recipes/qwen38/gdn.py:547` pins
`recurrence_policy="sequential_k1"`, `flare/recipes/qwen38/attention.py:184`
pins `attention_splits=16`, and `flare/preparation/tuning.py` records both
as fixed policy. The cross-capacity prefix gate still requires bitwise
equality.

**Required change.** Unchanged from the first set, now with measured
motivation: make the cross-capacity cold-versus-restored predicate a
tolerance, keep bitwise equality for same-capacity Python-versus-Rust replay,
remove both pins from the prepared queries, and A/B each policy from item
3.1's table against the timeline from item 3.2, one at a time. The
hyperconnection FP32 weights are the more faithful choice; measure their cost
before changing them. Keep every pinned configuration as a debug switch.

**Acceptance.** The prefix gate passes under free autotuning with the
tolerance predicate, and each policy change carries a before-and-after
timeline.

### 3.4 Kernel count is the lever

**Required change.** Once item 3.2 exists, record the kernel count per decode
layer and treat fusion of adjacent small operators, hyperconnection
normalization and projection in particular, as the performance workstream.
The 3.9 ms weight-traffic floor in the design is unreachable for either
implementation at the current kernel count and should be labeled as a
bandwidth bound rather than a target.

## 4. Repository hygiene, remaining

### 4.1 Commit the working tree

**Current state.** One commit; 115 untracked or modified entries.

**Required change.** Unchanged. Apply item 4.2 first, then commit in logical
groups.

### 4.2 Move validation archives out of the tree

**Current state.** New artifacts go to `/data/sparkgap-artifacts`. The 46
existing archives and 449 MB remain in `docs/validation`, and `.gitignore` has
no pattern for them.

**Required change.** Move the existing `*.tar.gz`, `*.json.gz`, and JSON over
1 MB to the artifact store, record the paths in the validation index, and add
the patterns to `.gitignore`.

### 4.3 Commit the FlashInfer changes and pin them

**Current state.** The editable install is in place and setup rejects a dirty
checkout, but `~/projects/flashinfer` is dirty at
`98fb87a461183f98d6a7049ca947fb136cc6fb59` and the pin still names that
upstream revision.

**Required change.** Commit the checkout's changes on a branch, update the pin
in `[tool.flare.source-revisions]`, and submit the `air_top_p` change
upstream separately.

**Acceptance.** Setup passes on a clean checkout at the pinned SHA.

### 4.4 Restore the design document to a specification

**Current state.** `docs/core-design.md` is 1,082 lines and still contains the
receipt narrative listed in the first set. None of the section-7 edits from
the first set have been applied.

**Required change.** Apply the first set's section 4.4 and section 7, plus:
the input-packing contract from item 6.2 below, the determinism property from
item 2.2, the amended acceptance from item 2.1, and the measurement
requirements from items 3.2 and 3.3. Add one "Qualification status"
subsection in section 5 that points to the validation index.

### 4.5 CPU bootstrap smoke test

**Current state.** `tests/preparation/test_model_tree.py` constructs the
canonical module tree on the meta device. It does not drive `prepare_model`
through the `Prepared*` constructors, so a constructor-signature mismatch
between recipe modules still passes the CPU suite.

**Required change.** Unchanged from the first set.

### 4.6 Receipt prose

**Current state.** New receipts such as the serving oracle follow conditions,
measurement, result, conclusion. `acceptance-gaps.md` and the validation index
remain check-count narrative.

**Required change.** Unchanged from the first set, applied to the remaining
receipts.

## 5. Feature gaps, remaining

### 5.1 JPEG and EXIF orientation

Unchanged from the first set. No JPEG decoder is present in `helios/Cargo.toml`.

### 5.2 Tool calls combined with structured output

**Current state.** `helios/src/server/request.rs` still rejects nonempty tools
with a structured-output grammar and rejects forced tool selection.

**Required change.** Unchanged from the first set: a union grammar of the
tool-call envelope with per-tool argument schemas and the final-content
schema. Item 1.6 makes this the same mechanism that speculative grammar rounds
use, so implement it once.

### 5.3 Bounded top-k

Deferred; record as such in the design's sampling section with the measured
general-path cost of 120 to 170 µs per step.

## 6. Architecture, remaining

### 6.1 Make the Rust model layer an explicit per-model contract

**Current state.** `helios/src/model.rs` still declares `VOCABULARY` and
Qwen4Exp buffer semantics. Speculation adds more model-specific packing to
the same module, which makes the trait boundary more urgent, not less.

**Required change.** Unchanged from the first set's item 6.2. Define the trait
before the speculative inputs are added so they land on the Qwen4Exp
implementation rather than in the runtime.

### 6.2 Typed errors

**Current state.** `Result<T, String>`; eight substring matches on error text
in `runtime.rs`.

**Required change.** Unchanged from the first set's item 6.3.

### 6.3 Split the server loop

**Current state.** `helios/src/server/runtime.rs` is 1,202 lines and
`run_loop` is 343 lines, both larger than at the first set.

**Required change.** Unchanged from the first set's item 6.4. Do it before the
speculative round handling is added to the loop.

### 6.4 Move tuning fixtures out of the recipe modules

**Current state.** Benchmark-input classes remain in `gdn.py`, `moe.py`,
`vision.py`, and `hyperconnection.py`.

**Required change.** Unchanged from the first set's item 6.5.

### 6.5 Migrate scripts onto the qualify package

**Current state.** `qualify/` exists with 17 modules including shared
provenance. `scripts/` has grown to 21 files and none were migrated.

**Required change.** Move each script's logic into `qualify` and reduce the
script to an entrypoint. Start with `qualify_server.py` and
`qualify_graph_execution.py`, the two largest.

### 6.6 Smaller items

- Add `SAFETY` comments to the `unsafe` blocks outside `helios/src/cuda.rs`.
- Return resolved bindings from `validate_uploads` in `execution/transport.rs`.
- Replace the `FLARE_GRAMMAR_TRACE` environment switch with a debug
  configuration field.
- Feature-gate the `ReplayWorker` Python binding as a qualification primitive.
- Record the scratch, metadata, and state dataclass convention once in
  `flare/recipes/qwen38/__init__.py`.

## 7. Corresponding document edits

| Document and section | Edit |
|---|---|
| mtp-design, Graph catalog, control and scheduling | One round in flight for speculative rounds (1.1); catalog widths per item 1.3; grammar rounds per item 1.6 |
| mtp-design, Memory and compact recurrence | Add state-write cost per item 1.4 |
| mtp-design, Sampling correctness | Stochastic chain as primary path per item 1.5 |
| mtp-design, Research comparison | Remove per item 1.7; replace with one paragraph of adopted mechanisms |
| mtp-design, Implementation dependencies | Reorder per item 1.2 |
| core-design, Runtime contract | Add the determinism property (2.2) and the per-model input-packing contract (6.1) |
| core-design, Correctness acceptance | Replace the oracle hierarchy with item 2.1; tolerance-based cross-capacity predicate per item 3.3 |
| core-design, Performance acceptance | Add items 3.2 and 3.4; relabel the 3.9 ms figure as a bandwidth bound |
| core-design, sections 1, 2, 5 | Remove receipt narrative; add the qualification-status subsection (4.4) |
| core-design, Sampling and RNG | Record bounded top-k as deferred (5.3) |
