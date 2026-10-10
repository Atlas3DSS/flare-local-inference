# Serving acceptance coverage

The target is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, on NVIDIA SM120.
SM121 requires its own hardware qualification. The
[core design](../core-design.md) defines the runtime contract;
[the correction plan](../core-corrections-closeout.md) tracks implementation work.

## Execution contract

Python loads, plans, tunes, warms, and captures. Rust owns serving after handoff.
Ordinary execution uses ABI 2; speculation uses ABI 3 vector completions.
At a 4,096-token prefill limit, the catalogs contain 28 ordinary or 39
speculative graphs. Sampling always uses the common FlashInfer implementation;
temperature zero is a point-mass distribution through that path.

Ordinary unconstrained decoding permits two submissions in flight. Grammar
constraints and all speculative rounds impose a completion barrier. Copy
speculation supports every batch-1 width 1–8, batch-2 width 1–4, and batch-4
width 1–2. Learned MTP preparation exists but has not passed full serving
qualification.

## Correctness and functional coverage

| Area | Established behavior | Remaining acceptance |
|---|---|---|
| Text arithmetic | [Layer tracing](text-layer-comparison.md) locates the first cross-stack difference at hyperconnection normalization and the reference's own variation at MoE. | Run the fixed 82-prompt continuation corpus against two reference runs; extend identical-input component checks. Whole-logit L2 is diagnostic. |
| Vision arithmetic | [Serving comparison](vision-serving-reference.md) records the encoder discrepancies. | Qualify component arithmetic and full-capacity image outputs. |
| Ordinary replay | [Graph execution](graph-execution-parity-prefill4096.md) and [shared scratch](unified-scratch.md) qualify their recorded catalogs, ownership, and allocation stability. | Repeat the relevant gates after operator or catalog changes. Historical dual-sampler results do not qualify the common-sampler speculative catalog. |
| Copy transaction | [Native serving](speculative-copy-serving.md) exercises all supported shapes and accepted copied drafts. [State replay](copy-state-replay.md) passes identical Python/native commands and independent W1/Wn all-accepted literal comparisons across all 14 shapes, including selected-column snapshots at endpoints 65–72. | Complete ordinary-decoder comparisons, every accepted-prefix length, cancellation, pressure, and image-conditioned speculation. |
| Agentic output | [The 24-context workload](speculative-agentic-serving.md) has 16 manual passes; eight cases fail API or output requirements. | Resolve engine/API defects and compare unchanged requests against ordinary and reference arms to separate model-output limitations. |
| Prefix restore | [Exact checkpoints](../core-corrections-closeout.md#serving-findings-that-affect-copy-qualification) retain instruction, prompt, and terminal endpoints. GPU transfer checks and full-model restored/direct-split comparisons pass at endpoints 10 and 4,032. [HTTP checks](exact-checkpoint-http.md) cover short instruction/tool prefixes, prompt retries, images, pressure, and cancellation. | Complete HTTP residue/failure/terminal-reuse coverage and learned-draft state across branches. Calibrate partition-sensitive floating-state/operator comparisons; splitting BF16 prefill at 10 differs from monolithic arithmetic. |
| Grammar speculation | Host-side cloned grammar state produces per-candidate masks; independent W1/Wn singleton-language comparisons accept copied drafts and preserve exact output/state. | Cover rejection, mixed requests, empty masks, EOS/cancellation, and grammar-state commits beyond singleton-language cases. |
| Learned MTP | [Identical-input operator checks](mtp-operators.md) isolate projection differences and verify normalization on common operands. | Full startup, prefill absorption, drafting, acceptance/rejection, draft-cache restore, and actual HTTP workloads. |
| Image and context capacity | [Aggregate images](server-http-vision-aggregate.md) and [context-limit execution](server-http-context-limit.json) exercise their recorded capacities and resource accounting. | Capacity execution does not establish image arithmetic or long-context model accuracy. |
| Failure lifecycle | [Typed native errors](../../native/src/error.rs) preserve causes, numeric CUDA codes, and variant-based status mapping. CPU tests, installed startup rejection, all 22 [fatal-completion HTTP/SSE checks](native-worker-failure.md), and full-model capacity rejection pass with private worker phases. | Preserve first failure and ownership retention. Multiple waiting-request HTTP fanout and failed CUDA drain remain outside the installed failure test's scope. |

A request may consume at most `context_limit - 1` tokens because its final
sampled token remains pending. Prefix snapshots represent complete model state
at exact rendered-instruction, final-prompt, and terminal consumed endpoints.
They include normalized recurrence/history, raw selector state, and privately
restored partial pages; complete pages remain immutable and shared. A KV page
match alone is insufficient. Restoring the same execution partition is exact;
BF16 recurrence does not guarantee equality between arbitrary prefill partitions.

## API and architecture gaps

- Images accept PNG/JPEG with EXIF orientation. All 26 decoder/checkpoint fixtures
  pass [installed HTTP admission](jpeg-exif-http.md); four invalid image requests
  are rejected. Two constrained outputs exhaust their unchanged 64-token budgets
  in legal JSON whitespace, leaving output completion and vision accuracy open.
- Native string-stop matching preserves raw token usage while suppressing the stop
  and same-token suffix in JSON/SSE. [Installed checks](output-protocol-http.md)
  include the unchanged stopped code-edit request. The complete agentic workload
  rerun remains open; string stops combined with structured grammar or custom token
  stops remain unsupported.
- Nonempty tools with a final-content schema, required/named tool selection,
  and disabling parallel tool calls remain unsupported.
- The static native `ModelContract` validates model metadata and graph geometry,
  packs Qwen inputs, and interprets completions. Its output protocol supplies
  separate reasoning/tool-call parsers and common incremental JSON/SSE presentation.
- Native errors distinguish admission, execution, and frontend failures without
  error-text dispatch. Installed fatal-completion HTTP/SSE checks pass. Private
  worker phases preserve transactional ordering and pass full-model checkpoint,
  pressure, and cancellation checks. Upload ranges are resolved once into reusable
  descriptors; CPU and real-CUDA transport checks pass.
- CPU bootstrap tests invoke real preparation constructors, public plans, and
  scratch declarations. They stop before tuning, compilation, and capture.
  Operator benchmark fixtures live under `preparation/fixtures`. CPU coverage
  includes complete ordinary/copy/learned-MTP constructor contracts, scratch
  requirements, producer bytes/layouts, and trial-resource release.

Explicit request budgets remain authoritative. Increasing a fixture's budget
or removing constraints does not resolve its failed output requirement.

## Performance coverage

[Serving measurements](serving-performance.md) characterize ordinary decoding
with the recorded sampler and operator policies. Native scheduling overhead is
small in that measurement; the captured kernels account for most step time.
Those runs do not measure the speculative common-sampler replay floor.

Required measurements are named per-kernel B1/B4 timelines, per-layer kernel
counts, fixed-width speculative costs, recurrent-state write cost, and matched
stochastic workload arms at concurrency 1/2/4. Policy changes require numerical
qualification and individual before/after measurements. Production MoE uses
b12x's default numerical policy, which permits nondeterministic atomic reductions.

Bounded top-k is deferred. The [general sampler measurement](sampling-flashinfer.md)
records approximately 120–170 microseconds per step over its measured conditions.
A compact candidate path must preserve threshold ties, filtering, and RNG
semantics before any performance comparison is valid.
