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
| Copy transaction | [Native serving](speculative-copy-serving.md) exercises all supported shapes and accepted copied drafts; [state replay](copy-state-replay.md) passes identical Python/native commands across all 14 shapes. | Independent one-token comparisons, every accepted-prefix length, cancellation, pressure, and image-conditioned speculation. |
| Agentic output | [The 24-context workload](speculative-agentic-serving.md) has 16 manual passes; eight cases fail API or output requirements. | Resolve engine/API defects and compare unchanged requests against ordinary and reference arms to separate model-output limitations. |
| Prefix restore | [Aligned prefix replay](prefix-replay-prefill4096.md) checks cold versus restored execution with fixed reduction policies. | Retain the final prompt checkpoint as well as the terminal checkpoint; qualify copied and learned-draft state across branches. Calibrate cross-capacity floating-state comparisons before freeing tuning policies. |
| Grammar speculation | Host-side cloned grammar state produces per-candidate masks. | Require an accepted copied draft and byte-identical constrained output against independent one-token execution; cover rejection and grammar-state commit. |
| Learned MTP | [Identical-input operator checks](mtp-operators.md) isolate projection differences and verify normalization on common operands. | Full startup, prefill absorption, drafting, acceptance/rejection, draft-cache restore, and actual HTTP workloads. |
| Image and context capacity | [Aggregate images](server-http-vision-aggregate.md) and [context-limit execution](server-http-context-limit.json) exercise their recorded capacities and resource accounting. | Capacity execution does not establish image arithmetic or long-context model accuracy. |
| Failure lifecycle | [Startup rejection](native-startup-validation-abi2.md) and [worker failure](native-worker-failure.md) cover their documented validation and drain cases. | Preserve status mapping, first failure, and ownership retention during native refactoring. |

A request may consume at most `context_limit - 1` tokens because its final
sampled token remains pending. Prefix snapshots represent complete model state
at aligned consumed-token boundaries, including recurrence and history; a KV
page match alone is insufficient for restoration.

## API and architecture gaps

- Images accept PNG/JPEG with EXIF orientation. Decoder and checkpoint-admission
  fixtures pass; JPEG/EXIF live HTTP coverage remains open.
- String stops remain unimplemented in the installed server.
- Nonempty tools with a final-content schema, required/named tool selection,
  and disabling parallel tool calls remain unsupported.
- Reasoning and tool-call parsing have separate model-owned interfaces and
  a common incremental JSON/SSE presentation path. Model-specific input packing
  still requires an explicit native contract.
- Native typed errors and worker-loop decomposition remain open. Upload ranges
  are resolved once into reusable descriptors; CPU and real-CUDA transport checks pass.
- CPU bootstrap tests invoke real preparation constructors, public plans, and
  scratch declarations. They stop before tuning, compilation, and capture.

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
qualification and individual before/after measurements. Production deterministic
reductions remain a contract; a nondeterministic diagnostic is a distinct arm.

Bounded top-k is deferred. The [general sampler measurement](sampling-flashinfer.md)
records approximately 120–170 microseconds per step over its measured conditions.
A compact candidate path must preserve threshold ties, filtering, and RNG
semantics before any performance comparison is valid.
