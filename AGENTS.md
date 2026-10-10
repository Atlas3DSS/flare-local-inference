# Flare design principles

Flare is a local inference server for NVIDIA SM120/SM121 GPUs. Python prepares
models with PyTorch, Transformers, and FlashInfer, including b12x. Helios, the
Rust core in `helios/`, serves requests through prepared CUDA graphs.

## Preserve canonical model implementations where they fit

- Treat Transformers as the canonical source for model structure and semantics.
  Preserve its forwards where their tensor layouts, state handling, and emitted
  GPU work fit the serving contract. Avoid independently reproducing model
  logic that can be reused directly.
- Inspect the installed model's actual call sites before choosing an extension
  point. Prefer attention, mask, expert, kernel, and loading interfaces when
  their contracts fit. Interface availability does not imply that every model
  uses it or that it controls the entire operation we need to optimize.
- Replace the smallest coherent module that gives control over the required
  dataflow. A whole attention module may need replacement to avoid expanding
  latent KV or constructing dense masks before the attention dispatcher. A
  whole MoE module may need replacement to overlap shared and routed experts.
- Use instance-local module replacement or supported construction hooks when
  sufficient. Avoid global monkey patches and Transformers forks without a
  concrete need. Do not force b12x through a Transformers integration that adds
  complexity without improving reuse or execution.
- Judge reuse by the GPU work it records and the semantics it preserves.
  Python dispatch during preparation and capture is not per-token overhead
  during native graph replay. A canonical forward is not inherently too slow,
  and a rewritten forward is not inherently faster.

## Flexible recipes, reusable components, shared native machinery

- Recipes in `flare/recipes/` own model-specific composition, checkpoint mapping,
  state layouts, and graph preparation. They may retain canonical forwards or
  compose prepared components as the model requires.
- Put reusable execution mechanisms in `flare/components/`. Recipes should be
  able to use components independently without adopting a rigid model framework.
  Shared/routed expert overlap, projections, collectives, and state transfers
  are examples of mechanisms that should not be copied between recipes.
- Helios uses explicit model contracts for model-specific input packing, state
  interpretation, and output parsing. Keep transport, scheduling, cache
  machinery, and request handling shared rather than duplicating them per model.

## Python prepares; Helios serves

- Python loads weights, resolves and autotunes kernels, plans storage, warms
  execution paths, and captures supported graph variants before serving starts.
  It then enters the blocking native server call.
- Helios owns sockets, request processing, tokenization, chat templates,
  scheduling, cache operations, graph replay, and output streaming. Serving
  must not call back into Python.
- CUDA graphs are the model execution boundary. Do not introduce an eager
  serving fallback, runtime compilation, graph recapture, or GPU allocation in
  steady-state execution. Reject requests outside prepared capacities explicitly.
- Keep graph executables, captured buffers, resource owners, and the CUDA
  context alive for their full native use. Model state and metadata must be
  explicit storage that Helios can manage without Python execution.

## Use FlashInfer and plan memory explicitly

- Flare is fully coupled to FlashInfer/b12x by design. It has no backend or
  device abstractions. Use FlashInfer/b12x and NVIDIA CUDA interfaces directly;
  do not introduce interchangeable kernel backends or device portability layers.
- Prefer proven b12x operators and FlashInfer sampling over duplicate kernels.
  Inspect their supported preparation, tuning, and workspace interfaces before
  introducing custom implementations.
- Use b12x's autotuning and warmup facilities during preparation. Do not pin
  kernel plans or tuning choices without a demonstrated requirement.
- Supply caller-owned scratch through the kernel planning interfaces. Reuse
  storage across sequential work; preserve separate storage for overlapping
  operations and persistent state. Aliasing must follow actual lifetimes and
  stream dependencies, including overlap captured inside a single graph.
- Treat KV, recurrent state, and other model histories as explicit cache state.
  Prefix reuse, branching, and speculative rollback must preserve all relevant
  state, not just token positions and KV pages.

## Optimize the serving path

- Apply the recipe requirements below during implementation. Inspect the
  corresponding model in the user's vLLM tree as a performance reference for
  fusion boundaries, precision, sharding, overlap, and kernel plans. Reuse
  FlashInfer/b12x mechanisms directly; vLLM is not a runtime dependency.
- Measure GPU graph replay separately from native scheduling and end-to-end
  request latency. Compare equivalent shapes and execution configurations when
  using another server as a performance reference.
- Attribute gaps to emitted kernels, memory traffic, dependencies, or host work
  before changing architecture. Preserve correctness when changing layouts,
  fusion boundaries, scratch reuse, or stream overlap.
- Keep ordinary tests and raw benchmark measurements. Do not create development
  audit reports, receipts, provenance bundles, or hash inventories.

## Recipe performance requirements

- **Fuse projections sharing an input.** Pack compatible weights once during
  loading, including their quantization scales and TP slices. Candidates include
  Q/K/V, MLP gate/up, MLA query/KV low-rank inputs, and recurrent attention's
  Q/K/V, beta, and forget-gate input projections. Preserve each output's model
  semantics and calibrated scales. Inspect the full dependency chain rather
  than reproducing one kernel call per canonical module.
- **Respect layouts after fusion.** A column slice of a fused output can be
  contiguous for batch one and strided for multiple rows. Check downstream b12x
  input, output, stride, and alignment contracts for every graph capacity.
  Allocate reusable staging storage where needed; do not introduce runtime
  allocation through an implicit `.contiguous()`.
- **Overlap independent branches.** Shared experts should be considered for
  overlap with routing and routed experts; recurrent output-gate projections
  can overlap independent input projections. Use
  [ConcurrentSharedExpert](flare/components/expert_overlap.py) and
  [ConcurrentProjectionChain](flare/components/projection_overlap.py) where their
  contracts fit. Measure the relevant row capacities: branches competing for
  bandwidth or SM capacity can make overlap slower.
- **Make overlap safe through compilation.** Declare every concurrently live
  plan with [ScratchArena.register_parallel](flare/preparation/scratch.py).
  Sequential work reuses storage; concurrent work gets disjoint storage sized
  from the tuned plans. Refresh bindings after arena resizing. The fork must
  follow input production, and the join must precede output consumption or
  input/scratch reuse. Custom ops must expose reads, writes, aliases, and join
  dependencies so compilation preserves those lifetimes.
- **Separate weight storage from arithmetic precision.** Keep BF16 weights in
  BF16 when the required accumulation/output is FP32; use
  [UnquantizedProjection](flare/components/linear.py) for that contract. Do not
  expand router weights to FP32 merely because routing requires FP32 logits.
  Preserve genuinely FP32 checkpoint weights and model-required arithmetic.
- **Let b12x select activation precision and kernels.** Use
  `QuantizedProjection(..., activation_mode="auto")` where the model's numerical
  contract permits it. Packed FP4/FP8 weights do not imply quantized activations
  are fastest at every row count. Keep production autotuning enabled and do not
  pin tile shapes, split-K, MoE plans, or a GEMM implementation to conceal an
  integration defect. Check what the selected plan actually executes.
- **Shard the vocabulary projection under TP.** Use
  [PreparedVocabularyProjection](flare/components/vocabulary.py) to project
  contiguous vocabulary partitions and gather logits before designated-rank
  sampling. Keep all ranks participating, including resume and other paths
  that produce logits. Respect the collective's supported geometry and retain
  its resources through native execution. A large head on one rank can dominate
  otherwise sharded decode.
- **Fuse residual and normalization boundaries.** Use b12x fused residual,
  normalization, activation, and mHC operations when their rounding matches the
  model contract. Consider adjacent layers as well as operations within a
  layer; [PreparedMHC](flare/components/mhc.py) supports fused post/pre mixing.
  Avoid materializing an intermediate solely because a Python module ends.
- **Compile complete prepared forwards.** Use
  [compile_forward](flare/preparation/compilation.py), which enables full-graph
  static compilation and autotuning for the remaining PyTorch operations while
  preserving native mutation contracts. Complete b12x preparation, tuning,
  binding refresh, and compiled warmup before capturing each serving graph.
  Retain compiled kernel owners until native replay has drained.
- **Prepare the whole serving shape range.** Include batch-one decode, packed
  decode, partial/inactive rows, and configurable chunked prefill through at
  least 4096 tokens. Test alternating graph variants with shared scratch and
  changing cache/recurrent state. Small-batch success does not establish stride,
  alias, or workspace correctness for prefill.
- **Measure complete GPU steps.** Include output projection, collectives, and
  sampling even when a reference engine executes them outside its main graph.
  For TP, measure the interval spanning all participating ranks. Reuse valid
  baselines; use kernel traces or component measurements to attribute gaps,
  then check gains through the actual Helios service. Correctness failures block
  performance claims. Verify allocation stability and scratch/input lifetime
  under repeated native replay, including poisoning reusable storage after its
  last permitted use.
