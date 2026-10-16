# Flare design principles

## Required setup and commit review

On entering a clone or worktree, and after pulling changes, re-read this file and
[the design self-review](docs/design-review.md). Before committing, ensure the
repository's `.githooks/commit-msg` hook is enabled by following that document's
installation instructions. Preserve any existing active hooks when installing.

Before every commit, follow the staged-diff review and acknowledgment instructions
in `docs/design-review.md`. A rejected acknowledgment requires reading that
document and reviewing the staged change before retrying; do not treat the
rejection as a formatting repair.

Flare is a local inference server for NVIDIA SM120/SM121 GPUs. Ignition, the
Python package in `ignition/`, prepares models with PyTorch, Transformers, and
FlashInfer, including b12x. Its `flare_heliopause` extension is built by the
`heliopause/` crate. The Rust CLI in `flare/` links Helios, the runtime library
in `helios/`, to serve requests through prepared CUDA graphs. Both Flare and
Heliopause depend on Helios; Helios has no Python or preparation-tool dependency.

## Task tracking

Use [docs/TODO.txt](docs/TODO.txt) for miscellaneous follow-up items. Keep entries
brief and link to detailed design or performance notes when needed. Update or
remove entries as work is completed. Keep this file focused on durable project
guidance rather than task lists or progress notes.

## Build and run model checkpoints

Run commands from the repository root:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

Keep model-specific launch instructions and hardware setup in
[model preparation and serving](docs/model-preparation.md). Read its relevant
model and hardware sections before preparing or validating a model; it also
covers build dependencies, checkpoint formats, and graph capacities. The
[CLI design](docs/cli-design.md) defines flags, configuration, signing, and
package selection.

The preparation dependency pins a Git revision of `local-inference-lab/flashinfer`
in `pyproject.toml`, with its resolved commit in `uv.lock`. To change it, update
the revision, run `uv lock --upgrade-package flashinfer-python`, then sync.
For local development, inspect the sibling checkout and its changes before
editing it, and install it explicitly with
`uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`.
`uv sync --locked --extra prepare --extra test` restores the Git dependency.

After changing `heliopause/` or Helios code used by the extension, rebuild the
installed extension:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

Inspect GPU occupancy with `nvidia-smi` before preparation or loading and select
available GPUs explicitly with `--gpus`. With `CUDA_VISIBLE_DEVICES` set, those
indices refer to the exposed device list. A package's TP size and GPU architecture
must match serving. Choose unused export filenames; do not delete packages merely
to reuse their names.

For serving-only Rust changes, rebuild Flare and load a compatible package.
Re-export after changing prepared kernels, capacities, or the package format.
`ignition run` finds `flare` on `PATH` before `flare/target/release/flare`; check
for stale installed executables when testing a local build. Stop Flare with
Ctrl-C or SIGTERM so native replay drains before CUDA resources are released.

For development without export, `ignition serve` requires Heliopause's
`qualification` Cargo feature; follow the
[development launch instructions](docs/cli-design.md#development-without-package-export).

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

Helios's shared runtime policy and separation from model computation are
non-negotiable architectural invariants. Correct output, performance gains,
delivery pressure, and intended cleanup do not justify violating them. A
temporary model-specific policy path is still a violation. Resolve the
architectural boundary before committing the change.

- Keep Helios as simple as possible. Model-specific execution behavior,
  composition, and optimization belong in Ignition and its captured programs.
  Limit native specialization to concrete cache storage requirements and how
  that storage is exposed to captured graphs. Express differences through cache
  schemes and prepared layout metadata, with contracts reusable by every model
  that has those requirements. Model state-update computation belongs in
  Ignition; native code manages its declared storage without implementing
  per-model algorithms.
- Admission, page reservation and growth, eviction, preemption, readmission,
  and offload are shared runtime policy and mechanisms. Do not select them by
  model or recipe identity, duplicate them in model contracts, or hide model
  allowlists behind capability flags. A cache scheme may require different page
  geometry, state storage, or restore constraints; keep those differences in
  its storage contract and let shared policy consume that contract.
- `PreparedContract` is validated Protobuf data, never a runtime policy extension
  point. Declare concrete graph value sources, storage views, dependencies,
  frontend syntax, and capacities; shared code validates and binds them. Never add per-model boolean flags,
  including `supports_*`, `can_*`, or default-false opt-ins, to enable or disable
  runtime behavior. Enum values, callbacks, schema names, and metadata fields
  must not encode the same model allowlist. Every member must have a meaning
  independent of model identity, and its consumers must implement shared behavior
  from actual data, geometry, and dependencies. Review both declarations and all
  consumers. Tool/reasoning parsers and image preparation/layout implementations
  use independent typed enum identifiers; those identifiers cannot select
  scheduling, cache policy, text-graph input packing, or completion behavior.
- A runtime defect reproduced on one model must be traced through every
  supported cache scheme before choosing the implementation boundary. Apply
  the fix to every compatible contract. Any restriction must follow from a
  concrete storage or execution constraint expressed in the contract. For
  example, incremental page reservation must depend on the cache's growth and
  continuation requirements; testing it with GLM does not justify a GLM-only
  scheduler path. Validate affected cache schemes and request lifecycles, and
  report untested models separately from implementation restrictions.
- Introducing a captured-graph operation family requires prior explicit approval
  from project owner Luke Alonso <lalonso@gmail.com>. Shape and capacity variants
  of an existing family do not constitute a new type when its operation contract
  stays unchanged. A new operation contract requires approval even if it reuses
  an existing graph kind or enum value. A design acknowledgment does not grant
  that approval.
- Recipes in `ignition/recipes/` own model-specific composition, checkpoint mapping,
  state layouts, and graph preparation. They may retain canonical forwards or
  compose prepared components as the model requires.
- Put reusable execution mechanisms in `ignition/components/`. Recipes should be
  able to use components independently without adopting a rigid model framework.
  Shared/routed expert overlap, projections, collectives, and state transfers
  are examples of mechanisms that should not be copied between recipes.
- Keep native input packing, frontend processing, transport, scheduling, cache
  machinery, and request handling shared. Express model-dependent behavior through
  prepared metadata and frontend assets; keep native specialization focused on
  cache storage contracts and graph cache bindings.

## Python prepares; Helios serves

- When replacing internal formats or execution paths, remove superseded loaders,
  fallback paths, and migration-only tests. Retain backward compatibility only
  when it is an explicit requirement.
- Python loads weights, resolves and autotunes kernels, plans storage, warms
  execution paths, and captures supported graph variants before serving starts.
  Ignition exports a runtime package and closes the prepared model. `ignition run`
  then replaces its Python process with the Rust Flare executable through `execv`.
- Helios owns sockets, request processing, tokenization, chat templates,
  scheduling, cache operations, graph replay, and output streaming. Serving
  must not call back into Python.
- CUDA graphs are the model execution boundary. Do not introduce an eager
  serving fallback, runtime compilation, graph recapture, or GPU allocation in
  steady-state execution. Reject requests outside prepared capacities explicitly.
- Keep graph executables, captured buffers, resource owners, and the CUDA
  context alive for their full native use. Model state and metadata must be
  explicit storage that Helios can manage without Python execution.
- `ignition serve --direct [--eager]` is the one Python execution path, a
  diagnostic and qualification tool requested by the project owner for
  reproducing model faults without Helios ([direct execution](docs/direct-execution.md)).
  It is never production serving or a fallback, and only its explicit flags
  select it.

## No environment-variable switches

- Flare, Helios, recipes, and components never read an environment variable to
  choose behavior. Feature toggles, experiment switches, tuning overrides,
  debug modes, and alternate code paths selected through the environment are
  prohibited, including temporary ones.
- A setting has one of three homes: the recipe declaration, the configuration
  file, or a command-line flag, as [the CLI design](docs/cli-design.md)
  specifies. A setting that only experiments need is an advanced flag carried
  through the plan into the recipe's `PreparationConfig`, so the plan, its
  printed report, and its recorded measurements describe what was prepared.
- `CUDA_VISIBLE_DEVICES` is the only variable Flare interprets: it renumbers the
  GPUs that `--gpus` indexes. Before starting Helios, Flare narrows that mask to
  the selected GPUs and passes contiguous local ordinals to the runtime. It
  sets `CUDA_FORCE_PRELOAD_LIBRARIES=0` and `CUDA_DISABLE_JIT=1` as fixed driver
  policy: serving loads compiled cubins without driver compilation.
  `ignition.cli.environment` sets the variables that
  PyTorch, b12x, Triton, and the Hugging Face hub read, from Flare's
  configuration, before importing them. Detecting a terminal's capabilities is
  not configuration.
- Tests may read variables that opt into GPU tests and locate their
  checkpoints, such as `FLARE_RUN_CUDA_TESTS`. A test variable must not change
  what the code under test does.

## Model-card sampling and structured runtime metadata

- Agents adding or changing a recipe must read the model card and manually verify
  its recommended sampling parameters for the default serving mode. Declare the
  complete defaults in the Ignition recipe and cite the recommendation at that
  declaration. Never infer sampling policy from `generation_config.json`,
  `config.json`, Transformers defaults, or another model. Model-card recommendations
  take precedence over conflicting config files; unspecified filters are disabled
  and penalties are neutral. Explicit request sampling parameters override defaults.
- Ignition validates checkpoint configuration for structural data, including
  EOS and template tokens, image preprocessing, and vision position geometry.
  Export only the fields Helios consumes in first-class Protobuf messages.
  Runtime assets are the tokenizer and chat template; do not package raw generation,
  tokenizer, processor, or model configuration files. Helios consumes prepared
  settings without per-model sampling policy or config-file interpretation.

## Use FlashInfer and plan memory explicitly

- Flare is fully coupled to FlashInfer/b12x by design. It has no backend or
  device abstractions. Use FlashInfer/b12x and NVIDIA CUDA interfaces directly;
  do not introduce interchangeable kernel backends or device portability layers.
- Prefer proven b12x operators and FlashInfer sampling over duplicate kernels.
  Inspect their supported preparation, tuning, and workspace interfaces before
  introducing custom implementations.
- Use b12x's autotuning and warmup facilities during preparation. Do not pin
  kernel plans or tuning choices without a demonstrated requirement.
- Constructors declare b12x requests through `ignition.preparation.tuning.declare`;
  they must not compile or prime default plans. Submit all operators across the
  complete serving shape catalog to one preparation batch, then bind selected
  plans through `register_finalizer`. Scratch sizing and serving bindings must
  follow tuning.
- Show startup work through `ignition.progress.phase`. Weight loading and kernel
  preparation retain a phase activity bar while their detailed b12x panels are
  transient and share the phase's Rich console. Use `DirectWeightLoader` for
  loading progress, including draft-model weights. Only the root TP rank
  renders progress.
- Supply caller-owned scratch through the kernel planning interfaces. Reuse
  storage across sequential work; preserve separate storage for overlapping
  operations and persistent state. Aliasing must follow actual lifetimes and
  stream dependencies, including overlap captured inside a single graph.
- Treat KV, recurrent state, and other model histories as explicit cache state.
  Prefix reuse, branching, and speculative rollback must preserve all relevant
  state, not just token positions and KV pages.

## Optimize the serving path

- Apply the recipe requirements below during implementation. Inspect the
  model's fusion boundaries, precision, sharding, overlap, and selected kernel
  plans. Reuse FlashInfer/b12x mechanisms directly.
- Measure GPU graph replay separately from native scheduling and end-to-end
  request latency. Compare equivalent shapes and execution configurations when
  using another server as a performance reference.
- Attribute gaps to emitted kernels, memory traffic, dependencies, or host work
  before changing architecture. Preserve correctness when changing layouts,
  fusion boundaries, scratch reuse, or stream overlap.
- Keep ordinary tests and raw benchmark measurements. Do not create development
  audit reports, receipts, provenance bundles, or hash inventories.

## Recipe performance requirements

- **Read quantization from checkpoint metadata.** Use `config.json` declarations
  to choose supported weight formats. Keep geometry and encoding validation
  explicit; do not hardcode a checkpoint's per-layer precision assignments.
  Ordinary NVFP4, CSF NVFP4, and MXFP4 expert storage use the same b12x MoE
  execution API. Dense weight formats likewise share `blockscaled.mm`; format
  handling belongs in loading and preparation, not recipe inference branches.
  Reject fusion of NVFP4 projections except routed MoE w1/w3, whose dedicated
  loader validates the shared scales. Never silently rescale or requantize
  weights to make a fusion possible.
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
  [ConcurrentSharedExpert](ignition/components/expert_overlap.py) and
  [ConcurrentProjectionChain](ignition/components/projection_overlap.py) where their
  contracts fit. Measure the relevant row capacities: branches competing for
  bandwidth or SM capacity can make overlap slower. Graph node construction
  order also affects scheduling. `ConcurrentSharedExpert(..., defer_launch=True)`
  records routed kernels before shared kernels while preserving the fork and
  join dependencies; measure this ordering in the complete model graph.
- **Make overlap safe through compilation.** Declare every concurrently live
  plan with [ScratchArena.register_parallel](ignition/preparation/scratch.py).
  Sequential work reuses storage; concurrent work gets disjoint storage sized
  from the tuned plans. Refresh bindings after arena resizing. The fork must
  follow input production, and the join must precede output consumption or
  input/scratch reuse. Custom ops must expose reads, writes, aliases, and join
  dependencies so compilation preserves those lifetimes.
- **Separate weight storage from arithmetic precision.** Keep BF16 weights in
  BF16 when the required accumulation/output is FP32; use
  [UnquantizedProjection](ignition/components/linear.py) for that contract. Do not
  expand router weights to FP32 merely because routing requires FP32 logits.
  Preserve genuinely FP32 checkpoint weights and model-required arithmetic.
- **Let b12x select activation precision and kernels.** Use
  `QuantizedProjection(..., activation_mode="auto")` where the model's numerical
  contract permits it. Packed FP4/FP8 weights do not imply quantized activations
  are fastest at every row count. Keep production autotuning enabled and do not
  pin tile shapes, split-K, MoE plans, or a GEMM implementation to conceal an
  integration defect. Check what the selected plan actually executes.
- **Shard the vocabulary projection under TP.** Use
  [PreparedVocabularyProjection](ignition/components/vocabulary.py) to project
  contiguous vocabulary partitions and gather logits before designated-rank
  sampling. Keep all ranks participating, including resume and other paths
  that produce logits. Respect the collective's supported geometry and retain
  its resources through native execution. A large head on one rank can dominate
  otherwise sharded decode.
- **Fuse residual and normalization boundaries.** Use b12x fused residual,
  normalization, activation, and mHC operations when their rounding matches the
  model contract. Consider adjacent layers as well as operations within a
  layer; [PreparedMHC](ignition/components/mhc.py) supports fused post/pre mixing.
  Avoid materializing an intermediate solely because a Python module ends.
- **Compile complete prepared forwards.** Use
  [capture_forwards](ignition/preparation/capture.py), which restores compiled
  forwards before tracing and compiles missing variants with static shapes and
  autotuning. Workers perform Dynamo tracing, storage lowering, and Inductor
  compilation from tensor and selected-plan metadata; CUDA resources remain in
  the owning rank. Preserve native mutations, storage aliases, and stream dependencies;
  bind process-local kernel handles and caller-owned tensors at startup. Complete
  b12x preparation, tuning, binding refresh, and compiled warmup before capturing
  each serving graph. Retain compiled kernel owners until native replay has drained.
- **Bound preparation cache backing by tuning and capture needs.** Keep virtual
  serving capacity separate from physical preparation backing. Back only the
  KV pages touched by tuning and capture, plus the null row, simultaneously live
  request/verification rows, and disjoint snapshot destinations. Do not physically
  allocate the planned prefix-cache capacity or fill remaining VRAM with KV during
  export; Helios sizes serving caches after graph upload. Check every fixture,
  state-clear operation, and snapshot/restore path against the backed extent.
  Describe the full virtual shapes with CUDA tensor metadata while compiling. Pass a
  `materialize` callback to `capture_forwards` to bind a bounded physical working
  set after compiler workers exit and before warmup and capture. Validate compiled
  guards against those tensors. Helios sizes the serving cache independently;
  temporary compiler contexts must not reduce its budget.
- **Share scratch across the complete graph catalog.** Vision, prefill, decode,
  speculative, and resume graphs execute serially and should reuse caller-owned
  activation and private buffer pools. Allocate per-variant views with
  `GraphBufferPool` cursors; keep simultaneously live values disjoint within a
  graph, including retained verification state and overlapping expert branches.
  Use the shared operator arena for kernel workspaces and one CUDA graph pool for
  compiler-created temporaries. Inspect constructor allocations as well as captured
  allocations; sharing the CUDA graph pool alone does not share recipe scratch.
  Constants and request-owned state must retain independent storage.
  Prime indexed operators with private valid metadata: aliased serving scratch
  is initialized by the complete graph, not by each operator's constructor.
  Keep shared buffer-pool layouts identical across TP ranks, including storage
  reserved for designated-rank sampling, so shared compiled forwards retain the
  same storage aliases on every rank.
- **Release preparation temporaries at phase boundaries.** Retire loading,
  conversion, tuning, and warmup pools when their temporary owners are gone;
  retain graph pools and captured addresses through export and replay. A live
  tensor can pin otherwise unused space in a shared allocator segment; retiring
  a pool cannot release a segment that still holds a weight. Account for
  simultaneous scratch, cached allocator segments, physical cache backing,
  and graph-executable/driver allocations. Report allocated and reserved tensor
  bytes separately from device-free memory. Validate the complete requested graph
  catalog near capacity; a smaller concurrency does not establish builder fit.
- **Derive preparation memory policy from hardware.** Shared preparation code
  detects integrated GPUs and budgets compiler workers and tuning races against
  available system RAM. Recipes pass byte headroom, not hardware flags. Forward
  compilation uses memory-bounded parallel workers on both integrated and discrete
  GPUs; artifact binding starts no additional compiler pool.
- **Compile tensor-parallel forwards once, sharded across ranks.** Pass the
  preparation coordinator to `capture_forwards`. Each forward variant compiles on
  one rank that selects it, from that rank's own forward, with a compiler pool
  budgeted for its host's ranks. Ranks on one host share its cache directory;
  ranks on other hosts receive the artifacts they lack. Every rank loads the same
  artifacts, binds its tensors and native handles, and participates in ordered
  warmup and local capture. Keep shard offsets and native handles in runtime
  bindings when they do not change the computation. For distinct rank behavior,
  select a variant key through `compilation_variant` on each rank. Every
  selected variant must compile before collective warmup; ranks must not retrace
  or compile on a binding mismatch.
- **Capture zero-draft rounds as decode.** Verification width one contains only
  the pending target token. Use ordinary decode attention and direct cache updates;
  do not inherit prefill kernels, speculative staging, or draft rollback work
  merely because the graph belongs to the speculative catalog. Retain the draft
  context updates and recurrent metadata required to alternate with wider rounds.
  Compare the complete width-one graph against ordinary decode, attribute any
  extra cost, and test transitions in both directions with partial/inactive rows
  and cache-boundary crossings. Apply this check in each recipe's existing
  preparation and tests.
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
