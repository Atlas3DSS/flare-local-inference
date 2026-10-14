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

## Build and run model checkpoints

Run commands from the repository root. Ignition prepares a source checkpoint
into a **runtime package**: a single `.safetensors` file with two `U8` tensors.
`flare.payload` holds stored tensors before graph, kernel, and frontend data;
`flare.metadata` holds the bounded Protobuf index at the end of the file.
The protobuf describes independently addressed resources. Tensor-parallel
packages contain one resource bundle per rank, with all ranks' storage before
execution resources.
Final packages use a format-4 signed envelope and encrypted cubins and graph
records. Ignition creates a per-user development signing key when `[publishing]`
is absent. `ignition run` passes the exact export key and publisher to Flare for
unattended approval; direct `flare run` needs the corresponding publisher profile
and first-use acceptance. See [signing and development](docs/cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the file without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

Set up the preparation environment and build the native CLI:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

Build dependencies include a C/C++ compiler, CMake, Ninja, pkg-config, Protobuf's
`protoc`, OpenSSL headers, Python headers, and liburing headers. On Ubuntu:

```sh
sudo apt-get install build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

The JPEG decoder builds its bundled libjpeg-turbo source. Preparation requires
the CUDA toolkit, including `nvcc` and CUPTI; configure `[paths] cuda` when it
is not supplied by the Python environment.

The preparation dependency targets `main` in `local-inference-lab/flashinfer`;
`uv.lock` records the resolved commit. Run
`uv lock --upgrade-package flashinfer-python` to advance it to the branch tip,
then sync the environment. For local FlashInfer development, install the sibling
checkout explicitly with `uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`;
`uv sync --locked --extra prepare --extra test` restores the Git dependency.
Inspect that checkout and its local changes before modifying it. After changing
Rust code in `heliopause/`, or Helios code used by the extension, rebuild the
installed extension explicitly:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

Inspect GPU occupancy with `nvidia-smi` before preparation or loading. Select
available GPUs explicitly on shared hosts. `--gpus 2` means one rank on GPU 2;
`--gpus 0,1` means two ranks on GPUs 0 and 1. With `CUDA_VISIBLE_DEVICES` set,
indices refer to the exposed device list. Each tensor-parallel size requires a
separately prepared package; a TP2 package cannot be loaded as TP1. Qwen TP1 and TP2 and
GLM TP4 have been exercised end to end; Qwen TP4 still requires model validation.

Export Qwen with up to five MTP draft tokens per round:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 2 \
  --mtp-draft-tokens 5 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-tp1-mtp5.safetensors --verbose

.venv/bin/ignition export qwen3.8-flash --gpus 0,1 \
  --mtp-draft-tokens 5 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-tp2-mtp5.safetensors --verbose
```

Choose an unused output filename; export appends `.safetensors` if needed and
refuses to overwrite one. Do not
delete an existing package merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the package, and exits.
Compilation is enabled by default; `--no-compile` is for development.
Qwen defaults to five MTP draft tokens, with explicit flags taking precedence.
Changing this default does not change an already exported package.

N-gram tables use indirect delivery exclusively: every graph consumes compact
GPU row buffers filled by Helios. Qwen PLE tables are stored once in the package.
Helios reads the 26.8 GiB table from disk by default on every GPU;
`flare run --embedding-table-storage ram` keeps it in resident host memory.
The graph publishes GPU-produced lookup IDs before layer
0 and waits for row delivery at the layer-1 consumer. Disk and RAM delivery
use the same prepared model graphs; the flag selects table storage only.

DeepSeek V4.1 Flash supports TP4 text generation, reasoning, and DSML tool calls.
DSpark is its learned draft method, defaulting to five draft tokens per round
with adaptive verification based on observed acceptance and GPU round cost;
vision graphs are not prepared. Its 188.8 GiB of Engram tables are
stored once in the package and read on demand through bounded direct-I/O
staging by default; `flare run --embedding-table-storage ram` loads them into
ordinary resident memory instead.
Both modes gather compact rows into shared pinned buffers, copy them to each
rank, and signal captured consumer events. Preparation appends the immutable
checkpoint slices without materializing the tables. Select four available
GPUs explicitly:

```sh
.venv/bin/python -m ignition export deepseek-v4.1-flash --gpus 0,4,6,8 \
  --max-concurrency 4 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/deepseek-v41-flash-tp4-c4-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v41-flash-tp4-c4-dspark5.safetensors \
  --gpus 0,4,6,8 --bind 127.0.0.1:8001
```

DeepSeek's logical cache pages cover 128 tokens across compressed latent and
index regions with different byte sizes. Continuation rows retain sliding
windows, pending compression state, partial pages, DSpark context windows, and the
last hidden state. Both use runtime-sized virtual ranges; exported packages
contain their layouts and capacity bounds without cache payloads.
Engram indexing runs in the graph over committed history and candidate tokens.
The three preceding tokens live in continuation state, including reused prefixes.
DSpark proposals, target verification, and accepted-state commit use the shared
native speculative contract; Helios contains no DSpark model implementation.

DeepSeek V4 Flash uses `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, including its
vision encoder and checkpoint-integrated DSpark model. The `deepseek-v4-flash`
alias prepares five draft tokens by default, with adaptive verification. It
accepts up to four images per request and has no offloaded embedding tables.
Use four available GPUs for preparation and serving:

```sh
.venv/bin/python -m ignition export deepseek-v4-flash --gpus 3,4,5,6 \
  --max-concurrency 4 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors \
  --gpus 3,4,5,6 --bind 127.0.0.1:8001
```

V4 uses 128-token logical pages containing 32 C4 latent records with a retrieval
index, or one C128 latent record with dense history attention. Each latent page
stores FP8/BF16 payloads followed by its own scale footer. Continuation rows
preserve the complete 128-token compression interval, C4 overlap, sliding
windows, partial pages, and DSpark context. Only accepted tokens update the
continuation state. Bidirectional image spans stay within one prefill chunk;
the scheduler shortens a preceding text chunk when necessary. Image preprocessing,
feature delivery, text execution, and output parsing all run natively in Helios.

MiMo V2.6 Pro MOPD uses `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` with checkpoint
MXFP4 experts, block-FP8 dense projections, and BF16 draft and vision weights.
The `mimo-v2.6-pro-mopd` alias prepares text, up to four images per request,
and seven DFlash draft tokens with adaptive verification. Audio is not prepared.
Select eight available GPUs:

```sh
.venv/bin/ignition export mimo-v2.6-pro-mopd --gpus 0,1,2,3,4,5,6,7 \
  --max-concurrency 4 --mtp-draft-tokens 7 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/mimo-v26-pro-mopd-tp8-c4-dflash7.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/mimo-v26-pro-mopd-tp8-c4-dflash7.safetensors \
  --gpus 0,1,2,3,4,5,6,7 --bind 127.0.0.1:8001
```

Global attention uses paged KV storage; continuation rows retain sliding-window
KV, partial pages, and accepted DFlash context. DFlash consumes selected target
layer features and predicts an eight-position block containing the pending
token and seven proposals. Helios uses the shared speculative execution contract;
draft modeling and state updates are captured by Ignition. The vision encoder
publishes compact image features for causal text prefill. Loading with
`--mtp-draft-tokens 0` selects ordinary execution without draft-state updates.

Both Ignition commands accept `--config PATH`; the default configuration is
`~/.config/flare/config.toml`. Storage locations are configured with
`[paths] hub_cache`, `kernel_cache`, and `package_cache`; `[memory] reserve`
leaves GPU memory available for other processes during preparation and serving.
Ignition additionally keeps 1 GiB for warm-up and capture. Qwen preparation backs
only a prefill chunk plus two KV pages per captured request row, unless an
explicit preparation page count is supplied programmatically. It reserves the
full virtual extent in the package. Helios sizes its caches
after uploading the selected graphs, keeping a separate 64 MiB driver margin.
Ignition captures power-of-two
batch capacities through the concurrency limit
and every verification width through the draft-token maximum. Flare can reduce
concurrency and draft tokens at load time; prefill capacity stays prepared.

Load an exported `.safetensors` package directly with the native CLI:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp2-mtp5.safetensors \
  --gpus 0,1 --bind 127.0.0.1:8001
curl --fail http://127.0.0.1:8001/health
```

Select a subset of a package's captured execution capacities:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp1-c4-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification --bind 127.0.0.1:8001
```

`--mtp-draft-tokens N` limits each round to N drafts plus its pending input.
`--mtp-draft-tokens 0` selects ordinary prefill, decode, and prefix-cache graphs
without draft execution or draft-state updates. Every speculative package also
contains these ordinary graphs. Adaptive width-one rounds retain draft state so
later rounds can resume speculation. Ignition's `--no-mtp` excludes the draft model
during preparation. `--mtp-adaptive-verification` selects among the loaded
widths; `--no-mtp-adaptive-verification` selects the fixed width. Omitted flags
use the package's serving defaults. Limits cannot exceed captured capacity;
missing required batch/width combinations fail before GPU initialization.
Concurrency three requires the batch-four bucket. Only selected graphs and
required kernel functions are loaded. Weights, MTP weights, and transport buffers
retain their prepared layouts. Continuation rows and KV pages occupy separate
fixed virtual ranges with declared maximum extents. Helios backs those ranges
according to selected concurrency, verification width, and available GPU memory;
reducing execution state leaves more physical memory for KV pages.

The selected GPU count must match the package, and each GPU must match its
rank's architecture and memory requirements. Flare serves all ranks in one
process with one execution thread per rank. Its dashboard runs on an
interactive terminal. Use `--api-key-file PATH` to supply a serving API key;
exported packages do not contain the key. Stop Flare with Ctrl-C or SIGTERM
so native replay drains before CUDA resources are released.

DGX Spark's GB10 is SM121 and shares physical DRAM with the CPU. Prepare an
SM121 package on Spark; an SM120 package does not satisfy the architecture
check. GPU weights, KV, registered host buffers, resident embedding tables,
compiler workers, and the OS compete for that memory. Set `[memory] reserve`
for host activity as well as driver headroom; `reserve = "4G"` leaves four GiB
outside the model's cache budget. RAM embedding tables load before cache
sizing. Disk embedding tables use bounded staging without a resident table.

Spark's NVML interface does not report VRAM capacity. Ignition identifies an
integrated device through CUDA and uses Linux `MemAvailable`, capped by CUDA's
device total, for planning. Qwen preparation and Helios use the same available
RAM budget when allocating weights and caches; swap is not GPU capacity.
CUDA free-memory queries exclude reclaimable page cache on GB10. Compiler worker
budgets account for both CPU and GPU allocations in shared RAM. Helios sizes
its caches independently after loading.
VMM allocations and registered host mappings remain distinct CUDA
memory types despite sharing DRAM; do not dereference a device allocation on
the CPU or replace system-scope synchronization with volatile accesses. See
[NVIDIA's Spark CUDA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

A model too large for one Spark runs tensor-parallel across two or four Sparks,
one rank per host, reducing through b12x RoCEnante over their ConnectX-7 ports:
`ignition export --ranks N --rendezvous ADDR:PORT` on rank zero with
`ignition join` on each peer writes one package on rank zero, and
`flare run PACKAGE --hosts H1,...` serves it, starting each peer's rank through
that host's `flare agent`. See [serving across DGX Sparks](docs/multi-host.md)
for setup, the agent's token, and failures.

For preparation followed immediately by serving, use the cached workflow:

```sh
.venv/bin/ignition run qwen3.8-flash --gpus 0,1 --port 8001 --verbose
```

`ignition run` caches packages under `paths.package_cache` (default
`~/.cache/flare/packages`) and replaces its process with Flare via `execv`.
It finds `flare` on `PATH` first, then `flare/target/release/flare`; check for
stale installed executables when testing a local build. Changes to preparation
inputs select a different cache entry. To test a serving-only Rust change,
rebuild Flare and load an existing compatible package directly. Re-export when
changing prepared kernels, capacities, or the package format.

Use `.venv/bin/ignition export --help` and `.venv/bin/ignition run --help` for
preparation flags. See [the runtime package design](docs/runtime-package-design.md)
for resource ownership and [the CLI design](docs/cli-design.md) for configuration.

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

- Keep Helios as simple as possible. Model-specific execution behavior,
  composition, and optimization belong in Ignition and its captured programs.
  Limit model-specific Helios support to K/V cache management and how cache
  storage is exposed to captured graphs, through contracts kept as general as
  possible. Model state-update computation belongs in Ignition; native code
  manages its declared storage without implementing per-model algorithms.
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
  prepared metadata and frontend assets; keep native model-specific execution
  contracts focused on cache management and graph cache bindings.

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
- `ignition serve --direct` is the one Python execution path, a
  diagnostic and qualification tool requested by the project owner for
  reproducing model faults without Helios ([direct execution](docs/direct-execution.md)).
  It is never production serving or a fallback, and only its explicit flag
  selects it.

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
- **Bound preparation KV backing by tuning and capture needs.** Describe the
  full virtual shapes with CUDA tensor metadata while compiling. Pass a
  `materialize` callback to `capture_forwards` to bind a bounded physical working
  set after compiler workers exit and before warmup and capture. Validate compiled
  guards against those tensors. Helios sizes the serving cache independently;
  temporary compiler contexts must not reduce its budget.
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
