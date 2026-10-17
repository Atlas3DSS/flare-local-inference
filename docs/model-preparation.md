# Model preparation and serving

## Runtime packages

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
and first-use acceptance. See [signing and development](cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the file without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

## Environment setup

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

The preparation dependency tracks the `main` branch of `local-inference-lab/flashinfer`
in `pyproject.toml`; `uv.lock` records the resolved commit. To refresh it, run
`uv lock --upgrade-package flashinfer-python`, then sync the environment.
For local FlashInfer development, install the sibling
checkout explicitly with `uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`;
`uv sync --locked --extra prepare --extra test` restores the Git dependency.
Inspect that checkout and its local changes before modifying it. After changing
Rust code in `heliopause/`, or Helios code used by the extension, rebuild the
installed extension explicitly:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

## GPU selection

Inspect GPU occupancy with `nvidia-smi` before preparation or loading. Select
available GPUs explicitly on shared hosts. `--gpus 2` means one rank on GPU 2;
`--gpus 0,1` means two ranks on GPUs 0 and 1. With `CUDA_VISIBLE_DEVICES` set,
indices refer to the exposed device list. Each tensor-parallel size requires a
separately prepared package; a TP2 package cannot be loaded as TP1. Qwen TP1 and TP2 and
GLM TP4 have been exercised end to end; Qwen TP4 still requires model validation.

## Preparation resources

`--compile-workers N` sets the CPU compiler process budget per host. Memory
limits can reduce it; local preparation ranks share the budget. Set
`[preparation] compile_workers = N` for a persistent default. Omitting the
setting retains automatic sizing, including shared-memory limits on DGX Spark.

`--tuning-gpus LIST` adds local GPU replicas for benchmarking b12x candidates.
The list begins with `--gpus` and continues in complete TP groups: for example,
`--gpus 0 --tuning-gpus 0-3 --compile-workers 24` prepares a TP1 package with
four tuning GPUs. Each replica needs memory for its model and preparation
storage. Helpers release their resources after tuning; serving ranks then
compile and capture their forwards. Kernel compilation runs in CPU workers,
independently of the number of tuning GPUs.

Multi-host `--ranks` preparation retains its existing sharding and transport;
`--tuning-gpus` is local-only and cannot be combined with it. An explicit
`--compile-workers` budget applies separately on each host.

## Exporting runtime packages

Choose an unused output filename; export appends `.safetensors` if needed and
refuses to overwrite one. Do not
delete an existing package merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the package, and exits.
Compilation is enabled by default; `--no-compile` is for development.

## Nemotron 3.5 Bolt 74B-A9B 24GB

`nemotron-bolt-24gb` prepares
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one SM120/SM121 GPU.
The recipe includes the C-RADIO image encoder and accepts up to four still
images per request. Video is not prepared. LiLiCorr uses the matching
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` checkpoint and defaults to
seven draft tokens with adaptive verification.

```sh
.venv/bin/python -m ignition export nemotron-bolt-24gb --gpus 0 \
  --max-concurrency 8 --prefill-chunk-size 4096 --mtp-draft-tokens 7 \
  --speculative-token-capacity 64 \
  --output /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors \
  --gpus 0 --bind 127.0.0.1:8001
```

Use `--no-mtp` during preparation to omit the draft model. Mamba-2 prefill
uses b12x's packed selective scan; decode and speculative verification use
FlashInfer state updates. Continuation rows retain BF16 recurrent state,
convolution history, partial FP8 KV pages, and LiLiCorr context and rollback
records. Prefix reuse and speculative acceptance preserve that complete state
through the shared Helios cache contract. Image features enter causal prefill
and may cross chunk boundaries.

Shared experts overlap routing and routed experts for graph capacities through
64 rows. Larger capacities execute those branches serially. LiLiCorr's separately
scaled NVFP4 gate and up projections also overlap. Prepare with
`--no-stream-overlap` to place those projections and expert branches on the main stream.
Routing excludes padded target rows using the packed batch's valid-token count.
The 64-row speculative capacity covers eight requests with seven draft tokens each.
For concurrency 32 with seven draft tokens per request, use
`--max-concurrency 32 --speculative-token-capacity 256`.

## Qwen3.8 27B

`qwen3.8-27b` prepares `nvidia/Qwen3.8-27B-NVFP4` for text generation on one GPU.
It uses the canonical Qwen3.5 dense decoder with tensor-scaled FP8 attention
projections and NVFP4 MLP and vocabulary weights. KV pages and Gated DeltaNet
continuations reside in device memory; the recipe has no host-mapped tables,
vision graphs, or learned draft model. Prepare on the same GPU architecture
as the serving device:

```sh
.venv/bin/ignition export qwen3.8-27b --gpus 0 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-27b-tp1-c4.safetensors --verbose
```

## Qwen3.8 Flash

Export Qwen with up to five MTP draft tokens per round:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 2 \
  --mtp-draft-tokens 5 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-tp1-mtp5.safetensors --verbose

.venv/bin/ignition export qwen3.8-flash --gpus 0,1 \
  --mtp-draft-tokens 5 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-tp2-mtp5.safetensors --verbose
```

Qwen defaults to five MTP draft tokens, with explicit flags taking precedence.
Changing this default does not change an already exported package.

N-gram tables use indirect delivery exclusively: every graph consumes compact
GPU row buffers filled by Helios. Qwen PLE tables are stored once in the package.
Helios reads the 26.8 GiB table from disk by default on every GPU;
`flare run --embedding-table-storage ram` keeps it in resident host memory.
The graph publishes GPU-produced lookup IDs before layer
0 and waits for row delivery at the layer-1 consumer. Disk and RAM delivery
use the same prepared model graphs; the flag selects table storage only.

## DeepSeek V4.1 Flash

DeepSeek V4.1 Flash supports TP4 text generation, reasoning, and DSML tool calls.
DSpark is its learned draft method, defaulting to five draft tokens per round
with separate proposal and verification graphs, using native conditional
confidence and measured execution cost for adaptive verification;
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
  --gpus 0,4,6,8 --host 127.0.0.1 --port 8001
```

DeepSeek V4.1's logical cache pages cover 128 tokens across compressed latent and
index regions with different byte sizes. Continuation rows retain sliding
windows, pending compression state, partial pages, DSpark context windows, and the
last hidden state. Both use runtime-sized virtual ranges; exported packages
contain their layouts and capacity bounds without cache payloads.
Engram indexing runs in the graph over committed history and candidate tokens.
The three preceding tokens live in continuation state, including reused prefixes.
DSpark proposals, target verification, and accepted-state commit use the shared
native speculative contract; Helios contains no DSpark model implementation.

## DeepSeek V4 Flash

DeepSeek V4 Flash uses `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF`, including its
vision encoder and checkpoint-integrated DSpark model. The `deepseek-v4-flash`
alias prepares five draft tokens by default, with adaptive verification. It
accepts up to four images per request and has no offloaded embedding tables.
Use two or four available GPUs for preparation and serving:

```sh
.venv/bin/python -m ignition export deepseek-v4-flash --gpus 3,4,5,6 \
  --max-concurrency 4 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors \
  --gpus 3,4,5,6 --host 127.0.0.1 --port 8001
```

V4 uses 256-token logical pages containing 64 C4 latent records with an FP8
retrieval index, or two C128 latent records with dense history attention. Each latent page
stores FP8/BF16 payloads followed by its own scale footer. Continuation rows
preserve the complete 128-token compression interval, C4 overlap, sliding
windows, partial pages, and DSpark context. Only accepted tokens update the
continuation state. Bidirectional image spans stay within one prefill chunk;
the scheduler shortens a preceding text chunk when necessary. Image preprocessing,
feature delivery, text execution, and output parsing all run natively in Helios.
Packages with 128-token V4 pages are incompatible and must be re-exported.

## MiMo V2.6 Pro MOPD

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
  --gpus 0,1,2,3,4,5,6,7 --host 127.0.0.1 --port 8001
```

Global attention uses paged KV storage; continuation rows retain sliding-window
KV, partial pages, and accepted DFlash context. DFlash consumes selected target
layer features and predicts an eight-position block containing the pending
token and seven proposals. Helios uses the shared speculative execution contract;
draft modeling and state updates are captured by Ignition. The vision encoder
publishes compact image features for causal text prefill. Loading with
`--mtp-draft-tokens 0` selects ordinary execution without draft-state updates.

## Step-5 Preview

Step-5 Preview uses `stepfun-ai/Step-5-Preview-NVFP4`. The
`step5-preview-nvfp4` alias prepares text, up to four images per request,
and three causal MTP draft tokens with adaptive verification. Checkpoint NVFP4
routed experts use W4A4. Routed experts declared BF16 in checkpoint metadata
are converted to NVFP4 during loading and use b12x W4A16 with BF16 activations;
the converted weights are retained in captured graphs and exported packages.
Shared, dense, and MTP MLP gate/up weights are packed and converted to MXFP8
during loading. Target-model projections select BF16 or quantized activations
through autotuning. MTP projections retain BF16 activations to bound rounding
error through the predictor feedback chain. Down and attention projections remain
BF16, with FP32 residual accumulation. Paged and sliding-window
K/V, including MTP windows, use FP8 E4M3 with unit descales;
sliding attention consumes live candidates in BF16 before committing them to FP8 history;
Q/K normalization and rotary computation retain their BF16 rounding before the
FP8 store. With CSF scales and shared scratch, the concurrency-16/MTP3 graph
catalog, including vision, fits on four RTX PRO 6000 Blackwell GPUs:

```sh
.venv/bin/ignition export step5-preview-nvfp4 --gpus 0,1,2,3 \
  --max-concurrency 16 --mtp-draft-tokens 3 --no-mtp-adaptive-verification \
  --speculative-token-capacity 16 --prefill-chunk-size 4096 --expert-scale-compression \
  --output /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors \
  --gpus 0,1,2,3 --host 127.0.0.1 --port 8001
```

Add `--expert-scale-compression csf` to compress all Step-5 routed-expert block
scales losslessly during loading. Both checkpoint W4A4 experts and converted
W4A16 experts use CSF, retaining their activation modes. Serialized layers share
one pair of expansion buffers, whose contents are zero-initialized on export.
One transient loading pool reuses conversion scratch across layers and retires
after weight loading, retaining live weights for graph export. The separate
graph-capture pool remains alive. Vision, prefill, decode, and draft activations
reuse typed graph buffers. MTP feedback, sampling, metadata, and resume scratch
reuse a separate buffer pool across graph variants, with disjoint buffers within
each graph. Sequential predictors share their norm-concatenation output. Kernel
workspaces use the shared operator arena, with separate slices for overlapping
expert branches. Compiler-created intermediates use one shared CUDA graph pool.
The memory plan estimates 6% lower resident weight storage; actual memory depends
on encoded exceptions and is reported after loading. Larger concurrency requires
separate preparation and memory validation.

Sliding attention keeps a 512-token continuation window. Sparse attention uses
128-token logical pages containing FP8 keys and values and an FP8 retrieval
index for eight-token regions. Continuation rows also preserve partial index
regions and accepted MTP state. Accepted target tokens, hidden states, and embeddings
remain in continuation storage until the following round consumes them in three
packed predictor passes. Candidate KV never enters committed predictor windows.
Region scoring, selection, and attention use
b12x kernels. Attention output gating is fused into the direct output or split
merge store, preserving BF16 rounding of attention and sigmoid before their
product. SwiGLU clamps the gate after SiLU. Helios manages these resources
through the shared cache and speculative execution contracts. The perception encoder
prepares 504-pixel local crops and a 728-pixel overview. Serving defaults follow
the model card: temperature 1, top-p 0.95, no top-k filter, and neutral penalties.

One-shot TP reductions with up to 36 rows fuse the residual update and
zero-centered RMSNorm into the collective, preserving FP32 residual storage and
BF16 normalized outputs. Attention and dense MLP projections enter the fused
collective directly as BF16; routed MLP updates enter as FP32. Attention and dense
MLP reductions round to BF16 before residual addition; routed MLP reductions retain
FP32. MLP reductions consume the following layer's input-normalization weight.
The final target and predictor MLPs
retain their separate final residual update. DMA prefill and hierarchical
collectives use separate reduction and normalization kernels.

## Kimi K3

Kimi K3 uses the local `/data/models/Kimi-K3-MXFP4-CSF` checkpoint. The
`kimi-k3-mxfp4-csf` alias prepares text generation, reasoning, XTML tool calls,
and up to four images per request on sixteen GPUs. It preserves MXFP4 experts
with compressed scales, BF16 projections, and FP32 KDA recurrent state.
Its 93-layer decoder combines KDA, unrotated latent attention, attention
residuals, and latent SiTU experts. DFlash2 uses
`lightseekorg/kimi-k3-dflash2` revision
`e77935fb4804e17eb55085bffd045eae1d779769`, downloaded to
`/data/models/Kimi-K3-DFlash2`, and defaults to seven draft tokens per round.

```sh
.venv/bin/python -m ignition export /data/models/Kimi-K3-MXFP4-CSF --gpus 0-15 \
  --max-concurrency 4 --mtp-draft-tokens 7 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-dflash2-7.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-dflash2-7.safetensors \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --host 127.0.0.1 --port 8001
```

The draft consumes prefix residuals from target layers 19, 37, 66, 78, and 90
(zero-based). Its five noncausal MLA layers predict an eight-position block;
grouped convolutions and a predecessor-conditioned candidate selector generate
seven proposals. Only accepted target features enter the draft's paged context,
and KDA verification records replay the accepted prefix into committed state.
Loading with `--mtp-draft-tokens 0` selects ordinary execution; exporting with
`--no-mtp` excludes draft weights and state.

MLA latent records occupy 128-token pages. Continuation rows preserve KDA
convolution and recurrent state, partial latent pages, and the normalized
last hidden state. The MoonViT encoder accepts a combined 65,536 image patches
per request and publishes hidden-column-sharded features for causal text
prefill. XTML encoding distinguishes protocol tokens from literal marker text.
Serving defaults follow the model card's agentic sampling recommendation:
temperature 1, top-p 1, no top-k filter, and neutral penalties. Reasoning is
always enabled; requests to disable thinking are ignored without an error.
Effort defaults to `max`; requests can select `low`, `high`, or `max` and
override sampling defaults. Preserve assistant reasoning in conversation history.

## Storage and graph capacities

Ignition commands accept `--config PATH`; the default configuration is
`~/.config/flare/config.toml`. Storage locations are configured with
`[paths] hub_cache`, `kernel_cache`, and `package_cache`; `[memory] reserve`
leaves GPU memory available for other processes during preparation and serving.
Ignition additionally budgets 1 GiB for warm-up and capture. Preparation KV backing
covers a prefill chunk plus boundary pages per captured request row, unless an
explicit preparation page count is supplied programmatically. Continuation
backing covers the null row, live request and verification rows, and snapshot
destinations used during preparation. Planned prefix-cache slots remain virtual;
they do not require physical backing during export. The memory report separates
the declared continuation extent from `preparation_state_rows` and
`preparation_state_bytes` where those extents differ. Both caches retain their
full virtual extents in the package. Helios
sizes its caches after uploading the selected graphs, keeping a separate 64 MiB
driver margin.
At package load time, `flare run --kv-cache-limit context` bounds the shared
KV pool by one prepared context; `concurrency` uses one context per selected
active request, and the default `memory` fills the available budget. Runtime
`--memory-limit` accepts an occupied size (`30G`), free reserve (`-30G`), or
fraction of total device memory (`0.8`). It replaces the exported serving
reserve and includes model resources as well as caches. `ignition run` forwards
both flags without selecting a different prepared package. See
[runtime memory limits](cli-design.md#runtime-memory-limits) for rounding,
minimum capacities, and integrated-memory accounting. Preparation retains its
own existing memory policy.
Ignition captures power-of-two
batch capacities through the concurrency limit
and width one at every bucket. Wider speculative and verification shapes require
`batch * width <= speculative_token_capacity`; the preparation flag
`--speculative-token-capacity` defaults to the larger of padded concurrency and
maximum verification width. Separate proposal catalogs cover every bucket and
load-time draft depth independently of that limit. Flare can reduce
concurrency and draft tokens at load time; prefill capacity stays prepared.
With `--mixed-prefill`, Qwen also captures mixed prompt/decode graphs: every
prompt bucket at width one and one balanced prompt bucket at each wider
verification depth, for each batch capacity of at least two. The ordinary and
draft-state-maintaining catalogs remain distinct. Mixed capture is disabled by
default and independent of the MTP flags.
See [mixed graph coverage](scheduling-runtime-design.md).

## Serving runtime packages

Load an exported `.safetensors` package directly with the native CLI:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp2-mtp5.safetensors \
  --gpus 0,1 --host 127.0.0.1 --port 8001
curl --fail http://127.0.0.1:8001/health
```

Select a subset of a package's captured execution capacities:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp1-c4-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification --host 127.0.0.1 --port 8001
```

`--mtp-draft-tokens N` limits each round to N drafts plus its pending input.
`--mtp-draft-tokens 0` selects ordinary prefill, decode, and prefix-cache graphs
without draft execution or draft-state updates. Every speculative package also
contains these ordinary graphs. Adaptive width-one rounds retain draft state so
later rounds can resume speculation. Ignition's `--no-mtp` excludes the draft model
during preparation. `--mtp-adaptive-verification` selects among the loaded
widths; `--no-mtp-adaptive-verification` selects the fixed width where captured,
with width-one execution available at larger batch capacities. Omitted flags
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

## DGX Spark

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
that host's `flare agent`. See [serving across DGX Sparks](multi-host.md)
for setup, the agent's token, and failures.

## Cached and development serving

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
preparation flags. See [the runtime package design](runtime-package-design.md)
for resource ownership and [the CLI design](cli-design.md) for configuration.

For model development, `ignition serve` hands prepared graphs directly to Helios
without exporting a package. It requires Heliopause's `qualification` Cargo
feature. Select idle devices explicitly; see [the development launch example](cli-design.md#development-without-package-export)
for the extension build and serving commands.
