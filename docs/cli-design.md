# Ignition preparation and Flare serving commands

Flare serves the model architectures its recipes support on NVIDIA SM120 and
SM121 GPUs; each recipe pins qualified default checkpoints. Its command line accepts only what the engine cannot determine by itself,
expressed in the user's terms. A setting the engine can read from the
checkpoint, measure on the host, or derive from another setting is not a
parameter anywhere: not a flag, not a configuration key, not a debug override.

## 1. Principles

- **Intent, not mechanism.** A parameter names a decision only the user can
  make: which GPUs, how many simultaneous requests, where to listen.
- **The engine is the recipe.** Per-model knowledge lives in `ignition/recipes/`
  as qualified code. A recipe declares its checkpoints, their pinned revisions,
  supported tensor-parallel sizes, concurrency range, memory cost model, and
  template defaults. The command line never needs to know which model it runs.
- **Derived means derived.** Context length, page count, prefix-cache size,
  snapshot slots, graph capacities, worker counts, and body limits are computed
  from the recipe, the host, and the user's intent. They are printed, never
  typed. The recipe sets the prefill chunk and the speculation settings;
  advanced flags override them.
- **Memory follows the model and the host.** Weights, graphs and
  scratch, live state, and a prefix cache sized from `--max-concurrency` come
  first. KV pages take everything else: the GPU's free memory at startup minus
  engine headroom. The page pool holds active requests and every cached prefix,
  so memory beyond what active requests need keeps more history resident.
  Nothing asks for a fraction or a size.
- **Estimate roughly, allocate from the device.** Before weights load, Flare
  estimates memory from the recipe's declared sizes, only to choose GPUs and to
  refuse a model that cannot fit. Allocation follows the memory the device has
  free during preparation, and `run` prints what it allocated. An estimate that
  does not fit fails before any socket opens, with the numbers and the settings
  that would make it fit.
- **No environment variables.** Flare reads none, with one exception:
  `CUDA_VISIBLE_DEVICES` restricts which GPUs the caller exposed, which changes
  the GPU each index names, so Flare inspects it to interpret `--gpus`.
  Variables that underlying libraries read are set by Flare from its own
  configuration before those libraries are imported.
- **Two levels of precedence.** Flags override the configuration file, which
  overrides the recipe. Nothing else exists.

## 2. Commands

The Python distribution and source package are named `ignition`. Install the
preparation environment with `uv sync --locked --extra prepare --extra test`,
then use `.venv/bin/ignition` or `.venv/bin/python -m ignition`.

```
ignition run     [MODEL] [flags]     cache a runtime package and launch Flare
ignition export  [MODEL] --output FILE [flags]  prepare a native model package
```

Every command has exactly one name. `MODEL` is a recipe alias, a Hugging Face
repository id with an optional `@revision`, or a local checkpoint directory;
with no `MODEL`, `run` and `export` use `defaults.model` from the configuration file and
fail with a usage error if it is unset.

Ignition exposes preparation and development launch commands. It prints
preparation progress without a banner. The Rust Flare executable displays the
Flare banner on a terminal and reports model loading through Helios.

`export` imports the `flare_heliopause` CPython extension in Ignition's process.
Ignition performs weight loading, tuning, compilation, warmup, and capture. It writes compiled CUDA modules,
allocation contents, graph descriptions, and frontend assets to one `.flaretensors` file
that must not exist, then exits without opening a listener. `run` caches the resulting package in
`paths.package_cache` and replaces its Python process with the Rust `flare` executable. Cache keys
include checkpoint identity, preparation settings, dependency versions, and
preparation source contents. Qwen and GLM provide native package storage support.
Tensor-parallel packages contain one resource bundle per selected GPU and load
in one process with one execution thread per rank. API keys are excluded from
the package.

Build the public serving executable, which links the Helios library:

```sh
cargo build --release --manifest-path flare/Cargo.toml
flare/target/release/flare run PACKAGE --gpus 0 --bind 127.0.0.1:8000
```

The native device ordinal follows CUDA's visible device enumeration. Optional
`--api-key-file PATH` supplies authentication at launch. Loading requires
the package's GPU architecture and sufficient GPU and mapped host memory.
Recorded addresses must be available; an address collision fails startup.
The package records prepared concurrency, the available prefill and speculation
graphs, and maximum virtual extents for cache storage. `--max-concurrency N`
and `--mtp-draft-tokens N` select supported reductions at load time;
`--mtp-adaptive-verification` and `--no-mtp-adaptive-verification` select the
verification policy. Omitted flags use package defaults. Only selected graphs
and required kernels load. Continuation rows and KV pages receive physical
storage according to those selections and available GPU memory.

## 3. Parameters

This is the complete inventory. Everything a user can set is listed here with
its home: the command line, the configuration file, or both.

### 3.1 Command-line flags

`run` accepts the following flags. `export` accepts the same preparation flags,
requires `--output FILE`, and excludes `--host` and `--port`.

| Flag | Meaning | Default |
|---|---|---|
| `MODEL` | Alias such as `qwen3.8-flash`, Hugging Face repository id such as `org/name` or `org/name@revision`, or checkpoint directory | `defaults.model` |
| `--gpus LIST` | GPU indices as NVML enumerates them: `0`, `0,1`, `0-3`. The count is the tensor-parallel size and must be one the recipe supports. | First GPUs that fit |
| `--max-concurrency N` | Maximum simultaneously active requests, 1 through the recipe's limit | `defaults.max_concurrency`, else the recipe default |
| `--host ADDR` | Listening address. `run` only. | `server.host`, else `127.0.0.1` |
| `--port N` | Listening port. `run` only. | `server.port`, else `8000` |
| `--verbose` | Per-phase detail on stderr | off |
| `--config PATH` | Configuration file | `~/.config/flare/config.toml` |

Advanced flags appear in their own section of `--help`. The recipe sets each
value; the flag overrides it.

| Flag | Meaning | Default |
|---|---|---|
| `--prefill-chunk-size N` | Most prompt tokens processed in one prefill step, a multiple of 64. A larger chunk reaches a long prompt's first token sooner; while it runs, other active requests wait for their next token. Command line only. | The recipe's chunk |
| `--no-mtp` | Disable multi-token speculation: load no draft model, keep no verification rows, and decode one token per step. | Speculation on |
| `--mtp-draft-tokens N` | Most draft tokens verified in one round, up to the recipe's limit (7 for Qwen and GLM; 5 for DeepSeek). Graphs are captured for every count from 0 through `N`, and verification adapts among them from measured acceptance. | 5 |
| `--no-mtp-adaptive-verification` | Default to fixed-width serving. Capture still includes every smaller width at every batch capacity. | Adaptive |
| `--mtp-proposal-source {mtp,copy}` | Where drafts come from: the checkpoint's MTP module, or `copy` for continuations of the request's own history, which loads no draft model. Qwen only. | `mtp` |
| `--history-drafts` | Also draft continuations copied from the request's own history. Qwen only. | off |
| `--no-compile` | Development only: capture graphs without compiling forwards. A recipe never sets it. | off |

The speculation flags exist on the command line only. `--no-mtp` excludes the
others. The `Select GPUs.` startup line states the settings in effect: `up to
N draft tokens` while verification adapts, `exactly N draft tokens` when it is
fixed.

Both Ignition commands accept `--config`.

Names follow one rule: the flag says what the user is deciding, in the word a
user would use. `--gpus` rather than `--device` and `--tp`, because the user
picks GPUs and tensor parallelism follows. `--max-concurrency` rather than
`--max-active` or `--max-num-seqs`, and with the `max` because it is a cap,
not a target. `--host` and `--port` because they are the names every HTTP server uses.

### 3.2 Configuration file

`~/.config/flare/config.toml`. All keys optional. Written on first `run` with
the resolved values present as comments.

```toml
[server]
host = "127.0.0.1"
port = 8000
api_key = ""                      # empty: no authentication

[defaults]
model = "qwen3.8-flash"
max_concurrency = 4

[memory]
reserve = "0"                     # GPU memory left untouched for processes that start later

[models."qwen3.8-flash"]          # per-model overrides of [defaults]
max_concurrency = 2

[paths]
hub_cache = "~/.cache/huggingface/hub"
kernel_cache = "~/.cache/flare/kernels"
package_cache = "~/.cache/flare/packages"
cuda = "/opt/cuda"                # used only when the toolkit is not found in installed packages
```

`[models.<alias>]` overrides `[defaults]`; flags override both.

Default request parameters, meaning the values a request receives for fields
it does not set, such as thinking, reasoning effort, sampling, and output
limit, are a separate mechanism that this specification does not yet define.
Until it exists, requests receive the checkpoint's template and
`generation_config.json` defaults and the engine's output limit.

`memory.reserve` exists for one case: a process that starts after Flare and
needs GPU memory. Processes already running when Flare starts need no
setting, because supply is free memory rather than card size. It is the one
memory number a user may ever type, it is the number they actually know, and
it lives only in the configuration file.

The prefix cache is not a setting. It holds model states at instruction,
prompt, and answer boundaries so that later requests repeating a prefix skip
recomputing it. A finished exchange keeps its prompt and answer states, so the
engine sizes the cache at eight states per concurrent request, about four
recent exchanges each, and gives it at most a quarter of the memory left after
weights, graphs, and live state. A state costs about 117 MiB on the Qwen3.8
recipe. Each active request additionally holds three states while it runs,
counted as live state. Under memory pressure, KV pages shrink first and the prefix cache has a
floor of 8 states.

### 3.3 Parameters that do not exist

Each of these was a flag on the previous entrypoint, is a flag on vLLM, or is
a field in a lil manifest. None is a parameter in Flare.

| Candidate | Verdict | Reason |
|---|---|---|
| `--context`, `max_model_len` | Not a parameter | Context is the checkpoint's maximum. KV pages are one shared pool and admission reserves each request's prompt plus output limit against it, so four 128K requests and one 262K request both fit without a setting. A recipe that preallocates per request derives its capacity from `--max-concurrency`. |
| `--memory`, `--gpu-memory-utilization`, `kv_cache_memory_bytes`, `--pages` | Not a parameter | A fraction is relative to the card and blind to other processes; a size asks the user to subtract their other workload in their head. Weights come from the index, graphs and scratch from capture, and pages from the free memory that remains. The only memory number is `memory.reserve`, in the configuration file, for processes that start later. |
| `--snapshot`, `--model-path` | Not a parameter | A directory is a valid `MODEL`. The old name also collided with state snapshots. |
| `--device`, `--tp` | Not a parameter | `--gpus` lists GPUs; tensor parallelism is their count. |
| `max_num_batched_tokens` | Not a parameter | The recipe's prefill chunk; `--prefill-chunk-size` is the advanced override. |
| `--headroom`, `--snapshot-budget`, `conversation_cache` | Not a parameter | Ignition keeps 1 GiB for compiled warm-up and graph capture. Helios sizes its caches after graph upload and keeps 64 MiB for driver bookkeeping, streams, and events. Both leave the configured `memory.reserve` untouched. The prefix cache is sized from `--max-concurrency`. |
| `--mtp`, width lists, fixed width, proposal source, copy | Advanced flags | Speculation is distribution-preserving and faster when the recipe prepares it, so every recipe with an MTP module enables it and adapts the draft count from measured rounds. `--no-mtp` and the `--mtp-*` flags override the recipe: to measure a setting, to pin one, or to free the draft model's memory on a host where it does not fit. One draft-token count sets the depth and `--no-mtp-adaptive-verification` pins it; there is no list of counts, no separate fixed count, and no flag that turns speculation on. |
| `--max-active` | Renamed | `--max-concurrency`. |
| `--bind`, `--listen` | Renamed | `--host` and `--port`. |
| `--thinking`, `--template-defaults FILE`, reasoning effort, sampling defaults, `--max-tokens` | Not command-line parameters | These are default request parameters, a separate mechanism not yet specified. Launch flags do not set per-request behavior. |
| `--memory-report FILE` | Removed | `run` prints the allocated capacities; the server exposes metrics over HTTP. |
| `--prepare-only`, `prepare`, `plan` | Removed | `run` prepares and caches a runtime package; `export` writes one to an explicit destination. |
| `--revision` | Part of `MODEL` | `org/name@revision`. An alias uses the revision its recipe pins. |
| `--dtype`, `--quantization`, `--kv-cache-dtype`, `--block-size`, kernel backends, `cudagraph_mode`, `--enable-prefix-caching`, `--enable-chunked-prefill` | Not a parameter | Recipe decisions, qualified. |
| `--served-model-name` | Not a parameter | Requests may name the alias or the repository id; `/v1/models` lists both. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Not a parameter | `--max-concurrency` and the derived chunk. |
| `queue_capacity`, `event_capacity`, `frontend_workers`, `max_body_bytes` | Not a parameter | Derived: queue is four times concurrency, frontend workers equal concurrency, event capacity is fixed, body limit is the recipe's image count times its per-image limit plus 1 MiB. |
| `compile_workers` | Not a parameter | Derived from CPU count. |
| `CUDA_HOME` | Not a parameter | Located from installed packages, then `[paths].cuda`. |
| `CUDA_VISIBLE_DEVICES` | Read only to interpret `--gpus` | `--gpus` selects. If the caller set it, integer entries are NVML indices, other entries are GPU UUID prefixes, `--gpus` indices are relative to that enumeration, and the `Select GPUs.` startup line says so. Flare then exports the selected GPUs' UUIDs to its own processes. |
| `--api-key` | Configuration only | Secrets do not go on command lines. |
| `--seed`, `--temperature`, stop sequences | Request fields | Per request. |

## 4. Derived settings

| Setting | Derived from |
|---|---|
| Tensor-parallel size | Count of `--gpus`, or the smallest recipe-supported size whose estimate fits the first GPUs with enough free memory |
| Context limit | The checkpoint maximum |
| KV pages | All remaining supply: free GPU memory minus engine headroom, `memory.reserve`, weights, graphs and scratch, live state, and the prefix cache; divided by the recipe's per-token page cost. Active requests are guaranteed an equal share, up to the checkpoint's maximum context; the rest holds cached prefixes |
| Prefix cache | Eight states per `--max-concurrency` request, at most a quarter of the memory left after weights, graphs, and live state; floor 8 states under pressure |
| Live state | The recipe's state rows for `--max-concurrency` requests: one row per request slot, the recipe's speculative rows per slot while speculating, and three checkpoint rows per active request |
| Decode graph capacities | 1 through `--max-concurrency`, rounded to the recipe's captured set |
| Prefill chunk | The recipe's chunk, 4,096 tokens for both recipes; `--prefill-chunk-size` overrides |
| Speculative catalog and draft counts | Recipe, overridden by `--no-mtp` and the `--mtp-*` flags; the draft count adapts from measured rounds unless fixed |
| Kernel selections | Autotuned during preparation; cached by checkpoint revision, recipe version, GPU, and settings |
| HTTP queue, workers, event capacity, body limit | `--max-concurrency` and the recipe's image limits |
| Compile workers | CPU count |
| Generation defaults | Checkpoint `generation_config.json`, pending the default-request-parameter mechanism |
| Template defaults | The checkpoint template's own defaults, pending the same mechanism |
| Served names | Recipe alias and repository id |

## 5. Command behavior

### 5.1 `ignition run`

1. Resolve `MODEL`, discover GPUs through NVML, and derive the memory plan.
   Validate the requested listener address. The selected GPU count determines
   the number of rank bundles.
2. Select a package cache entry from the checkpoint, preparation settings,
   dependencies, and preparation sources, holding a lock through publication.
3. On a cache miss, import `flare_heliopause`, download missing weights, prepare
   the model, export all resources, and close the prepared model. These steps
   run in one Python process per rank. Preparation failures leave no published package.
   A cache hit skips preparation entirely.
4. Apply the selected GPU visibility and replace ignition with `flare run PACKAGE`.
   Pass the listener address and an anonymous file descriptor for the API key.
5. Flare loads the model through Helios and starts HTTP. The completed
   `Start server.` row reports the listening URL.

The Qwen recipe sizes its KV pages from free memory after graph capture.
The package retains that capacity on subsequent loads. Insufficient device
memory or a recorded-address collision fails native startup.

Serving prints no per-request output. SIGINT and SIGTERM stop admission,
give active responses five seconds to finish, drain GPU work, and exit 0.
During preparation either signal ends Ignition immediately.

Preparation exit codes: 0 success, 2 usage, 3 fetch, 4 capacity, 5 preparation.
Native startup and runtime failures exit 1.

### 5.2 `ignition export`

Resolve the model and preparation settings as for `run`, then prepare and write
a runtime package to `--output FILE`, appending `.flaretensors` if needed.
The destination must not exist. Export
closes the prepared model and exits without starting a listener.

### 5.3 `flare run`

Animate the Flare banner during loading on a terminal. Prepare the model,
initialize GPUs, allocate model memory, load weights, prepare kernels and
execution, allocate cache, load model configuration and frontend assets, and
start the HTTP server. Helios reports each stage with a ten-column bracketed
progress bar, a result, and elapsed time.
Memory loading advances by bytes; kernel and execution preparation
advance by resource counts. Stages with no known total show a cycling bar.
Completed stages show `[    OK    ]`; a failed stage shows `[  FAILED  ]`.
A narrow terminal truncates the description while preserving status and timing.
Redirected output contains plain start and completion lines without animation.
After startup, a usable terminal switches to the live metrics dashboard.
Shutdown restores the terminal after draining active execution.

## 6. Capacity errors

An estimate that does not fit fails `run` before loading weights with the
requirement, what is available, and the options that satisfy it:

```
error: needs 68.3 GiB weights + 7.0 GiB graphs + 5.2 GiB live state + 0.9 GiB prefix cache
       + 0.8 GiB minimum pages (4 × 8,192 tokens) = 82.2 GiB;
       has 95.6 GiB free − 1.0 GiB headroom − 21.0 GiB reserve = 73.6 GiB.
       Options: [memory] reserve = "12G".
```

The requirement is the smallest estimate the engine accepts: the prefix cache
at its floor and 8,192 tokens of pages per active request. Every listed option
satisfies it on its own; the candidates are a smaller `reserve`, a lower
`--max-concurrency`, `--no-mtp`, and more GPUs. The error cannot occur without
a `reserve`, another process on the GPU, or a model that does not fit at all,
because pages and the prefix cache otherwise shrink to supply.

The estimate is rough. A model near the limit can pass it and exhaust GPU
memory during preparation; `run` reports that as a capacity error too, quoting
the failed allocation.

The same form applies to GPU selection, "needs 2 GPUs with 95 GiB; 1
selected; use --gpus 0-1", and to requests rejected at admission over HTTP,
whose response body names the exceeded capacity and the setting that raises
it.

## 7. Caches and the first run

Compiled kernels and tuning selections live under `[paths].kernel_cache`,
keyed by checkpoint revision, recipe version, GPU, and settings. A second
`run` with the same settings neither compiles nor tunes. Changing
`--max-concurrency`, `--gpus`, or the speculation flags can add graph shapes
and incur partial tuning.

`run` prints the resolution, the selected GPUs with the settings in effect,
and each preparation phase with its counts and elapsed time. For example:

```
[    OK    ] Resolve checkpoint.  qwen3.8-flash · dc5b8ab2 · cached  0.3s
[    OK    ] Select GPUs.  GPU 0 · concurrency 4 · up to 5 draft tokens  0.2s
[    OK    ] Inspect checkpoint.  3.6s
[    OK    ] Load weights.  15.8s
[    OK    ] Load MTP weights.  1.8s
[    OK    ] Load PLE table.  128/128  5.0s
[    OK    ] Allocate model state.  0.0s
[    OK    ] Construct graph catalog.  30/30  1.4s
[    OK    ] Prepare and tune kernels.  6.4s
[    OK    ] Measure graph memory.  30/30  5.9s
[    OK    ] Compile model forwards.  17/17  117.3s
[    OK    ] Allocate KV cache.  6.90 GiB per GPU · 270,016 tokens  0.3s
[    OK    ] Warm compiled forwards.  17/17  53.5s
[    OK    ] Prepare serving graphs.  30/30  4.2s
[    OK    ] Export native package.  /path/to/runtime-package  35.0s
```

After export, `run` replaces Ignition with Flare, which displays its banner and
model-loading stages. A package cache hit skips the preparation stages.

Preparation phases come from `ignition.progress.phase`; on a terminal each shows an activity
bar while it runs. `Select GPUs.` names the GPUs by index, followed by
`of CUDA_VISIBLE_DEVICES=...` when the caller restricted them, and reads
`speculation off` under `--no-mtp`. `Allocate KV cache.` reports the pages
allocated. A console too narrow for a row shortens its detail text and keeps
the status and elapsed time. `--verbose` adds each phase's detail lines.

## 8. Model resolution

A recipe declares each qualified checkpoint as an alias, a repository id, a
pinned revision, and an architecture. `MODEL` resolves in this order:

1. An existing directory is a checkpoint directory.
2. A string containing `/` is a Hugging Face repository id, optionally
   followed by `@revision`. A declared repository at its pinned revision, or
   with no revision, is the qualified checkpoint.
3. Anything else is an alias.

Any other directory or repository is served by the recipe whose architecture
its `config.json` names. For a repository, Flare reads `config.json` and the
safetensors index from the hub without downloading tensors; without
`@revision` it uses the repository's current commit and reports it. The
estimate sizes weights from that index and the recipe's placement. The
recipe's alias remains a served name. An
architecture no recipe supports, or an alias no recipe declares, is a usage
error listing the supported aliases; a repository the hub cannot provide is a
fetch error.

The hub cache follows the Hugging Face hub library's convention unless
`[paths].hub_cache` is set. An alias downloads its pinned revision.

## 9. Recipe obligations

For the command line to stay this small, every recipe declares:

- aliases, repository ids, revisions, architectures;
- each revision's stored tensor bytes by placement, device or mapped host
  memory, so the estimate needs no cached snapshot; another checkpoint's placement
  scales from these by its index's total size;
- supported tensor-parallel sizes and the concurrency range;
- a memory cost model: bytes per cached token, bytes per snapshot, bytes of
  live state per active request, and a graph-and-scratch estimate for a cold
  host;
- a prefill chunk and the decode capacities;
- its speculation settings, whether drafts can be copied from request history,
  the draft model's share of weights and pages, and its costs with speculation
  disabled;
- image limits.

Model-specific validation lives here, not in the entrypoint. A recipe that
cannot honor the settings reports a capacity error in the section 6 form.

## 10. Non-goals

Topology files, model families, launch override blocks, per-host environment
maps, container and multi-node discovery, multiple models in one process,
and serving an architecture no recipe supports.
Multi-node tensor parallelism across DGX Spark hosts, if it arrives, is a
separate command with its own discovery, not a layer inside `run`.

## Runtime capacity selection

`flare run PACKAGE --max-concurrency N --mtp-draft-tokens K` loads only the
captured graph variants needed for N concurrent requests and at most K drafts
per round. `--mtp-adaptive-verification` enables width selection;
`--no-mtp-adaptive-verification` loads only width K+1. With omitted flags, the
package supplies defaults. K=0 selects ordinary prefill, decode, resume, and
prefix-cache graphs without draft generation or draft-state maintenance. Adaptive
width-one rounds retain draft state for later wider rounds. MTP weights remain
loaded. Capacity increases and missing variants are rejected before GPU
allocation. Fixed storage retains its prepared layout. Continuation rows and KV
pages are sized after graph upload within their reserved virtual ranges; reducing
concurrency or verification width leaves more physical memory for KV pages.

`--embedding-table-storage disk|ram` selects storage for PLE and Engram tables.
Both modes deliver compact GPU rows to the same prepared graphs; disk is the default.
