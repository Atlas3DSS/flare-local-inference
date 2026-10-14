# Ignition preparation and Flare serving commands

Flare serves the model architectures its recipes support on NVIDIA SM120 and
SM121 GPUs. Recipes declare default checkpoints and supported preparation
capacities. The command line selects models, GPUs, concurrency, storage, and
listeners. The engine derives memory and execution geometry from those choices;
advanced preparation flags expose the explicit overrides listed below.

See [native inference APIs](api.md) for Chat Completions, raw Completions,
Anthropic Messages, stateless Responses, authentication, and client configuration.

## 1. Principles

- **Intent, not mechanism.** A parameter names a decision only the user can
  make: which GPUs, how many simultaneous requests, where to listen.
- **The engine is the recipe.** Per-model knowledge lives in `ignition/recipes/`
  as qualified code. A recipe declares its checkpoints, their pinned revisions,
  supported tensor-parallel sizes, concurrency range, memory cost model, and
  template defaults. The command line never needs to know which model it runs.
- **Derive execution geometry.** Context length, page count, GPU prefix-cache
  size, snapshot slots, graph capacities, worker counts, and body limits follow
  the recipe, host, and selected capacities. Advanced flags override the recipe's
  prefill chunk and speculation settings.
- **Memory follows the model and the host.** Weights, graphs and
  scratch, live state, and a prefix cache sized from `--max-concurrency` come
  first. KV pages take everything else: the GPU's free memory at startup minus
  engine headroom. The page pool holds active requests and every cached prefix,
  so memory beyond what active requests need keeps more history resident.
  The configuration's GPU reserve leaves memory for processes started later.
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
allocation contents, graph descriptions, and frontend assets to one `.safetensors` file
that must not exist, then exits without opening a listener. `run` caches the resulting package in
`paths.package_cache` and replaces its Python process with the Rust `flare` executable. Cache keys
include checkpoint identity, preparation settings, dependency versions, and
preparation source contents, signing-key fingerprint, publisher namespace, and
source URL. Qwen, GLM, and DeepSeek provide native package storage support.
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

### 3.1 Ignition command-line flags

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
| `--official-source URL` | Published repository URL under the configured publisher namespace | `[publishing] official_source` |
| `--publisher-keys PATH` | Additional publisher profiles passed to Flare; repeatable, `run` only | Embedded keys plus the local development profile when applicable |
| `--trust-store PATH` | Remembered execution trust passed to Flare; `run` only | `~/.config/flare/trust.toml` |
| `--verify-weights` | Check the complete stored allocation/table region; `run` only | off |

Advanced flags appear in their own section of `--help`. Recipe values apply when flags are omitted; explicit flags take precedence.

| Flag | Meaning | Default |
|---|---|---|
| `--prefill-chunk-size N` | Most prompt tokens processed in one prefill step, a multiple of 64. A larger chunk reaches a long prompt's first token sooner; while it runs, other active requests wait for their next token. Command line only. | The recipe's chunk |
| `--mixed-prefill` | Capture mixed prompt/decode graphs alongside separate graphs. Supported by Qwen with concurrency of at least two, with or without speculation. Command line only; existing packages retain their captured catalog. | Disabled |
| `--no-mtp` | Disable multi-token speculation: load no draft model, keep no verification rows, and decode one token per step. | Speculation on |
| `--mtp-draft-tokens N` | Most draft tokens verified in one round, up to 7. Graphs are captured for every count from 0 through `N`, and verification adapts among them from measured acceptance. | 5 |
| `--no-mtp-adaptive-verification` | Default to fixed-width serving. Capture still includes every smaller width at every batch capacity. | Adaptive |
| `--mtp-proposal-source {mtp,copy}` | Where drafts come from: the checkpoint's MTP module, or `copy` for continuations of the request's own history, which loads no draft model. Qwen only. | `mtp` |
| `--history-drafts` | Also draft continuations copied from the request's own history. Qwen only. | off |
| `--no-compile` | Development only: capture graphs without compiling forwards. A recipe never sets it. | off |

The speculation flags exist on the command line only. `--no-mtp` excludes the
others. The `Select GPUs.` startup line states the settings in effect: `up to
N draft tokens` while verification adapts, `exactly N draft tokens` when it is
fixed.

Both Ignition commands accept `--config`.

### 3.2 Native serving flags

`flare run PACKAGE` loads an exported file without the Ignition configuration
reader. `flare --help` lists commands; `flare run --help`, `flare run -h`, and
`flare help run` describe every serving option, its defaults, and accepted values.
Help also works after the package path and other options, without opening the
package or initializing CUDA. Options may precede or follow `PACKAGE`;
value-taking options accept both `--option VALUE` and `--option=VALUE`.
Help exits with status 0; usage errors go to stderr with status 2. Its options are:

| Flag | Meaning | Default |
|---|---|---|
| `--publisher-keys PATH` | Additional JSON verification profiles; importing a key does not approve execution | Embedded profiles |
| `--dangerously-approve-key FINGERPRINT` | Explicitly approve this full lowercase SHA-256 key fingerprint; requires the matching publisher flag | Interactive first use or remembered trust |
| `--dangerously-accept-publisher NAMESPACE` | Accept this exact publisher namespace; requires the matching key flag | Interactive first use or remembered trust |
| `--trust-store PATH` | Persistent key/namespace acceptance file | `~/.config/flare/trust.toml` |
| `--verify-weights` | Verify the complete stored allocation/table region before CUDA initialization | off |
| `--gpus N[,N...]` | Selected GPUs in rank order; indices follow an inherited `CUDA_VISIBLE_DEVICES` list when present | `0` |
| `--bind IP:PORT` | Listener address | `127.0.0.1:8000` |
| `--api-key-file PATH` | File containing the serving API key | No authentication |
| `--max-concurrency N` | Reduce concurrency within captured batch capacities | Package default |
| `--mtp-draft-tokens N` | Reduce the maximum draft count; zero selects ordinary execution without draft generation, absorption, or draft-state maintenance | Package default |
| `--mtp-adaptive-verification` / `--no-mtp-adaptive-verification` | Adaptive or fixed selection among captured widths | Package default |
| `--embedding-table-storage disk\|ram` | PLE and Engram table storage; both modes deliver compact GPU rows | `disk` |

Selected concurrency and draft depth cannot exceed prepared capacity.
Concurrency three requires the batch-four bucket. Fixed verification loads
width `N+1` for each required batch; adaptive verification loads the permitted
widths. Missing variants fail before GPU allocation. Continuation rows and
KV pages receive physical backing after graph upload, within fixed virtual
extents. Reducing execution state can leave more memory for KV pages.

### Signing and local development

Final exports always carry signatures and encrypted executable resources.
Configure published exports with all three values:

```toml
[publishing]
signing_key = "/private/flare-signing-key.pem"
namespace = "local-inference-lab"
official_source = "https://huggingface.co/local-inference-lab/foobar"
```

The key must be an unencrypted Ed25519 PKCS#8 PEM file. Keep it private;
runtime publisher profiles contain only its verification key. `--official-source`
overrides the configured repository URL. A partial publishing configuration is
an error and never falls back to local credentials.

With no publishing configuration, Ignition creates and reuses
`development-signing-key.pem` beside its configuration file, with owner-only
permissions. It also writes `development-publisher.json` containing the public
verification profile. These packages use the explicit namespace and source
`local-development`, with no claim to be an official published checkpoint.

`ignition run` supplies Flare with the development profile when applicable and
automatically passes the exact fingerprint and namespace from its export
credentials through Flare's two dangerous-approval arguments. It does not infer
approval from the cached package's claimed publisher. An unrecognized key still
prints its first-use warning; subsequent loads remember that exact pair. There
is no unsigned runtime mode and no blanket approval for development keys.

Publisher identity, key, source, and verification disclosures appear only on
first acceptance of a signing-key fingerprint and publisher namespace. On a
terminal, the banner animation finishes before those disclosures and the
acceptance prompt. Remembered acceptance suppresses the disclosures, while
every load still verifies signatures and executable resources.

To load a locally exported package directly, supply its public profile:

```sh
flare run model.safetensors \
  --publisher-keys ~/.config/flare/development-publisher.json --gpus 0
```

First interactive use requires typing `local-development`. For unattended
loading, supply both dangerous-approval flags with the exact fingerprint and
namespace. `flare trust revoke FINGERPRINT --trust-store PATH` removes acceptance;
omitting the path uses the default trust store. No official release key is
required for independent preparation and signing.

### 3.3 Configuration file

`~/.config/flare/config.toml`. All keys optional. Written on first `run` with
the resolved values present as comments.

```toml
[server]
host = "127.0.0.1"
port = 8000
api_key = ""                      # Empty: no authentication.

[defaults]
model = "qwen3.8-flash"
max_concurrency = 4

[memory]
reserve = "0"                     # GPU memory left for processes started later.

[models."qwen3.8-flash"]          # Per-model overrides of [defaults].
max_concurrency = 2

[paths]
hub_cache = "~/.cache/huggingface/hub"
kernel_cache = "~/.cache/flare/kernels"
package_cache = "~/.cache/flare/packages"
cuda = "/opt/cuda"                # Used when installed packages provide no toolkit.
```

`[models.<alias>]` overrides `[defaults]`; flags override both.

Default request parameters, meaning the values a request receives for fields
it does not set, such as thinking, reasoning effort, sampling, and output
limit, are a separate mechanism that this specification does not yet define.
Requests receive the checkpoint's template defaults, the recipe's manually
verified model-card sampling defaults, and the engine's output limit.
Sampling defaults are explicit package metadata; config-file sampling values
do not select them. Explicit request values override those defaults.

`memory.reserve` exists for one case: a process that starts after Flare and
needs GPU memory. Processes already running when Flare starts need no
setting, because supply is free memory rather than card size. It lives in the configuration file and is exported as the package's serving
reserve.

The GPU prefix cache is derived. It holds model states at instruction,
prompt, and answer boundaries so that later requests repeating a prefix skip
recomputing it. A finished exchange can retain its prompt and answer states, so the
engine exposes up to eight snapshot slots per concurrent request, about four
recent exchanges each. Snapshot slots share physical backing with KV pages and
acquire it only when retaining a checkpoint. Unpinned checkpoints are evicted
to admit requests. If active requests or transfers own all suitable backing,
the engine skips optional checkpoint retention. There is no dedicated snapshot
memory fraction or minimum resident checkpoint count.

### 3.4 Configuration boundaries

Configuration assigns each decision to one owner:

| Decision | Owner or interface |
|---|---|
| Checkpoint and revision | `MODEL`, including a local directory or `org/name@revision` |
| GPUs and TP size | `--gpus`; rank count follows the selected list |
| Concurrent requests | `--max-concurrency` |
| Listener | Ignition `--host` and `--port`; native Flare `--bind` |
| Context limit | Checkpoint maximum; admission still requires sufficient shared cache capacity |
| KV pages and GPU prefix-cache rows | Derived from available memory, graph cost, and concurrency |
| GPU memory for later processes | `[memory] reserve` |
| Preparation headroom | 1 GiB for compiled warmup and capture |
| Serving driver margin | 64 MiB after graph upload |
| Prefill chunk | Recipe default, overridden by `--prefill-chunk-size` |
| Speculation | Recipe default, overridden by `--no-mtp` and the advanced speculation flags |
| Weight format, KV precision, page geometry, kernel plans | Checkpoint metadata and recipe preparation |
| Queue, frontend workers, event capacity, body limit | Concurrency and recipe image limits |
| Compiler workers | Host CPU capacity |
| CUDA toolkit | Installed packages, then `[paths] cuda` |
| Served names | Recipe alias and repository ID, exposed through `/v1/models` |
| API key | Ignition configuration or native `--api-key-file`; excluded from packages |
| Seed, sampling controls, output limit, and stops | Request fields, with declared defaults for missing values |

Explicit request values take precedence over defaults. Speculation's benefit
depends on measured acceptance and round cost. Reducing the draft limit at
load time retains draft weights; freeing those weights requires preparation
with `--no-mtp`.

Host prefix retention is specified separately in the proposed
[prefix-cache offload design](prefix-cache-offload.md). Its host-memory and
disk settings require the implementation and measurements defined there.

## 4. Derived settings

| Setting | Derived from |
|---|---|
| Tensor-parallel size | Count of `--gpus`, or the smallest recipe-supported size whose estimate fits the first GPUs with enough free memory |
| Context limit | The smaller of the prepared context ceiling and loaded KV capacity after per-request draft-page storage; admission also checks available shared capacity |
| KV pages | All remaining supply after engine headroom, `memory.reserve`, weights, graphs, scratch, and execution rows. Retained snapshot rows borrow page backing; the load display reports KV capacity before checkpoint retention |
| Prefix cache | Up to eight snapshot slots per `--max-concurrency` request, acquired on demand from shared KV backing and evicted under pressure |
| Live state | One null row plus the recipe's execution rows per active request slot; GLM verification commits into one live row |
| Decode graph capacities | 1 through `--max-concurrency`, rounded to the recipe's captured set |
| Prefill chunk | The recipe's chunk, 4,096 tokens for Qwen, GLM, and DeepSeek; `--prefill-chunk-size` overrides |
| Speculative catalog and draft counts | Recipe, overridden by `--no-mtp` and the `--mtp-*` flags; the draft count adapts from measured rounds unless fixed |
| Kernel selections | Autotuned during preparation; cached by checkpoint revision, recipe version, GPU, and settings |
| HTTP queue, workers, event capacity, body limit | `--max-concurrency` and the recipe's image limits |
| Compile workers | CPU count |
| Sampling defaults | Complete Ignition recipe declaration, manually verified against the model card and exported in typed Protobuf fields |
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

Preparation records cache layouts and maximum virtual extents. Helios sizes
physical row and page counts from selected execution capacities and free
memory after graph upload. Insufficient memory or a recorded-address collision
fails native startup.

Serving prints no per-request output. SIGINT and SIGTERM stop admission,
give active responses five seconds to finish, drain GPU work, and exit 0.
During preparation either signal ends Ignition immediately.

Preparation exit codes: 0 success, 2 usage, 3 fetch, 4 capacity, 5 preparation.
Native startup and runtime failures exit 1.

### 5.2 `ignition export`

Resolve the model and preparation settings as for `run`, then prepare and write
a runtime package to `--output FILE`, appending `.safetensors` if needed.
The destination must not exist. Export
closes the prepared model and exits without starting a listener.

### 5.3 `flare run`

Animate the Flare banner during loading on a terminal. Validate the package,
initialize GPUs, allocate model memory, load weights and compiled kernels,
instantiate graphs, allocate caches, load frontend assets, and start HTTP.
Loading performs no kernel compilation or graph capture. Helios reports each stage with a ten-column bracketed
progress bar, a result, and elapsed time.
Terminal bars use cyan completed segments, dark gray remaining segments, and
half-cell boundaries. Stages without a known total animate a cyan-to-gray
gradient across all ten cells.
The `Verify package` stage times signed metadata authentication before model
loading. `--verify-weights` adds a timed `Verify stored data` stage after publisher
acceptance. Executable resources are verified as part of kernel and graph
preparation.
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

`run` prints checkpoint resolution, selected GPUs, preparation settings, and
each phase's counts and elapsed time. The phases include weight loading,
`Prepare embedding tables`, kernel tuning, forward compilation, KV allocation,
graph capture, and package export. A package cache hit skips preparation.

After export, `run` replaces Ignition with Flare, which displays its banner
and native loading stages.

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
- each revision's stored tensor bytes by placement, device or auxiliary table
  storage, so the estimate needs no cached snapshot; another checkpoint's placement
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
