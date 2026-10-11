# Flare command-line interface

Flare serves the model architectures its recipes support on NVIDIA SM120 and
SM121 GPUs; each recipe pins qualified default checkpoints. Its command line accepts only what the engine cannot determine by itself,
expressed in the user's terms. A setting the engine can read from the
checkpoint, measure on the host, or derive from another setting is not a
parameter anywhere: not a flag, not a configuration key, not a debug override.

## 1. Principles

- **Intent, not mechanism.** A parameter names a decision only the user can
  make: which GPUs, how many simultaneous requests, where to listen.
- **The engine is the recipe.** Per-model knowledge lives in `flare/recipes/`
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

```
flare run     [MODEL] [flags]     prepare and serve
flare models                      list supported checkpoints and what fits on this host
flare pull    MODEL               download a checkpoint into the hub cache
flare status  [--url URL]         summarize a running server
flare bench   [--url URL]         measure a running server
flare config                      print the effective configuration and file locations
flare skill                       print the agent guide
```

Every command has exactly one name. `MODEL` is a recipe alias, a Hugging Face
repository id with an optional `@revision`, or a local checkpoint directory;
with no `MODEL`, `run` uses `defaults.model` from the configuration file and
fails with a usage error if it is unset.

`models`, `status`, `bench`, and `config` print a report and exit. A terminal
receives the report with styled headings, labels, and status words. Redirected
output is the same text without styles, wrapping, or trailing spaces, and
`--json` replaces the report with one JSON document. Only `run` prints the
banner.

No command previews or rehearses `run`. `models` answers what fits on the
host; `run` performs the preparation, whose tuned kernels and compiled
forwards later runs reuse, and states what it allocated.

## 3. Parameters

This is the complete inventory. Everything a user can set is listed here with
its home: the command line, the configuration file, or both.

### 3.1 Command-line flags

`run` accepts these and nothing else.

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
| `--mtp-draft-tokens N` | Most draft tokens verified in one round, 0 through 7. Graphs are captured for every count from 0 through `N`, and verification adapts among them from measured acceptance. | The recipe's count: 3 for Qwen, 5 for GLM |
| `--no-mtp-adaptive-verification` | Verify exactly the draft-token count every round. Only that count's graphs are captured. | Adaptive |
| `--mtp-graph-budget N` | Most speculative graphs captured across batch sizes, at least one per batch size and at most 56. Small batches and deep drafts are captured first. | 8 |
| `--mtp-proposal-source {mtp,copy}` | Where drafts come from: the checkpoint's MTP module, or `copy` for continuations of the request's own history, which loads no draft model. Qwen only. | `mtp` |
| `--history-drafts` | Also draft continuations copied from the request's own history. Qwen only. | off |
| `--no-compile` | Development only: capture graphs without compiling forwards. A recipe never sets it. | off |

The speculation flags exist on the command line only. `--no-mtp` excludes the
others. The `Select GPUs.` startup line states the settings in effect: `up to
N draft tokens` while verification adapts, `exactly N draft tokens` when it is
fixed.

`models`, `status`, `bench`, and `config` accept `--json`.
`status` and `bench` accept `--url`, defaulting to the configured host and
port. Every command that reads the configuration file accepts `--config`.

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
| `--headroom`, `--snapshot-budget`, `conversation_cache` | Not a parameter | Headroom is an engine constant, 1 GiB: the free memory kept after the final KV allocation for the transient working memory of compiled warm-up and graph capture, measured at 0.83 GiB for one Qwen rank. Serving allocates nothing, and the CUDA context is counted in graphs and scratch. The prefix cache is sized from `--max-concurrency`. |
| `--mtp`, width lists, fixed width, proposal source, copy | Advanced flags | Speculation is distribution-preserving and faster when the recipe prepares it, so every recipe with an MTP module enables it and adapts the draft count from measured rounds. `--no-mtp` and the `--mtp-*` flags override the recipe: to measure a setting, to pin one, or to free the draft model's memory on a host where it does not fit. One draft-token count sets the depth and `--no-mtp-adaptive-verification` pins it; there is no list of counts, no separate fixed count, and no flag that turns speculation on. |
| `--max-active` | Renamed | `--max-concurrency`. |
| `--bind`, `--listen` | Renamed | `--host` and `--port`. |
| `--thinking`, `--template-defaults FILE`, reasoning effort, sampling defaults, `--max-tokens` | Not command-line parameters | These are default request parameters, a separate mechanism not yet specified. Launch flags do not set per-request behavior. |
| `--memory-report FILE` | Removed | `run` prints what it allocates and `status` reports the pool of a running server. |
| `--prepare-only`, `prepare`, `plan` | Removed | `run` performs the preparation and leaves the same caches; `models` gives the estimate of what fits. A separate preview would print figures that allocation does not follow. |
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

### 5.1 `flare run`

1. Resolve `MODEL`. Unknown: usage error listing supported aliases.
2. Discover GPUs through NVML. Apply `--gpus`, or select the first GPUs whose
   estimate fits at the smallest supported tensor-parallel size.
3. Estimate memory from the recipe's declared sizes and the GPUs' free memory:
   weights from the checkpoint index, a rough figure for graphs and scratch,
   live state, the prefix-cache floor, and minimum pages. The estimate does not
   fit: capacity error with options.
4. Ensure the snapshot is cached; download with progress if not. Failure:
   fetch error.
5. Load, tune, warm, capture, allocate KV pages, freeze. Failure: preparation
   error. Exhausted GPU memory, fewer than the minimum pages for
   `--max-concurrency`, or less than `memory.reserve` left free: capacity
   error.
6. Bind `--host` and `--port`. Address in use: usage error.

The Qwen recipe sizes its KV pages from the free memory it measures after graph
capture. The GLM recipe takes its page count from the estimate.

The completed `Start Helios.` line includes `listening on ADDR` in dim text before
the elapsed time. Serving prints nothing
per request. On a terminal, `run` then shows the native metrics dashboard;
with stderr redirected it prints nothing further. No flag controls the
dashboard. SIGINT and SIGTERM stop admission, give active responses five
seconds to finish, drain GPU work, release, and exit 0. During preparation
either signal ends the process immediately.

Exit codes: 0 success, 2 usage, 3 fetch, 4 capacity, 5 preparation, 6 runtime.

### 5.2 `flare models`

```
qwen3.8-flash
  checkpoint     local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2
  architecture   Qwen4ExpForConditionalGeneration
  size           95.1 GiB stored · 68.3 GiB on GPUs · 26.8 GiB PLE table in host memory
  context        262,144 tokens
  input          text and up to 4 images per request
  speculation    up to 3 draft tokens per round · MTP module
  GPUs           1, 2, or 4
  concurrency    4 by default · at most 64
  cached         yes
  fits here      1 GPU · 4 requests of about 104K tokens, or 1 of 262K

glm-5.3-flash
  checkpoint     local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd
  architecture   Glm5NextForConditionalGeneration
  size           165.5 GiB stored · 165.1 GiB on GPUs
  context        1,048,576 tokens
  input          text
  speculation    up to 5 draft tokens per round · MTP module
  GPUs           2 or 4
  concurrency    4 by default · at most 64
  cached         no · downloaded by the first run
  fits here      no
                 glm-5.3-flash needs 2 or 4 SM120 or SM121 GPUs; this host has 1

MODEL also accepts org/name[@revision] on Hugging Face or a checkpoint directory of a listed architecture.
```

Each block describes one recipe through the checkpoint its alias names: the
pinned repository and revision, the `config.json` architecture the recipe
serves, where the stored tensors reside, the checkpoint's context, accepted
inputs, speculation, and the supported GPU counts and concurrency range. The
alias that `defaults.model` names is marked `defaults.model`. `fits here` is
an estimate at the recipe's default concurrency and speculation: the GPUs it
takes, about how many tokens each concurrent request gets, and how many
requests could hold the full context instead. When the estimate does not fit,
it reads `no` followed by the capacity error `run` would print. `--json` emits
one object per alias with these declarations in full, including the complete
revision, and the estimate or the error.

### 5.3 `flare pull`

Download `MODEL` into the hub cache with progress: an alias's pinned revision,
or a repository at the named or current revision. Print the snapshot path.
Cached files are not re-downloaded.

### 5.4 `flare status`

Read `/metrics` and print: model, uptime, active and queued requests, decode
tokens per second over the last minute, prefix-cache hit rate and cached
tokens, free pages and snapshots, speculative acceptance when enabled.

### 5.5 `flare bench`

Run the fixed workload from `qualify/performance.py` and print time to first
token for a cold 512-token prompt, a cold 4,096-token prompt, and a
cached-prefix request; inter-token latency median and p99 at concurrency 1,
2, and 4; and decode throughput at each. Each figure carries its request
count. `bench` reports what the server delivered and makes no claim about
hardware limits. Time to first token is measured at the client; inter-token
latency and throughput use the server's per-token completion timestamps, so
tokens accepted in one speculative round show intervals near zero.

The workload covers the Qwen recipe. `bench` renders its prompts with the
checkpoint's tokenizer from the hub cache, and it imports `qualify`, which a
source checkout provides and the installed package does not.

### 5.6 `flare config`

Print the effective configuration after precedence, each value labeled with
its source, and the configuration file, hub cache, and kernel cache paths.

### 5.7 `flare skill`

Print the embedded agent guide: commands, the configuration file, how to add
a recipe, how to read a capacity error. Works offline.

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
and each preparation phase with its counts and elapsed time:

```
[    OK    ] Resolve checkpoint.  qwen3.8-flash · dc5b8ab2 · cached  0.3s
[    OK    ] Select GPUs.  GPU 0 · concurrency 4 · up to 3 draft tokens  0.2s
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
[    OK    ] Start Helios.  listening on 127.0.0.1:8000  0.8s
```

Phases come from `flare.progress.phase`; on a terminal each shows an activity
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
