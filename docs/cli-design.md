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
  typed. The recipe sets the prefill chunk; an advanced flag overrides it.
- **Memory follows the model and the host.** Weights, measured graphs and
  scratch, live state, and a prefix cache sized from `--max-concurrency` come
  first. KV pages take everything else: the GPU's free memory at startup minus
  engine headroom. The page pool holds active requests and every cached prefix,
  so memory beyond what active requests need keeps more history resident.
  Nothing asks for a fraction or a size.
- **Measure, then decide, then say so.** Capacity is estimated from the
  checkpoint index before weights load, confirmed against the allocator after
  capture, and printed. A plan that does not fit fails before any socket
  opens, with the numbers and the settings that would make it fit.
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
flare prepare [MODEL] [flags]     tune, capture, and populate caches; do not serve
flare plan    [MODEL] [flags]     print the derived plan without loading weights
flare models                      list supported checkpoints and what fits on this host
flare pull    MODEL               download a checkpoint into the hub cache
flare status  [--url URL]         summarize a running server
flare bench   [--url URL]         measure a running server
flare config                      print the effective configuration and file locations
flare skill                       print the agent guide
```

Every command has exactly one name. `MODEL` is a recipe alias, a Hugging Face
repository id with an optional `@revision`, or a local checkpoint directory;
with no `MODEL`, `run`, `prepare`, and
`plan` use `defaults.model` from the configuration file and fail with a usage
error if it is unset.

`plan`, `models`, `status`, `bench`, and `config` print a report and exit, as
does `prepare` after its startup phases. A terminal receives the report with
styled headings, labels, and status words. Redirected output is the same text
without styles, wrapping, or trailing spaces, and `--json` replaces the report
with one JSON document. Only `run` and `prepare` print the banner.

## 3. Parameters

This is the complete inventory. Everything a user can set is listed here with
its home: the command line, the configuration file, or both.

### 3.1 Command-line flags

`run`, `prepare`, and `plan` accept these and nothing else.

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
| `--prefill-chunk-size N` | Most prompt tokens processed in one prefill step, a multiple of 64. A larger chunk reaches a long prompt's first token sooner; while it runs, other active requests wait for their next token. `run`, `prepare`, and `plan`. Command line only. | The recipe's chunk |
| `--no-compile` | Development only: capture graphs without compiling forwards. `run` and `prepare` only. A recipe never sets it. | off |

`prepare`, `plan`, `models`, `status`, `bench`, and `config` accept `--json`.
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
weights, graphs, and live state. The plan prints its size and cost, about
117 MiB per state on the Qwen3.8 recipe, for information. Each active request
additionally holds three states while it runs; the plan counts those as live
state. Under memory pressure, KV pages shrink first and the prefix cache has a
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
| `--mtp`, `--no-mtp`, widths, fixed width, proposal source, copy | Not a parameter | Speculation is distribution-preserving and faster when the recipe prepares it, so it is on. Width is adaptive from measured rounds. Disabling or pinning it is a qualification need, served by the Python preparation API, not the command line. |
| `--max-active` | Renamed | `--max-concurrency`. |
| `--bind`, `--listen` | Renamed | `--host` and `--port`. |
| `--thinking`, `--template-defaults FILE`, reasoning effort, sampling defaults, `--max-tokens` | Not command-line parameters | These are default request parameters, a separate mechanism not yet specified. Launch flags do not set per-request behavior. |
| `--memory-report FILE` | Replaced | `plan --json` prints the estimate; `prepare --json` prints the measured figures. |
| `--prepare-only` | Replaced | `flare prepare`. |
| `--revision` | Part of `MODEL` | `org/name@revision`. An alias uses the revision its recipe pins. |
| `--dtype`, `--quantization`, `--kv-cache-dtype`, `--block-size`, kernel backends, `cudagraph_mode`, `--enable-prefix-caching`, `--enable-chunked-prefill` | Not a parameter | Recipe decisions, qualified. |
| `--served-model-name` | Not a parameter | Requests may name the alias or the repository id; `/v1/models` lists both. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Not a parameter | `--max-concurrency` and the derived chunk. |
| `queue_capacity`, `event_capacity`, `frontend_workers`, `max_body_bytes` | Not a parameter | Derived: queue is four times concurrency, frontend workers equal concurrency, event capacity is fixed, body limit is the recipe's image count times its per-image limit plus 1 MiB. |
| `compile_workers` | Not a parameter | Derived from CPU count. |
| `CUDA_HOME` | Not a parameter | Located from installed packages, then `[paths].cuda`. |
| `CUDA_VISIBLE_DEVICES` | Read only to interpret `--gpus` | `--gpus` selects. If the caller set it, integer entries are NVML indices, other entries are GPU UUID prefixes, `--gpus` indices are relative to that enumeration, and `plan` says so. Flare then exports the selected GPUs' UUIDs to its own processes. |
| `--api-key` | Configuration only | Secrets do not go on command lines. |
| `--seed`, `--temperature`, stop sequences | Request fields | Per request. |

## 4. Derived settings

| Setting | Derived from |
|---|---|
| Tensor-parallel size | Count of `--gpus`, or the smallest recipe-supported size whose per-rank weights leave room for the recipe default concurrency |
| Context limit | The checkpoint maximum |
| KV pages | All remaining supply: free GPU memory minus engine headroom, `memory.reserve`, weights, measured graphs and scratch, live state, and the prefix cache; divided by the recipe's per-token page cost. Active requests are guaranteed an equal share, up to the checkpoint's maximum context; the rest holds cached prefixes |
| Prefix cache | Eight states per `--max-concurrency` request, at most a quarter of the memory left after weights, graphs, and live state; floor 8 states under pressure |
| Live state | The recipe's state rows for `--max-concurrency` requests: one row per request slot, the recipe's speculative rows per slot, and three checkpoint rows per active request |
| Decode graph capacities | 1 through `--max-concurrency`, rounded to the recipe's captured set |
| Prefill chunk | The recipe's chunk, 4,096 tokens for both recipes; `--prefill-chunk-size` overrides |
| Speculative catalog and widths | Recipe; width adaptive from measured rounds |
| Kernel selections | Autotuned at prepare; cached by checkpoint revision, recipe version, GPU, and plan |
| HTTP queue, workers, event capacity, body limit | `--max-concurrency` and the recipe's image limits |
| Compile workers | CPU count |
| Generation defaults | Checkpoint `generation_config.json`, pending the default-request-parameter mechanism |
| Template defaults | The checkpoint template's own defaults, pending the same mechanism |
| Served names | Recipe alias and repository id |

## 5. Command behavior

### 5.1 `flare run`

1. Resolve `MODEL`. Unknown: usage error listing supported aliases.
2. Discover GPUs through NVML. Apply `--gpus` or select the first that fit.
   None fit: capacity error.
3. Compute the plan from the checkpoint index, the recipe cost model, the
   last measured graph sizes on this host if any, and free GPU memory. Weights,
   graphs, live state, the prefix-cache floor, and minimum pages do not
   fit: capacity error with options.
4. Ensure the snapshot is cached; download with progress if not. Failure:
   fetch error.
5. Load, tune, warm, capture, size pages against the measured allocator,
   freeze. Failure: preparation error. If measured graphs exceed the estimate,
   pages shrink and the corrected figures are printed; if the minimum for
   `--max-concurrency` can no longer be honored, capacity error.
6. Bind `--host` and `--port`. Address in use: usage error.

The completed `Start Helios.` line includes `listening on ADDR` in dim text before
the elapsed time. Serving prints nothing
per request. On a terminal, `run` then shows the native metrics dashboard;
with stderr redirected it prints nothing further. No flag controls the
dashboard. SIGINT and SIGTERM stop admission, give active responses five
seconds to finish, drain GPU work, release, and exit 0. During preparation
either signal ends the process immediately.

Exit codes: 0 success, 2 usage, 3 fetch, 4 capacity, 5 preparation, 6 runtime.

### 5.2 `flare prepare`

Steps 1 through 5 of `run`, then print the plan with measured figures and
exit. Exists so a machine can pay the tuning and capture cost once.

### 5.3 `flare plan`

Steps 1 through 3 of `run`, then print and exit. Does not require the
snapshot to be cached and initializes no CUDA context beyond NVML.

```
qwen3.8-flash
  checkpoint      local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2
  architecture    Qwen4ExpForConditionalGeneration
  GPUs            0 · RTX PRO 6000 Blackwell · tensor-parallel 1
  concurrency     4 requests
  context         262,144 tokens
  prefill chunk   4,096 tokens
  host memory     26.8 GiB · PLE table mapped from the checkpoint

  memory                GiB
  ───────────────────────────────────────────────────────────────────────────────────────────
  weights              68.3   checkpoint tensors
  graphs and scratch    7.0   estimate; never prepared on this host
  live state            5.2   4 active requests
  prefix cache          3.4   30 states for prompt reuse
  KV pages             10.7   418,880 tokens; full context for 4 requests would take 26.8 GiB
  headroom              1.0   kept free through warm-up and graph capture
  reserve               0.0   [memory] reserve
  ───────────────────────────────────────────────────────────────────────────────────────────
  free                 95.6   GPU 0

  guarantees 4 × 104,704 tokens at full concurrency; a single request up to 262,144
```

The memory rows account for the GPU's free memory: KV pages take what the rows
above them, headroom, and the reserve leave. The KV pages row adds what full
context for every active request would take when the pool is smaller than that.
Weights include the draft model that speculation loads. Live state holds 45
rows of 117 MiB at concurrency 4: a null row, four request rows, seven
speculative rows per request, and three checkpoint rows per active request.
The prefix cache would hold 32 states at this concurrency; a quarter of the
14.1 GiB left for cache and pages limits it to 30.
After a preparation on the host, the weights and graphs rows show the
measured figures and the graphs row reads `measured on this host`. With
several GPUs the table is headed `memory per GPU` and its last row is the
least free memory among them. When `CUDA_VISIBLE_DEVICES` is set, a line under
`GPUs` says that indices follow it.

### 5.4 `flare models`

```
qwen3.8-flash
  checkpoint     local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2
  architecture   Qwen4ExpForConditionalGeneration
  size           95.1 GiB stored · 68.3 GiB on GPUs · 26.8 GiB PLE table in host memory
  context        262,144 tokens
  input          text and up to 4 images per request
  speculation    up to 7 draft tokens per step
  GPUs           1, 2, or 4
  concurrency    4 by default · at most 64
  cached         yes
  fits here      1 GPU · 4 requests of 104,704 tokens, or 1 of 262,144

glm-5.3-flash
  checkpoint     local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd
  architecture   Glm5NextForConditionalGeneration
  size           165.5 GiB stored · 165.1 GiB on GPUs
  context        1,048,576 tokens
  input          text
  speculation    none
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
the plan at the recipe's default concurrency: the GPUs it takes, the tokens
each concurrent request is guaranteed, and how many requests could hold the
full context instead. When no plan fits, it reads `no` followed by the
capacity error `flare plan` would print. `--json` emits one object per alias
with these declarations in full, including the complete revision, and the plan
or the error.

### 5.5 `flare pull`

Download `MODEL` into the hub cache with progress: an alias's pinned revision,
or a repository at the named or current revision. Print the snapshot path.
Cached files are not re-downloaded.

### 5.6 `flare status`

Read `/metrics` and print: model, uptime, active and queued requests, decode
tokens per second over the last minute, prefix-cache hit rate and cached
tokens, free pages and snapshots, speculative acceptance when enabled.

### 5.7 `flare bench`

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

### 5.8 `flare config`

Print the effective configuration after precedence, each value labeled with
its source, and the configuration file, hub cache, and kernel cache paths.

### 5.9 `flare skill`

Print the embedded agent guide: commands, the configuration file, how to add
a recipe, how to read a capacity error. Works offline.

## 6. Capacity errors

A plan that does not fit fails before loading weights with the requirement,
what is available, and the options that satisfy it:

```
error: needs 68.3 GiB weights + 7.0 GiB graphs + 5.2 GiB live state + 0.9 GiB prefix cache
       + 0.8 GiB minimum pages (4 × 8,192 tokens) = 82.2 GiB;
       has 95.6 GiB free − 1.0 GiB headroom − 21.0 GiB reserve = 73.6 GiB.
       Options: [memory] reserve = "12G".
```

The requirement is the smallest plan the engine accepts: the prefix cache at
its floor and 8,192 tokens of pages per active request. Every listed option
satisfies it on its own; the candidates are a smaller `reserve`, a lower
`--max-concurrency`, and more GPUs. The error cannot occur without a `reserve`, another process on the GPU,
or a model that does not fit at all, because pages and the prefix cache
otherwise shrink to supply.

The same form applies to GPU selection, "needs 2 GPUs with 95 GiB; 1
selected; use --gpus 0-1", and to requests rejected at admission over HTTP,
whose response body names the exceeded capacity and the setting that raises
it.

## 7. Caches and the first run

Compiled kernels and tuning selections live under `[paths].kernel_cache`,
keyed by checkpoint revision, recipe version, GPU, and plan. A second `run`
with the same plan neither compiles nor tunes. Changing `--max-concurrency` or
`--gpus` can add graph shapes and incur partial tuning; `prepare` prints how
many selections were reused and how many measured.

`run` and `prepare` print the resolution and the plan, then each preparation
phase with its counts and elapsed time, then the measured sizing:

```
[    OK    ] Resolve checkpoint.  qwen3.8-flash · dc5b8ab2 · cached  0.3s
[    OK    ] Plan memory.  1 GPU · concurrency 4 · 271,360 KV tokens  0.2s
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
[    OK    ] Measure memory.  4,219 pages · 270,016 tokens · 32 cached states  0.0s
[    OK    ] Start Helios.  listening on 127.0.0.1:8000  0.8s
```

Phases come from `flare.progress.phase`; on a terminal each shows an activity
bar while it runs. `Measure memory.` reports the pages allocated after graph memory
was measured, which can differ from the plan. `--verbose` adds each phase's
detail lines.

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
`@revision` it uses the repository's current commit and reports it. The plan
sizes weights from that index and the recipe's placement, then the first
`prepare` measures them. The recipe's alias remains a served name. An
architecture no recipe supports, or an alias no recipe declares, is a usage
error listing the supported aliases; a repository the hub cannot provide is a
fetch error.

The hub cache follows the Hugging Face hub library's convention unless
`[paths].hub_cache` is set. An alias downloads its pinned revision.

## 9. Recipe obligations

For the command line to stay this small, every recipe declares:

- aliases, repository ids, revisions, architectures;
- each revision's stored tensor bytes by placement, device or mapped host
  memory, so `plan` needs no cached snapshot; another checkpoint's placement
  scales from these by its index's total size;
- supported tensor-parallel sizes and the concurrency range;
- a memory cost model: bytes per cached token, bytes per snapshot, bytes of
  live state per active request, and a graph-and-scratch estimate for a cold
  host;
- a prefill chunk, the decode capacities, and the speculative catalog;
- image limits.

Model-specific validation lives here, not in the entrypoint. A recipe that
cannot honor a plan reports a capacity error in the section 6 form.

## 10. Non-goals

Topology files, model families, launch override blocks, per-host environment
maps, container and multi-node discovery, multiple models in one process,
and serving an architecture no recipe supports.
Multi-node tensor parallelism across DGX Spark hosts, if it arrives, is a
separate command with its own discovery, not a layer inside `run`.
