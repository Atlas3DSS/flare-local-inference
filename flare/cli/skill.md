# Flare agent guide

Flare serves a closed set of qualified checkpoints on NVIDIA SM120 and SM121
GPUs through an OpenAI-compatible HTTP API. The command line takes only what
the engine cannot determine: which GPUs, how many simultaneous requests, and
where to listen. Everything else is derived and printed.

## Commands

```
flare run     [MODEL] [--gpus LIST] [--max-concurrency N] [--host ADDR] [--port N]
flare prepare [MODEL] [--gpus LIST] [--max-concurrency N] [--json]
flare plan    [MODEL] [--gpus LIST] [--max-concurrency N] [--json]
flare models  [--json]
flare pull    MODEL
flare status  [--url URL] [--json]
flare bench   [--url URL] [--json]
flare config  [--json]
flare skill
```

- `MODEL` is an alias from `flare models`, a Hugging Face repository id such
  as `org/name` or `org/name@revision`, or a checkpoint directory. Without it,
  `run`, `prepare`, and `plan` use `defaults.model`. A repository or directory
  that no recipe declares is served by the recipe for its `config.json`
  architecture, under that recipe's alias as well as its own name.
- `--gpus` takes NVML indices: `0`, `0,1`, `0-3`. Their count is the
  tensor-parallel size. Without it, Flare takes the first GPUs that fit.
- `plan` loads no weights and is the fastest way to learn what a host can
  serve. `prepare` pays tuning and graph capture once and records the measured
  memory so later plans are exact. `run` prepares and then serves.
- `--prefill-chunk-size N` on `run`, `prepare`, and `plan` overrides the
  recipe's prefill chunk: the most prompt tokens processed in one step. A larger
  chunk reaches the first token of a long prompt sooner; while it runs, other
  active requests wait for their next token.
- `run`, `prepare`, and `plan` also accept `--verbose` and `--config PATH`.
  `--no-compile` on `run` and `prepare` skips forward compilation for
  development startup.
- `status` summarizes a running server from its metrics. `bench` runs a fixed
  workload against a running `qwen3.8-flash` server from a source checkout and
  prints time to first token, inter-token latency, and throughput.
- `run` shows `listening on ADDR` as detail on the completed `Start Helios.` line. SIGINT and
  SIGTERM stop admission, give active responses five seconds, release the
  GPUs, and exit 0.

Exit codes: 0 success, 2 usage, 3 fetch, 4 capacity, 5 preparation, 6 runtime.

## Configuration file

`~/.config/flare/config.toml`; every key is optional. `flare config` prints
each effective value with its source. Flags override the file, which overrides
the recipe.

```toml
[server]
host = "127.0.0.1"
port = 8000
api_key = ""                      # empty: no authentication

[defaults]
model = "qwen3.8-flash"
max_concurrency = 4

[memory]
reserve = "0"                     # GPU memory left for processes that start later

[models."qwen3.8-flash"]          # per-model overrides of [defaults]
max_concurrency = 2

[paths]
hub_cache = "~/.cache/huggingface/hub"
kernel_cache = "~/.cache/flare/kernels"
cuda = "/opt/cuda"                # used when no toolkit is found in installed packages
```

With `api_key` set, every route except `/health` requires
`Authorization: Bearer KEY`. Unknown keys are usage errors: context length,
page counts, cache sizes, memory fractions, and speculation settings are not
parameters.

Flare reads no environment variables except `CUDA_VISIBLE_DEVICES`, which
restricts and renumbers the GPUs that `--gpus` indexes; `plan` reports it.

## Reading a plan

```
needs    weights + graphs + live + cache (N states) + pages (C × context) = total
has      free − headroom − reserve = supply
```

`needs` is what every active request at the checkpoint's maximum context would
take. `has` is the GPU's free memory minus a fixed 1 GiB headroom and
`memory.reserve`. KV pages take all of `has` that weights, graphs, live state,
and the prefix cache leave; pages beyond the active requests' share hold cached
prefixes. The prefix cache holds model states at prompt and answer boundaries so
later requests that repeat a prefix skip recomputing it; it is sized at eight
states per concurrent request, at most a quarter of the memory left after
weights, graphs, and live state, and shrinks to 8 states when pages would fall
below 8,192 tokens per active request. The last line states what is guaranteed:
tokens per request at full concurrency, and the longest single request.
Tensor-parallel plans print per-GPU figures.

## Reading a capacity error

```
error: needs ... GiB weights + ... GiB graphs + ... GiB live state + ... GiB prefix cache
       + ... GiB minimum pages (C × 8,192 tokens) = ... GiB;
       has ... GiB free − 1.0 GiB headroom − ... GiB reserve = ... GiB.
       Options: [memory] reserve = "12G", or --max-concurrency 2, or --gpus 0-1.
```

The first two lines are the smallest plan Flare accepts; the third is the
supply. Each listed option makes the plan fit on its own: apply one and rerun.
`No setting makes MODEL fit` means the weights exceed the selected GPUs.

HTTP rejections use the same form. A 400 names the request's prompt and output
limit against the server's context. A 429 or 503 names the exhausted request or
page capacity and the setting that raises it.

## Adding a recipe

A recipe is a package under `flare/recipes/` with two parts:

1. `recipe.py` declares a `Recipe`: each checkpoint's alias, repository id,
   pinned revision, architecture, and stored bytes by placement; the supported
   tensor-parallel sizes; the concurrency range; and per-rank costs (bytes per
   state row, bytes per KV page, and a cold-host estimate of graphs and
   scratch). It supplies `prepare(snapshot, plan, ...)`, which builds the
   bootstrap's configuration from a plan, and `usage(prepared)`, which reports
   each rank's device memory. The module must import without PyTorch.
2. `bootstrap.py` loads weights, declares and tunes kernels, captures every
   serving graph, and returns a prepared model for the native server.

Register the recipe in `flare.recipes.recipes()`. `flare prepare` rejects a
recipe whose declared row or page size differs from what its bootstrap
allocated, and records measured weights and graph memory under
`[paths].kernel_cache`.
