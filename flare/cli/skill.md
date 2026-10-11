# Flare agent guide

Flare serves a closed set of qualified checkpoints on NVIDIA SM120 and SM121
GPUs through an OpenAI-compatible HTTP API. The command line takes only what
the engine cannot determine: which GPUs, how many simultaneous requests, and
where to listen. Everything else is derived and printed.

## Commands

```
flare run     [MODEL] [--gpus LIST] [--max-concurrency N] [--host ADDR] [--port N]
flare models  [--json]
flare pull    MODEL
flare status  [--url URL] [--json]
flare bench   [--url URL] [--json]
flare config  [--json]
flare skill
```

- `MODEL` is an alias from `flare models`, a Hugging Face repository id such
  as `org/name` or `org/name@revision`, or a checkpoint directory. Without it,
  `run` uses `defaults.model`. A repository or directory
  that no recipe declares is served by the recipe for its `config.json`
  architecture, under that recipe's alias as well as its own name.
- `--gpus` takes NVML indices: `0`, `0,1`, `0-3`. Their count is the
  tensor-parallel size. Without it, Flare takes the first GPUs that fit.
- `models` loads no weights and is the fastest way to learn what a host can
  serve; its `fits here` line is a rough estimate. `run` prepares and then
  serves: it tunes kernels and captures graphs, which later runs with the same
  settings reuse from the kernel cache, and it allocates KV pages from the
  memory the GPUs have free.
- `--prefill-chunk-size N` on `run` overrides the
  recipe's prefill chunk: the most prompt tokens processed in one step. A larger
  chunk reaches the first token of a long prompt sooner; while it runs, other
  active requests wait for their next token.
- Speculation is on for every recipe with an MTP module. Advanced flags on
  `run` override the recipe: `--no-mtp` disables it;
  `--mtp-draft-tokens N` sets the most draft tokens per round, and
  verification adapts among 0 through N; `--no-mtp-adaptive-verification`
  verifies exactly N every round; `--mtp-graph-budget N` sets the number of
  speculative graphs captured. On
  `qwen3.8-flash`, `--mtp-proposal-source copy` drafts from the request's own
  history without the draft model and `--history-drafts` adds such drafts to
  the MTP module's. `--no-mtp` frees the draft model's weights, its pages, and
  seven state rows per request slot.
- `run` also accepts `--verbose` and `--config PATH`. `--no-compile` skips
  forward compilation for development startup.
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
page counts, cache sizes, and memory fractions are not parameters, and
speculation is set only by the advanced flags.

Flare reads no environment variables except `CUDA_VISIBLE_DEVICES`, which
restricts and renumbers the GPUs that `--gpus` indexes; the `Select GPUs.`
startup line reports it.

## What fits and what was allocated

`flare models` prints, for each alias, the pinned repository and revision, the
architecture its recipe serves, sizes, context, inputs, speculation, supported
GPU counts, the concurrency range, whether the checkpoint is cached, and under
`fits here` an estimate at default settings or the capacity error. The estimate
adds the checkpoint's weights, a rough figure for graphs and scratch, state for
the concurrent requests, and a prefix cache, and gives KV pages the rest of the
GPU's free memory less a fixed 1 GiB headroom and `memory.reserve`. Reports are
plain text when redirected; `--json` carries the same facts in full.

`flare run` states what it does as it starts: the `Select GPUs.` line names
the GPUs, the concurrency limit, and the speculation settings, and the
`Allocate KV cache.` line gives the pages actually allocated. `flare status`
reports the free pages and cache use of a running server.

## Reading a capacity error

```
error: needs ... GiB weights + ... GiB graphs + ... GiB live state + ... GiB prefix cache
       + ... GiB minimum pages (C × 8,192 tokens) = ... GiB;
       has ... GiB free − 1.0 GiB headroom − ... GiB reserve = ... GiB.
       Options: [memory] reserve = "12G", or --max-concurrency 2, or --gpus 0-1.
```

The first two lines are the smallest estimate Flare accepts; the third is the
supply. Each listed option makes the estimate fit on its own: apply one and
rerun; `--no-mtp` is listed when dropping speculation is enough. `No setting
makes MODEL fit` means the weights exceed the selected GPUs. The estimate is
rough: a model near the limit can pass it and run out of GPU memory during
preparation, which `run` also reports as a capacity error.

HTTP rejections use the same form. A 400 names the request's prompt and output
limit against the server's context. A 429 or 503 names the exhausted request or
page capacity and the setting that raises it.

## Adding a recipe

A recipe is a package under `flare/recipes/` with two parts:

1. `recipe.py` declares a `Recipe`: each checkpoint's alias, repository id,
   pinned revision, architecture, and stored bytes by placement; the supported
   tensor-parallel sizes; the concurrency range; its speculation settings; and
   per-rank costs with and without speculation (bytes per state row, bytes per
   KV page, and a rough estimate of graphs and scratch). It supplies
   `prepare(snapshot, plan, ...)`, which builds the bootstrap's configuration
   from the selected GPUs and settings, and `usage(prepared)`, which reports
   each rank's device memory. The module must import without PyTorch.
2. `bootstrap.py` loads weights, declares and tunes kernels, captures every
   serving graph, and returns a prepared model for the native server.

Register the recipe in `flare.recipes.recipes()`.
