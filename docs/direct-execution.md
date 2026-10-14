# Direct execution: reproducing model faults without Helios

`ignition serve --direct` is a diagnostic and qualification path, requested by
project owner Luke Alonso, for reproducing and bisecting model-level faults. It
is not production serving and never replaces it. Production serving loads an
exported package in Helios, which owns scheduling, input packing, graph replay
and output without Python ([AGENTS.md](../AGENTS.md), "Python prepares; Helios
serves"). Direct execution prepares the model in the Ignition process, then
drives the prepared graphs from Python: a small fixed scheduler decides each
step, packs the same graph inputs Helios packs, and replays the captured graph
on every rank. With `--eager` it calls the forward each graph captured instead.
Nothing selects either implicitly: `--direct`, and `--eager` with it, must be
given, and neither `run`, `export` nor Flare accepts them.

A fault that reproduces through direct graph replay does not depend on Helios's
scheduler, transport frames or native replay. A fault that reproduces under
`--eager` does not depend on CUDA graph capture or replay; one that disappears
under `--eager` points at what replay does differently (see "Eager forwards"
below). Preparing with `--no-compile` additionally separates compiled forwards
from the recipe's Python forwards. Every run prepares from the checkpoint;
there is no package.

## Usage

On one host, GLM-5.3 Flash at TP1 on a GB300:

```sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/ignition serve glm-5.3-flash --gpus 0 \
  --max-concurrency 8 --port 8000 --direct
CUDA_VISIBLE_DEVICES=1 .venv/bin/ignition serve glm-5.3-flash --gpus 0 \
  --max-concurrency 8 --port 8000 --direct --eager
```

Across DGX Sparks, rank zero adds `--ranks N --rendezvous ADDR:PORT` and each
peer runs `ignition join` exactly as for in-process serving
([multi-host.md](multi-host.md)); peers learn the execution mode from rank zero:

```sh
.venv/bin/ignition serve glm-5.3-flash --ranks 4 --rendezvous 10.200.0.12:29630 \
  --gpus 0 --max-concurrency 16 --host 0.0.0.0 --port 8000 --direct
.venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 4   # and ranks 2, 3
```

Every host needs the same Flare, Ignition and FlashInfer sources, as for any
multi-host preparation. Preparation flags (`--max-concurrency`,
`--prefill-chunk-size`, `--mtp-draft-tokens`, `--no-mtp-adaptive-verification`,
`--no-mtp`, `--no-context-parallel`, `--no-stream-overlap`, `--no-compile`)
apply as they do to `ignition serve`; direct execution runs what they prepared.
`--verbose` also prints every step's graph, row count and GPU time.

The listener answers non-streaming `POST /v1/chat/completions`, `GET
/v1/models`, `GET /health` and `GET /metrics`, so
`scripts/repro_glm_batched_speculation.py` runs against it unchanged. Replies
split `<think>` reasoning from content. A request whose completion record
carries a device error fails with HTTP 500 and the message Helios uses,
`prepared model reported device error N`, with the code in
`flare_direct.device_error`. `/metrics` reports
`flare_speculative_rounds_total{batch,width}` as Helios does, plus steps by
kind, device errors by code, request outcomes and prefix-cache counters. Each
retired request prints one line on stderr: prompt and reused tokens, generated
tokens, speculative rounds and outcome.

## Design

**Scheduler** (`ignition/direct/scheduler.py`, CPU only). FIFO admission
places a request in a free slot when its pages fit: every page its prompt,
output limit and widest verification round can reach, plus the draft page, as
Helios reserves them. Pages are reference counted and handed out smallest
first. Admitted requests are initialized together, or restored together when a
cached checkpoint's consumed prefix is exactly their prompt, then a restored
request runs the resume graph. Prompts prefill one chunk per step, padded to
the smallest prepared capacity. Every decoding request joins one round: the
smallest prepared speculative batch at the fixed width, or at the width with
the most expected tokens per measured graph time, from each request's
acceptance estimate as Helios keeps it. Decode rounds and prefill chunks
alternate while both have work. A prompt not yet cached is snapshotted at its
endpoint and published as a checkpoint that owns the prompt's complete pages;
later identical prompts share those pages and restore the partial page into a
private one, as in Helios.

**Packing.** Each step's uploads are the byte ranges Helios's `paged_text`
contract writes for that graph kind: state-row indices, row slots and
generations, page and draft page tables, token IDs, query lengths, sampling
flags, state-transfer commands and admission rows. Bytes it does not write
keep their resident values, as with native partial uploads. Completion records
are validated against the step before any state changes; a record that
contradicts it stops execution.

**Execution** (`ignition/direct/executor.py`). A rank copies each upload through
pinned staging on one stream, writes the frame index and submission sequence,
replays the graph, and waits for it before the next step. It checks any
collective runtime that reports transport failures only when asked, as Helios
does after each frame.

**Eager forwards.** With `--eager`, every rank releases its captured graphs
and, for each step, calls the forward that step's graph captured, compiled
where preparation compiled it, on the same buffers after the same uploads.
Scheduling, packing and completion handling do not change. The forwards are
built for capture and make no per-step host decisions, so they launch the
kernels the graph recorded, with the same arguments except for the addresses
of temporaries. Eager steps are bound by the host: on a GB300, GLM-5.3 Flash's
compiled forwards took 0.26 to 0.34 s per prefill chunk or speculative round
at every capacity exercised. What graph replay does and eager execution does
not reproduce:

- Replay runs the recorded launches from fixed addresses in the graph pool;
  eager forwards run their Python each step and take temporaries from the
  caching allocator, at other addresses. A fault that reads stale or
  uninitialized temporary storage can therefore behave differently.
- Replay schedules captured branches by their recorded dependencies, without
  launch gaps; eager execution orders them by stream and event as the Python
  launches them, so the timing of any race differs.
- A collective that orders itself across streams or checks transport health
  per call does so in eager execution; under replay its stream order is the
  graph's and health is checked after the step.
- Step times include Python launch time. The adaptive verification width is
  chosen from them, so it can differ from graph replay's choice; preparing
  with `--no-mtp-adaptive-verification` fixes the width in both modes.

**Ranks.** Rank zero decides every step and sends it to each peer as one
message over that peer's executor connection, the connection Helios uses for
per-step coordination, then executes it. Every rank therefore runs the same
steps in the same order with the same inputs and the same execution mode, and
the collectives inside each graph or forward join their GPU work. Rank zero
alone reads completion records. Stopping rank zero (SIGINT or SIGTERM) takes
effect between steps; it ends every peer's loop, then the group closes as after
in-process serving.

## Limits

- Prefix reuse covers exact full-prompt hits only. Partial prefixes, system
  boundaries and terminal checkpoints are not captured.
- One step at a time: no pipelined submission, and no mixed prefill and decode
  graphs.
- No streaming, grammar masks, tool-call parsing, raw completions, Responses or
  Messages APIs, images or copied proposals.
- Eager forwards allocate their temporaries from the caching allocator after
  the graphs release their pool; preparation leaves only its headroom free
  beyond that, so `[memory] reserve` may need raising for large prefill chunks.
- GLM-5.3 Flash retains its forwards for `--eager`; other recipes support graph
  replay only, and `--eager` with them fails once preparation has finished.
