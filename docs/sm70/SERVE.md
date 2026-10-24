# Serve on a V100

`flare run` loads an exported package and serves it over HTTP without Python.
This page covers the SM70 serve command. The upstream
[serving guide](../model-preparation.md#serving-runtime-packages) and the
[`flare run` flag table](../cli-design.md) are the general reference.

## The command

This is the command the fork runs in production, one process per card:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors \
  --publisher-keys /data/flare/keys/development-publisher.json \
  --trust-store /data/flare/keys/trust.toml \
  --verify-weights \
  --gpus 0 \
  --memory-limit 31G \
  --max-concurrency 4 \
  --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification \
  --prefix-cache-memory 16G \
  --host 127.0.0.1 --port 8300
```

Accept the signing key once before running it unattended
([EXPORT.md](EXPORT.md#signing-keys-and-the-trust-store)).

## Flags

| Flag | What it does | Why this value on SM70 |
| --- | --- | --- |
| `PACKAGE` | The `.safetensors` primary. Its payload blobs must sit beside it. | The package from [EXPORT.md](EXPORT.md). |
| `--publisher-keys PATH` | Adds a publisher verification profile. Importing a key does not approve it. | `development-publisher.json` from preparation. |
| `--trust-store PATH` | The file of accepted key fingerprints and namespaces. Flare reads it and records new acceptances in it. | Keep it with the keys, so a service account loads without prompting. The default is `~/.config/flare/trust.toml`. |
| `--verify-weights` | Checksums the stored weight allocation of the served rank before loading GPU resources. It adds a `Verify stored data` stage. | It catches a damaged or partly copied package before serving. It costs minutes on a spinning disk (389.7 s, STATUS.md line 66) and about 114 s normally (line 63). |
| `--gpus 0` | Serves one rank on GPU 0. With `CUDA_VISIBLE_DEVICES` set, indices refer to the visible list. | The package is TP1; the count must match. |
| `--memory-limit 31G` | Caps Flare's occupied memory on the GPU at 31 GiB: weights, graphs, scratch, execution state and caches. KV pages get what is left. | A V100 32 GB card holds 32 GB. 31G leaves the rest for the CUDA context and driver. Every measurement in these docs ran at 31G. |
| `--max-concurrency 4` | The most requests decoding at once. More requests queue. | 4 is the prepared maximum. The S1L attention kernel is fixed at four agents, and SM70 MTP preparation admits at most four requests. You may lower it; you cannot raise it. |
| `--mtp-draft-tokens 3` | The most MTP draft tokens verified per round. Verification width is drafts + 1. | See [draft tokens](#draft-tokens-3-or-4) below. |
| `--no-mtp-adaptive-verification` | Uses the fixed width every round instead of choosing a width from measured acceptance. | See [fixed-width verification](#fixed-width-verification) below. |
| `--prefix-cache-memory 16G` | Host-RAM tier for prefix-cache checkpoints. Finished prompt and answer states copy out of GPU memory into registered host memory and can be restored later instead of re-prefilled. | Long coding-agent prompts are prefill-bound on one card, so restoring a prefix saves the most time. Budget 16 GiB of host RAM per serving process. The benchmarks used 8G (BENCHMARK.md). |
| `--host 127.0.0.1 --port 8300` | The listener. The default port is 8000. | Loopback only. Put a router or proxy in front for remote clients, and use one port per card. |

Flags the fork does not set here, and their defaults:

- `--kv-cache-limit` defaults to `memory`: KV pages fill the memory left under
  the limit. The benchmark harness set `concurrency`, which caps the pool at one
  32,768-token context per request
  ([BENCHMARK.md](BENCHMARK.md#what-was-run)).
- `--scheduler-policy` defaults to `balanced`. On an earlier package,
  `--scheduler-policy agent` scored higher on a real 34k-token agent trace
  (53.6 against 50.8 emitted tokens per GPU-second at batch 4, STATUS.md
  line 30), but that measurement used adaptive verification and
  `--mtp-pipeline`. It has not been measured on the release package.
- `--mtp-pipeline` queues a second speculative round behind the running one. It
  changed the round period by at most 0.1 % (STATUS.md line 10). It is off.
- `--api-key-file PATH` enables bearer authentication. Without it, anyone who
  can reach the port can use it.

### Draft tokens: 3 or 4

The package carries verification widths 1 to 5, so `--mtp-draft-tokens` can be
anything from 0 to 4 at load time without re-exporting.

- 3 is the recipe's declared default, and the fork's production setting. The
  fork's real-trace agent runs on 2026-10-21 used it (fixed width 4, STATUS.md
  lines 21, 32).
- 4 is what every number in [README-SM70.md](../../README-SM70.md) was measured
  with.
- The run log has no comparison of 3 against 4 on the release package. If you
  choose between them, measure both on your own traffic
  ([BENCHMARK.md](BENCHMARK.md)).
- 0 serves without speculation (ordinary prefill and decode graphs).

### Fixed-width verification

With adaptive verification, the width of each round depends on which requests
share the batch. Under sampling, that changes the random draws, so the same
seeded request can produce different text alone and next to other requests. The
fork's measurements on 2026-10-21 showed this:

- Fixed width 4 matched solo outputs 4 of 4 times. Adaptive verification differed
  in 2 of 2 checks. Fixed width also produced 44.9 against 39.8 emitted tokens
  per GPU-second on a real 34k-token agent trace (STATUS.md line 21).
- After an adaptive-policy fix, and with the agent scheduler policy and
  `--mtp-pipeline`, adaptive was within a few percent of fixed width (53.5
  against 52.0 emitted tokens per GPU-second at batch 4), but still matched only
  1 of 2 exactness checks (line 27).

Fixed width became the default for combined speculative catalogs in `45f8801`.
The serve command also states it explicitly. Under greedy decoding every width
gives the same tokens, because verification is exact
([ARCHITECTURE.md](ARCHITECTURE.md#3-s1l-decode-verification-and-draft-attention)).
Under sampling, the sequence of widths decides the random draws, so a fixed
width keeps each request's output independent of the requests next to it.

## Startup

Startup takes minutes. Redirected output shows one line per stage:

1. `Verify package`: signature and metadata.
2. Publisher acceptance: silent once the trust store holds the key.
3. `Verify stored data` (with `--verify-weights`).
4. `Load weights`: about 20 GB from disk.
5. `Plan cache` and `Allocate cache`: graphs uploaded, then KV pages backed.
6. `Start server`: the listener is up.

The cache lines report the memory plan. For the release package at 31G with the
benchmark's KV cap, the run log shows `21.15 GiB loaded`, a `9.76 GiB` cache
budget and `KV capped at 4 contexts` (STATUS.md line 63), and the KV pool was
131,584 tokens (lines 65, 66). The pool at the default `--kv-cache-limit memory`
was not measured. Read your own `Allocate cache` line.

Wait for readiness by polling `/health`, not by sleeping. Allow at least
1800 s when several packages load from one spinning disk
([EXPORT.md](EXPORT.md#loading-several-packages-at-once)).

Stop Flare with Ctrl-C or SIGTERM. It stops admitting requests, gives active
responses five seconds, drains GPU work and exits 0.

An example systemd unit, assuming the paths above and a `flare` user:

```ini
[Unit]
Description=Flare SM70 on GPU 0
After=network-online.target

[Service]
User=flare
ExecStart=/opt/flare/flare run /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors --publisher-keys /data/flare/keys/development-publisher.json --trust-store /data/flare/keys/trust.toml --verify-weights --gpus 0 --memory-limit 31G --max-concurrency 4 --mtp-draft-tokens 3 --no-mtp-adaptive-verification --prefix-cache-memory 16G --host 127.0.0.1 --port 8300
KillSignal=SIGTERM
TimeoutStartSec=1800
TimeoutStopSec=120
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## API

Flare serves the same routes on every package
([native inference APIs](../api.md)):

| Route | Use |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI chat, streaming with `"stream": true` |
| `POST /v1/completions` | Raw text or token IDs |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages (`anthropic-version: 2023-06-01`) |
| `POST /v1/messages/count_tokens` | Exact prompt token count |
| `GET /v1/models` | One model card; `max_model_len` is the served context limit |
| `GET /health` | Readiness and liveness |
| `GET /metrics` | Prometheus text, or JSON with `?format=json` |

The served model name is `Qwen3.8-27B`. Sampling defaults follow the model card:
temperature 1.0, top-p 0.95, top-k 20. Requests are clamped to the model-author
ranges: temperature 0.7 to 1.0, top-p 0.8 to 0.95, top-k fixed at 20
(`ignition/recipes/qwen38_dense/recipe.py`). A request for temperature 0 runs at
0.7. `--unclamped-sampling` lifts the ranges for validation only.

```sh
curl -s http://127.0.0.1:8300/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}], "max_tokens": 256}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8300/v1"`. Each
request may use up to 32,768 tokens of prompt plus output.

## Health and metrics

`GET /health` returns `{"status":"ok"}` with HTTP 200 while the runtime is
healthy, and HTTP 503 with an error object otherwise. Before `Start server`
nothing listens, so connections are refused. Neither `/health` nor `/metrics`
needs the API key, and neither reports request content.

`GET /metrics?format=json` returns counters and gauges. Useful fields:

| Field | Meaning |
| --- | --- |
| `active` | Requests in a model slot (at most 4) |
| `pending` | Requests waiting for a slot |
| `free_pages`, `total_pages`, `page_tokens` | KV pages free and in the pool; pages hold 128 tokens |
| `reused_tokens` | Prompt tokens served from GPU prefix-cache hits |
| `host_hits`, `host_reused_tokens` | Prefix restores from the host store |
| `generated_tokens`, `accepted_drafts`, `verification_drafts` | Output, accepted draft tokens, draft positions verified |

The Prometheus text also carries `flare_speculative_rounds_total` and
`flare_speculative_gpu_nanoseconds_total` by batch and width, which
[BENCHMARK.md](BENCHMARK.md#gpu-step-per-round) uses for GPU round times.

## Several cards behind a router

Each V100 serves its own process and its own package copy; there is no
tensor parallelism on SM70. For N cards:

1. Start one `flare run` per card: `--gpus 0`, `--gpus 1` and so on, each on its
   own port (8300, 8301, ...). The cards can share one package directory and
   one trust store.
2. Start them a few minutes apart, or allow 1800 s each, when they read the same
   disk.
3. Put an HTTP router in front, for example nginx, HAProxy, Envoy or LiteLLM.
   This repository does not include one.
4. Health-check each backend with `GET /health`. Send no traffic until it
   returns 200, and take a backend out on 503 or connection errors.
5. Keep each conversation on the same backend (session affinity). The prefix
   cache is per process, so a conversation that moves to another card loses its
   cached prefix and prefills again.
6. For load-aware routing, read `active` and `pending` from
   `/metrics?format=json`. A backend with `active` 4 queues new requests.
7. To restart a card, drain it in the router first, wait for its `active` to
   reach 0, then send SIGTERM.

A minimal nginx example with hash affinity on a client-supplied header:

```nginx
upstream flare_sm70 {
    hash $http_x_session_id consistent;
    server 127.0.0.1:8300 max_fails=2 fail_timeout=10s;
    server 127.0.0.1:8301 max_fails=2 fail_timeout=10s;
}
server {
    listen 8080;
    location / {
        proxy_pass http://flare_sm70;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
```

Clients must send an `X-Session-Id` header that is stable per conversation;
without it, every request hashes to the same backend. `proxy_buffering off`
keeps streamed tokens flowing. Open-source nginx has no
active health checks; use a router that polls `/health`, or an external check
that edits the upstream list.
