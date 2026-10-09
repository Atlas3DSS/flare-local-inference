# Fatal completion validation through the native HTTP runtime

All 22 checks pass across two fresh native server instances on GPU
`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f` with typed native errors and the
`qwen4exp.graph-inputs.v2` input schema and private worker phases. The
[run result](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/worker-failure-phases/receipt.json)
records the installed extension, checks, and metrics. Its directory contains
graph manifests, configuration, HTTP requests/responses, and raw SSE.

## Input and failure path

Each server receives a complete 25-variant ABI-2 catalog with real CUDA graph handles
and transport-compatible storage. Graphs contain one deliberate ABI violation:
they write `1` to `completion_records[:, :, 7]`, whose reserved field must be
zero. A real HTTP chat request admits one physical request and launches one
initialization graph through the unmodified installed native API.

`GraphExecutor::poll` reads the device completion and rejects it with
`invalid or stale device completion record`. The resulting execution error
terminates the model worker, clears health, fails the waiting response, and
drains the model and completion-copy streams. The HTTP listener remains available
to report the unhealthy worker.

| Observation | JSON case | SSE case |
|---|---|---|
| Health before request | HTTP 200 | HTTP 200 |
| Waiting response | HTTP 500, exact fatal error | HTTP 200, one terminal error, no `[DONE]` or successful finish |
| Health after worker exit | HTTP 503 | HTTP 503 |
| Subsequent chat request | HTTP 503 | HTTP 503 |
| Native `healthy` and `wait` | false | false |
| Model requests admitted / failed | 1 / 1 | 1 / 1 |
| Initialization graphs launched | 1 | 1 |
| Free pages / snapshots / request slots after drain | 4096 / 4 / 4 | 4096 / 4 / 4 |
| `close()` result | exact fatal error; `drained=true` | exact fatal error; `drained=true` |
| Borrowed owner | retained before close; released afterward | retained before close; released afterward |

Weak references verify owner retention through handoff and release only after
native close confirms drain. Both instances finish without a GPU process.
Native sources and the installed library remain unchanged during execution.

The graph catalog performs no checkpoint inference and contains no model
weights. Exact checkpoint frontend assets are loaded and verified by native
bootstrap. The failure is a fatal completion-protocol violation, rather than a
request-local device status error, Rust panic, or unrecoverable CUDA fault.
Multiple waiting-request fanout and failed CUDA drain are not exercised.

## Reproduction

The harness binds a separate ephemeral loopback listener per case and uses the
returned `Server.address`. It does not contact an existing model server.

```sh
CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
CUDA_HOME=/opt/cuda timeout 120s .venv/bin/python \
  scripts/qualify_worker_failure.py --run \
  --device-uuid GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
  --artifacts /tmp/sparkgap-worker-failure-reproduction
```
