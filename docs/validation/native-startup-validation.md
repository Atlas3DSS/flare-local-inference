# Native startup rejection before CUDA and listener binding

All 14 invalid manifest/configuration cases pass against installed native library
SHA256 `735bf767999b26affeacc5ae574b6e52c6a5e1dc18f42cc208ee8b9f0b2a18ab`.
Each case calls `flare_native.Server` in a separate CPU process. Every call
rejects with its expected error and releases the borrowed Python owner. The
recorded syscalls contain zero socket, bind, listen or connect attempts, zero
CUDA-library or NVIDIA-device opens, and zero thread-creation attempts.

The [receipt](native-startup-validation.json) records exact inputs, error strings,
native-library hash, source hashes and syscall-trace hashes. The
[compressed artifacts](/data/sparkgap-artifacts/validation-receipts/native-startup-validation-artifacts.tar.gz) contain all
input descriptors, stdout/stderr, traces and the exact qualified harness source.
Sources and the installed library remained unchanged during the passed run.

## Cases and execution boundary

The cases cover an unsupported ABI version, null context, overlapping buffer
spans, invalid completion layout, absent batch-four resume graph, absent
512-token prefill graph, absent semantic input binding, incompatible page-table
shape, invalid model context size, absent frontend assets, asset checksum
mismatch, invalid frontend-worker count, zero native request-queue capacity,
and invalid event-queue capacity.

The fixture contains all 25 declared ABI-1 graph variants and all required
Qwen4Exp input shapes. Its addresses and handles are synthetic nonzero integers;
they are never dereferenced. The otherwise valid fixture reaches asset validation,
and the queue-limit cases load the exact checkpoint frontend assets before
rejecting. These are installed-constructor metadata and configuration tests;
they do not construct a Python `PreparedModel`, allocate model weights, initialize
CUDA or execute inference.

The Python extension invokes `Bootstrap::load`, which validates model semantics
before loading assets. `Server::start` then validates HTTP limits and starts the
model runtime before creating the HTTP thread. The listener bind follows model
worker readiness. Relevant implementations are
[`NativeServer::new`](../../native/src/python.rs),
[`Bootstrap::load`](../../native/src/server/config.rs),
[`ModelRuntime::start`](../../native/src/server/runtime.rs), and
[`Server::start`](../../native/src/server.rs).

`strace 7.2` records process threads and network/open/ioctl calls. Its socket,
bind, listen and ioctl guards return `EPERM` if reached. No network guard was
triggered. Python starts the harness through `runpy` so CPython's script-opening
`FIOCLEX` does not conflict with the ioctl guard. No CUDA library was loaded and
no NVIDIA device was opened. The archive preserves two harness diagnostics:
the direct script launcher failed before native construction on `FIOCLEX`, and
three initially paraphrased error expectations differed from the actual errors.

CPU reproduction with an unused output directory:

```sh
.venv/bin/python scripts/qualify_startup_validation.py \
  --artifacts /tmp/sparkgap-startup-validation-reproduction
```

## Fatal model-worker qualification

[`qualify_worker_failure.py`](../../scripts/qualify_worker_failure.py) prepares a
separate runtime-failure test. Its default invocation prints a CPU-only plan.
GPU/HTTP execution has not been performed for this receipt.

The proposed input is a complete tiny graph catalog with transport-compatible
storage. Its initialization graph writes a nonzero reserved completion field.
`GraphExecutor::poll` must reject the record and poison the executor;
`ModelRuntime` must clear health, fail the waiting response, drain both CUDA
streams and release scheduler resources. Separate fresh server instances test
JSON HTTP 500 and SSE terminal errors, followed by `/health` HTTP 503 and rejected
subsequent admission. This exercises production runtime error propagation without
changing production source or triggering an unsafe CUDA fault. It does not force
a Rust panic, test multiple waiting requests, establish failed-drain behavior or
validate checkpoint inference.

After assigning one idle GPU and matching the harness manifest version to the
installed library:

```sh
CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
CUDA_HOME=/opt/cuda timeout 120s .venv/bin/python \
  scripts/qualify_worker_failure.py --run \
  --device-uuid GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
  --artifacts /tmp/sparkgap-worker-failure
```
