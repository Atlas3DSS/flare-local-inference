# Flare on V100 (SM70)

This is an Apache-2.0 fork of [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference)
that adds a Volta (Tesla V100, SM70) execution path for one recipe,
`qwen3.8-27b` (`nvidia/Qwen3.8-27B-NVFP4`), with MTP speculative decoding and
up to 4 concurrent requests of 32,768 tokens. Upstream behaviour on other GPUs
is otherwise unchanged; the upstream README is kept as
[README-UPSTREAM.md](README-UPSTREAM.md).

## Requirements

- One Tesla V100 32 GB per served model (one GPU per process, TP1).
- An NVIDIA driver that supports CUDA 12.9 (R580 is the last driver branch
  that supports Volta) and a CUDA 12.x toolkit with `sm_70` support; CUDA 13
  cannot compile for `sm_70` ([BUILD.md](docs/sm70/BUILD.md)).
- Disk for the checkpoint (21,921,428,072 bytes of stored weights) and the
  prepared package (about 20 GB) ([EXPORT.md](docs/sm70/EXPORT.md)).
- For the export gates, FP32 references generated once; the long reference
  needs a separate 96 GB GPU
  ([EXPORT.md](docs/sm70/EXPORT.md#2-prepare-public-inputs-and-fp32-references)).

## Quickstart

On a Linux host with CUDA 12.9, Rust, `uv` and the build packages from
[BUILD.md](docs/sm70/BUILD.md), run from an empty directory. Set `THREADS` to
the CPU threads you allocate and `V100_GPU_UUID` to the card. Preparation runs
every export gate, exports only if all pass, and prepares 512-token prefill
chunks ([EXPORT.md](docs/sm70/EXPORT.md#3-prepare-and-export)):

```sh
git clone -b atlas/sm70-public https://github.com/Atlas3DSS/flare-local-inference.git && cd flare-local-inference
cargo build --release --locked --manifest-path flare/Cargo.toml
uv sync --locked --extra prepare --extra test
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch --reinstall-package torchvision torch==2.14.1+cu126 torchvision==0.29.1+cu126
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git /data/src/cutlass
.venv/bin/hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80 --local-dir /data/models/Qwen3.8-27B-NVFP4
mkdir -p /data/flare-run /data/flare-refs /data/flare-keys /data/flare-cache
.venv/bin/python tools/generate_prompts.py --output /data/flare-run/prompts.jsonl
# Generate /data/flare-refs/reference.safetensors and long-code-reference.safetensors: docs/sm70/EXPORT.md section 2.
CUDA_VISIBLE_DEVICES="$V100_GPU_UUID" bash tools/prepare_replay.sh .venv/bin/python --root /data/flare-run --checkpoint /data/models/Qwen3.8-27B-NVFP4 --prompts /data/flare-run/prompts.jsonl --compile-workers "$THREADS" --cutlass-include /data/src/cutlass/include --kernel-cache /data/flare-cache --signing-dir /data/flare-keys --short-reference /data/flare-refs/reference.safetensors --long-reference /data/flare-refs/long-code-reference.safetensors --measure /data/flare-run/tau.json --output /data/flare-run/model.safetensors --report /data/flare-run/prepare.json
./flare/target/release/flare run /data/flare-run/model.safetensors --publisher-keys /data/flare-keys/development-publisher.json --trust-store /data/flare-keys/trust.toml --verify-weights --gpus 0 --memory-limit 31G --max-concurrency 4 --mtp-draft-tokens 3 --no-mtp-adaptive-verification --prefix-cache-memory 16G --host 127.0.0.1 --port 8300
```

On first load, type `local-development` when Flare asks you to trust the
development signing key. When `curl http://127.0.0.1:8300/health` returns
`{"status":"ok"}`, the OpenAI-compatible API is at
`http://127.0.0.1:8300/v1`. A first request:

```sh
curl -sN http://127.0.0.1:8300/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}], "max_tokens": 64, "stream": true, "return_token_ids": true, "return_token_timestamps": true, "stream_options": {"include_usage": true, "continuous_usage_stats": true}}'
```

Optional `flare run` flags, all off by default
([LIMITATIONS.md](docs/sm70/LIMITATIONS.md)):

- `--prefill-srpt`: when several long prompts arrive together, finish their
  prefills one after another instead of round-robin, so the first ones get
  their first tokens much earlier. Trade-off: requests that finished prefill
  decode slowly while the others' chunks run, and short prompts can wait
  behind a started long prompt.
- `--prefill-srpt-short-tokens 2048 --prefill-srpt-decode-priority-tokens 4096`
  (with `--prefill-srpt`): let short prompts interleave and let decoding
  requests go first. Trade-off: later agents' first tokens come later, and
  under decode-heavy load the longest prompts' worst case is worse
  ([details](BENCHMARKS-SM70.md#refined-prefill-ordering-opt-in-not-in-the-release-binaries)).
- `--prefix-cache-restore-wait`: under KV pressure, a preempted request waits
  for its host-cached prefix instead of prefilling it again. Trade-off: that
  request waits, at most as long as the recompute would take.

## What to expect

Measured on one V100 with package `f21b9abb` and server binary `48b69ed4`,
release benchmark protocol; see [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md).

| Measurement | Value |
| --- | ---: |
| Prefill, 1 request, 2,048 / 30,720 tokens | 994 / 866 tok/s |
| Short prompts, decode per agent, 1 / 4 agents | 83.2 / 64.9 tok/s |
| Four cold 28.7k-token agents, first-token p50 | 132 s default, 83 s with `--prefill-srpt` |

## Numerics

The export gates require speculative decoding to match ordinary greedy
decoding byte for byte, and a 30-prompt comparison against an FP32 reference
found no measurable difference from the SM120 build in mean or p99 KL
([ARCHITECTURE.md](docs/sm70/ARCHITECTURE.md#distance-from-the-fp32-reference)).

## Limitations

See [LIMITATIONS.md](docs/sm70/LIMITATIONS.md). In short:

- The KV pool is 124,928 tokens at `--memory-limit 31G`; four 32,768-token
  contexts do not fit at once.
- Preemption under KV pressure is not output-exact: a preempted request can
  continue differently after it resumes.
- The opt-in prefill ordering trades decode speed or worst-case latency for
  earlier first tokens; it is off by default.

## Documentation

- [README-SM70.md](README-SM70.md): results, trade-offs and the full quickstart
- [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md): tables with receipts
- [docs/sm70/EXPORT.md](docs/sm70/EXPORT.md): preparing a package
- [docs/sm70/SERVE.md](docs/sm70/SERVE.md): the serve command and API
- [docs/sm70/VALIDATE.md](docs/sm70/VALIDATE.md): validation gates
- [docs/sm70/BENCHMARK.md](docs/sm70/BENCHMARK.md): how the numbers were measured
- [docs/sm70/ARCHITECTURE.md](docs/sm70/ARCHITECTURE.md): the SM70 path and its changes
- [docs/sm70/LIMITATIONS.md](docs/sm70/LIMITATIONS.md): what does not work or was not tested
- [docs/sm70/WHY-NOT-LOW-BIT.md](docs/sm70/WHY-NOT-LOW-BIT.md): comparison with low-bit routes
- [README-UPSTREAM.md](README-UPSTREAM.md): the upstream NVIDIA README

## Reviews

Every change in this fork was reviewed by a reviewer who did not write it; the
review trailers are in `git log`.
