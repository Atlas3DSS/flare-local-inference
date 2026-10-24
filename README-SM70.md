# Flare on SM70 (Tesla V100 PCIe 32 GB)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/sm70/img/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/sm70/img/hero-light.svg">
  <img alt="Decode-only, Qwen3.8-27B NVFP4 on one Tesla V100 PCIe 32 GB, memory limit 31G, package model-32k-tonight-a, 2026-10-22: median decode 90.4 tok/s at 1 agent and 70.7 tok/s per agent at 4 agents." src="docs/sm70/img/hero-light.svg">
</picture>

This fork adds an SM70 (Volta) execution path to
[flare-local-inference](https://github.com/NVIDIA/flare-local-inference).
Upstream Flare targets SM120/SM121 GPUs. This branch serves one recipe on one
Tesla V100 32 GB:

- `qwen3.8-27b` (`nvidia/Qwen3.8-27B-NVFP4`, revision `482ca0f`), TP1;
- the checkpoint's MTP head for speculative decoding, draft depth up to 4;
- up to 4 concurrent requests and up to 32,768 tokens per request.

The port is maintained in this fork. It is not supported upstream.

**Who it is for.** Use it if you have V100 32 GB cards and want to serve
Qwen3.8 27B for coding agents or other local clients through an
OpenAI-compatible API. It also documents how to keep speculative decoding
bit-exact on a GPU without BF16 tensor cores. For any other model, GPU, more
than 4 concurrent requests or contexts above 32,768 tokens, use upstream Flare.

**What is in the tree.** The code is the release tree `0e4c81f`, validated on
2026-10-22. It is followed by a comment-only scrub (`8889ec8`) and documentation
commits. On top of the SM70 base port, the release tree carries four changes,
explained in [ARCHITECTURE.md](docs/sm70/ARCHITECTURE.md):

- the scheduler hang fix (`e81534a`, `718a187`);
- prefill projections on the pgemm kernel (`8ad725c`);
- the S1L decode, verification and draft attention (`0aea71a`, `ca16f00`);
- GDN replay-commit (`d3258bc` to `7cb1bc7`).

## Measured results

All rows: Qwen3.8-27B NVFP4, one Tesla V100 PCIe 32 GB, TP1. Unless a row names
another package, the package is `model-32k-tonight-a`, prepared from `0e4c81f`
for a 32,768-token context, 4 agents and MTP draft depth 4, and the server ran
at `--memory-limit 31G`, `--max-concurrency 4` and `--mtp-draft-tokens 4` with
fixed-width verification.
Each source is a line of the fork's run log (`STATUS.md`). The quoted fragments
are in [BENCHMARK.md](docs/sm70/BENCHMARK.md#sources), which also explains how
to reproduce each row. The full tables are in
[BENCHMARKS-SM70.md](BENCHMARKS-SM70.md).

The two tables measure different things. Do not compare a number from one table
with a number from the other.

### End to end, real coding-agent traces

End-to-end numbers include prefill, scheduling and every request in the
workload.

| Measurement | Value | Hardware | Memory limit | Concurrency | Package | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Generated tokens per second, end to end | 47.23 tok/s | V100 PCIe 32 GB | not stated in the source line | not stated in the source line | `best24`, a different package, **not** the release | 2026-10-22 | STATUS.md line 68 |
| Release package | not measured | V100 PCIe 32 GB | 31G | 4 | `model-32k-tonight-a` | n/a | none |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/sm70/img/e2e-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/sm70/img/e2e-light.svg">
  <img alt="End to end on real coding-agent traces, Tesla V100 PCIe 32 GB, 2026-10-22: 47.23 generated tok/s on package best24, a different package from the release. The release package was not measured on real traces. Prefill was about 85 % of busy time in an earlier real-trace analysis." src="docs/sm70/img/e2e-light.svg">
</picture>

**Caveat.** The release package has no real-trace end-to-end measurement. The
47.23 tok/s row describes a different package and is listed only because it is
the real-trace figure the run log reports for the same day. On real coding-agent
traces one card is prefill-bound: an earlier analysis put prefill at about 85 %
of busy time (STATUS.md line 37). End-to-end runs on fixed prompts, which are
not real traces, are in [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md): 238.0 tok/s
aggregate output on short prompts at 4 agents (line 66), and a TTFT p50 of
90.7 s for four 22.4k to 30.2k-token prompts at 4 agents (line 65).

### Decode-only

Decode-only numbers are per-request decode rates or GPU step times. They exclude
prefill and queueing.

| Workload | Value | Hardware | Memory limit | Concurrency | Package | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Short prompts, median decode | 90.4 tok/s | V100 PCIe 32 GB | 31G (a) | 1 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 66 |
| Short prompts, median decode per agent | 70.7 tok/s | V100 PCIe 32 GB | 31G (a) | 4 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 66 |
| 4 prompts of 16.6k-18.6k tokens, median decode | 82.6 tok/s | V100 PCIe 32 GB | 31G (a) | 1 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 64 |
| 4 prompts of 16.6k-18.6k tokens, median decode per agent | 42.6 tok/s | V100 PCIe 32 GB | 31G (a) | 4 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 64 |
| 4 prompts of 16.6k-18.6k tokens, GPU step per 1 x 5 round | 49.5-49.6 ms | V100 PCIe 32 GB | 31G (a) | 1 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 64 |
| 4 prompts of 22.4k-30.2k tokens, median decode | 82.2 tok/s | V100 PCIe 32 GB | 31G | 1 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 65 |
| 4 prompts of 22.4k-30.2k tokens, median decode per agent | 37.2 tok/s | V100 PCIe 32 GB | 31G | 4 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 65 |

(a) Lines 64 and 66 do not restate the memory limit. The script that ran every
leg sets `--memory-limit 31G`, and both lines report the same 131,584-token KV
pool as the 31G run on line 65.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/sm70/img/latency-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/sm70/img/latency-light.svg">
  <img alt="Decode-only GPU step per 1 x 5 speculative round, 1 agent, four 16.6k to 18.6k-token prompts: replay-commit with pre-S1L draft 55.1 ms, S1L draft without replay-commit 50.4 ms, release (replay-commit + S1L) 49.5 ms." src="docs/sm70/img/latency-light.svg">
</picture>

### Capacity

| Measurement | Value | Hardware | Memory limit | Concurrency | Package | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| KV pool | 131,584 tokens | V100 PCIe 32 GB | 31G | 4 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md lines 65, 66 |
| KV pool before replay-commit | 98,304 tokens | V100 PCIe 32 GB | 31G | 4 | baseline package of 2026-10-21 | 2026-10-21 | as cited on STATUS.md line 49 |
| Four prompts of 22.4k-30.2k tokens (110,022 in total) | all resident, no eviction or restore | V100 PCIe 32 GB | 31G | 4 | `model-32k-tonight-a` | 2026-10-22 | STATUS.md line 65 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/sm70/img/kv-pool-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/sm70/img/kv-pool-light.svg">
  <img alt="KV pool at memory limit 31G with 4 agents: 98,304 tokens before replay-commit and 131,584 tokens with it, +33,280 tokens (+33.9 %). Four prompts of 22.4k to 30.2k tokens (110,022 prompt tokens) stay resident with zero evictions." src="docs/sm70/img/kv-pool-light.svg">
</picture>

The graphics illustrate the cited numbers. The tables and their sources are the
authoritative text.

These runs used a `flare` binary built before this branch's Helios commits, and
a benchmark harness that is not in this repository. See
[BENCHMARK.md](docs/sm70/BENCHMARK.md#what-was-run) before comparing your own
numbers.

## Trade-offs and benefits

The usual way to run a 27B model on a V100 is to requantize it into a format
the card handles well. This port instead serves NVIDIA's own NVFP4 package
(4-bit weights with FP8 block scales, FP8 attention projections) with the
checkpoint's weights unchanged. Volta has no FP4, FP8 or BF16 tensor-core
instructions, so the kernels decode every packed weight exactly to FP16 and
accumulate in FP32. [WHY-NOT-LOW-BIT.md](docs/sm70/WHY-NOT-LOW-BIT.md) has the
detail and the sources.

- **Given up:** extra decode work per weight compared with native FP4/FP8 tensor
  cores; prefill-bound long prompts (TTFT p50 90.7 s for four 22.4k-30.2k
  prompts at 4 agents, STATUS.md line 65); one recipe, TP1, at most 4 requests of
  32,768 tokens; 816 upstream commits not in this branch; old cards, with power
  not measured; no real-trace end-to-end number for the release.
- **Gained:** no requantization, calibration or retraining; speculative
  decoding that emits exactly the ordinary greedy tokens (strict 47/47,
  invariance 28/28, line 62); a 131,584-token unquantized BF16 KV pool at 31G
  that holds four 22-30k-token agents with zero evictions (lines 49, 65);
  70.7 tok/s per agent at 4 agents, decode-only (line 66); prefix caching; 32 GB
  cards at used-market prices.

**What makes it useful** is the combination: upstream's NVFP4 package on
hardware without FP4 instructions, with its outputs measured against an
independent FP32 reference (KL 0.0011 short, 0.0024 long; not bit-identical,
and not compared with SM120); one committed recurrent row per request, so long
multi-agent contexts stay resident; exact MTP verification with the draft
attention on tensor cores; every package the fork measured gated before export;
and end-to-end and
decode-only results published apart. It suits multi-agent, long-context and
prefix-heavy serving on V100 32 GB cards. It is not for training, BF16
workloads, other models, or speed records.

| Route | Numerics relative to the NVFP4 checkpoint | Measured here? | Source |
| --- | --- | --- | --- |
| This port | Weights unchanged; speculation exact; KL to FP32 reference 0.0011 / 0.0024 | Yes | STATUS.md lines 49-66, commits `8ad725c`, `ca16f00` |
| llama.cpp GGUF K-quants | Requantized from BF16; its own quantization error | No | [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) |
| INT8 DP4A (superl8) | 3-bit GGUF on INT8 kernels; reports 21.3 tok/s decode for Qwen3.6-27B Q3_K_S on a CMP 100-210 | No | [scoreboard](https://github.com/jajmangold/superl8/blob/main/bench/qwen3-scoreboard.json) |
| W8A8 / SmoothQuant | INT8 weights and activations, calibrated | No | [SmoothQuant](https://arxiv.org/abs/2211.10438) |
| FP16 dense | About 54 GB; does not fit one 32 GB card | No | two bytes per parameter |

## Quickstart

On a Linux host with one V100 32 GB, CUDA 12.9, Rust, `uv` and the build
packages from [BUILD.md](docs/sm70/BUILD.md), run from an empty directory:

```sh
git clone -b atlas/sm70-public https://github.com/Atlas3DSS/flare-local-inference.git && cd flare-local-inference
cargo build --release --locked --manifest-path flare/Cargo.toml
uv sync --locked --extra prepare --extra test
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch --reinstall-package torchvision torch==2.14.1+cu126 torchvision==0.29.1+cu126
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git /data/src/cutlass
.venv/bin/hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80 --local-dir /data/models/Qwen3.8-27B-NVFP4
sed -n '/^cat > prepare_sm70.py/,/^EOF$/p' docs/sm70/EXPORT.md | sh
.venv/bin/python prepare_sm70.py --checkpoint /data/models/Qwen3.8-27B-NVFP4 --cutlass-include /data/src/cutlass/include --signing-dir /data/flare/keys --kernel-cache /data/flare/kernel-cache --output /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors
./flare/target/release/flare run /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors --publisher-keys /data/flare/keys/development-publisher.json --trust-store /data/flare/keys/trust.toml --verify-weights --gpus 0 --memory-limit 31G --max-concurrency 4 --mtp-draft-tokens 3 --no-mtp-adaptive-verification --prefix-cache-memory 16G --host 127.0.0.1 --port 8300
```

On first load, type `local-development` when Flare asks you to trust the
development signing key. Flare records that in the trust store, so later loads
do not ask. When `curl http://127.0.0.1:8300/health` returns `{"status":"ok"}`,
the OpenAI-compatible API is at `http://127.0.0.1:8300/v1`. The last line is
the fork's production serve command, with 3 draft tokens; the tables above were
measured with 4. [SERVE.md](docs/sm70/SERVE.md) explains each flag. Change the
`/data` paths if you cannot write there.

Preparation takes about 25 minutes on one card (STATUS.md lines 49, 55). The
script exports without the fork's export gates; [VALIDATE.md](docs/sm70/VALIDATE.md)
explains what that means.

## Documentation

| Document | Contents |
| --- | --- |
| [BUILD.md](docs/sm70/BUILD.md) | Toolchain, building Flare and the Python environment, checking the SM70 kernels, common failures |
| [EXPORT.md](docs/sm70/EXPORT.md) | Preparing a V100 package, the export gates and goldens, signing keys and the trust store, sizes and times |
| [SERVE.md](docs/sm70/SERVE.md) | The serve command flag by flag, the API, health and metrics, several cards behind a router |
| [VALIDATE.md](docs/sm70/VALIDATE.md) | The validation gates in order, with commands, pass criteria and the release results |
| [BENCHMARK.md](docs/sm70/BENCHMARK.md) | How each number above was measured and how to reproduce it |
| [ARCHITECTURE.md](docs/sm70/ARCHITECTURE.md) | Why Volta needs its own path, and the four changes: hang fix, pgemm, S1L attention, GDN replay-commit |
| [LIMITATIONS.md](docs/sm70/LIMITATIONS.md) | What does not work or was not tested, upstream divergence, reporting issues |
| [WHY-NOT-LOW-BIT.md](docs/sm70/WHY-NOT-LOW-BIT.md) | This port against the usual low-bit routes on V100-class cards: trade-offs, benefits, sources |
| [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md) | Every measured number, with its source line |

For everything that is not SM70-specific, the upstream documents apply:
[model preparation and serving](docs/model-preparation.md),
[CLI design](docs/cli-design.md), [native inference APIs](docs/api.md) and
[signing](docs/signing.md).
