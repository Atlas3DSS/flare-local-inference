# Flare on SM70 (Tesla V100 32 GB)

This branch adds an SM70 (Volta) execution path to Flare for one recipe:
`qwen3.8-27b` (`nvidia/Qwen3.8-27B-NVFP4`) on one V100 32 GB GPU, TP1, with
the checkpoint's MTP head for speculative decoding. Upstream Flare targets
SM120/SM121. This port is maintained in a fork and is not supported upstream.

The release tree is `atlas/tonight-a` at `0e4c81f` (base `718a187`, S1L decode
and draft attention `ca16f00`, prefill projections `8ad725c`, GDN replay-commit
`7cb1bc7`), plus comment and documentation commits on `atlas/sm70-public`.
Measured numbers are in [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md).

## Why Volta needs its own path

V100 tensor cores execute FP16 `mma.m8n8k4` only. Volta has no BF16 or FP8
tensor-core instructions and no FP8 conversion instructions, and the
FlashInfer/b12x kernels Flare uses elsewhere target newer architectures. The SM70
path therefore:

- holds BF16 values in FP32 tensors and rounds them to BF16 with integer
  round-to-nearest-even, so the model's BF16 rounding points are kept without
  BF16 storage (`ignition/components/sm70_precision.py`);
- runs NVFP4 weight-only projections (E2M1 times the E4M3 block scale, FP32
  dot, then the checkpoint global scale) and tensor-FP8 projections (E4M3
  activation rounding kept, combined scale applied after the dot) from packed
  weights, on FP16 tensor-core tiles with FP32 accumulation; tensor-FP8 prefill
  feeds exact FP16 operands (`sm70_projection*.py`, `sm70_dense.*`,
  `sm70_prefill.*`, `sm70_pgemm.cuh`);
- runs decode and verification attention on a tensor-core kernel that converts
  BF16 K/V blocks to power-of-two-scaled FP16 exactly, with an FP32 path for
  blocks that do not admit that conversion (`sm70_decode_attention.*`);
- runs the Gated DeltaNet recurrence in Triton with floating-point fusion
  disabled, and commits speculative rounds by replaying the accepted prefix
  from stored FP32 operands (GDN replay-commit, `sm70_gdn.py`).

## What works

Validated on the release tree on 2026-10-22:

- Preparation and serving of `qwen3.8-27b` on one V100 32 GB, TP1.
- MTP speculative decoding with draft depth 4 (verification widths 1 to 5).
  Fixed-width verification is the serving default.
- Up to 4 concurrent requests. The decode attention kernel is fixed at four
  agents.
- Served context up to 32,768 tokens per request. Preparation refuses a wider
  page table.
- Prefix caching, and preemption that keeps the preempted request's
  checkpoint for readmission.
- Bit-exact speculative verification. Accepting k verification rows leaves the
  same state as k single-row greedy decodes (the `forced_prefix_*` gates).
- At `--memory-limit 31G` with 4 agents, the KV pool holds 131,584 tokens
  (98,304 before replay-commit). Four prompts of 22.4k to 30.2k tokens stay
  resident together.

## What does not work or was not validated

- **No `ignition export` command line for SM70 yet.** The recipe declares a
  262,144-token context, which the SM70 attention path refuses, and the CLI
  preparation path does not pass a CUTLASS include directory, which SM70 MTP
  preparation requires. The validated packages were prepared by calling
  `ignition.recipes.qwen38_dense.bootstrap.prepare` directly with a
  `PreparationConfig` (see [Prepare](#prepare)). That harness is not part of
  this repository.
- Other recipes and models on SM70 were not run.
- More than 4 concurrent requests, or a context above 32,768 tokens.
- Tensor parallelism on SM70. The recipe is single-GPU.
- Quantized (NVFP4) activations. SM70 NVFP4 is weight-only.
- Two scheduler fixes from `atlas/sm70-launch` are not in this tree: `b25e756`
  (greedy readmission re-derives delivered tokens when the preempted checkpoint
  cannot be kept) and `0f17156` (host-store admission deadlock in a
  one-context pool). This tree's `e81534a` separately makes a refused request
  retry when a host-store copy-out pin is released.
- Adaptive verification (`--mtp-adaptive-verification`) makes sampled outputs
  depend on batch composition. Use the fixed-width default when outputs must
  not depend on co-scheduled requests.
- Long prompts are prefill-bound. With four fixed prompts of 22.4k to 30.2k
  tokens, TTFT at 4 agents was 90.7 s (p50). An earlier analysis
  of real coding-agent traces on one card put prefill at about 85 % of busy
  time.
- Shared changes were not re-run on SM120/SM121. These include the
  `qwen38_dense` recipe declaration, which now enables MTP by default
  (3 draft tokens, at most 4) and caps concurrency at 4 for every architecture,
  and the Helios scheduler changes.
- Windows was not built or run.
- No end-to-end real-trace measurement of the release package appears in the
  cited run log. See [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md).

## Build

Set up the environment and build the native CLI as described in
[model preparation](docs/model-preparation.md#environment-setup). For Volta:

- Use a CUDA toolkit whose `nvcc` targets `sm_70`. The SM70 components were
  validated with CUDA 12.9 and a PyTorch cu126 build. Other toolkits were not
  tested.
- SM70 kernels compile with `nvcc` from PyTorch's `CUDA_HOME` during
  preparation. They are cached under the configured kernel cache
  (`[paths] kernel_cache`), keyed on the source bytes.
- The SM70 prefill library includes CUTLASS warp-level headers
  (`cutlass/gemm/warp/mma_tensor_op_sm70.h`, `mma_tensor_op_policy.h` and the
  headers they include). Provide a CUTLASS `include` directory.
  `ignition/components/CUTLASS-LICENSE.txt` carries the CUTLASS BSD-3-Clause
  license.

## Prepare

The validated package used these `PreparationConfig` values for
`ignition.recipes.qwen38_dense.bootstrap.prepare`:

| Field | Value |
| --- | --- |
| `context` | `32768` |
| `max_active` | `4` |
| `prefill_chunk_size` | `128` |
| `mtp` | `True` |
| `speculative_widths` | `(1, 2, 3, 4, 5)` (draft depth 4) |
| `cutlass_include` | a CUTLASS `include` directory |
| `kernel_cache` | a writable directory |

The release run log records the 32,768-token context, 4 agents and draft
depth 4. Prefill chunk 128 is the value recorded in the component validations
(commits `8ad725c` and `ca16f00`).

Prepare on a V100, the same architecture as the serving device. Preparation of
the release package took 54.6 min on one card, including CPU contention from a
concurrent test run.

## Serve

```sh
./flare/target/release/flare run model-32k.safetensors --gpus 0 --memory-limit 31G
```

- `--scheduler-policy agent` suits coding agents that block on each tool
  result.
- Fixed-width verification is the default. Leave
  `--mtp-adaptive-verification` off when outputs must not depend on batch
  composition.
- Startup verifies stored data and loads about 20 GB of weights. From a
  spinning disk this took 389.7 s plus 101.8 s, so allow for it in readiness
  timeouts.

## Gates passed

All results are for the release tree (`0e4c81f`) and its package on one V100
32 GB, 2026-10-22.

| Gate | Result |
| --- | --- |
| Strict export validation, including every `forced_prefix_*` check | 47/47 |
| Long-sequence invariance | 28/28 |
| Commit growth, including a negative capacity round | 4/4 |
| Consumer identity, components, accuracy (full-vocabulary KL) | pass |
| Fixed coding-token goldens, exact identity against the prior package | pass |
| Native server, MTP depth 4, 1 agent and 4 agents | 24/24 requests ok |
| Long context, 4 agents x 22.4k-30.2k prompts at 31G | 24/24 ok, all four resident, no eviction |
| CPU suites (`tests/recipes`, `tests/preparation`, `tests/cli`, replay tests) | no new failures against the `718a187` base |

The CPU suites had 16 failures, 1,544 passes, 860 skips and 100 errors, against
16, 1,536, 855 and 100 on the base. Every failure and error also occurs on the
base, apart from one cluster-export test that passed 3/3 when run alone. It
failed only while the suite shared the CPU with a preparation run.
