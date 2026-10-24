# Why not another low-bit route

The usual way to run a 27B model on a V100-class card is to requantize it into
a format the old card handles well. This port does something else: it serves
NVIDIA's own NVFP4 package of Qwen3.8-27B on Volta, with the checkpoint's
weights unchanged. This page compares the two approaches, lists what each one
gives up, and says what this port is useful for.

## What the served model is

The package serves `nvidia/Qwen3.8-27B-NVFP4` at revision `482ca0f`. This is
upstream's own low-precision format, not a full-precision model:

- MLP and vocabulary weights are NVFP4: 4-bit E2M1 values with an FP8 (E4M3)
  scale per block of 16 values and a per-tensor FP32 scale;
- attention projections are tensor-scaled FP8 (E4M3);
- norms and the recurrent state stay in higher precision. On SM70 they are held
  in FP32, with BF16 rounding applied where the model rounds to BF16.

Volta has no FP4, FP8 or BF16 tensor-core instructions. Its tensor cores
multiply FP16 only. Every E2M1 and E4M3 value is exactly representable in FP16,
so the SM70 kernels decode the packed weights to FP16 in registers, without
rounding, and accumulate in FP32
([ARCHITECTURE.md](ARCHITECTURE.md#why-volta-needs-its-own-path)). The weights
the tensor cores see are the checkpoint's weights. Nothing is requantized,
calibrated or retrained.

### What is proven, and what is not

Proven by the export gates on the release package (STATUS.md line 62):

- **Speculative decoding is exact on this port.** Accepting k verification rows
  leaves the same bytes of committed state as k single-token greedy decodes
  (`forced_prefix_*`, part of strict validation 47/47). Speculative and
  non-speculative greedy outputs are identical over 512 tokens, alone and
  co-scheduled (invariance 28/28).
- **Rebuilds are reproducible.** A rebuilt package reproduces the fixed
  coding-token goldens token for token (STATUS.md lines 54, 62).
- **Replay-commit is exact.** The replayed recurrent state equals the old
  candidate state byte for byte, compared as `uint8` views so that `-0.0` and
  `+0.0` count as different (`95b953f`; STATUS.md lines 43, 46).

Measured, not exact: closeness to an independent FP32 Hugging Face reference
forward of the same checkpoint, which uses no Flare code. The accuracy gate
records full-vocabulary KL(reference ‖ SM70). For the component packages just
before the release merge, the commit messages record a short-context
`kld_mean` of 0.0011443 and a long-context one of 0.0024010 (`8ad725c`,
`ca16f00`). The release run records the measurement as complete without
restating the values (line 62). The KL is small, but it is not zero. SM70
outputs are not bit-identical to that reference.

Not measured: a comparison with Flare on SM120. The SM70 kernels are new code
with their own reduction orders. SM70 runs NVFP4 projections weight-only: it
does not quantize activations to FP4. So nothing here shows that a V100 emits
the same tokens as an SM120 GPU.

## The usual options on a V100-class card

None of these routes was measured by this fork. The figure quoted for INT8 DP4A
is from a public project, with its own caveats, and does not compare directly
with ours.

**GGUF K-quants with llama.cpp** (for example Q4_K_M or Q3_K_S).
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
requantizes a higher-precision checkpoint into its own block formats and runs
them on NVIDIA GPUs, Volta included. This works on a wide
range of models and cards. Applied to this model, it is a different set of
weights: a K-quant is made from the BF16 model, not from NVIDIA's NVFP4 package,
and it has its own quantization error. Its server has its own slot and
prompt-cache design, which we have not compared with Flare's paged KV pool and
prefix cache.

**INT8 DP4A kernels.** Volta has no INT8 tensor-core instructions, but its CUDA
cores have the `dp4a` 4-way INT8 dot product. The public
[superl8](https://github.com/jajmangold/superl8) project and its
[superl8-serve](https://github.com/jajmangold/superl8-serve) server build on
it, for V100 and for the CMP 100-210, a V100-class card with tensor cores
disabled. Its
[scoreboard](https://github.com/jajmangold/superl8/blob/main/bench/qwen3-scoreboard.json)
reports, for `Qwen3.6-27B-Q3_K_S` on a "V100-labelled CMP 100-210 fleet card"
with a 256-token prompt and 128 new tokens: 21.3 tok/s steady decode and
7.2 tok/s end to end, without speculative decoding. The row is marked
`local-provenance`, with the checkpoint currently absent. That is a different
model version, a 3-bit GGUF quantization, a card without tensor cores and a
different harness. It does not compare directly with any number here.

**W8A8 (SmoothQuant style).**
[SmoothQuant](https://arxiv.org/abs/2211.10438) moves activation outliers into
the weights so that both can be INT8. It needs calibration data and changes the
numerics. On Volta, INT8 matrix products run on `dp4a`, not on tensor cores.

**FP16 dense.** Converting the weights to FP16 keeps them simple to run on
Volta's tensor cores, but a 27B-parameter model is about 54 GB in FP16 (two
bytes per parameter). It does not fit on one 32 GB card at all, let alone with
room for KV pages.

**BF16.** Volta has no BF16 arithmetic in hardware. The model's BF16 rounding
points have to be emulated, as this port does in FP32.

## What this port did

It kept upstream's NVFP4 package and upstream's model computation, and added
Volta kernels for the parts that need them
([ARCHITECTURE.md](ARCHITECTURE.md)):

| Change | What it buys | Evidence |
| --- | --- | --- |
| SM70 base port: exact FP16 decode of FP4/FP8 weights, FP32-held BF16, tensor-core attention, FP32 Gated DeltaNet | The NVFP4 package prepares and serves on a V100 at all | strict 47/47, invariance 28/28 (STATUS.md line 62) |
| pgemm prefill projections (`8ad725c`) | Register-B packed GEMM with a device live-row count; same bits as the previous kernel | 0 mismatches in captured-graph identity (commit message) |
| S1L decode, verification and draft attention (`6b808ef`, `0aea71a`, `ca16f00`) | Bit-exact multi-row verification, and MTP draft steps on tensor cores instead of a scalar sweep | 1 x 5 round 49.5-49.6 ms vs 55.1 ms without the S1L draft (STATUS.md line 64) |
| GDN replay-commit (`d3258bc` to `7cb1bc7`) | One committed recurrent-state row per request instead of five: 16 x 166 MiB returned to KV pages | KV pool 98,304 to 131,584 tokens at 31G, +33.9 % (line 49) |
| Scheduler hang fix (`e81534a`, `718a187`) | 4-agent serving with the host prefix store keeps making progress | progress tests: 4 passed, 1 ignored on `fe058dc` |

## Trade-offs and benefits

**What we give up on Volta:**

- **Arithmetic work per weight.** With no FP4 or FP8 tensor cores, every packed
  weight is decoded to FP16 in registers before the FP16 multiply. On SM120 the
  tensor cores consume FP4 and FP8 directly. We have not measured an SM120 run
  for comparison.
- **Prefill and time to first token.** Long prompts are prefill-bound. Four
  prompts of 22.4k to 30.2k tokens at 4 agents had a TTFT p50 of 90.7 s
  (STATUS.md line 65). On real coding-agent traces prefill took about 85 % of
  busy time (line 37).
- **Scope.** One recipe, one card per process (TP1), at most 4 concurrent
  requests and 32,768 tokens each. NVFP4 is weight-only; quantized activations
  are not used. Other recipes, tensor parallelism, Windows and the shared changes
  on SM120 are untested ([LIMITATIONS.md](LIMITATIONS.md)).
- **Upstream distance.** The branch is based on upstream `e7b8c01`. Upstream
  `main` (`e19549a`, 2026-10-23) has 816 commits that are not in this branch,
  96 of them on its first-parent line.
- **Old hardware.** The V100 is several generations old. Power per generated
  token was not measured.
- **No end-to-end real-trace number for the release.** The only real-trace
  figure, 47.23 tok/s, is for a different package (line 68).

**What we gain:**

- **The checkpoint's own weights.** No requantization, no calibration set, no
  retraining, and no second quality evaluation of a new quantization.
- **Exact speculative decoding.** MTP verification at every width gives the
  same tokens as ordinary greedy decoding, alone or co-scheduled (line 62).
- **A large lossless KV pool.** 131,584 tokens at `--memory-limit 31G` with
  4 agents (lines 49, 65, 66). Four prompts of 22.4k to 30.2k tokens, 110,022 in
  total, stay resident with zero evictions or restores (line 65). On SM70 the KV
  pages hold BF16, unquantized; upstream's recipes default to FP8 KV.
- **Concurrency.** 70.7 tok/s per agent at 4 agents and 90.4 tok/s at 1 agent on
  short prompts; 37.2 tok/s per agent and 82.2 tok/s at 1 agent for the
  22.4k-30.2k prompts (lines 65, 66). These are decode-only rates.
- **Prefix caching** in GPU memory and in a host-memory tier, shared with
  upstream.
- **Cheap cards.** 32 GB V100s are available second-hand at used-market prices.

## What makes this unique and useful

Each part below exists elsewhere in some form. What this port offers is the
combination.

- **Upstream's NVFP4 package on hardware without FP4 instructions.** The
  checkpoint's FP4 and FP8 values are used exactly, and the outputs are measured
  against an independent FP32 reference (KL above), not assumed.
- **One committed recurrent row per request.** GDN replay-commit keeps long
  multi-agent contexts resident: four 22-30k-token agents with zero evictions at
  31G (line 65).
- **MTP speculative decoding on Volta.** The S1L kernel serves target decode,
  verification and draft attention on tensor cores, and its row contract makes
  verification bit-exact.
- **Gated and reviewed changes.** Every package the fork measured was exported
  only after the strict, invariance, commit-growth and golden gates passed. The kernel commits
  record independent reviews; [ARCHITECTURE.md](ARCHITECTURE.md) lists the
  review history of each change, including the two that record a single review
  chain or none.
- **A reproducible measurement discipline.** End-to-end and decode-only results
  are published in separate tables, every row names its hardware, memory limit,
  concurrency, package, date and source line, and missing measurements are
  listed as missing ([BENCHMARK.md](BENCHMARK.md)).

## What it is useful for, and what not

Useful for:

- serving several coding agents with long contexts from one V100 32 GB;
- agent and lab fleets built from several such cards behind a router
  ([SERVE.md](SERVE.md#several-cards-behind-a-router));
- prefix-heavy workloads, where restored prefixes avoid re-prefilling;
- studying exact speculative decoding and bit-exact kernels on GPUs without
  BF16 tensor cores.

Not for:

- training or fine-tuning;
- BF16 workloads, or any model other than this package;
- more than 4 concurrent requests per card, or contexts above 32,768 tokens;
- workloads where time to first token on long prompts matters most;
- speed comparisons: these numbers are not a record claim, and they compare
  only with runs that report the same columns.

## Comparison

| Route | Numerics relative to the NVFP4 checkpoint | Measured here? | Source |
| --- | --- | --- | --- |
| This port (SM70 Flare) | Checkpoint weights unchanged. Speculative decoding exact. KL to an FP32 reference 0.0011 / 0.0024 (short / long). | Yes | STATUS.md lines 49-66; commits `8ad725c`, `ca16f00` |
| llama.cpp GGUF K-quants | Requantized from BF16 into a different format; its own quantization error | No | [llama.cpp quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) |
| INT8 DP4A (superl8) | 3-bit GGUF weights on INT8 `dp4a` kernels; Qwen3.6-27B Q3_K_S: 21.3 tok/s steady decode, 7.2 tok/s end to end, CMP 100-210 | No | [qwen3-scoreboard.json](https://github.com/jajmangold/superl8/blob/main/bench/qwen3-scoreboard.json) |
| W8A8 / SmoothQuant | INT8 weights and activations, calibrated; on Volta via `dp4a` | No | [SmoothQuant](https://arxiv.org/abs/2211.10438) |
| FP16 dense | FP16 conversion of the BF16 model (FP16 has a smaller range than BF16), about 54 GB; does not fit one 32 GB card | No (arithmetic) | two bytes per parameter |
