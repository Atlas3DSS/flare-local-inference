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
restating the values (line 62). The KL is small, but it is not zero. SM70 outputs are not bit-identical to that
reference. The reference runs the same 4-bit checkpoint, so this KL measures the
port's arithmetic only. It does not include the error of NVIDIA's NVFP4
quantization relative to the BF16 model, which this port inherits unchanged and
did not measure.

Measured since, but not exact: a comparison with Flare on SM120 (2026-10-23,
ordinary graphs, 3 public prompts x 47 positions). Full-vocabulary
KL(SM70 ‖ SM120) averaged 0.0015 to 0.0024 per prompt, top-1 agreement was
0.979 to 1.000, and one of three independent greedy continuations parted after
9 tokens ([comparison](ARCHITECTURE.md#sm70-and-sm120-comparison-ordinary-graphs);
receipt [sm70-vs-sm120-comparison.json](receipts/sm70-vs-sm120-comparison.json)).
The SM70 kernels are new code with their own reduction orders, and SM70 runs
NVFP4 projections weight-only: it does not quantize activations to FP4. The
comparison is not hardware-only, since each architecture keeps its own recipe
arithmetic and KV format, and it covers neither MTP nor native serving. So it
does not show that a V100 emits the same tokens as an SM120 GPU.

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

## Strata and Project Maya

Two public engines serve large models on V100s. Neither serves Qwen3.8-27B, and
this fork has measured neither on its release, so nothing below is a
like-for-like comparison.

**[Strata](https://github.com/Niko1221/Strata)** is a C++/CUDA engine for
Qwen3.8-Flash-Next, which its README describes as a 125-billion-parameter model,
and for Qwen3.6-35B-A3B. It keeps attention and the most-used experts in VRAM,
all experts in RAM, and computes RAM-resident experts on the CPU
([HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md)).

- **Volta.** A community-tested build, outside its supported cards, compiled
  with CUDA 12.x because CUDA 13 dropped Volta
  ([NVIDIA_V100.md](https://github.com/Niko1221/Strata/blob/main/docs/NVIDIA_V100.md)).
  The V100 build sets the compile-time experimental flag
  `-DSTRATA_EXPERIMENTAL_SM60=ON`. Strata's sm_70 tensor-core prompt attention
  (`prompt_attn_v70_kernel` on `mma.m8n8k4`) and its chunked GDN path are
  compiled only under that flag. Builds without it, including the ready-made
  engine for compute capability 7.5 and newer, contain neither
  ([qsa_prompt_attn.cu](https://github.com/Niko1221/Strata/blob/main/src/kernels/cuda/qsa_prompt_attn.cu),
  [gdn_chunk.cu](https://github.com/Niko1221/Strata/blob/main/src/prefill/gdn_chunk.cu)).
  In the V100 build the tensor-core prompt attention is the default for int8,
  FP16 and K8V4 KV, and `STRATA_PROMPT_ATTN_OLD=1` selects the older kernel
  (NVIDIA_V100.md).
- **Weights.** GGUF quantizations: Strata's own 1- to 3-bit sizes and
  Unsloth's 4- to 8-bit files
  ([MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)).
  The V100 measurements use Unsloth's UD-IQ4_XS with an int8 KV cache.
  Strata's own table puts UD-IQ4_XS at 88.91 % the same top token as Q8_0, with
  mean KL 0.101
  ([QUANTS.md](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md)).
  Its default Volta prompt-path expert kernels (MMQ, `dp4a`) quantize
  activations to 8 bits (q8_1). The opt-in FP16 tensor-core expert kernel uses
  FP16 activations instead, "so the answers differ" (NVIDIA_V100.md).
- **Speed on a V100.** One V100-PCIE-32GB with UD-IQ4_XS and MTP: decode
  38-51 tok/s (NVIDIA_V100.md). One community V100-SXM2-32GB: 28.16 and 28.64
  tok/s, then 30.92 and 32.94 tok/s with the opt-in Volta decode kernels
  ([COMMUNITY_BENCHMARKS.md](https://github.com/Niko1221/Strata/blob/main/docs/COMMUNITY_BENCHMARKS.md)).
  "Bit-identical" in that report means the opt-in kernels produced the same
  tokens as Strata's default kernels, not the same tokens as a reference.
- **Several cards.** Strata splits layers across cards, and the cards take turns
  on one request. A split adds expert-cache room, not speed for a single stream
  (NVIDIA_V100.md, "Limits";
  [MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)).
  A multi-card figure is one stream, not a sum of streams. Strata answers one
  request at a time by default; `--parallel N` serves several (its README).
- **Our own run.** On 2026-10-20 this fork ran Strata with Qwen3.8-Flash-Next
  on three V100s: 56-64 tok/s total across a 3-V100 layer split, single stream,
  decoding with MTP. The quantization used was not recorded. Divided by three
  cards, that is 18.7-21.3 tok/s per card, but in a layer split the cards take
  turns on one stream, so the division understates Strata. Its own 1-card V100
  figure is 38-51 tok/s.
- **Shared design.** Strata already commits its Gated DeltaNet state by
  replaying accepted tokens. Its verify half writes no state, and its commit half
  replays the first accepted tokens from stored inputs
  ([verify_kernels.hpp](https://github.com/Niko1221/Strata/blob/main/include/strata/kernels/verify_kernels.hpp)).
  That is the same idea as this port's GDN replay-commit
  ([ARCHITECTURE.md](ARCHITECTURE.md#4-gdn-replay-commit)).

**[Project Maya](https://github.com/mw00/project-maya)** builds on the Strata
engine; its MIT license credits Niko1221 and the Strata contributors. It serves
GLM-5.3-Flash, which its README gives as 321 B parameters with about 18 B active
per token, and DeepSeek-V4.1-Flash-1017 (748 B). It keeps the most-used experts
on the GPU, the next in RAM and the rest on NVMe.

- **Weights.** Maya's own GGUF quantizations of Z.ai's FP8 release: Maya-S
  96.5 GB, Maya-S24 94.7 GB (4-bit attention, for cards of 24 GB or less),
  Maya-M 116 GB and Maya-L 156.3 GB. Maya-L is IQ3_S
  ([issue #130](https://github.com/mw00/project-maya/issues/130)). Against the
  FP8 model, the README reports the same next token 83 % of the time for Maya-S
  and Maya-S24, 86 % for Maya-M and 90 % for Maya-L, with 97.7-99.2 % of its
  zero-shot accuracy. By our arithmetic (file size x 8 / 321 B parameters), that
  is about 2.4 bits per parameter on average for Maya-S and 3.9 for Maya-L.
- **Speed on V100s**, from user reports linked in the README's table:
  - one V100-PCIE-32GB, Maya-S24: 26.8 tok/s, mean of three 256-token answers
    from `./maya.sh --bench`, with 65-72 % of each token's experts in VRAM and
    the rest computed by the CPU
    ([#171](https://github.com/mw00/project-maya/issues/171));
  - four V100-PCIE-32GB, Maya-S (its v2 IQ2_XXS file): 69.3 tok/s, mean of
    three answers, with every expert in VRAM
    ([#327](https://github.com/mw00/project-maya/issues/327));
  - four V100-PCIE-32GB, Maya-L, concurrency 1, 24-30-token prompts: 26.7 tok/s
    on v1.1.0, and 48.25 tok/s on v1.1.3 after `--calibrate` (#130).
- **What `--calibrate` does.** It measures the host's CPU expert-lane threads and
  its PCIe share (the maintainer's reply in #130). It tunes the host side. It
  does not change precision. In #130, calibration alone raised decode by 34 %
  over the same version without it.
- **Concurrency.** The README says up to 8 requests can be answered together.
  #130 states concurrency 1; the `--bench` reports give a mean over answers and
  do not state concurrency.

### What the comparison shows

| Engine | Model | Weights on the V100 | Cards / streams | Decode tok/s | Per card | Agreement with a reference | Source |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| This port | Qwen3.8-27B | NVIDIA NVFP4 + FP8, unchanged | 1 / 1 | 90.4 | 90.4 | KL 0.0011 / 0.0024 vs an FP32 forward of the same checkpoint | STATUS.md line 66 |
| This port | Qwen3.8-27B | same | 1 / 4 | 70.7 per stream | 70.7 per stream | same | STATUS.md line 66 |
| Strata | Qwen3.8-Flash-Next, 125 B | UD-IQ4_XS GGUF, int8 KV | 1 / 1 | 38-51 | 38-51 | 88.91 % top token vs Q8_0 | NVIDIA_V100.md, QUANTS.md |
| Strata (our run, 2026-10-20) | Qwen3.8-Flash-Next | not recorded | 3 / 1 | 56-64 tok/s total across a 3-V100 layer split, single stream | 18.7-21.3 if divided by 3; Strata's own 1-card figure is 38-51 | not measured | fork notes |
| Project Maya | GLM-5.3-Flash, 321 B (18 B active) | Maya-S24, about 2.4 bits/param | 1 / `--bench` | 26.8 | 26.8 | 83 % next token vs FP8 | #171 |
| Project Maya | GLM-5.3-Flash | Maya-S v2 (IQ2_XXS) | 4 / `--bench` | 69.3 | 17.3 | 83 % (Maya-S) | #327 |
| Project Maya | GLM-5.3-Flash | Maya-L (IQ3_S), about 3.9 bits/param | 4 / 1 | 48.25 | 12.1 | 90 % | #130 |

Read the table with these limits:

- **Different models.** Strata and Maya serve mixture-of-experts models about
  4.6 and 12 times larger in total parameters (125 B and 321 B against 27 B),
  often with part of the experts in host RAM. Maya's model activates about 18 B
  parameters per token, fewer than this dense 27B model. This port serves a 27B model that fits on one card. Decode rates across
  different models do not rank the engines.
- **Different references.** Our KL compares the port with an FP32 forward of the
  same 4-bit checkpoint, so it measures kernel arithmetic. Strata's and Maya's
  figures compare their quantizations with a higher-precision release, so they
  measure quantization error. The numbers are not on one scale. All three routes
  serve weights of about 2 to 4 bits.
- **Different concurrency.** This port serves up to 4 requests per card
  together, with a shared KV pool and prefix cache. The Strata and Maya V100
  figures are one stream where stated; their multi-card figures are one stream
  spread over the cards.

What the evidence supports:

- **No per-card speed advantage to explain.** In the published V100 figures,
  Strata and Maya do not decode faster per card than this port; they decode
  slower. Their numbers do not show speed bought by lower precision, because
  they show no speed gain over this port at all. Our numbers do not show that
  this engine is faster on their models either; nobody has run the same model
  on both.
- **Their advantage is capacity.** They fit 125 B to 748 B-parameter models on
  cards that cannot hold them, by quantizing to 2-4 bits and tiering experts
  across VRAM, RAM and NVMe. This port cannot serve those models at all.
- **This port's advantage is multi-agent serving of a mid-size model.** It keeps
  NVIDIA's weights unchanged, runs 4 streams per card with exact speculative
  decoding, and keeps four 22-30k-token contexts resident.

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
  tensor cores consume FP4 and FP8 directly. We have not measured SM120 speed
  for comparison.
- **Prefill and time to first token.** Long prompts are prefill-bound. On the
  release binary, effective prefill is 719-807 tok/s, one request at a time: a
  30,720-token prompt waits 42.7 s for its first token alone and 170.2-171.0 s
  when four arrive together ([BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md)). On
  real coding-agent traces prefill took about 85 % of busy time (STATUS.md
  line 37).
- **Scope.** One recipe, one card per process (TP1), at most 4 concurrent
  requests and 32,768 tokens each. NVFP4 is weight-only; quantized activations
  are not used. Other recipes, tensor parallelism, Windows and the shared changes
  on SM120 are untested ([LIMITATIONS.md](LIMITATIONS.md)).
- **Upstream distance.** The branch is based on upstream `e7b8c01`. Upstream
  `main` (`e19549a`, 2026-10-23) has 816 commits that are not in this branch,
  96 of them on its first-parent line.
- **Old hardware.** The V100 is several generations old. Power per generated
  token was not measured.
- **No end-to-end real-trace number for the release binary.** The real-trace
  figure, 47.23 tok/s at best (line 68), used the 2026-10-22 release-tree
  package with a pre-release binary.

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
  31G (line 65). Strata uses the same verify-then-replay idea for its own GDN
  layers; what this port adds is the byte-exact proof against the old candidate
  path and the freed rows going to a shared KV pool for four requests.
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
| Strata | Different model (125 B MoE) as GGUF, UD-IQ4_XS on V100: 88.91 % top token vs Q8_0; 38-51 tok/s on one V100, one stream | No (our 2026-10-20 run: 56-64 tok/s on 3 V100s, one stream) | [Strata](https://github.com/Niko1221/Strata), [QUANTS.md](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md) |
| Project Maya | Different model (321 B MoE) as its own 2-4-bit GGUF: 83-90 % next token vs FP8; 26.8 tok/s Maya-S24 on one V100, 48.25 tok/s Maya-L on four | No | [Project Maya](https://github.com/mw00/project-maya), [#130](https://github.com/mw00/project-maya/issues/130) |
| W8A8 / SmoothQuant | INT8 weights and activations, calibrated; on Volta via `dp4a` | No | [SmoothQuant](https://arxiv.org/abs/2211.10438) |
| FP16 dense | FP16 conversion of the BF16 model (FP16 has a smaller range than BF16), about 54 GB; does not fit one 32 GB card | No (arithmetic) | two bytes per parameter |
