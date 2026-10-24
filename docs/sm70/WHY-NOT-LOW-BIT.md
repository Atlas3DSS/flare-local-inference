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

Of these routes, only llama.cpp was measured by this fork, on the same model and
card type ([same model on llama.cpp](#same-model-on-llamacpp)). The figure
quoted for INT8 DP4A is from a public project, with its own caveats, and does
not compare directly with ours.

**GGUF K-quants with llama.cpp** (for example Q4_K_M or Q3_K_S).
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
requantizes a higher-precision checkpoint into its own block formats and runs
them on NVIDIA GPUs, Volta included. This works on a wide
range of models and cards. Applied to this model, it is a different set of
weights: a K-quant requantizes the checkpoint into its own block formats, with
its own quantization error. Its server has its own slot and prompt-cache
design. The measured comparison on this model follows below.

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

## Same model on llama.cpp

On 2026-10-23 the fork ran the same checkpoint on llama.cpp, on the same card
type (Tesla V100 PCIe 32 GB), with the same public prompts, prefill token IDs,
sampling and FP32 reference as the release benchmarks. This port's columns are
the previous package `fd7c24dc` on the release binary `d4ffb1fa`, which
prefills in 128-token chunks; where the current package `f21b9abb` was measured
with the same protocol, its figure is given beside them. Receipts:
[a1-summary.json](receipts/measurements/a1-summary.json),
[a1-llamacpp-kl.json](receipts/measurements/a1-llamacpp-kl.json) and
[gguf.sha256](receipts/measurements/gguf.sha256).

- **Engine.** llama.cpp at commit `8e5a58df` (unmodified upstream), built for
  `sm_70` only with CUDA 12.9.86, flash attention on, every layer on the GPU and
  F16 KV unless stated.
- **Weights.** GGUF files converted from `nvidia/Qwen3.8-27B-NVFP4` at
  `482ca0f`: a BF16-outtype file (SHA-256 `8b7e7e4f…`), which keeps the NVFP4
  MLP tensors and head as stored and converts the FP8 projections to BF16;
  Q4_K_M (`d4e22cb3…`, 4.82 bits per weight) and Q8_0 (`9f335777…`, 8.31 bits
  per weight) requantized from it. The head stays NVFP4 in both, because
  upstream llama.cpp loads the head's per-tensor scale (`output.scale`) only for
  an NVFP4 head and would drop it from a requantized one. The main GGUFs leave
  out the MTP head. For speculation, the MTP head was exported separately as a
  draft GGUF, repaired as described in
  [BENCHMARK.md](BENCHMARK.md#llamacpp-with-this-checkpoints-mtp-head)
  ([mtp-fixed.sha256](receipts/measurements/mtp-fixed.sha256)).

Decode with speculation on and off for both engines, in tokens per second per
request after the first token
([speculation-a1-summary.json](receipts/measurements/speculation-a1-summary.json),
[speculation-a2-summary.json](receipts/measurements/speculation-a2-summary.json),
[a1-spec-results.json](receipts/measurements/a1-spec-results.json),
[a2-spec-results.json](receipts/measurements/a2-spec-results.json),
[a1-flare-d0-run.json](receipts/measurements/a1-flare-d0-run.json)):

| Engine and speculation | 1 agent, short prompts | 1 agent, 28.7k-token context | 4 agents, short prompts, per agent | 4 agents, 28.7k-token context, per agent (a) |
| --- | ---: | ---: | ---: | ---: |
| This port, depth-4 MTP, previous package | 83.0 | 72.4 | 64.8 | 41.5 |
| This port, depth-4 MTP, package `f21b9abb` | 83.2 | 73.5 | 64.9 | 38.2 |
| This port, speculation off (`--mtp-draft-tokens 0`), previous package | 30.9 | 27.0 | 28.3 | 19.2 |
| llama.cpp Q4_K_M with the checkpoint's MTP head (`--spec-type draft-mtp`), best depth | 64.6 (depth 3) | 55.9 (depth 3) | 31.8 (depth 4) | 22.3 (depth 4) |
| llama.cpp Q4_K_M, no speculation | 33.2 | 28.2 | 20.5 | 14.3 |
| llama.cpp Q4_K_M, n-gram speculation (`ngram-simple`) | 32.8 | 28.0 | 20.8 | 13.1-14.7 |

Short prompts generate 128 tokens and long ones 512. (a) llama.cpp prefills the
four long prompts one after another, so its four-agent long value is the steady
per-agent rate while all four decode; this port's is the median decode rate of
the 12 long requests. The no-speculation llama.cpp row was measured again in
this run; the earlier same-model run gave 28.4 and 20.8 where this run gives
28.2 and 20.5.

In plain words:

- **Speculation off.** llama.cpp is 4-7 % faster at one agent (33.2 against
  30.9, 28.2 against 27.0). This port is 1.34-1.38 times faster at four agents
  (28.3 against 20.5, 19.2 against 14.3).
- **Speculation on.** This port is 1.28-1.30 times faster at one agent and
  1.86-2.04 times faster at four. Speculation speeds this port up 2.16-2.69
  times and llama.cpp 1.55-1.98 times. Both engines use the same MTP head and
  accept drafts at a similar rate at depth 4: 0.70 on short and 0.72 on long
  prompts on this port's server counters, against 0.69 at one agent and 0.70
  at four on llama.cpp. This port's advantage with speculation therefore comes
  from the cost of each verification step and from batching four agents, not
  from a higher acceptance rate.
- **n-gram speculation** gave no gain on these prompts.

Checks:

- llama.cpp's greedy output with every speculative variant (MTP depth 1-4, the
  BF16 draft head and n-gram) matched the run without speculation on 4 of 4
  short prompts, 128 tokens each.
- This port's sampled outputs with speculation off and at depth 4 cannot be
  compared token for token, because speculative sampling consumes randomness
  differently (0 of 24 short and 0 of 24 long requests matched). Its greedy
  identity with speculation rests on the export gates (strict validation and
  long-sequence invariance).
- The speculation-off run used the same depth-4 package with
  `--mtp-draft-tokens 0`, which the release server accepts. Its counters
  record no draft or verification steps and a 131,072-token pool.

Prefill and numerics, which speculation does not affect:

| Metric | This port, previous package | This port, package `f21b9abb` | llama.cpp Q4_K_M | llama.cpp Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| TTFT, 1 request, 2,048 / 8,192 / 22,528 / 30,720 tokens (s) | 2.54 / 10.43 / 30.39 / 42.75 | 2.06 / 8.53 / 25.08 / 35.46 | 3.25 / 11.57 / 34.21 / 49.27 | not measured |
| Effective prefill, 1 request, same lengths (tok/s) | 807 / 785 / 741 / 719 | 994 / 961 / 898 / 866 | 630 / 708 / 659 / 623 | not measured |
| KL(build ‖ FP32 reference), mean of 3 prompt means / max, 141 positions | 0.0011 / 0.021 | not repeated (a) | 0.0088 / 0.068 | 0.0013 / 0.034 |
| Top-1 agreement with the FP32 reference (of 141) | 135 | not repeated (a) | 137 | 136 |

(a) The SM70 capture was not repeated on the current tree. The current
package's export-accuracy output is byte-identical to the previous package's.

Read the tables with these notes:

- **Weights differ.** This port executes the checkpoint's NVFP4 and FP8 weights
  as stored. Q4_K_M and Q8_0 requantize them, with the head left NVFP4. KV is
  BF16 on this port and F16 on llama.cpp.
- **Numerics.** This port's KL row is the SM70 capture of ordinary graphs from
  the release tree ([distance from the FP32 reference](ARCHITECTURE.md#distance-from-the-fp32-reference));
  the release export's own accuracy measurement gives 0.0012 mean, 0.023 max and
  top-1 agreement on 136. Q8_0's mean KL from the FP32 reference (0.0013) is
  close to this port's, and Q4_K_M's (0.0088) is about eight times higher.
  llama.cpp matches the reference's top-1 token on one or two more of the 141
  positions than this port's capture. Q8_0 was measured for KL only.
- **TTFT.** Both columns are client times, including queueing and transport.
  llama.cpp's own prompt-evaluation rate is higher than its effective rate here:
  783 / 774 / 696 / 656 tok/s at 1 request.
- **Cards.** The two engines ran on two physical V100 PCIe 32 GB cards in the
  same host, alongside other work on other cards.

### Four long-context agents on one card

llama.cpp Q4_K_M ran with `-np 4 -c 131072`, 32,768 tokens per slot, in F16 and
q8_0 KV variants, with cold requests and 256 output tokens. This port's column
is the previous package's release long-context run, with the current
package's beside it.

| Four agents, one V100 PCIe 32 GB | llama.cpp Q4_K_M, F16 KV | llama.cpp Q4_K_M, q8_0 KV | This port, BF16 KV, previous package | This port, BF16 KV, package `f21b9abb` |
| --- | ---: | ---: | ---: | ---: |
| Peak device memory (nvidia-smi) | 24,356 MiB | 20,988 MiB | 31,282 MiB, pre-allocated (131,584-token pool at `--memory-limit 31G`) | 31,606 MiB, pre-allocated (124,928-token pool at `--memory-limit 31G`) |
| All four 32K-class contexts resident | yes | yes | yes | four 30,720-token contexts fit; four 32,768-token contexts do not |
| Four cold 30,720-token prompts: first token per agent (s) | 59.8 / 107.9 / 156.7 / 200.3 | 57.4 / 105.9 / 155.1 / 199.1 | 170.2-171.0, all four together | 139.7-142.4, all four together; 36.2 / 71.9 / 107.6 / 145.1 with `--prefill-srpt` |
| Four public 28.7k-token agent prompts, first send: first token per agent (s) | 57.8 / 102.2 / 146.8 / 186.3 | 54.7 / 99.4 / 144.3 / 184.1 | 157.96-159.80, all four together (b) | 123.65 / 125.49 / 138.65 / 142.08 (b) |
| Decode per agent while all four decode (tok/s) | 14.3; 13.9 after 30,720-token prompts | 13.4; 13.2 | 40.2-46.1, median 41.9 (depth-4 MTP) | 27.6-63.4, median 43.1 (first batch); median 38.2 over 12 requests |

(b) The previous package's batch is not labelled cold: no request reports
cache credit, but the server's reused-token counter moved during it
([BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md)). This package's first batch is cold; its four
decode windows overlap only partly.

- llama.cpp prefills the four requests largely one after another. The first
  agent starts decoding after about a minute and the last waits about 200 s.
  This port's previous package interleaves the four prefills, and none of them
  emits a token before all four have finished; with package `f21b9abb` the four
  first tokens of the cold batch arrive from 123.65 to 142.08 s. Total prompt throughput for four 30,720-token prompts
  is 608-613 tok/s on llama.cpp, 719 tok/s on this port's previous package and
  863 tok/s on package `f21b9abb`.
- The KV formats differ: F16 and BF16 store 2 bytes per element and q8_0 about
  1. On llama.cpp, q8_0 KV saves 3,368 MiB and costs 2-6 % of decode rate at
  these lengths.
- Output lengths differ (256 against 512 tokens), and the decode row carries the
  speculation and requantization notes above.
- No shared-prefix run was possible: the four public long prompts share only a
  95-character prefix, about 20 tokens.

### vLLM and SGLang on the same card

On 2026-10-24 the fork also tried vLLM and SGLang with this checkpoint on the
same card type ([engines-README.md](receipts/engines/engines-README.md),
[engines-summary.json](receipts/engines/engines-summary.json); one results file per
attempt in [receipts/engines/](receipts/engines/)). "Official" means the project's own
release wheel; "fork" means a community project.

- **Cannot serve on V100 today: official vLLM 0.32.0 and SGLang 0.5.22.**
  vLLM 0.32.0 stops before loading with `The quantization method
  modelopt_mixed is not supported for the current GPU. Minimum capability: 75.
  Current capability: 70.` SGLang 0.5.22 fails at its first CUDA operation with
  `CUDA error: no kernel image is available for execution on the device`. Their
  kernels and the PyTorch builds they pin contain no `sm_70` code
  ([kernel-archs.txt](receipts/engines/kernel-archs.txt)).
- **Official vLLM 0.18.0** (`bcf2be96`), the last release wheel with `sm_70`
  kernels, refuses this NVFP4 checkpoint (`Minimum capability: 89`). It does
  serve a community GPTQ-Int4 quantisation of the BF16 base model
  (`Max73333/Qwen3.8-27B-GPTQ-Int4-V100` at `c7ed2123`) on one card. That is a
  different quantisation source from the checkpoint the other rows run.
- **1Cat-vLLM v1.5.1** (`589a5d1e`), a community fork, serves the same NVFP4
  file as this port on one card. Its load uses almost the whole card: with its
  MTP head, and in both KL captures, the load peaked at 32,114-32,242 MiB,
  above the 31,744 MiB per-card cap the fork's measurements run under, so its
  MTP and KL runs could not complete there. In the measured run the driver's
  sampler saw 32,266 MiB in use, so that run may also have passed the cap
  briefly. A V100 owner without that cap is not bound by it; whether MTP and
  the KL capture then fit was not measured.

| Engine, one V100 PCIe 32 GB | Weights | Decode, 1 user (tok/s) | Decode per user, 4 users (tok/s) | TTFT, one 30,720-token prompt (s) | Mean KL to the FP32 reference |
| --- | --- | ---: | ---: | ---: | ---: |
| This port, MTP depth 4 (package `f21b9abb`, sampled at T 0.7) | NVFP4 checkpoint | 83.2 | 64.9 | 35.5 | 0.0011 |
| This port, speculation off (previous package, sampled at T 0.7) | NVFP4 checkpoint | 30.9 | 28.3 | n/a | same as above |
| llama.cpp `8e5a58df`, no speculation (sampled at T 0.7) | Q4_K_M from the checkpoint | 33.2 | 20.5 | 49.3 | 0.0088 |
| 1Cat-vLLM v1.5.1 (fork), no speculation, greedy | NVFP4 checkpoint | 29.3 | 27.9 | 32.9 | not measured (memory cap) |
| vLLM 0.18.0 (official), no speculation, greedy | GPTQ-Int4 of the BF16 base model | 39.9 | 30.7 | 191.2 | 0.0383 |
| vLLM 0.32.0 (official) | NVFP4 checkpoint | cannot serve on V100 | | | |
| SGLang 0.5.22 (official) | NVFP4 checkpoint | cannot serve on V100 | | | |

The vLLM-family rows are greedy; this port's and llama.cpp's rows were
sampled at temperature 0.7 ([same-model results above](#same-model-on-llamacpp)).
The vLLM 0.18.0 KL compares a different quantisation of the base model, run on
a kernel that vLLM itself logs as buggy, with a reference made from the NVFP4
checkpoint ([kl-vllm0180-gptq.json](receipts/engines/kl-vllm0180-gptq.json)). Its first
answer after launch took about 15.5 minutes (Triton compilation).

In plain words: without speculation, single-card decode is similar across
these engines (29-33 tok/s for the NVFP4 and Q4_K_M runs, 39.9 for the
GPTQ-Int4 checkpoint). This port's lead comes from its MTP speculation. For
prefill, the 1Cat-vLLM fork is ahead on one card: 32.9 s against this port's
35.5 s for a 30,720-token prompt.

**Four PCIe V100s.** 1Cat-vLLM with tensor parallelism over four
V100-PCIE-32GB cards on one PCIe switch, without NVLink, gave 63.0 tok/s for
one user, 50.3 per user with four (203.4 aggregate) and a 15.8 s first token
at 30,720 tokens ([t1-1cat151-tp4-fp16kv-results.json](receipts/engines/t1-1cat151-tp4-fp16kv-results.json),
[host-facts.txt](receipts/engines/host-facts.txt)). 1Cat's own published four-card numbers
are on NVLink-connected V100-SXM2 cards, which is different hardware.

| Four V100 PCIe 32 GB | Users | Decode per user (tok/s) | Aggregate (tok/s) | TTFT, 30,720 tokens (s) |
| --- | ---: | ---: | ---: | ---: |
| 1Cat-vLLM TP4, 1 user (measured, greedy) | 1 | 63.0 | 63.0 | 15.8 |
| 1Cat-vLLM TP4, 4 users (measured, greedy) | 4 | 50.3 | 203.4 | n/a |
| This port, four independent single-card servers | | pending: measurement not yet published | | |

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
| This port, package `f21b9abb`, depth-4 MTP (speculative) | Qwen3.8-27B | NVIDIA NVFP4 + FP8, unchanged | 1 / 1 | 83.2 | 83.2 | export-accuracy output byte-identical to the previous package's | [BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md#package-f21b9abb-512-token-prefill-chunks-on-server-binary-48b69ed4) |
| This port, package `f21b9abb`, depth-4 MTP (speculative) | Qwen3.8-27B | same | 1 / 4 | 64.9 per stream | 64.9 per stream | same | same |
| This port, previous package on the release binary, depth-4 MTP (speculative) | Qwen3.8-27B | NVIDIA NVFP4 + FP8, unchanged | 1 / 1 | 83.0 | 83.0 | KL 0.0011 vs an FP32 forward of the same checkpoint (141 positions) | [previous-package section](../../BENCHMARKS-SM70.md#previous-package-fd7c24dc-on-release-binary-d4ffb1fa-128-token-prefill-chunks) |
| This port, previous package on the release binary, depth-4 MTP (speculative) | Qwen3.8-27B | same | 1 / 4 | 64.8 per stream | 64.8 per stream | same | same |
| llama.cpp Q4_K_M, no speculative decoding (same model, same card type) | Qwen3.8-27B | Q4_K_M requantized from the checkpoint, NVFP4 head, F16 KV | 1 / 1 | 33.2 | 33.2 | KL 0.0088 vs the same FP32 forward | [a1-summary.json](receipts/measurements/a1-summary.json) |
| llama.cpp Q4_K_M, no speculative decoding | Qwen3.8-27B | same | 1 / 4 | 20.8 per stream | 20.8 per stream | same | same |
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
  serve weights of about 2 to 4 bits. A same-model quality run on Strata is not
  possible: Strata loads only Qwen3.8-Flash-Next and Qwen3.6-35B-A3B
  ([MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)), and its Qwen3.6 engine refuses any other file
  ([q36_engine.cu](https://github.com/Niko1221/Strata/blob/main/src/qwen35/q36_engine.cu), "runs Qwen3.6-35B-A3B (qwen35moe) files only"), so it
  cannot load this 27B model. No FP32 reference exists here for those two models,
  so the scales cannot be joined that way either. The same-model comparison that
  is possible is [llama.cpp](#same-model-on-llamacpp).
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
| 512-token prefill (`4acbecf0`) | Exact prefill fusions and a per-128-row attention split plan: larger chunks with the same tokens | prefill 23.1 % / 20.5 % faster at 2,048 / 30,720 tokens, same goldens and accuracy output; KV pool 124,928 tokens at 31G (-5.1 %) |
| Scheduler hang fix (`e81534a`, `718a187`) | 4-agent serving with the host prefix store keeps making progress | progress tests: 4 passed, 1 ignored on `fe058dc` |

## Trade-offs and benefits

**What we give up on Volta:**

- **Arithmetic work per weight.** With no FP4 or FP8 tensor cores, every packed
  weight is decoded to FP16 in registers before the FP16 multiply. On SM120 the
  tensor cores consume FP4 and FP8 directly. We have not measured SM120 speed
  for comparison.
- **Prefill and time to first token.** Long prompts are prefill-bound. With
  package `f21b9abb`, effective prefill is 866-994 tok/s, one request at a
  time: a 30,720-token prompt waits 35.46 s for its first token alone and
  139.7-142.4 s when four arrive together (previous package: 719-807 tok/s,
  42.7 s and 170.2-171.0 s; [BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md)).
  The measured reason is the projection arithmetic. The previous package
  prefills in 128-token chunks, and in a profile of one of its prefills on the
  release binary the projection GEMMs,
  where every packed weight is decoded to FP16, took 63-66 % of the time. The
  GDN recurrence took about 11 %, attention 7-10 %, and the GPU was idle only
  1.4-1.9 % ([prefill-profile.json](receipts/measurements/prefill-profile.json)). On real coding-agent traces
  prefill took about 85 % of busy time (STATUS.md line 37).
- **Scope.** One recipe, one card per process (TP1), at most 4 concurrent
  requests and 32,768 tokens each. NVFP4 is weight-only; quantized activations
  are not used. Other recipes, tensor parallelism, Windows and the shared changes
  on SM120 are untested ([LIMITATIONS.md](LIMITATIONS.md)).
- **Upstream distance.** The branch is based on upstream `e7b8c01`. Upstream
  `main` (`e19549a`, 2026-10-23) has 816 commits that are not in this branch,
  96 of them on its first-parent line.
- **Old hardware.** The V100 is several generations old. Power per generated
  token was not measured.
- **No end-to-end real-trace number for either release binary.** The real-trace
  figure, 47.23 tok/s at best (line 68), used the 2026-10-22 release-tree
  package with a pre-release binary.

**What we gain:**

- **The checkpoint's own weights.** No requantization, no calibration set, no
  retraining, and no second quality evaluation of a new quantization.
- **Exact speculative decoding.** MTP verification at every width gives the
  same tokens as ordinary greedy decoding, alone or co-scheduled (line 62).
- **A large lossless KV pool.** 131,584 tokens at `--memory-limit 31G` with
  4 agents for the 128-token-chunk packages (lines 49, 65, 66), and 124,928
  tokens for the current 512-token-chunk package. Four prompts of 22.4k to 30.2k tokens, 110,022 in
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
| llama.cpp GGUF K-quants | Requantized from this checkpoint (head kept NVFP4); KL to the same FP32 reference 0.0088 (Q4_K_M) and 0.0013 (Q8_0) | Yes, same model and card type: Q4_K_M 33.2 tok/s decode without speculation and 64.6 with the checkpoint's MTP head (this port 30.9 and 83.0), TTFT 49.3 s at 30,720 tokens (this port 42.75 s) | [same model on llama.cpp](#same-model-on-llamacpp) |
| INT8 DP4A (superl8) | 3-bit GGUF weights on INT8 `dp4a` kernels; Qwen3.6-27B Q3_K_S: 21.3 tok/s steady decode, 7.2 tok/s end to end, CMP 100-210 | No | [qwen3-scoreboard.json](https://github.com/jajmangold/superl8/blob/main/bench/qwen3-scoreboard.json) |
| Strata | Different model (125 B MoE) as GGUF, UD-IQ4_XS on V100: 88.91 % top token vs Q8_0; 38-51 tok/s on one V100, one stream | No (our 2026-10-20 run: 56-64 tok/s on 3 V100s, one stream) | [Strata](https://github.com/Niko1221/Strata), [QUANTS.md](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md) |
| Project Maya | Different model (321 B MoE) as its own 2-4-bit GGUF: 83-90 % next token vs FP8; 26.8 tok/s Maya-S24 on one V100, 48.25 tok/s Maya-L on four | No | [Project Maya](https://github.com/mw00/project-maya), [#130](https://github.com/mw00/project-maya/issues/130) |
| W8A8 / SmoothQuant | INT8 weights and activations, calibrated; on Volta via `dp4a` | No | [SmoothQuant](https://arxiv.org/abs/2211.10438) |
| FP16 dense | FP16 conversion of the BF16 model (FP16 has a smaller range than BF16), about 54 GB; does not fit one 32 GB card | No (arithmetic) | two bytes per parameter |
