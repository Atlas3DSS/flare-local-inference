# ENGINES: vLLM and SGLang on one Tesla V100 32 GB, same model as Flare (2026-10-24, <measurement>)

**Question.** Can vLLM and SGLang serve Qwen3.8-27B on one V100-PCIE-32GB? If so, how do they compare with the Flare SM70 port?

**Setup.**
- Host: <release-host> .
- Single-card rows: <card> and <card>. One TP4 row, requested by the owner: <cards 0-3>.
- Every GPU run used `<lease wrapper> <card> <label> 31 <min>`, inside its own `systemd-run --user -p Nice=10` unit. That gives one lease manager lease per card, and observe.py stops the run above 31,744 MiB per card.
- All leases and units are released.
- **Official** means the vLLM or SGLang project's own release wheel. **Fork** means a community project. No row mixes the two.

## Table

**Protocol** (as in run 2/run 6, except the sampling):
- **Short decode:** public prompts code-00..03, chat template, thinking off. Prompt lengths measured here: 578, 579, 580 and 582 tokens.
  - 128 output tokens, EOS ignored, greedy (T 0), 3 reps.
  - c1 = median of 12 requests.
  - c4 = 4 requests released together. Per-agent value = median of 12 requests. Aggregate = median over the 3 batches of the summed per-request rates.
- **TTFT:** `tools/benchmark_prefill.exact_prompt` (run id `sm70-public-release-c6fdc120`, source sha 622fdfae).
  - 30,720 prompt tokens, 32 output tokens, c1.
  - 1 warmup + 3 measured; the median is reported.
- **Decode tok/s** is measured at the client: (completion tokens − tokens in the first chunk) / (time of last chunk − time of first chunk). Token counts come from `continuous_usage_stats`.
- **KL** = KL(candidate‖FP32 reference). This is the same harness as run 2: `llama_kl.py`, 3 public prompts × 47 teacher-forced positions, full-vocabulary FP64. The table gives the mean of the 3 per-prompt means.

| Engine | Official / fork, version, commit | Weights | Cards | Serves | Decode c1 (tok/s) | Decode c4 per agent (aggregate) | TTFT 30,720 (s) / prefill (tok/s) | KL vs FP32 ref | Exact error / note |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| **Flare SM70** (reference) | this repo, release package f21b9abb | nvidia/Qwen3.8-27B-NVFP4 @ 482ca0f3 | 1 | yes | 83.2 with MTP d4; 30.9 with speculation off (run 6) | 64.9 with MTP d4; 28.3 with speculation off | 35.5 (= 30,720 / 866) / 866 | 0.0011 | release receipts; sampling T 0.7, not greedy |
| llama.cpp (reference, run 2/run 6) | official 8e5a58df | Q4_K_M requantized from the NVFP4 | 1 | yes | 33.2; 64.6 with MTP d3 | 20.5 (82.1); 31.8 with MTP d4 | 49.3 / 623 | 0.0088 | sampling T 0.7 |
| vLLM | **official 0.32.0** (current; PyPI default wheel, torch 2.13.0+cu130) | NVFP4 @ 482ca0f3 | 1 | **no** | – | – | – | – | `Value error, The quantization method modelopt_mixed is not supported for the current GPU. Minimum capability: 75. Current capability: 70.`. Torch also warned: `Your installed torch==2.13.0+cu130 does not include kernels for this GPU.` |
| vLLM | **official 0.32.0+cu129** (GitHub release wheel, torch 2.13.0+cu129) | NVFP4 @ 482ca0f3 | 1 | **no** | – | – | – | – | Same `modelopt_mixed ... Minimum capability: 75. Current capability: 70.` (run f2b). The first attempt died earlier with `OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory`, because the resolver pulled PyPI torchcodec 0.17.0, a CUDA 13 build. The rerun used torchcodec 0.16.0+cu129 |
| vLLM | **official 0.18.0** (PyPI wheel; tag v0.18.0 = bcf2be96; last release wheel with sm_70 SASS; torch 2.10.0+cu128) | NVFP4 @ 482ca0f3 | 1 | **no** | – | – | – | – | `Value error, The quantization method modelopt_mixed is not supported for the current GPU. Minimum capability: 89. Current capability: 70.` |
| vLLM | **official 0.18.0**, same wheel, **unpatched** | GPTQ int4 g128 sym: Max73333/Qwen3.8-27B-GPTQ-Int4-V100 @ c7ed2123 (the one allowed download) | 1 | **yes** | **39.9** | **30.7 (116.9)** | **191.2 / 161** | **0.0383** (max 0.679; top-1 130/141) | Kernels: TRITON_ATTN plus FLA Triton GDN. The server logs `Currently, the 4-bit gptq_gemm kernel for GPTQ is buggy. Please switch to gptq_marlin.` gptq_marlin is not available on sm_70. Ready after 562 s; the first request then took 368 s (Triton JIT) |
| 1Cat-vLLM | **fork** github.com/1CatAI/1Cat-vLLM v1.5.1, source 589a5d1e, wheel sha 757f2200 | NVFP4 @ 482ca0f3 (the same file Flare uses) | 1 | **yes** | **29.3** | **27.9 (111.4)** | **32.9 / 933** | not captured (see note 3) | FP16 compute and KV, `--gpu-memory-utilization 0.88 --max-num-batched-tokens 4096`, no speculation. Its load needs about the whole card (see note 3) |
| 1Cat-vLLM + MTP | fork v1.5.1 with `--speculative-config {"method":"mtp","num_speculative_tokens":4}` (the checkpoint's own MTP head) | NVFP4 @ 482ca0f3 | 1 | not measurable under the 31 GiB lease | – | – | – | – | Killed twice by the lease observer during load: `31744 MiB lease exceeded`. Process peaks were 32,174 and 32,114 MiB; target + MTP weights are 23.26 GiB |
| 1Cat-vLLM | fork v1.5.1, TP4 | NVFP4 @ 482ca0f3 | **4** (<cards 0-3>) | **yes** | **63.0** | **50.3 (203.4)** | **15.8 / 1,948** | not captured | FP16 KV, util 0.80, 8192-token chunks, no speculation. 27,688 MiB per card. Host facts in note 6 |
| SGLang | **official 0.5.22** (current; PyPI; torch 2.14.1+cu130, sglang-kernel 0.4.9) | NVFP4 @ 482ca0f3 | 1 | **no** | – | – | – | – | `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`, at the first CUDA op (`torch.ones` in `GroupCoordinator.__init__`). Same result with default flags and with `--dtype half --attention-backend triton --sampling-backend pytorch` |
| pmaxx73/qwen38-v100 (Docker recipe) | **fork** @ 8b15e9e4 (vLLM v0.18.0 + 2 python patches) | GPTQ above | 1 | not run as Docker | – | – | – | – | `bash: line 1: docker: command not found` on <release-host>; installing Docker needs root. Its patches applied cleanly to the official 0.18.0 wheel (`venv-vllm-0.18.0-pmaxx73`). That patched venv was never needed, because the unpatched official wheel already served the GPTQ checkpoint |

### TP4 row against four independent single-card servers (same four cards)

| Use of 4 cards | Streams | Per-stream decode (tok/s) | Aggregate decode (tok/s) | TTFT 30,720 |
| --- | ---: | ---: | ---: | ---: |
| 1Cat-vLLM TP4, c1 (measured) | 1 | 63.0 | 63.0 | 15.8 s |
| 1Cat-vLLM TP4, c4 (measured) | 4 | 50.3 | 203.4 | – |
| 4 × Flare, c1 each (arithmetic from release receipts, not a 4-card run) | 4 | 83.2 | 332.8 | 35.5 s |
| 4 × Flare, c4 each (arithmetic from release receipts, not a 4-card run) | 16 | 64.9 | 1,038 | – |
| 4 × 1Cat-vLLM TP1, c4 each (arithmetic from this job's c3 run) | 16 | 27.9 | 446 | 32.9 s |

## Notes that must travel with the table

1. **Sampling and speculation.**
   - Every engines row is greedy, as this brief asks. The Flare and llama.cpp rows were sampled at T 0.7 / top-p 0.95 / top-k 20 (run 2, run 6).
   - No engines number uses speculation. The Flare and llama.cpp MTP values are quoted from their own receipts.
   - 1Cat's published headline numbers use a separate DFlash2 draft model (incoai/Qwen3.8-27B-DFlash2, 3.85 GB). It was not downloaded: the brief allows one alternative checkpoint, and that went to the GPTQ.
2. **The weights differ in the GPTQ row.**
   - Max73333 quantized `Qwen/Qwen3.8-27B` (BF16) with GPTQModel 7.3.2. It was not made from the NVFP4 checkpoint.
   - The FP32 reference comes from the NVFP4 checkpoint. The GPTQ KL therefore mixes three things: two different quantizations of the base model, plus the kernel that vLLM itself labels buggy.
   - Per prompt, KL mean / max: 0.0221 / 0.063, 0.0618 / 0.679, 0.0310 / 0.157.
   - The capture used the offline `LLM` API in eager mode with max_num_seqs 1. The first capture attempt ran out of memory in the full-vocabulary sampler warm-up. The bench used compiled CUDA graphs.
   - The 1Cat rows run the same NVFP4/FP8 file as Flare, through 1Cat's TurboMind weight-only SM70 path (logged: `SM70 ModelOpt NVFP4 TurboMind dense path enabled (weight-only; activations remain half)`).
3. **1Cat on one card fills the card.**
   - Weights take 21.01 GiB. Right after weight loading there is a short spike that the lease observer saw in 4 of 6 single-card loads:
     - 32,302 MiB (util 0.90, 8192-token chunks);
     - 32,174 and 32,114 MiB (with MTP);
     - 32,242 and 32,232 MiB (the two KL captures: util 0.85, 2048 chunks; then util 0.75, eager, 1 sequence, 1024 chunks).
   - The spike is above the lease cap of 31,744 MiB and below the card's 32,768 MiB.
   - In the completed c3 run the observer's highest sample was 31,736 MiB. The driver's 0.5 s device sampler saw 32,266 MiB of device memory in use. A spike shorter than the 1 s observer sampling interval cannot be ruled out, so c3 may have briefly exceeded the cap unseen.
   - At util 0.83 with 8192-token chunks, the load fit but the KV budget did not: `ValueError: To serve at least one request with the model's max seq len (32768), (2.75 GiB KV cache is needed, which is larger than the available KV cache memory (1.52 GiB). Based on the available memory, the estimated maximum model length is 12288.`
   - The measured c3 configuration has KV 5.58 GiB = 65,536 FP16 tokens. 1Cat's own TP1 test used util 0.92 and 8192-token chunks on a V100-SXM2. A V100 owner without our 31 GiB lease cap is not bound by it; whether MTP or KL capture then fits was not measured.
   - KL for 1Cat could therefore not be captured under the lease rules. The TP4 row's KL was not attempted: <cards 0-3> were returned to the multicard manager right after the TP4 protocol.
4. **Memory and start-up, other rows.**
   - vLLM 0.18.0 GPTQ: weights 16.77 GiB, KV 5.99 GiB = 24,304 FP16 tokens (util 0.90), peak 28,976 MiB device.
   - 1Cat TP1 start-up: ready 202 s with warm caches. The first cold load read 20.4 GiB from the HDD in 277 s.
   - 1Cat TP4: ready in 240 s.
5. **vLLM 0.18.0 cannot parse a GPU UUID** in `CUDA_VISIBLE_DEVICES`: `ValueError: invalid literal for int() with base 10: 'GPU-f2b6541e-8121-c7fd-8a73-78d4b1743d3e'` (runs g0, g1, g2).
   - Workaround in `scripts/enginesenv.sh` (`ENGINES_INDEX_BINDING=1`): map the leased UUID to this host's index with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
   - 1Cat TP1 accepted the UUID directly. The TP4 run used the same index mapping.
6. **TP4 host facts** (`host-facts.txt`):
   - <cards 0-3> are nvidia-smi indices 1, 2, 3, 0: four V100-PCIE-32GB on one PCIe switch (topology PIX), each PCIe 3.0 x16, NUMA node 0, 250 W, no NVLink.
   - 1Cat logged `Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs`, so all-reduce went through NCCL over PCIe.
   - 1Cat's own TP4 numbers come from four NVLink-connected V100-SXM2-32GB cards (61.8 tok/s at 128K, NVFP4, no MTP). That is not this hardware.
7. **Offline kernel receipts** (`kernel-archs.txt`, cuobjdump 12.9):
   - vLLM 0.32.0 `_C_stable_libtorch` has sm_75..sm_120 images and no sm_70, in both the PyPI and the cu129 wheel.
   - SGLang `sgl_kernel/flash_ops` has sm_80/86/90 only. torch 2.13.0+cu129, 2.13.0+cu130 and 2.14.1+cu130 are built for sm_75+.
   - vLLM 0.18.0 `_C.abi3.so` has 31 sm_70 images. The 1Cat wheel's `_C` libraries contain sm_70 images only (45 + 26). torch 2.10.0+cu128 lists sm_70.
   - No CUDA 12 build of sglang-kernel 0.4.9 is published: the cu129 index stops at 0.4.6.post1.
8. **Short-prompt TTFT is not comparable with Flare's,** because Flare's short runs reuse prefixes. Here, 1Cat ran with `--enable-prefix-caching` and vLLM 0.18.0 with prefix caching off (its default for this model). Both logged a `Prefix cache hit rate: 0.0%` throughout, so every engines TTFT is a cold prefill. The 30,720 prompts are unique per case.

## What a V100 owner can and cannot do with this model today

**Official current releases do not run on a V100 at all.**
- vLLM 0.32.0 refuses this checkpoint's quantization on capability 7.0 before loading anything. Its kernels, and the PyTorch builds it pins, contain no sm_70 code.
- SGLang 0.5.22 fails at its first CUDA operation with "no kernel image is available".

**The last official vLLM wheel with sm_70 kernels, 0.18.0:**
- It also refuses the NVFP4 checkpoint (minimum capability 89).
- Unpatched, it does serve a community 4-bit GPTQ checkpoint of the same model on one card:
  - 39.9 tok/s single stream and 30.7 tok/s per agent at 4 agents;
  - 191 s to first token at 30,720 prompt tokens;
  - about 15.5 minutes from launch to first answer;
  - KL vs the FP32 reference 0.038, against 0.0011 for Flare and 0.0088 for llama.cpp Q4_K_M.

**The community fork 1Cat-vLLM v1.5.1 serves the same NVFP4 file Flare uses, on one card:**
- 29.3 tok/s single stream and 27.9 tok/s per agent at 4, without speculation;
- 32.9 s to first token at 30,720, the fastest single-card prefill measured here (Flare 35.5 s);
- its load uses essentially the whole 32 GB card, so its MTP head and a logit capture could not run under our 31 GiB lease.

**On four PCIe V100s (no NVLink),** 1Cat TP4 gives 63.0 tok/s single stream and 50.3 per agent at 4, with 15.8 s TTFT at 30,720.

**Against Flare on one card:** Flare's release reaches 83.2 tok/s single stream and 64.9 per agent with MTP, or 30.9 / 28.3 with speculation off. That is above every single-card vLLM-family decode number measured here.

## Files

- **Root of this folder:**
  - `*-results.json`: one per attempt. Each holds the command, the packages, every request with its token arrival times, the VRAM peaks and the server log tail.
  - `engines-summary.json`, `kl-vllm0180-gptq.json`, `downloads.sha256` (wheels, patches, GPTQ checkpoint files), `kernel-archs.txt`, `host-facts.txt`.
  - `V100-PORTS.md`: the researcher's survey of V100 ports.
- **`logs/`:**
  - every server log and 0.5 s VRAM CSV;
  - the lease observer's `<measurement>-*-outcome.json` and `<measurement>-*-manifest.json`;
  - install logs and a `uv pip freeze` per venv.
  - The f4 and c1 runs ended before a results file. Their server logs and outcome files are the receipts:
    - f4: the lease observer stopped it with `foreign compute process appeared` (pid 661679, 310 MiB on <card>, not a descendant of the lease child; where it came from was not verified). By then f4's log already held the `no kernel image` error.
    - c1: exceeded the lease cap.
- **`scripts/`:** `engines_driver.py` (protocol), `enginesenv.sh`, the per-row `run_*.sh` / `chain_*.sh` / `kl_*.sh` scripts, `vllm_logits.py`, `summarize_engines.py`, `collect.sh`, The KL metric is run 2's `../run 2/llama_kl.py`, run unchanged.
- **On <release-host>:**
  - `<release-host>/flare/compare-vllm-sglang/`: venvs, downloads, GPTQ checkpoint and raw logits `results/kl2-*.tf.f32`.
  - Labels: f = official feasibility, g = vLLM 0.18.0, c = 1Cat TP1, t = 1Cat TP4.
