# TP1 exact-checkpoint serving startup and shutdown

The vLLM run `qwen38-exact-tp1-bf16kv-fp32ssm-hostple-no-mtp-512-quantized-mxfp8`
served the exact checkpoint on one GPU at `http://127.0.0.1:38762` and reached
HTTP readiness in 298.094141 seconds. The controller verified the live argv,
environment, model ID, and 262144 context. The process used only
`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`. No configuration changes or startup
retries were required.

The checkpoint is
`/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, declared revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. vLLM is commit
`64fc1eec593397cbd765cc082e9ec90fbeb5ea66`; FlashInfer/b12x is commit
`98fb87a461183f98d6a7049ca947fb136cc6fb59`.
[The receipt](vllm-tp1-startup.json) fingerprints the launch, source receipts,
checkpoint identity evidence, native extensions, readiness, and shutdown.
[Raw inputs](vllm-tp1-startup-raw.json) preserve the original paths, byte counts,
SHA-256 hashes, and exact UTF-8 contents of the controller, source-isolation
wrapper, configuration, logs, and lifecycle receipts.
[Selected plans](vllm-tp1-startup-selected-plans.json) and
[selected program metadata](vllm-tp1-startup-program-metadata.json) preserve the
tuning-cache snapshot and available CuTe compilation metadata. The
[historical TP2 reference](exact-serving-reference.md) remains unchanged.

## Selected execution

The engine config confirms TP1/DP1/PP1/DCP1, BF16 model and KV, four sequences,
512 batched tokens, 262144 context, no speculative model, and b12x dense/MoE
kernels. The live environment selects quantized MXFP8 activations and FP32 SSM
storage is explicit in argv. Decoder graph replay is enabled; vision encoder
compilation and vision CUDA graphs are disabled, as in the pinned serving path.

Observed log selections:

- `B12xNvFp4LinearKernel` and `B12xMxfp8LinearKernel`.
- b12x GDN prefill and decode, dimension-first convolution history.
- Prepared `attention.qsa`, `moe.decode`, and `norm.hyperconnection` components.
- `FLASH_ATTN` vision attention; the pinned SM120 dispatch selects FA2, though
  this run does not include an instruction trace identifying the implementation.
- FlashInfer top-p/top-k sampling.
- A private 26.82 GiB CUDA-mapped host PLE table. The earlier generic Engram log
  says `table_memory='device'`; the b12x PLE allocation log confirms the explicit
  `ple_table_memory='ram'` setting takes effect.

Weight preparation completed 62 signatures with 500 candidate measurements and
290 compilations in approximately 95 seconds. State preparation completed 67
signatures with 736 candidate measurements and 1246 compilations in approximately
68 seconds. Final binding reused those preparations with zero compilations.
FlashInfer autotuning was disabled; b12x autotuning remained enabled.

The matching b12x selection-cache snapshot contains 18 exhaustive multi-candidate
race winners with executable program keys. Six GDN configurations select the
sequential algorithm. Five MoE configurations select the dynamic backend with
direct/grouped routes. Four GEMM records select A16 tiles with split-K=1; two
pointwise/reduction records and one Triton row record complete the snapshot.
The cache identifies records by hashed query key and does not retain a complete
request/layer/query mapping. Default-only plans, including QSA, are not an
exhaustive-race cache entry. No full per-capacity plan declaration dump was enabled.

## Memory and graphs

| Measurement | Logged result |
|---|---:|
| GPU total memory | 95.01 GiB |
| Desired GPU budget at 0.94 | 89.31 GiB |
| Model-loading memory | 67.9 GiB |
| Consumed weights plus non-Torch memory | 68.6 GiB |
| Peak activation memory | 1.88 GiB |
| Available/allocated KV cache | 18.74 GiB |
| Serving manager blocks | 6268 |
| Manager block width | 1520 tokens |
| Cache groups | 49 |
| Token-equivalent cache capacity | 684632 tokens |
| Reported maximum-context concurrency | 2.61 |
| Preliminary graph profile | 0.15 GiB, 4 seconds |
| Final graph capture | 0.04 GiB, 3 seconds |

The final capture completed four PIECEWISE graphs and three FULL graphs. The
configured sizes are `[1,2,4,8]`; FULL graphs cover one, two, and four tokens.
Request-boundary recurrent checkpoint caching is enabled. Preparation used 32
temporary cache blocks before allocating the 6268-block serving pool.

The measured cache capacity exceeds the independently calculated 7.178435 GiB
needed for one 262144-token request including boundary reserves and the null
block. The four-sequence scheduler limit does not imply room for four simultaneous
maximum-length contexts.

## Observed warnings and limits

Post-load processing logged two recovered expandable-segment allocation failures
at 01:18:38 and 01:18:54 on 2026-10-09 in America/Los_Angeles: each attempted a
20 MiB mapping with 2,686,976 bytes free. Model loading then returned successfully.
No traceback or allocation stack was emitted, so these warnings cannot be
attributed to a specific layer.

The pinned loader casts checkpoint FP32 HC normalization parameters to BF16 and
BF16 GDN `A_log` parameters to FP32, matching its saved TP2 behavior. The exact
checkpoint path does not imply arithmetic equivalence to the canonical eager
reference.

During the separately run semantic evaluation, the runtime JIT monitor reported
Triton compilation for `_stage_qsa_rope_positions_kernel` at 01:22:45 and
`_restore_auxiliary_state_kernel` at 01:22:46. These are observed warmup coverage
gaps. Canonical numerical acceptance gates remain open.

SIGTERM stopped the controller and API server with exit code 0 after
932.179041 seconds of controller lifetime. The assigned GPU had no remaining
compute processes. Source trees, checkpoint asset hashes and shard identities,
native extension hashes, and launch-bundle hashes match before and after the run.

The vLLM internal shutdown path used abort mode with timeout 0. It logged a forced
kill of the remaining engine process, followed by a Python resource-tracker
warning for one leaked semaphore. Exit code 0 and an empty GPU process list show
process termination and device release; the log does not establish graceful
internal engine teardown.
