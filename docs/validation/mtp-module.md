# Qwen MTP module: exact-checkpoint component comparison

The MTP module loads and executes the checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Its embedding and feedback operation agree bit-for-bit with the pinned vLLM module on three identical inputs. **Numerical agreement of the complete draft module remains unresolved:** full-vocabulary relative L2 is 21.9–30.2%, and two of three greedy IDs differ.

The [receipt](mtp-module.json) contains every stage metric and executed linear call. The [external evidence directory](/data/sparkgap-artifacts/mtp-implementation/module-oracle) retains the raw tensors, logs, source snapshots, failed runs, CPU diagnostics, [member manifest](/data/sparkgap-artifacts/mtp-implementation/module-oracle/manifest.json), and [archive](/data/sparkgap-artifacts/mtp-implementation/module-oracle/mtp-module.tar.gz).

The [identical-input operator comparison](mtp-operators.md) isolates projection
arithmetic as a source of the prefill key differences and compares both outputs
with FP64 dot products rounded to BF16. The component observations below retain
their original input and execution conditions.

## Conditions and model contract

The checkpoint contains one MTP decoder block and 4,637 MTP tensors, totaling **1,596,720,640 stored bytes**. Routed experts use NVFP4 W4A16, plain E4M3 block scales, per-expert FP32 global scales, 512 experts and top-10 routing. Their inputs remain BF16; target-model activation quantization and CSF decoding do not apply to these MTP tensors. Dense projections, shared experts and the shared vocabulary head are BF16. The MTP loader aliases the target embedding and head and preserves the checkpoint's up/gate expert layout.

The immutable feedback fixture is derived from recorded target layer-47 residuals in [vllm-layers.pt](/data/sparkgap-artifacts/text-layer-trace/vllm-moe-inprocess-repeat/vllm-layers.pt), SHA-256 `44aed1bafd133c18f7b038717d81bf22a8abc074d518bc71d25baed17cceb400`. The prompt is `[760, 6511, 314, 9338, 369]`, with teacher tokens `[11751, 13, 561]`. MTP receives `h[i]` and `x[i+1]` at position `i`: five prefill rows at positions 0–4, then one row at positions 5 and 6. The request-tail output indices are 4, 0 and 0. The fixture SHA-256 is `63b48403a3d81094169b94dc3effa8f2e52e94ad0074ee4873fa7896228b2a33`.

Flare executes captured capacity-8 and capacity-1 graphs using a **19,402,752-byte shared scratch arena**, BF16 KV, 64-token pages and 16 attention partitions. The reference executes `Qwen4ExpMTP` with its ordinary weight-loading, preparation and model operators in eager mode, without a target model or scheduler. Its source is vLLM `64fc1eec593397cbd765cc082e9ec90fbeb5ea66` and FlashInfer `98fb87a461183f98d6a7049ca947fb136cc6fb59`. Pinned QSA requires 48-token pages for its seven-draft raw ring; its source selects 16 partitions for five rows and 64 for one row. Both use identical logical causal positions and BF16 KV values. `VLLM_MTP_NVFP4_LM_HEAD=0` preserves the checkpoint's shared BF16 head.

## Numerical results

Relative L2 is `||Flare − vLLM||₂ / ||vLLM||₂`, evaluated in FP32. Each logits comparison includes all 248,320 vocabulary entries. KL is `KL(vLLM || Flare)` from FP32 log-softmax. No numerical acceptance threshold is assigned by this component diagnostic.

| MTP interval | Logit relative L2 | SG / vLLM argmax | Top-10 overlap | Attention output relative L2 | Key-history relative L2 |
|---|---:|---|---:|---:|---:|
| Five-row prefill | 0.289674 | 11 / 13 | 9/10 | 0.076661 | 0.002505 |
| Position 5 | 0.302094 | 271 / 1049 | 7/10 | 0.277231 | 0.002285 |
| Position 6 | 0.219072 | 6511 / 6511 | 8/10 | 0.126132 | 0.002116 |

The full receipt retains top-1 and runner-up margins, all per-row stage metrics and downstream routes. Those margins are observations on this fixture, not a bound on draft-token agreement.

Embedding and feedback outputs are storage-identical in all three intervals. Prefill hyperconnection normalized inputs differ by approximately `2.4e-6` relative L2; their projected-down outputs differ by `0.001727`. Continuation hyperconnection mixes agree exactly. The selected attention positions are exact and causal, and each backend's newly written key/value rows equal its own recorded projected values. Norm weights are byte-identical. RoPE tables have FP32 storage in Flare and BF16 storage in vLLM; the BF16 values consumed by the recipe agree exactly.

A CPU diagnostic evaluates each backend's recorded, post-normalization Q/K/V with FP32 scaled scores, softmax and value reduction. Its attended values differ from the corresponding GPU outputs by **0.115–0.174% relative L2**. Keeping Flare's query, gate and values while substituting only the recorded vLLM key history reduces projected attention error to **0.991%, 0.951% and 0.610%**. Substituting values alone leaves the larger discrepancy. Recorded scaled scores reach approximately 742–767 in magnitude; small key differences produce score changes up to 5.34 and probability changes up to 0.538. The inherited key differences therefore explain substantial attention amplification. The CPU formula does not replace the serving reference.

| Operator comparison | Evidence |
|---|---|
| Embedding and MTP feedback | Identical inputs and weights; bit-exact recorded outputs in all three intervals. |
| BF16 projections | Actual vLLM `torch.nn.functional.linear` calls recorded; Flare selects b12x SIMT at these row capacities. Prefill operands already differ slightly after normalization. A controlled identical-operand projection backend ablation was not run. |
| Attention | Actual positions, writes and norm/RoPE constants checked; each backend's own operands evaluated independently with the CPU dense formula. A GPU cross-backend attention test with identical substituted operands was not run. |
| Shared/routed experts and final mixer | Inputs and routes differ downstream of attention; their output differences do not isolate expert arithmetic. |

The reference's `VLLM_B12X_BF16_GEMV` environment value is unset and resolves to false. No BF16 dense module has an attached b12x GEMV plan. Actual functional-linear invocations cover QKV, attention output/index projections, hyperconnection projections, router and shared expert. The internal PyTorch GEMM algorithm was not profiled. **The cause of the initial key drift has not been isolated to a dispatch choice.** The observer-only reference rerun preserves all **138 recorded tensors bit-for-bit**.

## Runtime checks and retained failures

Two actual-checkpoint GPU tests pass: feedback against the published b12x operator reference, and complete-module capture/replay with scratch poisoning, immutable inputs, finite outputs, padded-row checks and stable allocations. All three teacher-forced Flare replay intervals have unchanged CUDA allocation counts. Focused CPU tests record 59 passes and three CUDA skips; the final observer/fixture suite records 11 passes. These suites overlap and are not summed.

The evidence retains a launch failure caused by absent GPU device nodes, and a reference-harness failure that used the compact selector's default row 0 instead of request-tail row 4. The latter was rejected for incompatible output geometry; the corrected execution sets the public `set_prefill_output_indices` explicitly. Two earlier CPU failures and their corrections are also retained. Source snapshots distinguish collection-time files from the hashes recorded during execution; unrelated concurrently edited orchestration files are listed explicitly rather than represented as the executed versions.

Full target/draft verification, acceptance, commit/rollback, end-to-end quality and MTP performance are not established by these component measurements.
