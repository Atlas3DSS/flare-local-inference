# Autotuned text-decoder execution and numerical comparison

The main repository's `scripts/qualify_decoder.py --tune` successfully loads the exact checkpoint, prepares two complete graphs, hands them to the native executor and generates four greedy tokens. **Canonical numerical equivalence remains unqualified:** logit relative L2 ranges from **15.89% to 24.39%**, despite matching all four greedy token IDs. The [receipt](decoder-autotuned.json) preserves execution, numerical comparison, selected configurations and verification separately.

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The prompt `The capital of France is` has IDs `[760, 6511, 314, 9338, 369]`; generated IDs `[11751, 13, 561, 6511]` decode to ` Paris. The capital`. The canonical reference uses the same prompt and teacher-forced continuation `[11751, 13, 561]`, so every compared step has identical context.

| Step | Greedy ID, both paths | Logit relative L2 | Hidden relative L2 | Maximum logit difference | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| 0 | 11751 | 0.1588927507 | 0.1591365337 | 1.9375 | 8 |
| 1 | 13 | 0.1820678860 | 0.2412264496 | 2.296875 | 9 |
| 2 | 561 | 0.1998210698 | 0.2490161061 | 2.78125 | 9 |
| 3 | 6511 | 0.2439029813 | 0.2779791951 | 3.1796875 | 5 |

Relative L2 is `norm(native.float() - reference.float()) / clamp_min(norm(reference.float()), 1e-12)`, separately for logits and final hidden state. Maximum difference uses the FP32-converted logits. Probabilities use FP32 softmax over the complete vocabulary. The recorded comparison specifies no numerical acceptance thresholds; matching argmax IDs does not establish equality of the distributions or hidden states.

An independent CPU run of the unchanged comparison script reproduced the complete recorded comparison JSON exactly. Before comparing the prepared decoder, that script rechecked the original canonical and native tensor hashes and reproduced all four metric rows from `/tmp/sparkgap-canonical-text-short-comparison.json`. The canonical artifact still has SHA-256 `f5df07709c7b18a55e24236b0488925a98475302a1374f9617b380651229b34a`; its pinned Transformers revision is `4cc2aa84301c9aa210b5513dab9fefae03981f6e`. The [reference specification](decoder-reference.json) describes the exact-weight loading, quantization arithmetic and canonical architecture used to produce it.

All four logits tensors and all four hidden tensors are also byte-identical to `/tmp/sparkgap-decoder-autotuned.pt`, the separately recorded unconstrained-autotuning execution. Its original receipt, source, log, tensors and comparison are archived. The GDN policy constraint therefore leaves this particular short-prompt result unchanged; the separate [595-check prefix qualification](prefix-replay-sequential-k1.md) exercises capacity-dependent continuation.

The execution uses one capacity-512 prefill graph and one single-row decode graph, context capacity 262,144, 4,096 KV pages and 46 hybrid-state slots. All four native replay intervals completed with **zero CUDA allocations**, finite full-vocabulary logits and the expected completion records. The final record reports four generated tokens and terminal status. Preparation used five selection-cache groups, measured no candidates, and finalized 2,422 declarations in 0.345756 seconds. All 36 GDN prefill declarations select sequential recurrence with `k_split=1`.

The command ran from `/home/luke/projects/sparkgap` with `PYTHONPATH` unset:

```bash
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
  CUDA_HOME=/opt/cuda \
  .venv/bin/python -u scripts/qualify_decoder.py --tune \
  --output /tmp/sparkgap-decoder-sequential-k1.json \
  --tensor-output /tmp/sparkgap-decoder-sequential-k1.pt
```

The execution tool observed exit **0** for session **82146**. Total elapsed time was **45.905245 seconds**, including checkpoint loading, preparation, capture and CPU observations; it is not an inference-latency measurement. Final allocated GPU storage was **85,264,703,488 bytes**, with **28,800,138,240 bytes** of mapped host PLE storage. No process-specific sampler mapping was captured during this execution.

The [archive](/data/sparkgap-artifacts/validation-receipts/decoder-autotuned.tar.gz) contains all four tensor artifacts required to reproduce the comparisons: canonical reference, historical native baseline, unconstrained-autotuning native output and policy-constrained native output. It also contains the unchanged metric script, independent verifier and logs, raw receipts, original reference qualification log, main-integration receipt, 32 verified main source/test files, 25 verified source files for the unconstrained run and 15 integration backup files. Every archive member was rehashed. The inventory records **226 individual tensors** across those artifacts.

The original reference receipt retains its constructor source digest. That constructor file subsequently acquired instrumentation, so its historical bytes are not included; the stored canonical tensor is included and matches its original digest exactly. The reference quantization implementation, reference tests and model descriptor still match their recorded hashes and are archived.

Archive SHA-256: `5dda5aaa64e160a65be38a79430c9b60aca491c17f98a326e54d111aaf845824`; **103 members**, **12,869,906 bytes**. Policy-constrained native tensor SHA-256: `61d45469d814941e682c7fbb668c0969a75fa128d68cad692844ab4fde9a1b13`. The formulas, reference artifact and numerical limitations are unchanged.
