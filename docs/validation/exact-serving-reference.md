# Exact-checkpoint vLLM serving reference

The saved vLLM run `candidate-qwen-64fc1eec59-verified` served the same local
checkpoint payload used by Flare. Its execution arithmetic differs from the
pinned Transformers eager component adapter. A [paired vision comparison](vision-serving-reference.md)
uses this implementation with the same checkpoint and records encoder differences;
no complete-model numerical bound is established. The canonical component
references and their acceptance gates remain unchanged.

The machine-readable [provenance receipt](exact-serving-reference.json) records
launch arguments, environment, source hashes, asset hashes, and filesystem
identity. Inspection read source and saved evidence without launching vLLM or
reading tensor payloads.

## Checkpoint identity

Repository: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`.
Declared revision: `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`.

The serving directory
`/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`
and the corresponding snapshot under
`/home/luke/.cache/huggingface/hub/` reference the same device/inode for all
21 safetensors shards. Nine metadata assets, including configuration, tensor
index, tokenizer, processors, generation configuration, and model card, have
matching SHA-256 hashes. This establishes local file identity at inspection;
it does not supply independently recomputed payload hashes or authenticate the
remote revision. The receipt retains every shard's inode, size, and mtime.

## Recorded implementation and execution

The saved run is under
`/home/luke/projects/vllm-lunar-leviathan-rebase-20261008/e2e/candidate-qwen-64fc1eec59-verified/`.
Its `actual-launch.json` verifies argv and environment against the rendered
configuration. Source receipts before and after tool evaluation match:

| Component | Commit | Git tree |
|---|---|---|
| vLLM | `64fc1eec593397cbd765cc082e9ec90fbeb5ea66` | `272a636af76f2231a897822d83d84d262ff50753` |
| FlashInfer/b12x | `98fb87a461183f98d6a7049ca947fb136cc6fb59` | `0884355f2d5bc3ae0e6ea3da927045972d56bbdf` |

Both worktrees were clean in those receipts and at inspection. vLLM is loaded
from `/home/luke/projects/vllm-lunar-leviathan`; FlashInfer is isolated at
`/tmp/lunar-rebase-20261008/flashinfer-pinned`. The environment uses a shared
virtualenv, so its installed package metadata alone is not a historical lock.
The server identifies itself as `0.30.1rc1.dev924+g64fc1eec5`.

Recorded settings include TP2, BF16 model dtype, FP8 KV cache,
`modelopt_mixed`, b12x dense/MoE/GDN kernels, vision data parallelism,
instanttensor loading, 4096 batched tokens, four requests, 262144 context,
prefix caching, and three-token embedded MTP. The encoder was neither compiled
nor CUDA-graph captured; decoder graph replay was enabled.

| Component | Observed selection and source arithmetic |
|---|---|
| Vision attention | Log lines 44–46 select `FLASH_ATTN`. The SM120 source default selects FA2 and calls noncausal `flash_attn_varlen_func`; it does not use Transformers eager attention. No compiled instruction trace was recorded. |
| Patch projection | `Conv3dLayer.forward_cuda` lowers this nonoverlapping geometry to `F.linear(..., bias)` on torch ≥2.9. The Transformers adapter retains `Conv3d`. |
| MXFP8 projections | Logs select `B12xMxfp8LinearKernel`. Stored E4M3/UE8M0 weights feed prepared b12x blockscaled plans. Activation mode defaults to `auto`; its heuristic selects A16 for eligible ≤8-row calls, quantized otherwise. Autotuning was enabled; individual selected regimes were not saved. |
| Vision NVFP4 MLP/merger | Logs select `B12xNvFp4LinearKernel`; the `W4A16_NVFP4` loader forces A16 and rejects an explicit quantized-activation override. |
| GDN prefill | Log line 48 explicitly selects b12x. Frozen b12x uses BF16 chunk operands, FP32 recurrent storage, and 16-token chunks. Normalized/transformed Q/K and inverse operands have explicit BF16 boundaries. The API has no all-FP32 precision option at this revision. Sequential versus segment-parallel selection was not logged. |
| GDN decode | Log line 49 explicitly selects b12x. Source uses FP32 recurrent registers and Q/K/update arithmetic, BF16-rounded beta, and BF16 output. Checkpoint `mamba_ssm_dtype=float32` propagates into state storage. |
| QSA | Logs show `attention.qsa` preparation. Recorded b12x dense/MoE selection routes Qwen4Exp through `nvidia/b12x_qsa.py` and prepared `b12x.attention.qsa` contexts. Per-capacity partition choices were not saved. |
| HC/PLE affine parameters | Loading logs report conversion of retained FP32 tensors into BF16 parameters. |
| MTP head | Logs record runtime NVFP4 quantization with BF16 activations; source applies this to the MTP private head. The target head remains checkpoint BF16 without the MXFP8-head override. |

Exact source paths and hashes supporting these statements are in the receipt.

## Launch and existing results

The recorded launch, from `/home/luke/projects/vllm-lunar-leviathan`, is:

```bash
/home/luke/projects/vllm-lunar-leviathan-rebase-20261008/e2e/with-source-isolation-pinned-fi.sh \
  /home/luke/.local/bin/lil run \
  /home/luke/projects/lil-launcher/recipes/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD.yaml \
  --config /home/luke/projects/vllm-lunar-leviathan-rebase-20261008/e2e/pulsar-lunar-leviathan-fi-pinned-v1.yaml \
  --model-path /data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD
```

The recipe, wrapper, runtime configuration, import-isolation hook, rendered
command, and actual argv/environment are fingerprinted. A repeat must verify
those fingerprints and explicitly coordinate the recorded two-GPU allocation;
the recipe resolves available devices and is not itself a fixed-device command.

`smoke-response.json` contains a successful `READY` response.
`tool-eval/results.json` records 15 completed scenarios and score 100, with
temperature 0, seed 42, and concurrency 1. Source receipts bracket that run,
and `raw-output-audit.json` retains its saved-output review.
[Flare tool qualification](tool-eval-native.json) repeats the same 15-scenario
configuration and scores 30/30, matching the recorded vLLM result. Neither run
establishes multimodal quality or canonical numerical parity.
