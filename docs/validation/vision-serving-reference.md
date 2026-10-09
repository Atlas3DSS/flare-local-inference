# Exact-checkpoint vision serving comparison

The isolated vision module from the established vLLM serving source completed
both image fixtures using the exact Qwen3.8 Flash Next NVFP4/MXFP8/CSF checkpoint.
Sparkgap flash and vLLM both exceed the canonical encoder's unchanged 3% relative
L2 limit. No serving-reference numerical acceptance bound is defined.

| Encoder comparison | Image 0 relative L2 | Image 1 relative L2 |
| --- | ---: | ---: |
| Sparkgap flash against vLLM | 13.258807% | 7.061926% |
| vLLM against canonical eager adapter | 12.224209% | 6.419347% |
| Sparkgap flash against canonical eager adapter | 7.800213% | 6.946120% |

All six output tensors are finite BF16 with shape [64, 2560]. The two images are
the 256×256 RGB fixtures defined by
[the canonical vision qualification](../../tests/recipes/test_qwen38_vision.py).
Each processor produces a [1, 16, 16] image grid and 256 patch rows. vLLM's actual
processor emits uint8 patches and its FusedMMInputNorm runs on the GPU. Its BF16
output is bit-exact to the canonical processor's normalized pixels cast to BF16:
zero differing elements out of 393,216 for each image.

The [machine-readable receipt](vision-serving-reference.json) contains source
fingerprints, package versions, launch environment, full resolved configuration,
every observed attention and quantized-linear invocation, fixture and output
hashes, and the retained failure log. Tensor artifacts, PNG fixtures, individual
receipts and raw logs are in /tmp/qwen38-vision-serving-reference.

## Executed serving implementation

- Checkpoint repository: local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD.
  Declared reference revision: dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2.
  The serving master and inspected HF snapshot use the same 21 shard inodes;
  metadata identity is documented in
  [the serving provenance receipt](exact-serving-reference.md).
- vLLM source: /home/luke/projects/vllm-lunar-leviathan,
  revision 64fc1eec593397cbd765cc082e9ec90fbeb5ea66.
- FlashInfer source: /tmp/lunar-rebase-20261008/flashinfer-pinned,
  revision 98fb87a461183f98d6a7049ca947fb136cc6fb59.
- Interpreter: /home/luke/projects/vllm-lunar-leviathan/.venv/bin/python,
  under the recorded source-isolation environment.
- Device: GPU-afd4b1ad-8a64-7057-4bde-241822724c7f, exposed as the only CUDA
  device. One NCCL rank; vision data-parallel mode gives a local vision TP size
  of one, matching each vision replica in the recorded TP2 serving setup.
- Loaded 499 vision tensors, 326,652,352 stored bytes, through the unchanged
  Qwen3_VisionTransformer load_weights and vLLM post-load quantization hooks.
  The bounded safetensors CPU iterator replaces only the recorded transfer
  mechanism; it does not replace parameter-loading or quantization arithmetic.
  No language-model or MTP weights were allocated.
- Executed the original Qwen3_VisionTransformer forward, including its patch
  projection, positional interpolation, rotary embedding, transformer blocks,
  and spatial merger. The multimodal encoder is eager in the recorded serving
  configuration.

All 54 observed attention invocations selected FLASH_ATTN, FlashAttention version
2, with FP8 attention disabled. The 108 observed MXFP8 invocations selected
quantized activation mode. All 112 observed W4A16 NVFP4 invocations selected A16
with tile M/N/K = 64/128/64 and split K = 1.

The unmodified vLLM weights-preparation coordinator derives 65,536 encoder rows
and 16,384 merger rows from its multimodal budget. The same policy ran with
autotuning enabled. Sparkgap uses its 256-patch prepared flash path.
The different preparation capacities and selected GEMM configurations remain
part of the comparison.

The module-level run preserves the local vision computation of data-parallel
serving. It does not execute the full TP2 engine, multimodal request scheduling,
or HTTP preprocessing path. The one-rank tuning environment can choose different
winners from an earlier distributed startup; historical per-query winners were
not retained in the serving log. Exact executed winners are retained here.

## Reproduction

The harness is
[compare_vision_serving_reference.py](../../scripts/compare_vision_serving_reference.py).
It has four sequential modes: fixtures, vllm, sparkgap, compare. The vllm mode
requires the exact clean source revisions above and rejects imports outside
those trees. Both GPU modes require the assigned GPU UUID. They must run only
after acquiring the device.

From the Sparkgap repository, export fixtures with its isolated environment:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/compare_vision_serving_reference.py \
      fixtures /home/luke/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD/snapshots/dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2 \
      --output /tmp/qwen38-vision-serving-reference

The receipt's launch object contains the exact vllm-mode argv, working
directory, and selected environment. Replay it in a fresh subprocess with
that environment merged into the calling environment. It starts only the
standalone vision harness, not vLLM serve.

Run sparkgap mode using the same snapshot/output arguments, Sparkgap's
interpreter, PYTHONPATH=., CUDA_HOME=/opt/cuda and the assigned CUDA_VISIBLE_DEVICES
UUID. Run compare mode on CPU to reproduce all three paired metrics.

The first vLLM attempt failed because the harness added a global
PreparationSession.freeze after weights preparation. The actual worker applies
a capture-local guard around decoder graph capture and leaves eager vision
outside that guard. Removing the additional freeze restored the source's
lifecycle; no reference source or installed environment changed. The failed log
is retained as vllm-run-1.log, followed by successful vllm-run-2.log. All reference
source hashes and tracked worktree status were checked after execution.
