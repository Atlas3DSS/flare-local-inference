# Teacher-forced text logits against the vLLM serving path

Flare's recorded text decoder differs from the pinned vLLM serving path by
**17.72–22.41% relative L2** over four identical token contexts. All four argmax
IDs agree. Two of four steps exceed the required bound of Flare–vLLM error
being no larger than vLLM–canonical error. **The reference varies between
requests:** two identical cold requests in one vLLM engine differ by
18.95–23.96% logit relative L2, with one of four argmax IDs changing. Their first
different recorded tensor is the layer-0 MoE output; every preceding recorded
layer-0 tensor, including the MoE input, is identical. The cross-implementation
discrepancy cannot be attributed to Flare from final logits alone. Numerical
acceptance requires component comparisons on identical operands and a measured
reference variance floor.

The [machine-readable receipt](text-serving-oracle.json) contains all three
pairwise comparisons, source identities, selected activation modes and the four
bound predicates. Full logits, worker observations, source snapshots, launch
configuration and logs are stored under
`/data/sparkgap-artifacts/text-serving-oracle/`. Its `artifact-index.json`
records each file's size and SHA-256.

## Conditions

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Both runs use the same
21 checkpoint shards. The vLLM process runs on the assigned RTX PRO 6000 Blackwell
GPU, UUID `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`.

- vLLM: clean `64fc1eec593397cbd765cc082e9ec90fbeb5ea66`; FlashInfer:
  clean `98fb87a461183f98d6a7049ca947fb136cc6fb59`.
- TP1, BF16 model/KV/convolution state, FP32 GDN recurrent state, mapped-host PLE,
  no speculative decoding, 512-token prefill limit and context limit 262,144.
- Flare tensors: the [recorded decoder execution](decoder-autotuned.md),
  with capacity-512 prefill, single-row decode, quantized MXFP8, sequential GDN
  with one key split and 16 QSA attention partitions.
- Canonical tensors: the stored Transformers diagnostic with MXFP8 and NVFP4
  activation quantization. Its FP32 recurrence intermediates, affine-weight
  precision and reduction boundaries differ from the serving implementations.

The prompt `The capital of France is` has IDs `[760, 6511, 314, 9338, 369]`.
The consumed continuation is `[11751, 13, 561]`. Context lengths are 5, 6, 7
and 8. Each observation contains all 248,320 vocabulary logits.

`qualify.vllm_text` uses the ordinary `LLM.from_engine_args` loader, scheduler,
model runner and cache path. vLLM's existing trace-replay sampler forces the
recorded continuation after model execution. `raw_logits`, full-vocabulary
logprobs and the flat output representation retain the original scores.
The collector scatters scores by token ID and verifies vocabulary coverage,
finiteness and the duplicated sampled-token score.

## Measurements

Relative L2 is `norm(actual - reference) / norm(reference)`, using FP32 logits.
The named right-hand implementation supplies the denominator. KL is
`KL(vLLM || Flare)` using FP32 log-softmax over the complete vocabulary.
Top-10 overlap counts shared token IDs.

| Step | Argmax, all paths | Flare / vLLM L2 | vLLM / canonical L2 | Flare / canonical L2 | Flare / vLLM top-10 | KL | Relative bound |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 11751 | 0.1923783 | 0.1796956 | 0.1588928 | 9 | 0.0648012 | Fails |
| 1 | 13 | 0.2033026 | 0.1627352 | 0.1820679 | 10 | 0.0190213 | Fails |
| 2 | 561 | 0.1772253 | 0.2014077 | 0.1998211 | 9 | 0.1466506 | Passes |
| 3 | 6511 | 0.2240819 | 0.2311742 | 0.2439030 | 6 | 0.0069344 | Passes |

This measurement covers one short prompt and the stored Flare implementation.
Generation agreement over a diverse corpus and the default 4,096-token graph
catalog require separate execution. Logit extraction and provenance collection
make these runs numerical measurements rather than inference benchmarks.

## Executed arithmetic and provenance

Worker observations before and after generation retain 2,809 prepared plans.
All 1,254 scalar MXFP8 plans declare and select `quantized`; none selects A16.
Plan selections, module bindings, configuration and session state remain equal
across generation. The vLLM hyperconnection norm weights are BF16. Six GDN
prefill configurations are selected, including a two-key-split configuration;
their complete queries and selections are in the worker observations.

All 345 mapped native libraries match their recorded filesystem identities.
The loaded FlashInfer sampling binary is copied into the artifact store by
SHA-256. The two missing source paths belong to PyTorch's synthetic `torch.ops`
and `torch.classes` modules. Checkpoint identities and qualification sources
remain unchanged through shutdown. The process exits zero and GPU use returns
to 2 MiB.

The first capture attempt completed model initialization but failed to serialize
the provenance RPC callable; it produced no logits. Its exit record and log are
preserved separately. The successful offline run enables vLLM's callable RPC
serialization and records that setting. Separate-process CPU checks reproduce
the rejection with the option disabled and pass callable/metadata round trips
with it enabled. The comparator, reconstruction and observation suite passes
58 tests with CUDA initialization forbidden in the relevant CPU checks.

## Reference repeatability

The same pinned vLLM launch, checkpoint, prompt and teacher continuation were
executed in another process without layer instrumentation. Relative L2 against
the first vLLM run is `[0.1892070, 0.2093741, 0.1668091, 0.2085139]`.
All four argmax IDs remain equal. The full vectors and launch records are in
`/data/sparkgap-artifacts/text-layer-trace/vllm-uninstrumented-repeat/`;
`reference-repeatability.json` in its parent directory records the comparisons.

An instrumented run captured all 48 materialized decoder residuals at all four
contexts with complete coverage and correct token IDs/positions. Its logits
differ from the first vLLM run by 18.42–21.77%, so its strict unchanged-logit
gate fails. Those tensors remain diagnostic observations. The cached FlashInfer
sampling binary changed between runs; the pinned greedy/raw-logit path uses
Torch top-k/gather and read-only greedy selection rather than that binary's
sampling/filter kernels.

Two identical requests in one instrumented engine, with a successful prefix
cache reset between completed requests, differ by
`[0.189528, 0.209964, 0.202320, 0.239640]` in full-logit relative L2.
The second context's logit argmax changes from token 11 to token 13; the first
request has a tie for the largest logit. All recorded
layer-0 inputs and operations through the MoE input are exactly equal. The first
different recorded operation is the MoE output at all four contexts. Layer-0
residual error is 0.29–0.39%; the difference propagates through the remaining
47 layers. Both captures pass tensor coverage, token-position and finiteness
checks, but fail the unchanged-original-logit gate. The failing exit and both
sets of tensors are retained under
`/data/sparkgap-artifacts/text-layer-trace/vllm-inprocess-repeat/`.

The selected vLLM MoE plans permit BF16 atomic output accumulation. The selected
Flare paths use deterministic reductions with different rounding boundaries.
For the vLLM dynamic kernel, ten routed experts and five intermediate slices
produce approximately 50 arrival-ordered BF16 additions per output element.
Separate routed/shared observations confirm identical router logits, all 80
expert IDs and weights, and shared-expert outputs between requests. Routed
outputs first differ by 0.60–0.88%; final logits differ by 16.63–20.30% in that
pair. The [layer comparison](text-layer-comparison.md) records these observations
and the matched-input normalization proof. Source pins, selected plans, module
bindings and configuration remain
unchanged across the two requests. Shutdown returns GPU use to 2 MiB.

## Reproduction

Capture into a fresh artifact directory:

```sh
.venv/bin/python -m qualify.vllm_text \
  --plan /data/sparkgap-artifacts/text-serving-oracle/tp1-quantized-mxfp8-rpc/source/plan.json \
  --output /data/sparkgap-artifacts/text-serving-oracle/reproduction
```

The supervisor checks source pins and checkpoint identities, applies the
recorded source isolation and GPU assignment, and records process exit.
Recompute the comparison from the retained tensors:

```sh
.venv/bin/python -m qualify.text_oracle \
  --flare /data/sparkgap-artifacts/text-serving-oracle/tp1-quantized-mxfp8-rpc/sparkgap.pt \
  --vllm /data/sparkgap-artifacts/text-serving-oracle/tp1-quantized-mxfp8-rpc/vllm.pt \
  --canonical /data/sparkgap-artifacts/text-serving-oracle/tp1-quantized-mxfp8-rpc/canonical.pt \
  --output /data/sparkgap-artifacts/text-serving-oracle/comparison-reproduction.json
```
