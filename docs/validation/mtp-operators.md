# MTP identical-input operator measurements

The prefill key projection differs between Sparkgap and pinned vLLM on identical
BF16 inputs and byte-identical checkpoint weights. For the measured prefill
inputs, Sparkgap's key projection exactly equals the FP64 mathematical dot
rounded to BF16. The vLLM projection differs in 577 of 2,560 values. QK
normalization produces identical values when given identical raw projections.

## Conditions

Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The immutable fixture is the five-row
MTP prefill followed by two single-row teacher-forced continuations used by the
[MTP module oracle](mtp-module.md). Fixture SHA-256:
`63b48403a3d81094169b94dc3effa8f2e52e94ad0074ee4873fa7896228b2a33`.

Each operator receives both implementations' recorded operands independently.
Sparkgap uses its prepared capacities 8 and 1 with CUDA graph replay. The reference
uses actual eager vLLM operators from commit
`64fc1eec593397cbd765cc082e9ec90fbeb5ea66` and FlashInfer
`98fb87a461183f98d6a7049ca947fb136cc6fb59`, at live row counts 5 and 1. An additional
reference prefill invocation zero-pads those same five rows to eight.

The probes preserve the reference's fused HC-down row stride of 336 and fused
QKV row stride of 13,312. Seven loaded projection weight views match each other
exactly; the CPU dot diagnostic also checks its six weight matrices against the
checkpoint. Independent RoPE probes receive common BF16 inputs
constructed on CPU from the recorded raw Q/K; those inputs are stimuli, while
the measured outputs come from each implementation's actual rotary operator.

## Results

Both implementations reproduce all 11 recorded operator boundaries at every
step exactly: **66/66** comparisons. All **210** repeated operator outputs are
value-bit identical. Sparkgap's six operator-graph replays and three module-graph
replays allocate zero. All **30** reference five-row versus padded eight-row
operator comparisons are exact over the five live rows.

Relative L2 uses the vLLM output norm as denominator. Ranges below cover the two
recorded input sources at the prefill step; they are measured precision
differences, without fitted acceptance thresholds.

| Operator | Relative L2, percent |
|---|---:|
| HC grouped normalization | 0.0002403 |
| HC down projection | 0.1723–0.1727 |
| HC scaled SiLU | 0 |
| HC up projection | 0.002339–0.002680 |
| HC gate/stream reduction | 0 |
| Q projection | 0.2133–0.2144 |
| K projection | 0.3688–0.3779 |
| V projection | 0.2095–0.2105 |
| Q and K normalization | 0 |
| Q rotary transform | 0.02100–0.02149 |
| K rotary transform | 0.006177–0.006190 |
| Attention output projection | 0.0001460–0.007373 |

The first recorded difference is one of 51,200 prefill HC normalization values,
with absolute difference `0.0009765625`; both continuation HC normalization
outputs are exact. The MTP HC norm checkpoint weight is BF16. Sparkgap passes
FP32 `1 + weight` through the public CuTe grouped-normalization implementation;
vLLM passes the BF16 raw weight through the public zero-centered Triton
implementation. The same-input projection measurements isolate further
differences independently of that preceding normalization difference.

The CPU mathematical-dot diagnostic multiplies the exact saved BF16 inputs by
the byte-matched checkpoint weights in FP64, then rounds the result to BF16.
For the prefill inputs recorded by Sparkgap:

| Projection | Output values | Sparkgap values differing from rounded dot | vLLM values differing from rounded dot |
|---|---:|---:|---:|
| HC down | 1,600 | 0 | 452 |
| HC up | 51,200 | 1 | 8 |
| Q | 61,440 | 2 | 11,994 |
| K | 2,560 | 0 | 577 |
| V | 2,560 | 0 | 510 |
| Attention output | 12,800 | 3 | 6 |

Sparkgap executes public b12x BF16 SIMT projections at these capacities. The
reference records actual `torch.nn.functional.linear` calls, with
`VLLM_B12X_BF16_GEMV=False`, fused HC-down/injection and fused QKV projections.
The reference's internal GEMM algorithm and accumulation precision were not
profiled. Its five-row and padded eight-row results are identical for every
measured operator.

The [module oracle](mtp-module.md) records subsequent amplification of prefill
key differences by attention. Together, these measurements locate an independent
source of that key drift in projection arithmetic and show that Sparkgap is
closer to the mathematical dot for these inputs. Whole-module logit relative L2
remains diagnostic under [the numerical acceptance contract](../core-design-corrections-2.md#21-replace-logit-relative-l2-as-an-acceptance-metric).

## Evidence and limits

The [receipt](mtp-operators.json) records all conditions and integrity checks.
The [artifact manifest](/data/sparkgap-artifacts/mtp-implementation/operator-oracle/manifest.json)
indexes 117 verified payloads, including raw operator tensors, all per-row
measurements, loaded-weight fingerprints, executed plans, the CPU dot diagnostic,
26 passing CPU tests and source copies. The
[archive](/data/sparkgap-artifacts/mtp-implementation/operator-oracle/operator-oracle.tar.gz)
has SHA-256 `a3c2fabdf9adb21ba2aa9802f129d4962f6171d2eafaaf9d3042c2f3b5f0c616`.

This fixture covers three teacher-forced contexts and 15 operator boundaries.
Expert operators, speculative acceptance/state transactions, broad generation
agreement and full serving performance require their own qualification. The
operator reference repetitions establish repeatability for these calls, not for
the full vLLM model. Production arithmetic and the serving-reference
configuration are unchanged.
