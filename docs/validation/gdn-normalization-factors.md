# GDN normalization-factor precision diagnostic

The unchanged exact-checkpoint layer-zero GDN test fails its first prefill output
assertion with both the installed BF16 tensor-core recurrence and an isolated
normalization-factor modification. Moving Q/K normalization after the tensor-core
dot products reduces relative L2 but leaves four pointwise failures.

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Both executions use GPU
`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f` and
`tests/recipes/test_qwen38_gdn.py::test_actual_checkpoint_prefill_decode_continuation_graphs[65]`.
The fixture packs 65 and one token, using seed 741 and the pinned canonical
Transformers reference. Inputs, references and assertions are unchanged.

| Measurement | Installed recurrence | Normalization after dot products |
|---|---:|---:|
| Output relative L2 | 0.0143300667 | 0.0135013591 |
| Relative L2 limit | 0.015 | 0.015 |
| Pointwise failures / output elements | 4 / 168,960 | 4 / 168,960 |
| Maximum absolute error | 0.002197265625 | 0.002197265625 |
| Pointwise relative tolerance | 0.025 | 0.025 |
| Pointwise absolute tolerance | 0.001121605469 | 0.001121605469 |

The modification retains raw BF16 Q/K operands for KKᵀ and QKᵀ MMA, then
multiplies by the corresponding FP32 reciprocal norms before applying the
existing decay, beta and scale factors. Other BF16 operand boundaries, persistent
FP32 state and KDA paths remain unchanged. Raw products can overflow for extreme
inputs that normalization before MMA would handle; this modification is not
installed in the serving stack.

Exactly one source differs among the 554 Python files recorded for each run:
`b12x/sequence/_shared/delta_prefill/_cute_kernels.py`. Its installed SHA-256 is
`6b091099326a7db4677fbad3935328c9ad8f15334d6bd7ec64b6719e1a0a3afe`;
the isolated modification is
`88a56cc441d755ae7d9274d69961e2f14b487d3b0f3aa9ac56762a61f4fed32d`.
All recorded sources remain unchanged during their respective executions.

The recorder copies observations to CPU after graph replay and invokes the
original assertion function. Both runs stop at the first failed output
comparison, before continuation and checkpoint assertions. No runtime performance
or complete-model numerical acceptance follows from these measurements.

[Machine-readable results](gdn-normalization-factors.json) and the
[evidence archive](/data/sparkgap-artifacts/validation-receipts/gdn-normalization-factors.tar.gz) retain both tensor artifacts,
logs, source hashes, kernel sources, patch, test, recorder and CPU mirror diagnostic.
All 20 archive payloads pass independent SHA-256 and length verification.
Archive SHA-256: `53844fa73efbd06187a58ffdf30b86380760accd6bc4d6f45f99c903632c5ef4`.
