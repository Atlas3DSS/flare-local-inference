# Decoder prefill capacity and workspace planning

CPU validation passes **223 tests with 51 CUDA skips**. Decoder input and shared intermediate buffers cover 3072, 4096, and 8192 token capacities; smaller graph views retain their parent storage. State preparation rejects token capacities exceeding any input token, position, or image-index buffer before constructing operators.

`DecoderWorkspaces.allocate(device, tokens=...)` obtains the default GDN and quantized MXFP8 scratch requirements from public b12x plans before operator priming. The 64 MiB initial allocation remains a floor. Batched tuning subsequently reserves the candidate envelope. `DecoderInputs.allocate` and `DecoderWorkspaces.allocate` retain their explicit 512-token API defaults; bootstrap supplies its configured capacity.

The metadata audit uses a declared SM120/170-SM device identity with device detection mocked and CUDA initialization prohibited. It allocates decoder tensors on `meta`, compiles no kernels, and reports `cuda_initialized=false`. Byte counts below exclude weights, persistent model/cache state, routed-expert workspace, tuning candidates, and graph storage.

| Tokens | Initial projection bytes | Initial recurrence bytes | Shared decoder intermediate bytes | QSA scratch bytes |
|---:|---:|---:|---:|---:|
| 512 | 67,108,864 | 67,108,864 | 196,874,240 | 166,879,232 |
| 3,072 | 67,108,864 | 74,370,048 | 1,171,210,240 | 201,252,864 |
| 4,096 | 67,108,864 | 74,373,120 | 1,560,944,640 | 214,990,848 |
| 8,192 | 67,108,864 | 74,376,192 | 3,119,882,240 | 269,975,552 |
| 16,384 | 103,973,120 | 74,382,336 | 6,237,757,440 | 379,944,960 |
| 65,408 | 414,589,184 | 74,416,128 | 24,896,291,840 | 1,038,564,864 |
| 65,472 | 415,006,976 | 74,416,128 | 24,920,650,240 | Rejected |
| 262,144 | 1,661,108,480 | 74,568,704 | 99,774,013,440 | Rejected |

GDN's sequential recurrence uses two windows of prepared tile records. At 4096 tokens its default scratch requires 74,373,120 bytes, exceeding the previous 67,108,864-byte allocation before batched tuning could resize it. The public metadata sizing preserves `recurrence_policy="sequential_k1"`, BF16 activations, FP32 state, and exact checkpoint parameter dtypes.

At context 262144, QSA's selector reserves 128 MiB for FP32 scores and requires space for 512 carried groups plus at least one additional group per query row. Capacity 65408 plans successfully; the next 64-aligned capacity, 65472, raises `ValueError: QSA 128 MiB score workspace cannot hold one top-k carry row`. Fixed 16-way attention partitions process at most 64 rows per attention launch; that bound does not change selector workspace sizing.

MXFP8 quantized projections acquire a capacity-specific dense lowering from 2048 rows onward. Dispatch uses the actual tensor shape, which is the graph capacity, rather than the device live-token count. Larger-capacity numerical and cold/cache equivalence require GPU qualification. The decoder's hidden checkpoint kernel also scales its integer mRoPE reduction vector to the next power of two of `3 * token_capacity`; this audit does not establish its compiled resource usage.

Short-convolution and PLE public planning use token-scaled storage and expose no 512-token maximum. The metadata and CPU shape checks establish allocation/preparation contracts; GPU compilation, graph replay, complete-model numerics, memory headroom, and serving throughput at the larger capacities remain separate qualifications.

Reproduce the CPU suite:

```bash
cd /home/luke/projects/sparkgap
env -u FLARE_RUN_CUDA_TESTS CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q tests/preparation tests/recipes
```

The archived metadata script requires the repository on `PYTHONPATH` when executed from an extracted evidence directory. The first suite launch placed `env -u` after an assignment and exited 127 before pytest; its error log is retained. The corrected launch exited 0 in 37.84 seconds.

[Structured receipt](prefill-capacity-planning.json) and [source, metadata script, and raw logs](/data/sparkgap-artifacts/validation-receipts/prefill-capacity-planning.tar.gz). Archive SHA-256: `e23b867d5b5be570deaf540eb3a4d090a8833b8f235c100cc6b6571dc29cce91`.
