# Full-support selected-attention partitions

The saved layer-3 Qwen4Exp attention operands at a 262,136-token prompt reproduce both recorded serving defaults bitwise: direct attention for the final 504-token prefill chunk captured at capacity 512, and split-64 attention for the final decode token at capacity 1. For each fixed partition count, every active output is bitwise identical across capacities 1, 2, 4, 16, 64, 128 and 512. All 56 variants pass the installed sparse-attention reference with `rtol=0, atol=0.02` in two independent process runs.

Split-16 with 64-row attention chunks uses 24.094 MiB of partial workspace. For the complete saved 504-row prefill workload it takes 735.52 / 734.56 µs, compared with 1,157.68 / 1,157.02 µs for split-64 with the same chunking and 465.58 / 464.94 µs for direct attention at capacity 512. For the saved single decode row, split-16 takes 18.99 / 19.07 µs versus split-64 at 14.06 / 14.24 µs. These CUDA event measurements cover selected attention only.

## Operands and execution

The input [capture receipt](qsa-long-context-operands.json) identifies `/tmp/sparkgap-qsa-long-context/attention-operands.pt`, SHA256 `d988ee3ec3c2b3be0fc0afc3e780178045b8bcf9535267eba7df0e81d8fc4e2c`. The artifact contains exact queries, selected positions, metadata, recorded attention outputs and referenced main K/V pages from checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`.

| Workload | Active query positions | Selected keys per row | Rows with 2,051 keys | Referenced physical pages |
|---|---|---:|---:|---:|
| Final prefill chunk, 504 rows | 261632–262135 | 2048–2051 | 126 | 436 |
| Final decode, one row | 262142 | 2051 | 1 | 98 |

Every selection is causal and contains no repeated positions. The recorded K/V page sets exactly cover all referenced physical pages. Replay preserves physical page IDs, table values and logical row order. The K/V values were read after the HTTP request drained; later decode appends do not change the earlier selected positions. Raw and compressed ring state is excluded from the attention-only artifact.

The harness is [qualify_qsa_long_partitions.py](../../scripts/qualify_qsa_long_partitions.py), sharing the capture implementation in [qualify_qsa_fixed_partitions.py](../../scripts/qualify_qsa_fixed_partitions.py). It uses installed low-level selected-position kernels without modifying the installed implementation or policy. The selected-forward source SHA256 is `835044758d5bf0a445d5c0957d69b588ee71aad1763d31370713581b7e9264b9`; native replay extension SHA256 is `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`. The receipt retains all installed b12x Python source fingerprints and sampling source fingerprints.

`direct1` means the direct BF16-output kernel without split/merge. It does not represent the fixed-S1 public API candidate, which uses partial output and a merge. The split variants execute the low-level split kernel followed by its merge, reusing one partial slab across consecutive row groups on the captured stream. Capacities below 504 partition the prefill rows into consecutive groups; larger capacities pad inactive rows with zero queries and `-1` request IDs, positions and selections. A padded output must be zero.

The independent reference invokes the installed `b12x.attention.qsa.reference.sparse_paged_gqa_reference` directly on CPU: FP32 QK, FP32 softmax, FP32 PV, then BF16 output. Only that CPU reference uses a bijective physical-page remap into compact saved storage. This reference differs from Transformers attention that rounds probabilities to BF16 before PV.

## Correctness and lifetime checks

Both runs pass:

- Two recorded default outputs reproduced bitwise before the replay matrix, four baseline checks in total.
- Fixed direct/S16/S32/S64 output equality across all seven capacities, including both serial and cooperative merge variants.
- Exact output equality across all repetitions and between processes.
- The unchanged sparse-reference `rtol=0, atol=0.02` gate for every active output. Maximum absolute error is 0.015625 for prefill and 0.0078125 for decode.
- Zero CUDA allocation-count growth for all 2,580 native replays, including baselines and warmups.
- Zero inactive outputs and unchanged immutable query, selection, metadata and cache tensors.
- Native worker drain before graph release, process exit 0, and GPU release with 2 MiB used and no compute process.

The source reduction tree for split merge is consistent with the capacity-invariant results: cooperative and serial merge schedules retain the same arithmetic tree for a fixed split count. Different partition counts are not bitwise interchangeable on these operands:

| Comparison | Prefill differing BF16 elements / 3,096,576 | Decode differing BF16 elements / 6,144 | Prefill maximum absolute difference | Decode maximum absolute difference |
|---|---:|---:|---:|---:|
| S16 versus S32 | 17,378 | 19 | 0.0078125 | 0.00390625 |
| S16 versus S64 | 20,235 | 21 | 0.0078125 | 0.0009765625 |
| S32 versus S64 | 13,435 | 19 | 0.0078125 | 0.00390625 |

The full-support observations extend the [short-prefix partition results](qsa-fixed-partitions.md), whose at-most-459 selected keys occupied only 15 nonempty 32-key tiles. These long-context operands occupy all 65 tiles, and therefore require multiple nonempty tiles per S16/S32 partition.

## Timings and workspace

Each table entry is the median CUDA event duration in microseconds for process 1 / process 2, with three warmups and 20 measured repetitions per variant. Events bracket every attention group required for the complete workload. Native graph replay drives the launches. CPU readback, reference computation, capture, selection, projection, cache writes, scheduler work and HTTP are outside the events. The two runs contain 2,240 measured timings; all raw samples and p10/p90 ranges are retained in the JSON receipt. The maximum change between process medians is 4.11%, on a short decode variant.

The first table processes the same 504 saved prefill rows for every capacity. Small capacities require many consecutive graph launches, so these durations are not model-batch throughput measurements.

| Capacity | Direct | S16 | S32 | S64 |
|---:|---:|---:|---:|---:|
| 1 | 64725.86 / 64724.82 | 8425.82 / 8426.05 | 6342.99 / 6340.88 | 5367.68 / 5364.64 |
| 2 | 32461.36 / 32467.02 | 4226.42 / 4225.73 | 3186.46 / 3189.41 | 3013.55 / 3012.88 |
| 4 | 16214.80 / 16216.13 | 2212.50 / 2212.16 | 1842.16 / 1843.36 | 2101.02 / 2100.70 |
| 16 | 4153.57 / 4155.23 | 1031.60 / 1027.74 | 1019.62 / 1018.54 | 1332.56 / 1332.66 |
| 64 | 1078.00 / 1071.02 | 735.52 / 734.56 | 885.73 / 884.72 | 1157.68 / 1157.02 |
| 128 | 596.14 / 599.49 | 683.17 / 684.26 | 852.74 / 851.52 | 1506.70 / 1508.54 |
| 512 | 465.58 / 464.94 | 710.66 / 710.86 | 1002.05 / 1001.66 | 1500.66 / 1501.15 |

The second table processes the same one saved decode row; all other rows are inactive padding.

| Capacity | Direct | S16 | S32 | S64 |
|---:|---:|---:|---:|---:|
| 1 | 149.94 / 150.00 | 18.99 / 19.07 | 15.63 / 16.27 | 14.06 / 14.24 |
| 2 | 138.53 / 138.29 | 19.50 / 19.41 | 15.55 / 15.52 | 13.63 / 13.98 |
| 4 | 140.08 / 140.27 | 19.71 / 19.74 | 16.45 / 17.15 | 16.34 / 16.27 |
| 16 | 138.69 / 138.88 | 28.24 / 28.93 | 28.18 / 28.05 | 38.53 / 38.37 |
| 64 | 139.07 / 139.84 | 71.23 / 71.36 | 90.56 / 90.78 | 122.67 / 123.41 |
| 128 | 139.81 / 139.04 | 128.18 / 128.26 | 169.23 / 169.22 | 326.16 / 326.21 |
| 512 | 312.37 / 311.07 | 535.36 / 535.44 | 829.63 / 828.74 | 1322.16 / 1321.84 |

Partial output plus LSE workspace is `capacity × splits × 24 × (256 + 1) × 4` bytes, independent of the number of consecutive groups:

| Attention chunk capacity | S16 MiB | S32 MiB | S64 MiB |
|---:|---:|---:|---:|
| 1 | 0.376 | 0.753 | 1.506 |
| 64 | 24.094 | 48.188 | 96.375 |
| 128 | 48.188 | 96.375 | 192.750 |
| 512 | 192.750 | 385.500 | 771.000 |

Direct attention has no partial/LSE workspace. The harness retains every variant simultaneously and allocates 5,669,829,632 bytes of CUDA tensors; that aggregate is not a serving workspace requirement. Capacity-128/512 split timings allocate full-capacity partial slabs. A public path bounded to 64 attention rows corresponds to the segmented capacity-64 measurement.

## Reproduction and evidence

From the repository root, with the saved operands and recorded installed build:

```bash
.venv/bin/python scripts/qualify_qsa_long_partitions.py --validate-only
CUDA_HOME=/opt/cuda CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
  .venv/bin/python scripts/qualify_qsa_long_partitions.py
CUDA_HOME=/opt/cuda CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
  .venv/bin/python scripts/qualify_qsa_long_partitions.py \
  --artifacts /tmp/sparkgap-qsa-long-partitions/repeat \
  --output /tmp/sparkgap-qsa-long-partitions/repeat-receipt.json
```

The harness refuses occupied GPU4, mismatched input or installed-source hashes, and existing output receipts. CPU validation does not initialize CUDA. The process runs completed in 18.66 and 18.48 seconds. No matrix checks failed. The separate [operand-capture failure receipt](qsa-long-context-operands-failure-before-http.json) retains the earlier HTTP-helper naming failure before any HTTP request was sent.

- [Compact receipt with all raw timings and comparison counts](qsa-long-partitions.json).
- [Compressed full receipts, harness sources, installed kernel/reference sources and logs](/data/sparkgap-artifacts/validation-receipts/qsa-long-partitions-evidence.tar.gz), SHA256 `8ff3e36eeb7f2fa4203a85a83f67e5374fec08102c37dcc2bd4b244eb35e1a0b`.
- `/tmp/sparkgap-qsa-long-partitions/attention-outputs.pt` and `repeat/attention-outputs.pt` retain every active output tensor; their hashes are in the receipt.
- `/tmp/sparkgap-qsa-long-partitions/independent-reference.pt` and its repeat retain the exact CPU-reference outputs.

Coverage is one checkpoint, one layer, one long prompt and its final prefill/decode operands. These results establish fixed-partition capacity invariance and the stated sparse-reference gate for those operands. Full-model logit/token parity and serving throughput require separate measurements.
