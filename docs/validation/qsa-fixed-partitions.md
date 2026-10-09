# Fixed-partition Qwen selected attention

Fixed direct, 16-way, 32-way and 64-way selected attention each produce bitwise
identical outputs across the tested query capacities. This holds for both the
11-row restored tail and the 459-row cold prefill recorded from layer 3 of
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`.

The [summary and raw timing samples](qsa-fixed-partitions.json) cover two
independent processes, 44 graph variants per process and 2,030 native graph
replays in total. Both processes pass recorded-baseline reproduction, repeat
stability, capacity invariance, inactive-output checks and immutable-input
checks. CUDA allocation counters remain unchanged for every replay. Both
workers drain and release their graph handles.

## Operands and execution

The input artifact is `/tmp/sparkgap-prefix-qsa-recording.pt`, SHA256
`1c9e3fdfc57e0cdc1f50420034ea03fd77e370b593f8aecfde3c62be72835397`.
The observations are `restored_full_capacity_tail.qsa` and `cold.qsa`.
They provide the BF16 query, selected logical positions, query/request metadata
and main KV pages after cache writes. The restored tail contains query positions
448–458; the cold prefill contains positions 0–458. Both use 24 query heads,
two KV heads, head dimension 256 and softmax scale 1/16.

The [harness](../../scripts/qualify_qsa_fixed_partitions.py) invokes the installed
low-level `launch_selected_paged_gqa_direct`, `launch_sparse_gqa_split` and
`launch_sparse_gqa_merge` functions. The selected-forward source SHA256 is
`835044758d5bf0a445d5c0957d69b588ee71aad1763d31370713581b7e9264b9`.
Native extension SHA256 is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
Sources and extension are identical between runs. No installed kernel,
production dispatch, graph catalog or package was changed.

Recorded direct-capacity-512 and split-16-capacity-16 tail outputs reproduce
bitwise before comparison proceeds. The cold prefill direct-capacity-512 output
also reproduces its saved model output bitwise. The tail reference tensor
artifact has SHA256
`3ef0c80ee8de7fa21d2d9bc844c56ab9942624cbe3fb72c0a31b05e0b000fa09`,
verified against the [capacity-isolation receipt](qsa-capacity-isolation.json).

Tail capacities are 1, 2, 4, 16, 64, 128 and 512. Cold-prefill capacities are
16, 64, 128 and 512. A capacity smaller than the active row count partitions
the saved rows into consecutive groups inside one graph. Each group has its
own immutable inputs and persistent output; groups reuse one capacity-sized
FP32 partial/LSE workspace. Query and selection padding is inactive. Every
active row is processed exactly once per graph replay.

Physical cache page IDs 0–7 and their values are preserved in an eight-page
pool. The unchanged block table is checked to reference only recorded pages
for all selected keys. The two main BF16 KV tensors occupy 1 MiB per workload.
Attention inputs remain immutable after native handoff. Python performs CPU
readbacks only after native completion fences.

## Attention time and workspace

Each process records three warmups and 20 measured replays per variant.
External CUDA events bracket all attention launches for the complete saved
workload. Preparation, selection, cache writes, projections, readbacks and HTTP
are excluded. Execution uses GPU4, an RTX PRO 6000 Blackwell Max-Q, driver
610.57.04, PyTorch 2.13.0+cu130 and CUDA_HOME `/opt/cuda`.

The table reports median microseconds from both processes. Workspace includes
FP32 partial outputs and LSE; it excludes immutable inputs and final outputs.

| Saved workload | Row grouping | Attention path | Median µs, run 1 / run 2 | Workspace MiB |
| --- | --- | --- | ---: | ---: |
| 11-row tail | 1 × capacity 16 | Direct | 111.46 / 111.25 | 0 |
| 11-row tail | 1 × capacity 16 | Split 16 | 27.47 / 28.11 | 6.023 |
| 11-row tail | 1 × capacity 16 | Split 32 | 29.33 / 29.33 | 12.047 |
| 11-row tail | 1 × capacity 16 | Split 64 | 39.55 / 39.62 | 24.094 |
| 459-row prefill | 1 × capacity 512 | Direct | 366.03 / 366.66 | 0 |
| 459-row prefill | 1 × capacity 512 | Split 16 | 579.49 / 579.87 | 192.750 |
| 459-row prefill | 8 × capacity 64 | Split 16 | 597.06 / 596.75 | 24.094 |
| 459-row prefill | 4 × capacity 128 | Split 16 | 550.18 / 550.77 | 48.188 |
| 459-row prefill | 8 × capacity 64 | Split 32 | 751.95 / 752.16 | 48.188 |
| 459-row prefill | 8 × capacity 64 | Split 64 | 1016.10 / 1017.42 | 96.375 |
| 459-row prefill | 1 × capacity 512 | Split 64 | 1377.76 / 1380.21 | 771.000 |

For these operands, split 16 with 64-row groups uses one eighth of the
full-capacity split-16 workspace and takes approximately 3% longer. It takes
approximately 63% longer than direct-capacity-512 attention. A 128-row group
reduces time to approximately 551 µs at twice the 64-row workspace size.
These measurements favor split 16 over split 32/64 for this recorded prefill;
they do not determine a decode policy.

The small-capacity measurements process all 11 saved tail rows, rather than
one independently recorded decode batch. For capacity 1, split 16 takes
156.66/155.10 µs across 11 sequential groups, while split 64 takes
117.71/116.86 µs. That tradeoff requires actual decode and longer-selection
measurements before choosing a serving policy.

Maximum between-process median variation is 2.33%. Some small-tail paths have
wider within-process p10–p90 intervals; all 20 samples and quantiles are retained.
Clocks were neither fixed nor sampled per case. The process holds every variant
simultaneously: total CUDA tensor allocation after capture is 4,168.63 MiB,
which is not a proposed serving workspace requirement.

## Arithmetic and coverage boundary

For a fixed split count, comparisons cross both merge implementations:
`capacity > splits` selects the serial implementation; other shapes use the
cooperative implementation. Source inspection shows the same split-pair sums
`s` and `s+32`, followed by adjacent butterfly stages 0–4. Cooperative execution
distributes stages 0–2 over four contiguous groups of eight and joins them in
stages 3–4. Maximum and denominator reductions retain the same lane/slot order.
The GPU results confirm bitwise equality across that dispatch boundary for the
saved operands.

All valid selected entries lie in 32-key tiles 0–14. The cold prefill has
1–459 selected keys per row; the tail has 449–459. Thus every split count of
16 or greater places at most one nonempty tile in a partition. Split 16, 32
and 64 happen to match each other bitwise throughout this matrix. This does
not establish their equivalence when more selection tiles are populated.

All split variants differ from direct attention:

| Workload | Different BF16 elements | Different rows | Relative L2 vs direct | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Restored tail | 7,201 / 67,584 | 11 / 11 | 0.000942656 | 0.015625 |
| Cold prefill | 276,132 / 2,820,096 | 427 / 459 | 0.000808833 | 0.015625 |

The direct and split differences reproduce the recorded implementation
behavior. No independent canonical attention, full-model numerical, or serving
performance gate is exercised. A recorded long-context selection that fills
the 2,051-entry selected-position buffer remains necessary to compare split
counts beyond the short-prefix partition coverage.

## Reproduction and artifacts

CPU provenance and shape validation:

```bash
.venv/bin/python scripts/qualify_qsa_fixed_partitions.py --validate-only
```

Captured GPU qualification, after assigning the idle GPU and choosing an absent
receipt path:

```bash
CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
CUDA_HOME=/opt/cuda .venv/bin/python -u scripts/qualify_qsa_fixed_partitions.py \
  --output /tmp/qsa-fixed-partitions-reproduction.json \
  --artifacts /tmp/qsa-fixed-partitions-reproduction
```

The [evidence archive](/data/sparkgap-artifacts/validation-receipts/qsa-fixed-partitions-evidence.tar.gz), SHA256
`7d28a777b5c4a446c17ac77f6d8a60f15471ad3413677f6fa23a4dbda69a32f3`,
contains both complete receipts, logs, executable harness and installed kernel
source. Full output tensors remain under `/tmp/sparkgap-qsa-fixed-partitions/`;
both processes produce the same tensor artifact SHA256
`2ee44129babeae365a9defd4e43684ccf05d0971afcafb4641b9cf61d1d2b938`.
The compact JSON records artifact paths, per-row comparison archive, source
hashes, allocation checks and every timing sample. Both GPU processes exit 0;
GPU4 returns to 2 MiB with no compute process.
