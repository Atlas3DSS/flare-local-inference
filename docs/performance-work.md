# GPU performance work

The [core design](core-design.md#performance-acceptance) defines service
baselines and the native replay overhead limit. Measure each optimization
against a baseline with the same checkpoint, GPU assignment, activation
precision, graph capacity, sampling settings, cache conditions, and completed
work. Correctness failures block performance claims.

## Measurement

Use `qualify.gpu_replay` for complete GPU graph duration, including vocabulary
projection, collectives, and sampling. Under tensor parallelism, measure the
interval spanning every rank. Collect batch-one and batch-four kernel
timelines with operator and layer attribution, kernel counts, summed kernel
durations, and graph elapsed time. Host graph-launch traces alone cannot
attribute GPU work.

Verify improvements through an exported `.safetensors` package loaded by
`flare run --embedding-table-storage ram`, including Helios's asynchronous
row delivery. Measure disk delivery separately when it is the deployment
configuration. For matching batch and verification width, graph rounds per
second are the change in `flare_speculative_rounds_total` divided by the
change in `flare_speculative_gpu_nanoseconds_total`, multiplied by
`1e9`. Emitted-token throughput also depends on draft acceptance.

Separate GPU replay time, native scheduling and submission, and HTTP latency.
Record raw measurements with their configuration, rerun noisy comparisons,
and retain workload regressions. The single-token weight-traffic estimate is
a bandwidth bound; the minimal replay harness supplies the measured floor.

## Fusion and precision

Fuse hyperconnection residual injection with the following RMSNorm using
FP32 gains and caller-owned output. Preserve PLE injection and inactive-row
mask boundaries. Pack compatible GDN QKV/Z and B/A projections and sparse
attention Q/K/V projections once during loading, including scale rows and TP
slices. Downstream kernels must accept their strides at every capacity.

Preserve BF16 intermediate rounding, FP32 normalization statistics, ATen
mean-reduction order, and FP32 sigmoid evaluation before multiplication and
BF16 rounding. Retain native ATen LayerNorm where decomposition changes
normalization or affine rounding.

Read formats and calibrated scales from checkpoint metadata. Routed MoE
gate/up fusion requires the dedicated loader's shared-scale validation.
Investigate a layer-wide gate/up activation scale only where the checkpoint
and operator contract permit it; retain the down projection's calibrated
scales. Never rescale or requantize weights to enable fusion.

## Complete-forward compilation

Compile prepared forwards with `torch.compile` after b12x tuning and before
capture. Opaque b12x calls expose their reads, writes, aliases, and native
launches; Inductor optimizes surrounding tensor operations. Preserve normal
functionalization, using byte storage for mixed-dtype aliases and typed
storage for aliases of one dtype.

Complete text forwards include MTP absorption, speculative verification, and
commit. Vision and maintenance graphs capture their prepared operations
directly. Chain verification exposes token and acceptance-counter mutations.
Conditional penalty histograms include only preceding candidate IDs, bounded
by verification width and the conditional-prefix offset.

`qualify.gpu_replay --check-compiled` compares logits and completion records
over 256 steps, plus final request, hybrid, and KV state, against uncompiled
graphs on the same model and buffers. This bytewise diagnostic requires
deterministic kernels. Plans with variable reduction order use component
bounds derived from measured repeatability.

Check graph breaks, escaping output lifetimes, shared scratch, native replay
allocations, and graph duration. The
[PLE delivery test](../tests/components/test_ple_delivery_cuda.py) exercises
asynchronous delivery; the
[PLE component tests](../tests/recipes/test_qwen38_ple.py) compare consumer
arithmetic across graph variants.

## Autotuning and kernel bands

Use b12x's numerical policy and autotuning, including dynamic expert routing
and atomic reductions where the model contract permits them. Measure GDN
recurrence schedules, QSA partitions, MoE reductions, and hyperconnection
projection precision independently. Preparation metadata must agree with the
actual submitted queries.

Measure every captured speculative shape. Total target rows, `batch ×
verification_width`, determine whether expert execution stays in the small-row
band or selects grouped kernels. The [MTP design](mtp-design.md) specifies
round-cost and acceptance measurements. Do not infer a speedup from draft
depth or a component benchmark alone.

## Independent branch overlap

Declare simultaneous scratch through `ScratchArena.register_parallel`.
Reserve the complete candidate envelope during tuning, then trim and refresh
bindings to the selected peak. Serial layers and graph variants reuse storage;
concurrent branches use disjoint regions.

Qwen's GDN input projections overlap through a prepared auxiliary stream at
capacities up to sixteen rows. Shared experts overlap routing and routed
experts through capacities of 256 rows, joining before gating and addition.
QSA index projection and selection overlap QKV projection, normalization,
rotary embedding, and KV writes through capacities of sixteen rows. The
selection join precedes attention.

Forks and joins must survive compilation. Their dependencies preserve inputs,
scratch, cache mutations, and final KV-pool bindings. Rust launches the
captured graph; its internal streams require no host intervention. Measure
complete graph latency, including synchronization and resource contention, at
each relevant row capacity and with each candidate node-construction order.

## Compressed scale loading

The [compressed scale layout](tp-checkpoint-layout.md) supports TP1, TP2, and
TP4 without changing packed weights or calibration. Measure partition reads,
GPU regrouping, native preparation, and upload separately from replay.
Scale-expansion tuning includes live producers and consuming expert work.
Poison reusable scale scratch between calls and verify alternate graph
capacities through 4,096-token prefill.
