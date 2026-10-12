# GPU replay optimization work

Measure each change independently with the batch-one decode corpus in
`qualify.gpu_replay`. Compare complete GPU graph duration, including the
vocabulary projection and sampling. Component correctness, graph capture,
caller-owned scratch, and allocation-stable replay are required.

1. Fuse hyperconnection residual injection and the following RMSNorm with FP32
   gains and caller-owned outputs. Preserve the PLE and inactive-row mask
   boundaries.
2. Pack GDN QKV/Z and B/A projections and sparse-attention Q/K/V projections.
   Share input quantization and projection launches without changing checkpoint
   precision or persistent state layout.
3. Compile the complete prepared model forward with `torch.compile` and
   autotuning. Keep b12x calls as opaque registered custom operators within the
   compiled graph so Inductor can optimize surrounding PyTorch tensor work.
   Complete compilation and tuning before CUDA graph capture; verify graph breaks,
   output lifetimes, shared scratch, replay allocations, and complete graph
   timing against the uncompiled path.
   Trace the prepared operators after b12x tuning. Preserve every device launch,
   express mutations through shared storage for every captured alias, and retain
   normal Inductor functionalization. Use byte storage for mixed-dtype views and
   typed storage for aliases of one dtype. Preserve BF16 intermediate rounding
   and ATen mean-reduction order, including FP32 normalization statistics.
   Preserve ATen FP32 sigmoid results before multiplication and BF16 rounding.
   Retain native ATen LayerNorm to preserve its BF16 output rounding and affine
   normalization semantics.
   Measure serving through an exported `.flaretensors` package loaded by
   `flare run --ngram-storage ram`, including Helios's asynchronous row delivery.
   Compare `flare_speculative_rounds_total` divided by
   `flare_speculative_gpu_nanoseconds_total` for matching batch and verification
   width, converting nanoseconds to seconds. Token throughput also depends on
   draft acceptance. GPU traces identify exposed copies, waits, and kernel work.
   `tests/components/test_ple_delivery_cuda.py` checks asynchronous PLE delivery;
   `tests/recipes/test_qwen38_ple.py` compares the PLE consumer with checkpoint
   reference outputs across graph variants. Preparation enables compilation for complete text
   forwards, including learned-MTP absorption and speculative rounds. Vision
   and maintenance graphs capture their prepared operations directly.
   FlashInfer chain verification exposes token and acceptance-counter mutations
   to compilation. Native replay tests cover compiled verification and request
   commit at widths 1–8, invalid distributions, and per-request RNG lane invariance.
   Conditional penalty counts include only preceding candidate IDs. The
   verification width and conditional-prefix offset bound the histogram updates;
   future draft positions require no updates.
   With `--check-compiled`, the replay qualification compares all 256 steps'
   logits and completion records, plus final request, hybrid, and KV state,
   against uncompiled graphs using the same prepared model and buffers. This
   bytewise diagnostic requires deterministic kernels; ordinary performance runs
   use b12x's default numerical policy without this check.
4. Tune MoE with b12x's default numerical policy, including dynamic direct routing
   and atomic reductions. Measure selected kernel plans and complete graph
   latency for the serving workload.
   TODO: Use one layer-wide input activation amax to derive a shared activation
   scale for the expert gate (w1) and up (w3) projections. Ensure every layer
   has this scale; retain per-expert activation scales for the down (w2)
   projection.
5. Overlap independent graph branches where separate scratch and explicit
   stream dependencies permit concurrent execution. Decode GDN input projections
   use separate views of the operator arena and one preparation-owned auxiliary
   stream at capacities up to 16 rows. An opaque projection-pair operation retains
   the fork and join through forward compilation. Scratch planning reserves the
   maximum concurrent candidate requirement, then trims to the selected peak;
   serial layers and graph variants reuse the same storage. Shared-expert
   projections run alongside routing and routed experts at capacities up to 256
   rows, using a disjoint arena suffix. A captured join precedes shared-expert
   gating and the final sum. Inductor combines the sigmoid, multiplication, and
   routed addition into one pointwise kernel while preserving BF16 intermediate
   rounding. Diagnostics explicitly materialize the shared-only result.
   MXFP8 QSA input projections overlap at capacities up to 16 rows. Index
   projection and b12x selection use the auxiliary stream while QKV projection,
   normalization, rotary embedding, and KV writes use the main stream. Separate
   workspace views isolate both projections and QSA. The b12x selection-stream join
   precedes attention, and explicit tensor operands preserve cache mutations
   and final KV-pool rebinding through compilation. Fork and rejoin auxiliary
   streams within Python's capture of a single CUDA graph;
   Rust continues to launch one graph per replay. Allocate workspace by
   simultaneous use across branches, then reuse it across serial layers.
   Verify the graph retains overlap after compilation and measure complete
   graph latency, including synchronization and resource contention.
6. TODO: Design a [TP-sliceable compressed checkpoint layout](tp-checkpoint-layout.md)
   for TP=1, TP=2, and TP=4, with export-time exception indexes and direct loading
   of partition views. Resolve native-tile boundary handling before implementing
   the storage format and b12x readers.
