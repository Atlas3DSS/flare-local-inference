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
   `scripts/qualify_decoder.py --tune --compile` checks decoder correctness.
   `qualify.gpu_replay flare --compile` measures compiled prefill, decode, and
   resume forwards through model bootstrap. Bootstrap recompiles these forwards
   after binding the final KV pools and retains their operator owners until
   execution drains. Vision and complete speculative rounds require separate
   compilation qualification before enabling compilation by default. Maintenance
   graphs retain their prepared native kernels.
   FlashInfer chain verification exposes token and acceptance-counter mutations
   to compilation. Native replay tests cover compiled verification and request
   commit at widths 1–8, invalid distributions, and per-request RNG lane invariance.
   With `--check-compiled`, the replay qualification compares all 256 steps'
   logits and completion records, plus final request, hybrid, and KV state,
   against uncompiled graphs using the same prepared model and buffers. This
   bytewise diagnostic requires deterministic kernels; ordinary performance runs
   use b12x's default numerical policy without this check.
4. Tune MoE with b12x's default numerical policy, including dynamic direct routing
   and atomic reductions. Measure kernel selection and complete graph latency
   against vLLM's b12x integration.
5. Overlap independent graph branches where separate scratch and explicit
   stream dependencies permit concurrent execution. Decode GDN input projections
   use separate views of the operator arena and one preparation-owned auxiliary
   stream at capacities up to 16 rows. An opaque projection-pair operation retains
   the fork and join through forward compilation. Scratch planning reserves the
   maximum concurrent candidate requirement, then trims to the selected peak;
   serial layers and graph variants reuse the same storage. Shared-expert
   projections run alongside routing and routed experts at capacities up to 256
   rows, using a disjoint arena suffix. A captured join precedes shared-expert
   gating and the final sum, which remain visible to Inductor. MXFP8 QSA input
   projections overlap at capacities up to 16 rows. Index projection and b12x
   selection use the auxiliary stream while QKV projection, normalization,
   rotary embedding, and KV writes use the main stream. Separate workspace
   views isolate both projections and QSA. The b12x selection-stream join
   precedes attention, and explicit tensor operands preserve cache mutations
   and final KV-pool rebinding through compilation. Fork and rejoin auxiliary
   streams within Python's capture of a single CUDA graph;
   Rust continues to launch one graph per replay. Allocate workspace by
   simultaneous use across branches, then reuse it across serial layers.
   Verify the graph retains overlap after compilation and measure complete
   graph latency, including synchronization and resource contention.
