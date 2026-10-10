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
   typed storage for aliases of one dtype. Preserve BF16 intermediate rounding.
   The qualification entry point is `scripts/qualify_decoder.py --tune --compile`;
   server bootstrap integration requires full-model parity and replay timing.
4. Evaluate MoE kernel selection against the runtime's deterministic-output
   contract and identical-input numerical checks.
5. Overlap independent graph branches where separate scratch and explicit
   stream dependencies permit concurrent execution. Evaluate QSA indexing
   alongside QKV projection and shared experts alongside routed experts, with
   joins captured before dependent operators consume their outputs. Fork and
   rejoin auxiliary streams within Python's capture of a single CUDA graph;
   Rust continues to launch one graph per replay. Allocate workspace by
   simultaneous use across branches, then reuse it across serial layers.
   Verify the graph retains overlap after compilation and measure complete
   graph latency, including synchronization and resource contention.
