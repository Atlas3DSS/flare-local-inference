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
4. Evaluate MoE kernel selection against the runtime's deterministic-output
   contract and identical-input numerical checks.
5. Overlap independent graph branches where separate scratch and explicit
   stream dependencies permit concurrent execution.
