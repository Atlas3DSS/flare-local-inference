# b12x Serving Stack Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Entry Points                                   │
│                                                                         │
│   cli.py ──────────────────┐     api/server.py ─────────┐              │
│   (interactive / one-shot) │     (FastAPI, OpenAI-compat)│              │
│                            ▼                             ▼              │
│                    ┌──────────────────────────┐                         │
│                    │     ServingEngine         │                         │
│                    │                          │                         │
│                    │  submit() / generate()   │                         │
│                    │  _step() loop            │                         │
│                    │  TP broadcast protocol   │                         │
│                    └────┬─────────┬───────────┘                         │
│                         │         │                                     │
│             ┌───────────┘         └──────────┐                          │
│             ▼                                ▼                          │
│   ┌──────────────────┐            ┌────────────────────┐               │
│   │  BatchScheduler  │            │    ModelRunner      │               │
│   │                  │            │                    │               │
│   │  continuous batch │            │  _forward_inner()  │               │
│   │  chunked prefill  │            │  compile_model()   │               │
│   │  preemption       │            │  capture_graphs()  │               │
│   └──┬───────────┬───┘            └──────┬─────────────┘               │
│      │           │                       │                              │
│      ▼           ▼                       ▼                              │
│  ┌────────┐ ┌──────────┐    ┌──────────────────────────┐               │
│  │ Radix  │ │ Request  │    │  for layer in layers:    │               │
│  │ Cache  │ │ Lifecycle│    │    hidden = layer(h, st) │               │
│  └───┬────┘ └──────────┘    └──────────┬───────────────┘               │
│      │                                 │                                │
│      ▼                                 │                                │
│  ┌─────────────┐                       │                                │
│  │  PagePool    │   ┌──────────────┐   │                                │
│  │  (KV cache)  │   │  SSMStatePool│   │                                │
│  │  64-tok pages│   │  (GDN state) │   │                                │
│  └──────────────┘   └──────────────┘   │                                │
└─────────────────────────────────────────┼───────────────────────────────┘
                                          │
          ┌───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TransformerLayer                                  │
│                                                                         │
│         norm1 ──► attn ──► + residual ──► norm2 ──► ffn ──► + residual │
│                                                                         │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │         Norm Blocks         │  │        Attention Blocks          │ │
│  │                             │  │                                  │ │
│  │  RMSNorm      GemmaRMSNorm │  │  B12xPagedAttention              │ │
│  │  (standard)   (1 + weight)  │  │    QKV → QK norm → RoPE         │ │
│  │                             │  │    → KV write → paged attn      │ │
│  │  make_norm() factory        │  │    → output gate → O proj       │ │
│  └─────────────────────────────┘  │    → TP allreduce               │ │
│                                   │                                  │ │
│                                   │  GDNLinearAttention              │ │
│                                   │    projections → conv1d          │ │
│                                   │    → chunk/fused recurrent       │ │
│                                   │    → gated RMSNorm → O proj     │ │
│                                   │    → TP allreduce               │ │
│                                   │                                  │ │
│                                   │  (future: MLA, DSA, ...)        │ │
│                                   └──────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         FFN Blocks                               │  │
│  │                                                                  │  │
│  │  MoEFFN                                                          │  │
│  │    routing (sigmoid / softmax, optional bias + renorm)           │  │
│  │    → b12x_sparse_moe_fp4  (NVFP4 fused kernel)                 │  │
│  │    → optional shared expert (TP-sharded dense BF16)             │  │
│  │    → TP allreduce                                                │  │
│  │                                                                  │  │
│  │  (future: DenseFFN, ...)                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Per-Step State                                  │
│                                                                         │
│  StepState { cos, sin, positions, page_table, cache_seqlens,           │
│              cu_seqlens_q, ssm_cache_indices, is_decode }              │
│                                                                         │
│  - Built once per step by ModelRunner                                  │
│  - Passed to every layer                                                │
│  - Each attention block pulls what it needs                            │
│  - Cache refs bound at init via bind_cache(), not per-step             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       CUDA Graph Capture                                │
│                                                                         │
│  CapturedGraph                                                          │
│    static StepState buffers (same tensor objects every replay)         │
│    static output buffer                                                 │
│    per-graph workspace pools (attn + MoE)                              │
│    per-layer output buffers                                             │
│                                                                         │
│  GraphPool                                                              │
│    decode graphs keyed by batch_size                                   │
│    prefill graphs keyed by batch_size (optional)                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       Tensor Parallelism                                │
│                                                                         │
│  TPGroup                                                                │
│    allreduce_sum_() over NCCL                                          │
│    tp_shard_dim0/dim1() with pad-and-shard for uneven TP              │
│                                                                         │
│  launch_tp()                                                            │
│    mp.Process per rank, worker health monitoring                        │
│    KV budget synced via all_reduce MIN                                 │
│    30-min NCCL timeout for first-pass kernel JIT                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      Model Recipes                                      │
│                                                                         │
│  recipe_minimax_m2.py                                                   │
│    62 layers × TransformerLayer(B12xPagedAttention, MoEFFN, RMSNorm)   │
│    256 experts, top-8, sigmoid routing, TP=2                           │
│                                                                         │
│  recipe_qwen3_5.py                                                      │
│    15 × TransformerLayer(B12xPagedAttention, MoEFFN, GemmaRMSNorm)    │
│    45 × TransformerLayer(GDNLinearAttention, MoEFFN, GemmaRMSNorm)    │
│    512 experts, top-10, softmax routing, shared expert, TP=any         │
│                                                                         │
│  (future: recipe_deepseek_v3.py, recipe_llama4.py, ...)               │
└─────────────────────────────────────────────────────────────────────────┘

Adding a new model:
  1. Write a recipe that picks blocks and composes TransformerLayer
  2. Register with @register_recipe("model_type")
  3. Done — engine, runner, graphs, TP all work unchanged

Adding a new attention type (e.g. MLA):
  1. Write MLAAttention(nn.Module) with bind_cache() + forward_from_state()
  2. Use it in a recipe: TransformerLayer(attn=MLAAttention(...), ...)
  3. Done — no changes to layer.py, runner.py, or cuda_graph.py
```
