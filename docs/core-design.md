# Flare core design: Python model preparation, Rust graph execution

## Purpose and founding principles

Flare is a local inference server for NVIDIA SM120 and SM121 GPUs. It serves
one to a few concurrent users running agent workloads: long contexts, repeated
turns with appended tool results, shared-prefix branches, and image inputs. It
supports exactly one kernel library and has no backend abstraction.

### Principles

- **One hardware target, one kernel library.** FlashInfer supplies GPU kernels,
  including its b12x CuTe DSL operators for the model and its sampling kernels.
  The full FlashInfer library is available to preparation. There is no
  dispatch layer, no fallback implementation, and no portable code path.
- **Graph capture is the native interface.** A captured CUDA graph holds the
  compiled kernels, their argument blobs, and the buffer addresses. The runtime
  launches graph executables and uploads metadata; it never needs the kernel
  library's calling convention. b12x being pure Python therefore costs nothing
  in steady state. It costs startup time and preparation complexity, both
  bounded by the preparation lifecycle in section 2.
- **Python prepares; Rust serves.** Python bootstraps the process, loads the
  model, and captures its graphs. Rust starts the server and owns every request
  operation, including sockets, tokenization, chat templates, image
  preprocessing, scheduling, grammar advancement, output decoding, and
  streaming. Serving never invokes a Python callback.
- **Explicit state.** Every mutable model quantity is caller-owned storage
  indexed by slot or page: KV pages, compressed selector keys, recurrent
  matrices, convolution histories, n-gram histories, and positions.
  Checkpoints, branches, and speculative rollback are copies and index updates
  over that storage, never reconstruction.
- **Admission decides capacity.** A request that does not fit the prepared
  capacities fails at admission with a concrete error. The steady state never
  allocates, compiles, or recaptures.
- **Model families own their semantics.** A Python recipe supplies checkpoint
  tensor mapping, replaced forwards over b12x operators, state layout, and native
  asset configuration. A statically selected Rust model contract supplies input
  packing, prompt interpretation, and output parsing. CUDA transport remains
  shared; unsupported model semantics require an explicit implementation before
  serving those requests.

### Scope relative to production servers

Local serving omits most of a production server's surface: disaggregated
prefill, multi-backend dispatch, broad quantization-format support, adapters,
pooling models, and cross-node parallelism. The features that matter for agent
workloads are prefix caching that includes recurrent state, multi-turn append,
cancellation, streaming, tool-call presentation, structured output, and
speculative decoding through the checkpoint's multi-token prediction (MTP)
head. The first five are core to this design, structured output has a stated
policy in section 3, and MTP follows base serving.

### Viability assessment

- **Kernel coverage.** b12x at the pinned revision provides operators for every
  region of the acceptance model: Gated DeltaNet (GDN) prefill and decode, QSA
  sparse attention with its selector, fused NVFP4 mixture-of-experts with
  compressed scale planes (CSF), hyperconnection, PLE n-gram hashing and
  embedding with a mapped-host table, MXFP8 and weight-only NVFP4 linear, BF16
  vocabulary projection, and varlen attention for vision. Each computes its
  per-call metadata on device from device-resident counts, so no attention or
  state operator needs host planning between replays. Required additions are a
  prepared GDN causal convolution factored from PLE state machinery, an
  independent preparation entry for the existing KV writer at 256-dimensional
  heads. FlashInfer supplies sampling; the server owns request state, penalties,
  grammar masks, and the per-request RNG contract.
- **Native runtime value.** At decode batch 1 the model's weight traffic sets a
  floor near 4 ms per token; section 5 gives the breakdown. The
  [native executor](../helios/src/execution/transport.rs) stages metadata in
  pinned buffers and keeps up to two submissions in flight. Rust also renders
  chat templates and tokenizes inputs without Python's global interpreter lock.
  Profile graph replay, completion frames, device request state, and metadata
  staging against minimal replay of the same graphs. Those measurements set
  optimization priorities; the scheduler and checkpoint policy remain
  Rust-owned, so requests never return to Python.
- **Principal risks and their coverage.** State correctness across six
  state-bearing mechanisms under chunking, batching, and checkpoints is covered
  by the acceptance cases in section 5. Reference provenance is covered by the
  locked environment and stored golden outputs. Startup cost is covered by the
  persistent kernel cache. Structured output's conflict with pipelined decode
  is covered by the one-in-flight policy.
- **Acceptance model.** The acceptance model front-loads nearly all mechanism
  complexity. Small captured graphs exercise the model-independent bundle,
  startup, and failure contracts; the complete Qwen graph catalog exercises
  model state and Python/Rust execution parity.

## 1. Target and execution contract

Build a local inference server in which Python constructs the model and captures
its executable CUDA graphs, then hands the prepared resources to Rust. Rust
starts the server and owns all request handling, scheduling, cache management,
and graph replay.

The first acceptance target is **one 96 GB RTX PRO 6000 Blackwell**, running text
and image requests against:

- Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`.
- Checkpoint revision: `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`.
- Architecture: `Qwen4ExpForConditionalGeneration`.
- Transformers preparation reference: `4cc2aa84301c9aa210b5513dab9fefae03981f6e`,
  installed in a locked environment for canonical architecture and processor
  semantics. Exact-checkpoint reference loading requires the quantization
  qualification described in section 5; golden outputs are stored with their
  inputs and loader provenance.
- Kernel library: FlashInfer
  `d019c06357ba22009f8112c2f8606b1edd174fcc`, including the CuTe DSL operators
  under `flashinfer/experimental/b12x` and FlashInfer sampling. Every operator launches from
  Python; captured CUDA graphs are the only native execution path.

The checkpoint combines 36 Gated DeltaNet layers, 12 sparse-attention layers,
512 routed experts per decoder layer, four hyperconnection streams, PLE n-gram
embeddings, and a vision encoder. Its stored formats include NVFP4 experts with
CSF scales, MXFP8 projections, and mixed-precision vision weights.
[Checkpoint](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD/tree/dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2)

Native multi-token prediction (MTP) follows base serving. Tensor parallelism and
SM121 remain architectural requirements, with separate qualification after the
single-GPU core.

### Runtime contract

- Every admitted model execution runs through prepared CUDA graphs.
- Preparation resolves kernels, allocates storage, warms execution paths, and
  captures every supported graph variant.
- Python performs bootstrap, model loading, preparation, and capture, then hands
  ownership of the prepared resources to the native core.
- Rust owns all runtime sockets, HTTP/SSE, queues, tokenization, chat-template
  rendering, image preprocessing, grammar advancement, sampling control,
  output decoding, cancellation, and health reporting.
- Rust never invokes Python to process a request, execute a model step, prepare
  attention metadata, update a cache, or sample a token. Strong Python resource
  references remain alive in the same process until native shutdown completes.
- Unsupported requests fail admission with a concrete capacity or feature error.
- Graph executable handles, their captured storage, and their CUDA context remain
  alive together.
- Steady-state serving performs no CUDA allocations, kernel compilation, or graph
  capture.

### Default capacities and behavior

| Setting | Default |
|---|---|
| Active requests | 4 |
| Decode graph capacities | 1, 2, 4 |
| Prefill scheduling | One request per chunk, interleaved with decode |
| Maximum prefill chunk | 4,096 tokens; configurable with `--prefill-chunk-size` |
| Prefill graph capacities | 1, 16, 64, 128, 256, 512, 1,024, 2,048, 4,096 tokens |
| Prefill tail placement | Smallest capacity that holds the tail |
| Maximum context | Checkpoint limit: 262,144 tokens |
| Main KV precision | BF16 |
| GDN recurrent state | FP32 |
| PLE table placement | CUDA-mapped host memory |
| Checkpoint boundaries | Exact rendered instruction, final prompt, and terminal consumed endpoints |
| Structured output | Host grammar engine; one submission in flight, with per-candidate masks during copy speculation |
| Vision acceptance | Images; video qualification deferred |
| Speculative decoding | Optional copy or learned-MTP preparation; one round in flight; qualification requirements in the MTP design |

The configured prefill chunk size must be a multiple of 64 between 64 and the
configured context limit. Its graph capacities are the sorted union of
`{1, 16}`, powers of two starting at 64 and not exceeding the configured chunk
size, and the exact chunk size. A non-power-of-two limit therefore has its own
captured graph. The native scheduler uses the declared limit and selects the
smallest captured capacity that contains each chunk or tail.
This configuration range is a structural contract, not a guarantee that every
size fits the model's kernel and device-memory limits. Preparation must validate
every declared operator query and workspace envelope; unsupported geometry or
insufficient memory fails startup without reducing the chunk size or omitting a
graph variant.

At the default prefill limit, ordinary execution declares 28 graphs. Speculative
execution replaces the three ordinary decode graphs with 14 verification shapes,
producing 39 graphs. Every variant uses the common FlashInfer sampler. The
catalog includes vision, prefill, resume, initialization, snapshot, and restore;
startup rejects incomplete declarations.

A context limit is distinct from aggregate cache capacity. Admission must reserve
enough resources for each request's prompt and permitted output. Aggregate page
capacity is the GPU memory remaining after prepared weights, operator
workspaces, the graph memory pool, live state, headroom, and the
checkpoint-state budget. At 24.75 KiB per cached token the acceptance target
yields on the order of 500,000 tokens; preparation reports the exact figure.

## 2. Python preparation and the Qwen recipe

### Retain the Transformers module tree and replace its forwards

Instantiate the canonical Transformers model on the meta device, keep its module
tree for weight naming, configuration, and vision-block structure, and stream
checkpoint tensors into their execution layouts.

Replace the forward methods of the text model, the decoder layer, sparse
attention with its indexer, Gated DeltaNet, the PLE layer, and the vision model
with prepared operations over packed inputs, fixed storage, explicit positions,
and cache metadata. The canonical forwards are unsuitable for capture: the
hyperconnection combine is inline in the decoder layer, the indexer runs a
Python loop per query with host synchronization and dense mask construction,
the PLE embedding moves ids between devices, position history is concatenated
onto the cache object every step, and the vision model loops over images on the
host.
[Transformers implementation](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

The [Qwen3.8 recipe](../flare/recipes/qwen38/model.py) keeps the Transformers
module tree and replaces its forwards with prepared operations.

Recipes control checkpoint loading, model composition, memory policy, tuning,
and graph-catalog construction. They compose concrete components from
`flare/components` and select preparation helpers from `flare/preparation`.
Shared components implement operations and resource ownership; each recipe
defines its own orchestration. Compatibility is enforced at the native handoff
through graph signatures, fixed device buffers, completion records, and owner
lifetimes.

Rust owns one serving lifecycle for HTTP admission, scheduling, cache ownership,
graph submission, and completion retirement. Model adapters declare input
layouts, capabilities, and prompt/output semantics through `ModelContract`.
Every adapter participates in the same native lifecycle. Model differences are
expressed as capabilities, buffer layouts, and semantic checks; adapters do not
implement separate admission, scheduling, cache-ownership, replay, or shutdown
loops.
The shared [graph catalog](../helios/src/model/catalog.rs) validates model limits,
buffer layouts, declared capacities, required operations, and every supplied
graph's input bindings. Native startup and graph selection use this common
implementation; adapters supply declarations and model-specific constraints.
Shared [request admission](../helios/src/model/admission.rs) owns request epochs,
sampling parameters, prompt tokens, output budgets, and grammar masks. The core
validates prompt limits and constructs admissions for every model. Model adapters
supply additional prompt metadata and validate its layout; Qwen includes rotary
positions and vision-feature indices, while GLM text requires no extra metadata.
The core validates scheduled rows and their admission generations before model
input packing. It computes row identities, query lengths, and final-prefill
sampling eligibility once. Both adapters use shared incremental page-table
uploads, with explicit page windows for target and shifted draft caches.

[Prepared projection components](../flare/components/linear.py) provide BF16,
FP32, and block-scaled b12x execution, tuning declarations, caller-owned outputs,
shared scratch, and captured projection overlap. Recipes supply weights and
geometry; the components have no Qwen-specific dimensions or state layout.
The shared [embedding component](../flare/components/embedding.py) prepares
b12x lookup into fixed output storage. Token IDs and live counts remain device
inputs; counted lookup zeros inactive rows. Qwen text and MTP use this component,
and checkpoint tests cover Qwen and GLM embeddings through native 4096-row
replay with exact outputs and stable allocation counters.
The shared [sampling component](../flare/components/sampling.py) owns request
epochs, Philox state, penalties, grammar masks, admission, FlashInfer sampling,
and completion records. Vocabulary size and EOS tokens come from the recipe.
Qwen extends request state with its PLE token history; the shared sampler has no
model-history dependency. Admission and sampling remain captured device work.
Shared [token-state operations](../flare/components/token_state.py) gather
pending tokens and positions, validate request epochs, and commit consumed
spans. Recipes can supply two-token history storage for previous-token
embeddings. Omitting history storage removes history loads and stores from
the kernels. Qwen supplies its PLE history and padding token explicitly.
Shared [routing and expert components](../flare/components/moe.py) retain FP32
route weights, configurable scoring and correction bias, fixed output buffers,
and b12x's default expert execution policy. Autotuning uses private activation
and routing fixtures with the recipe's router precision and scaling.

The shared [canonical constructor](../flare/components/model.py) creates meta
module trees without initializing CUDA. The [GLM 5.3 Flash recipe](../flare/recipes/glm53/model.py)
uses it for `Glm5NextForConditionalGeneration`. Its
[weight loader](../flare/recipes/glm53/weights.py) partitions KDA and MLA heads,
dense/shared MLP intermediates, and NVFP4 routed-expert intermediates for TP2/TP4.
Low-rank input projections, sparse indexers, normalization, and mHC weights are
replicated. Dense MLPs retain BF16 checkpoint precision; attention and shared
experts retain MXFP8. b12x slices compressed NVFP4 scales and rebases exceptions
before preparing GPU weights. Packed weight slices transfer directly through the
shared [weight loader](../flare/components/weights.py); scale expansion uses
caller-owned buffers that serialized layers can share. Loader contexts use one
PyTorch CPU thread for small compressed-scale planes and restore the caller's
thread count on exit. The shared
[prepared MLP](../flare/components/mlp.py) executes BF16 dense and MXFP8 shared
experts with the checkpoint's SwiGLU clamp and caller-owned intermediates.
The [MLA branch](../flare/recipes/glm53/mla.py) combines the replicated query and
KV low-rank inputs into one MXFP8 projection. Separate canonical RMSNorms read
views of the fused output and write contiguous query and latent buffers.
The [KDA branch](../flare/recipes/glm53/kda.py) packs Q/K/V, beta, and the
forget-gate input into one MXFP8 projection. A shared
[projection-chain component](../flare/components/projection_overlap.py) runs
the two output-gate projections on an auxiliary stream, overlapping the fused
input and forget-gate output projections. The branches use disjoint regions of
the planned scratch arena and join before convolution and recurrence. Batch-one
forget-gate input aliases the fused output; multirow graphs copy that slice into
caller-owned contiguous storage required by b12x GEMM. Convolution and recurrence
use b12x kernels and head-sharded FP32 recurrent state. Shared
[recurrent components](../flare/components/recurrent.py) provide state allocation,
device metadata, checkpoint export, and prepared bindings for Qwen GDN and GLM
KDA. Compiled KDA prefill graphs support chunks through 4096 tokens. KDA decode,
prefill, and MLP graphs include b12x output reductions and support native replay
on TP2 and TP4. The shared [prepared mHC component](../flare/components/mhc.py)
uses FP32 mixing coefficients, BF16 residual streams, fused weighted RMSNorm,
and caller-owned scratch. The [GLM decoder layer](../flare/recipes/glm53/decoder.py)
composes mHC, head-sharded KDA or sparse MLA, an attention all-reduce, fused post/pre mixing,
the sharded dense MLP or MoE, and a second all-reduce. MLP output and mixing
coefficients feed the following layer's fused post/pre operation; the final
layer materializes the residual streams for contraction. Graph variants support
decode and prefill through 4096 tokens.
Each rank hands its graph to its native replay worker. The
[GLM MoE composition](../flare/recipes/glm53/moe.py) keeps replicated router weights
in their stored BF16 format and produces FP32 logits. Routing uses sigmoid scores,
correction-biased top-8 selection, normalized weights
scaled by 2.5, and all 288 experts partitioned along the intermediate dimension.
The MXFP8 shared expert uses the same prepared MLP component as dense layers.

Dense-layer native replay matches the uncompiled prepared forward exactly.
MoE output qualification remains failing at `rtol=0.01, atol=0.001`: the default
b12x dynamic expert kernel uses BF16 atomic accumulation, and repeated eager
calls also differ. A 4096-token TP4 native replay preserves recurrent state,
attention, router logits, selected experts, routing weights, shared-expert output,
and replicated rank outputs exactly, but differs from the eager residual in
0.1% of elements under that pointwise check (maximum absolute error 0.0048828125).
The [complete text stack](../flare/recipes/glm53/stack.py) composes all 45 layers,
canonical token embeddings, final stream averaging and RMSNorm, and a sharded
vocabulary projection. Each graph capacity uses one residual buffer and one set
of mHC, KDA, MLA, dense-MLP, and MoE intermediates reused across serialized layers.
Every layer owns separate recurrent or cache state; graph variants share that
persistent state and the tuned operator arena. Embeddings and final normalization
are replicated. Every rank loads a contiguous vocabulary weight partition.
The shared [vocabulary component](../flare/components/vocabulary.py) projects
replicated hidden rows locally and gathers BF16 logits with b12x PCIe kernels.
Rank zero samples from the complete vocabulary. Local and gathered logit buffers
are caller-owned and shared across serial graph variants.
The [full-checkpoint TP test](../tests/recipes/test_glm53_stack_tp_cuda.py) exercises
prefill, decode, rank agreement, finite outputs and recurrent state, and stable
CUDA allocation counters through native replay.
Sampled generation through the Rust rank coordinator also passed on TP2 and TP4.
A 24-token chat prompt used a 128-token prefill graph, all 45 decoder layers,
and FlashInfer sampling. TP2 reached the requested 32-token budget; TP4 produced
29 tokens and stopped at a checkpoint EOS token. Peers stayed inside Rust from
handoff through coordinated drain, with stable allocator counters. The test's
rank-zero Python harness submitted native uploads and inspected completion
records; rank communication and replay were native. Replicated final hidden
outputs and consumed-token counts agreed after drain.
The shared [continuation storage](../flare/components/state.py) lays out aligned
typed regions in one byte row per null, live, or immutable slot. Qwen retains its
117,291,776-byte ordinary continuation layout. GLM binds KDA recurrence, convolution
history, and C4 index tails directly to strided views in the
[continuation pool](../flare/recipes/glm53/pool.py). Request consumed-token counts
alias its four live rows, and compiled generation saves each active request's
final normalized hidden output. A GLM row includes one 256-token latent page and
one C4 index page per MLA layer for unaligned endpoints. It occupies 75,420,672
bytes per TP2 rank or 38,515,712 bytes per TP4 rank; a 4-GiB snapshot budget holds
56 or 111 rows, respectively, excluding null/live rows and shared KV pages.
Complete TP2/TP4 generation passes with these strided bindings, saved hidden
outputs, exact replicated-rank agreement, and stable replay allocation counters.
Shared [state transfers](../flare/components/state_transfer.py) validate complete
row copies, resets, exact consumed endpoints, destination conflicts, and partial
page aliases before writing. A snapshot saves every KDA state, index tail, and
the unaligned physical latent/index page; restore writes the saved page into the
request's private tail page. Aligned endpoints copy no external KV bytes.
Immutable snapshots can restore into multiple independent live slots. Native
bytewise tests cover endpoints 1, 3, 4, 255, 256, 257, 511, 512, and 513 and verify
that rejected commands leave all pool and cache bytes unchanged.

[GLM maintenance](../flare/recipes/glm53/maintenance.py) composes shared request
admission and continuation transfers into init, restore, and snapshot graphs.
Eight BF16 flags carry per-command validation failures and participation through
a b12x all-reduce before any persistent write. Any rank's rejection suppresses
the command on all ranks and emits completion error 72 with zero counters.
Resident request state remains unchanged on failure. Native TP2/TP4 component
tests cover successful admission and branch restoration, plus an epoch,
endpoint, page, admission, or command-count fault on one rank. Replay allocation
and segment counters remain stable. These maintenance tests use Python/Gloo to
coordinate rank submissions to native replay workers.

The shared [resume state](../flare/components/resume.py) validates consumed
endpoints, pending samples, epochs, and rank participation before gathering saved
normalized hidden outputs. [GLM resume](../flare/recipes/glm53/resume.py) projects
the saved rows across vocabulary shards and gathers logits on every rank.
Rank zero performs FlashInfer sampling; peers record completion without
consuming tokens or changing continuation state.
Complete-checkpoint TP2/TP4 tests run native init, 267 live tokens in a compiled
512-token prefill graph, snapshot, two branch restores, resume, and decode through
the Rust rank coordinator. Each branch shares the full 256-token page and owns
its partial latent/index page. The snapshot includes three unfinished C4 tokens.
Resume logits match prefill logits byte-for-byte. Both branches preserve the
immutable snapshot and source KV pages and finish at a checkpoint EOS token.
The sampled continuations contain 18/18 tokens on TP2 and 18/26 tokens on TP4.
Peers remain inside Rust throughout execution and coordinated drain; allocation
and segment counters stay stable during replay. The rank-zero test harness
submits uploads and inspects results in Python.
The [GLM native contract](../helios/src/model/glm5next.rs) selects init, restore,
snapshot, resume, prefill, and decode graph capacities and packs request uploads.
It uses 256-token page tables and broadcasts each pending decode token to every
rank. Reset commands use source slot -1; exact restoration names the saved
consumed endpoint and a private partial-page destination. Qwen and GLM share
host admission staging for sampling parameters, prompt histograms, and grammar
masks. The GLM catalog declares no vision graphs; native startup omits the image
processor and rejects image inputs.

Native GLM prompt preparation uses the checkpoint chat template and preserves
exact instruction prefixes through the empty user header. Its output parser
shares incremental think/tool envelopes with Qwen while interpreting GLM
`<arg_key>/<arg_value>` payloads and preserving content whitespace. Tests compare
rendered bytes, token IDs, instruction boundaries, assistant continuations, and
parallel tool history against Transformers. GLM structured output with an open
reasoning prefix is unsupported; the template does not honor Qwen's
`enable_thinking=false` option. Native TP2/TP4 HTTP checks cover repeated-prompt
cache reuse, four concurrent requests, SSE, a 6,639-token prompt spanning
4096-token prefill chunks, and parsed tool calls. These checks exercise the
Python CLI handoff and the shared Rust server; independent full-model numerical
qualification remains incomplete.

The [GLM bootstrap](../flare/recipes/glm53/bootstrap.py) prepares 24 graphs per
rank for four active requests and a 4096-token prefill limit. Decoder variants
bind bounded views of one largest-capacity activation allocation; distinct
layers reuse those intermediates while persistent continuation state remains
separate. Recurrent autotuning borrows inactive continuation storage and clears
it before capture. The HTTP checks use 16,384-token context, 256 KV pages, a
1 GiB snapshot budget, and 4 GiB of reserved headroom per rank.

[GLM batch preparation](../flare/recipes/glm53/batch.py) converts native token,
slot, epoch, length, and page-table uploads into KDA and sparse-MLA metadata.
Epoch, pending-token, vocabulary, context, and physical-page checks precede cache
mutation. Eight BF16 flags represent per-slot validation errors and participation;
a b12x all-reduce makes every rank reject the same rows before packing tokens.
Requests whose resident epochs do not match are left untouched. Native TP2/TP4
tests cover a pending-token fault on rank zero and an epoch mismatch on one peer,
with identical packed metadata on every rank and continued consumption by valid
requests. The [generation composition](../flare/recipes/glm53/generation.py)
joins this preparation with the complete decoder and shared token consumption.
Rank zero samples through FlashInfer; peer ranks use shared completion recording
without vocabulary tensors or RNG work. Native transport tests cover admission,
65 live tokens in a 128-token prefill graph, decode, and both completion ABIs.

Shared [sparse MLA components](../flare/components/sparse_mla.py) prepare b12x
latent-cache writes and attention over caller-selected physical records. Recipes
supply geometry and device metadata; results copy into caller-owned output
storage before another operator reuses the arena. Arena finalization refreshes
bindings after tuning. Fixed-shape compiled graphs with 32-head decode and
16-head 4096-token prefill run through the native replay worker with exact cache
and output agreement against uncompiled execution and no allocator changes.
b12x owns the sparse kernels, latent-record encoding, and pooled-selection
expansion. The shared [sparse indexer](../flare/components/indexer.py) prepares
FP8 selection with private finite tuning operands, logical or physical output
indices, device visibility lengths, and shared scratch. Prefill uses one shared
page table; decode accepts a page table per query row. The prepared b12x call
exposes tensor dependencies to compilation, resets fused merge counters before
arena reuse, and retains unused score outputs in planned storage. Tuned decode
and 4096-row prefill graphs run through the Rust replay worker without allocator
activity. Selected IDs and scores match uncompiled execution, allowing either
candidate at an exactly equal-score cutoff tie.

b12x also provides prepared GLM C4 pooling and FP8 query rotation. Caller-owned
BF16 key/gate tails carry incomplete pools between chunks. Nine consecutive
65-token chunks crossing physical pages produce
the same cache and tail as ordered decode over the combined input. The strict
4096-token PyTorch pooling comparison remains failing at two FP8 bytes, each
one representable step apart. The [GLM indexer recipe](../flare/recipes/glm53/indexer.py)
composes checkpoint MXFP8 query/key/head-weight projections, BF16 pooling gates,
canonical LayerNorm, rotated FP8 queries, C4 pool updates, selection, and physical
slot expansion. A model page holds 256 raw tokens and one 64-record index page;
expanded physical indices address the main latent cache's 64-record pages.
Prefill graphs admit one request; decode supports packed rows from multiple
requests. Position -1 and negative state slots mask inactive queries and
requests. Three consecutive chunks, including 4096-token chunks, preserve
projection outputs, pool cache, sequence tails and physical selections through
compiled Rust replay. The compiler bridge retains ATen LayerNorm because
Inductor's decomposition changes BF16 rounding. Shared prepared batched
projections implement query absorption and value expansion directly from
strided MXFP8 KV weights and rowwise E8M0 scales. The b12x BMM supports 16/32-head
partitions through 4096 rows with no scratch, copies, or weight dequantization.
Its opaque compiler boundary exposes scale bytes as uint8 views. Checkpoint
projection tests cover compiled native replay, allocation stability, and the
BF16 matrix-product error bound against FP64.

The [GLM MLA recipe](../flare/recipes/glm53/mla.py) composes canonical low-rank
normalization, head-sharded queries, native query absorption, latent-cache
writes, C4 selection, sparse attention, value expansion, and the partial hidden
output projection. Each rank retains identical latent/index cache geometry;
the output requires a tensor-parallel reduction. Compiled native replay covers
both 16- and 32-head partitions, decode, and three consecutive prefill chunks
through 4096 tokens. Cache bytes, C4 tails, selected-token sets, inactive-row
masking, and allocation counters match uncompiled execution. Exact output
qualification remains failing for 32-head decode and both 4096-row cases.
The decode output projection uses BF16 atomic split-K accumulation and varies
on repeated eager calls. Prefill selection returns the same set in different
orders; attention replay matches eager execution exactly when both consume the
same ordered selection. No deterministic kernel policy or runtime sort is
imposed.

Sparse MLA decoder layers include the sigmoid MoE and both b12x reductions in
one compiled graph per rank. Native TP2/TP4 decode, TP2 65-token prefill, and
TP4 4096-token prefill preserve cache/tail state and allocator counters relative
to uncompiled execution; reduced outputs and selected experts agree across
ranks. Numerical qualification remains failing: atomic projection and expert
reductions, and unordered top-k attention, propagate differences through the
residual. Maximum observed residual absolute error is 0.015625 for the decode
and 65-token cases. The 4096-token case first differs at the reduced attention
output (maximum absolute error 0.00146484375). Dense KDA layer checks remain
exact on TP2/TP4 through 4096 tokens.

[Prepared PCIe reductions](../flare/components/collectives.py) embed b12x
single-channel TP2/TP4 all-reduce in compiled forwards with caller-owned outputs.
A bootstrap-owned IPC runtime reserves the largest configured BF16 hidden
activation: `max_tokens × hidden_size × 2` bytes. A 4096-token GLM prefill needs
32 MiB per collective input. The runtime and its IPC buffers persist across
serialized graph variants.
A [Gloo bootstrap coordinator](../flare/preparation/collectives.py) matches
collective declarations and barriers before kernel preparation. Completed ranks
continue exchanging readiness until all peers finish local preparation. Kernel
preparation failures propagate through the control group before job cleanup.
Per-rank native
workers can replay the captured reductions without Python between iterations.
The [native rank coordinator](../helios/src/execution/parallel.rs) owns rank-zero
execution and Unix control sockets to peer processes. Each peer enters Rust once
after Python prepares its graph bundle. Startup compares rank identities and
logical graph I/O schemas; CUDA handles and addresses stay process-local.
Before launch, every rank validates the same uploads and next fence. One step
may be outstanding, and its completion retires only after all ranks agree on
request epochs, consumed-token counts, and compatible status. Rank zero alone
returns sampled tokens. b12x performs GPU collectives inside the captured graphs.
Native TP2/TP4 tests cover admission, prefill, sampled decode, backpressure,
both completion ABIs, and allocation stability without Python peer-step calls.
Shutdown drains every rank before releasing graph owners and IPC mappings.
Disconnected ranks or graph-completion timeouts retain resources without
synchronizing a potentially stranded collective. Process supervision must
terminate the whole failed rank group before reclaiming those resources.
The native model scheduler owns either the local executor or this coordinator
through [RankExecutor](../helios/src/execution/ranks.rs), with no intervening
execution-worker queue. Native server startup accepts prepared peer sockets.
The Python server binding selects Qwen or GLM from the prepared input schema
once at startup. Both model runtimes use the same native rank coordinator and
HTTP server. The [Rust rank supervisor](../helios/src/processes.rs) watches
owned children through Linux pidfds. Unexpected peer exit or an expired drain
deadline terminates the rank group without running CUDA destructors. Peer
processes receive a parent-death signal if their root exits. Coordinated shutdown
allows peers to finish before Python reaps them. Process-level tests cover peer
crashes, parent exit, shutdown deadlines, and normal drain.
The TP2 CLI exits normally on SIGTERM and releases both CUDA contexts. Injecting
SIGKILL into one TP4 peer during a streamed HTTP response produces a control-socket
error, terminates all four owned ranks, and releases their CUDA contexts. Forced
exit leaves multiprocessing semaphore cleanup to Python's resource tracker.
The bootstrap must retain each PCIe runtime until every rank has drained replay.

The shared [tuning session](../flare/preparation/tuning.py) owns declaration
collection, candidate scratch reservation, autotuning, and binding finalization.
Recipes supply the workload schema, numerical policy, and search-limit metadata;
the [Qwen policy](../flare/recipes/qwen38/tuning.py) defines those values for its
operator catalog. Constructors declare operators without compiling or launching
them. Each weight or cache-state batch containing collective declarations uses
the coordinator until every rank finishes. Each rank tunes
its local operators, reserves and trims its scratch arena, and finalizes bindings
before graph capture. Candidate sharding across ranks is not supported.

| Model region | Prepared execution |
|---|---|
| Quantized linear projections | b12x MXFP8 and NVFP4 operators |
| Routed experts | b12x fused MoE with prepared CSF weights |
| GDN recurrence | b12x GDN prefill and decode |
| GDN causal convolution | Independent b12x short convolution factored from PLE history and state machinery, consuming raw projected QKV |
| Sparse attention and selector state | b12x QSA |
| Hyperconnections | b12x hyperconnection primitives; BF16 low-rank projection GEMMs outside the operator |
| PLE lookup | b12x NVFP4 PLE embedding over mapped host storage; server-owned n-gram history |
| PLE mixing and convolution | b12x PLE operators |
| Vision | Transformers block structure; b12x varlen attention, MXFP8 attention projections, weight-only NVFP4 MLPs through the block-scaled `w4a16` entry |
| Vocabulary projection | b12x BF16 vocabulary projection |
| KV writes | Existing b12x paged KV-write kernel through an independent prepared entry supporting 256/256 head dimensions |
| Request metadata, state maintenance, sampling | Captured server operations |

Ordinary tensor composition may remain expressed in PyTorch during capture.
Decoder hyperconnections fuse residual injection with the following mixer's
RMSNorm using FP32 gains, an in-place BF16 residual, and caller-owned normalized
scratch. The PLE injection boundary requires a separate normalization. MTP
fuses the attention-to-MLP boundary; its inactive-row mask precedes the final
standalone normalization.

GDN combines QKV/Z and B/A weights into two MXFP8 projections. Sparse attention
combines Q/K/V into one projection and keeps the indexer independent. MTP uses
the same Q/K/V grouping with BF16 weights. Loading concatenates serialized
values and scale rows before b12x packs the scales; it does not requantize
weights. Output slices share persistent packed scratch. The b12x convolution,
recurrence, and KV writer consume row-strided views without unpacking copies.

Operators that allocate their outputs per call, including standalone
hyperconnection combine, are acceptable inside capture
because the allocation resolves into the graph memory pool. Graph variants share
one pool only for intermediates whose lifetime ends within their graph. Values
that survive a graph boundary live in caller-owned persistent storage outside
the graph-private pool: request and cache state, snapshots, saved hidden
vectors, vision embeddings, and completion frames. Each graph copies escaping
results into that storage before completing. Graphs execute serially on the
model stream, including any captured side-stream joins; serialization alone
does not establish safe pooled-output lifetimes. Qualification alternates
decode, prefill, maintenance, and vision variants in different orders.

### Checkpoint loading

Use the exact checkpoint's safetensors index and quantization configuration.

The loader must:

- Account for every required tensor through an explicit recipe mapping.
- Preserve packed NVFP4 values, global/input scales, MXFP8 scale interpretation,
  and PLE hashing constants.
- Feed compressed expert scales into b12x's CSF preparation path.
- Share CSF expansion scratch across serialized layer execution.
- Convert normalization weights only where the selected b12x contract requires
  it, and qualify the numerical effect.
- Exclude MTP tensors explicitly while MTP is disabled.
- Release loading intermediates incrementally.
- Stream the PLE table into write-combined mapped host memory with sequential
  writes and never read it back on the CPU.

Avoid materializing a second full model or a fully dequantized expert table.

Native NVFP4 compressed-scale batches are cached under
`$XDG_CACHE_HOME/flare/nvfp4-scales`, defaulting to `~/.cache/flare/nvfp4-scales`.
Checkpoint file metadata, tensor ranges, partition geometry, and b12x's code
cache identity determine reuse. Warm loads transfer these batches through the
b12x direct loader. The cache contains compressed scales and exception indices;
packed expert weights continue to load from the checkpoint.

The local snapshot's safetensors headers give this storage breakdown:

| Component | Stored GiB | Initial placement |
|---|---:|---|
| Decoder routed experts | 59.94 | GPU |
| Other decoder tensors | 6.57 | GPU |
| Vision tensors | 0.30 | GPU |
| PLE tables | 26.82 | Mapped host memory |
| MTP tensors | 1.49 | Unloaded |
| **Total checkpoint** | **95.12** | |
| **GPU-bound stored tensors without MTP** | **66.82** | |

These are stored tensor sizes. Prepared layouts, scratch, activations, graphs,
and caches require additional memory.

Preparation produces a memory report covering persistent weights, mapped host
storage, operator workspaces, the shared graph memory pool, live and speculative verification
state slots, checkpoint snapshots, and page capacity.
Reserve 4 GiB of GPU headroom and a 4 GiB checkpoint-state budget by default;
allocate the remaining planned capacity to pages. Fail preparation if the
configured capacities cannot fit.

### Gated DeltaNet convolution

b12x's GDN packages treat convolution as external. The prepared
`sequence.short_conv` operator comes from the editable sibling `../flashinfer`
checkout. Reproducible environment setup verifies that checkout's clean status
and HEAD against the revision in `pyproject.toml`; it performs no patch application.
The PLE implementation in `sequence/ple` supplies
reusable history and state machinery: caller-owned state sized by dilation and
kernel width, state-slot ids, a fresh-state flag, accepted-token counts,
device-resident sequence and token counts, and a mixed prefill-plus-decode mode.

The independent prepared short-convolution operator shares that machinery.
GDN consumes raw projected QKV, stores its raw input history, and uses dilation
1 with SiLU over the projection width. The complete PLE operator also performs
gating, normalization, and residual addition; those operations do not belong in
GDN. Preserve PLE's own input-history and residual semantics when sharing the
convolution machinery. Qualify convolution and activation rounding against the
GDN references. PLE's Triton implementation is a correctness reference only,
so the factored operator needs a throughput-qualified path.

The prepared interface supports:

- Packed prefill and single-token decode.
- Caller-owned convolution state.
- Explicit request-to-state-slot mapping.
- Fixed token and sequence capacities.
- Inactive lanes that perform no state writes.
- Capture-safe execution without CPU sequence-length extraction.

The [GDN component tests](../tests/recipes/test_qwen38_gdn.py) compare exact
checkpoint execution with the canonical Transformers implementation. The
serving recipe consumes the prepared b12x operator directly.

### KV writes

Flare owns the cache-write contract: write projected K/V to the page and offset
identified by request metadata, then execute attention. QSA does not write the
main cache; writes precede its call.

b12x `attention/paged_decode` contains a capture-safe KV-write kernel driven by
a per-token slot mapping in which negative entries are skipped. Its
`[pages, page_size, kv_heads, head_dim]` layout matches the QSA main cache, and
the kernel supports the model's BF16 256-dimensional heads. Its public
`write_kv` entry requires a prepared attention plan, whose supported head
dimensions are only 192/128 and 128/128 at the pinned revision.

FlashInfer exposes the writer through `attention.paged_kv_write`, an independent
preparation entry with 256/256 support. The attention plan's supported dimensions
remain unchanged.
Slot mappings are contiguous int64 values `page * page_size + offset`; live
destinations must be unique and padded input rows must have valid backing
storage even when their negative mappings suppress writes. The writer is
captured inside the forward graph. Rust assigns pages; the captured GPU program
writes them.

### Sparse attention arithmetic

The Qwen recipe binds `qsa.Caps(attention_splits=16)` for every prefill and
decode capacity. Attention processes at most 64 rows per launch, requiring
24.094 MiB of FP32 partial-output/LSE scratch from the shared operator arena.
Chunks and layers reuse that storage. Selector
capacity and cache updates retain their declared row geometry. The fixed split
count is part of the b12x prepared-query identity, not an autotuning knob.

Row-dependent direct/split dispatch changes BF16 probability rounding. Fixed
partitions preserve the attention reduction order across graph capacities.
Removing the fixed policy requires calibrated cross-capacity component checks
and a measured kernel timeline; same-capacity replay remains bitwise.

### GDN prefill arithmetic

The Qwen recipe declares
`gdn_prefill.Caps(recurrence_policy="sequential_k1", ...)` for every prefill
capacity. It preserves BF16 intermediates and FP32 recurrent state, constraining
execution to sequential recurrence with `k_split=1`. Value splitting, pipeline
stages, and window tiles remain tunable. Key splitting and chunk-parallel
recurrence change reduction association and are excluded from this recipe's
search space. The policy is part of the prepared query and selection-cache
identity; b12x rejects incompatible explicit configurations.

The BF16 recurrence processes 16-token chunks relative to each prefill invocation.
An arbitrary consumed endpoint is a legal final-state boundary, but restarting
from that state can change chunk arithmetic. Sequential single-key-split policy
does not make computation invariant to splitting inside a chunk. Exact
snapshot transfer preserves the split execution's state and continuation.

Allowing other recurrence schedules requires calibrated state and operator
comparisons on identical operands. Identical graph inputs and restores of the
same execution partition must remain bitwise equal. Complete-model logit
distance is diagnostic and does not select the recurrence policy.

### Preparation lifecycle

`RecordedPreparationSession.declare()` collects requests without compiling,
priming, or binding kernels. Components use `flare.preparation.tuning.declare()`
to submit requests and `register_finalizer()` to defer serving bindings. After
every admitted capacity has an owner, bootstrap prepares two b12x batches:
weight operations, then operations that depend on cache or recurrent-state
layout. Both batches enable autotuning and coalesce equivalent declarations
across layers, target and draft models, and graph capacities. On the target
RTX PRO 6000, preparation defaults to eight compiler workers;
`B12X_COMPILE_WORKERS` overrides the count.

Qwen declares attention against bounded cache capacities and temporary page
pools. Its selected plans support replacement page pools with the same layout.
Bootstrap measures graph memory with uncompiled captures before allocating the
final pools, rebinds attention to those pools, then compiles and captures the
serving graphs. GLM's configured
page pools are allocated before its cache-state batch.

Searchable operations use deterministic finite fixtures with active metadata at
their declared geometry. Weighted operations use the actual checkpoint weights.
Each timed trial has an activation producer and, where needed, state restoration.
Serialized GDN and routed-expert candidates each share one fixture per request
and use the shared operator scratch arena. GDN state fixtures borrow the unused
hybrid state pool while preserving the serving row strides. After trial owners are released,
preparation synchronizes, restores every hybrid state region and PLE history to
its admission value, and synchronizes again before capture. A restoration failure
prevents capture and freeze.

Hyperconnection priming borrows the decoder's zero-initialized residual
storage through capacity views. Priming only reads that source; normalized and
projected outputs occupy separate scratch. Autotuning retains private seeded
fixtures.

One `flare.preparation.scratch.ScratchArena` supplies byte storage to quantized
projections, GDN, QSA, routed MoE, PLE, and vision attention across serialized
layers and graph executions. `arena.views(*shapes_and_dtypes)` returns contiguous
typed views with 256-byte-aligned, nonoverlapping regions for buffers needed by
simultaneous work. Serial invocations reuse the arena from offset zero.
`register_parallel(*plans)` declares operations whose scratch must coexist;
their aligned requirements are added within each group. Separate groups reuse
storage, so the reserved extent is the maximum across serial operations and
concurrent groups.
`reserve(nbytes)` grows the backing tensor during preparation. Candidate
measurement uses the maximum scratch requirement of the complete legal
candidate envelope, including alignment, rather than a sum of per-family,
per-layer, or per-candidate allocations. After selection and CUDA synchronization,
`trim(nbytes)` replaces that storage with the peak requirement of the selected
plans and concurrent groups. `freeze()` then rejects growth beyond the retained extent.

`RecordedPreparationSession.register_scratch(arena)` associates that storage
with the batched declarations. The session reserves the complete candidate
budget before measurement and keeps that extent stable throughout the trials.
It obtains the selected requirement from each plan's public
`memory_requirements()`, synchronizes before replacing storage, trims the
arena, and freezes the retained extent. `arena.finalize()` then refreshes
every binding and shared alias against that final allocation, including
singleton operations whose equivalent declarations share a prepared payload.
Standalone component tests using a plain b12x session prepare and bind eagerly;
model startup always uses the batched, tuned lifecycle.
Bound addresses must not change after capture.
The arena remains owned until all referencing graphs and native CUDA work drain.

One preparation-owned auxiliary stream overlaps independent GDN projections,
shared and routed experts, and QSA selection with its main QKV producer.
Concurrent branches use disjoint scratch views. Opaque operations preserve
forks, joins, input lifetimes, and tensor mutations through forward compilation.
Every branch rejoins before dependent outputs or reused scratch are consumed.
Python captures these dependencies inside one CUDA graph; Rust submits that
graph without calling Python or managing the internal streams.

Startup compiles complete text forwards with Inductor before capture, including
prefill absorption, ordinary decode, resume sampling, and speculative rounds.
Prepared b12x operations remain opaque calls; pointwise PyTorch work is visible
to fusion and autotuning. Compilation preserves shared-storage mutations and
retains prepared owners until graph teardown. Serialized Inductor callables
receive persistent tensors as runtime inputs; native call indices refer to each
forward's retained invocation table. Compiled forwards reside under
`$XDG_CACHE_HOME/flare/compiled`, defaulting to `~/.cache/flare/compiled`.
The cache stores compiled code without
checkpoint payloads or process-local kernel handles. Cache misses autotune with
the bound buffers before capture and serialization. Tests can set
`compile_decoder=False` to capture the same prepared operations without Inductor.
Serving never executes either Python forward after native handoff.

b12x retains priming `PreparedCall.owners` with prepared plans. Priming calls use
`prime_owners(arena, ...)` to retain the arena object without retaining obsolete
views or bindings from a larger allocation. Serving components own their
refreshed final bindings. Benchmark calls retain concrete trial views and
bindings through measurement and teardown. Explicit raw-tensor callers retain
their supplied storage owners. Replacing `arena.tensor` is insufficient to
release memory if another owner still references its previous storage; ownership
qualification checks those references as well as the reported arena extent.

Operator benchmark operands and trial-state arenas live in
[`preparation/fixtures`](../flare/preparation/fixtures/). GDN, fused MoE,
hyperconnection, variable-length attention, blockscaled GEMM, and MTP feedback
each provide a concrete fixture factory. Recipes retain plan declarations,
priming, selected bindings, and finalizers. Fixture factories preserve checkpoint
weight identity, tensor layouts and aliases, producer order, candidate scratch
requirements, and cleanup before capture; they do not own serving execution.

Scratch contents are volatile across operator calls. Kernels may not depend on
bytes left by a previous call. Model state, cache pages, request metadata,
weights, CSF expansion storage, and outputs that remain live while another
operator runs have separate storage. Decoder and vision activation buffers
therefore retain their existing lifetime boundaries. Graph variants share the
operator arena only under the native model stream's serialization and captured
side-stream joins. The preparation report records the initial allocation,
candidate requirement and reservation, selected requirement, final reservation,
and pending persistent bytes under `scratch_arena`. The model memory report records the retained
allocation as `operator_scratch_bytes`.

Selection caches are separate from compiled executable caches. The serving
namespace includes checkpoint revision, configuration and tensor-header hashes,
workload schema, and numerical policy. b12x also keys selection by compute
capability, SM count, query/configuration/semantic/candidate-contract versions,
encoded query, invocation, explicit configuration, and selected dependencies.
Completed cache entries can avoid a search; incompatible identities cannot reuse
the same entry. Compiled kernel objects persist across restarts independently.

The search preserves activation precision and arithmetic constraints. BF16
projections search every eligible b12x backend and `rows_per_tile` configuration
using private activation and output buffers with the serving tensor strides.
Hyperconnection down and residual-injection weights share one packed projection;
the canonical Transformers parameters remain views of that storage. Quantized MXFP8
wrappers retain default inner dense-GEMM lowerings; their existing benchmark
callbacks do not expose an inner search. Vision NVFP4 projections remain A16;
routed MoE remains activation-quantized. QSA fixes `attention_splits=16`, and GDN
prefill fixes sequential recurrence with one key split. The preparation report
records these limits, selections, cache hits, measurements, and memory release.

Preparation proceeds in a fixed order:

1. Validate checkpoint architecture, tensor mappings, and hardware.
2. Construct and patch the Transformers model; load packed weights and mapped
   PLE storage.
3. Allocate persistent state, live activation/output buffers, one operator scratch
   arena, and a one-page KV sizing pool. Construct every admitted variant and
   collect its weight and cache-state declarations without priming kernels.
4. Reserve the maximum candidate scratch envelope and keep its extent stable
   throughout the weight and cache-state preparation batches. Both batches
   enable autotuning.
5. Synchronize, trim the arena to the maximum selected scratch requirement,
   freeze its retained extent, and refresh selected bindings and aliases.
   Release trial owners, restore borrowed state, synchronize, and release
   unused allocator cache. Retain the trimmed arena for graph execution.
6. Warm and capture the uncompiled catalog to measure graph memory, then size the
   KV pools while preserving configured GPU headroom. Check actual allocator
   consumption before rebinding. If automatic sizing consumes headroom, fence
   and release the unbound pools, reduce their capacity using the measured
   deficit, and allocate again. An explicit capacity that cannot fit fails.
7. Rebind final KV storage, discard the sizing graphs, compile, warm and capture the final
   forward, sampling, vision, and state-maintenance graphs, and freeze preparation.
8. Validate catalog and frontend assets before native startup. Replay qualification
   checks changed inputs, allocation stability, and exclusive ownership separately.

Disabled or cancelled tuning, unresolved default selections, and failed binding
finalization prevent capture and freeze. Explicit overrides and singleton
configurations remain valid completed selections. Rust receives only the frozen
catalog and its owners; request processing never invokes the tuner or Python.
Native startup validates and consumes the model before binding the server socket,
as specified in section 3. Preparation success and cache reuse do not establish
numerical or performance acceptance.

## 3. Helios runtime and graph interface

Helios is the Rust serving core, packaged as `flare-helios` and imported by
Python as `flare_helios`.

### Process and ownership model

Start with a Python process containing a PyO3 Rust extension and a dedicated
native GPU worker. Python completes bootstrap and invokes native server startup;
Rust binds the listening sockets and owns the server lifetime and request paths.

`Server<M>` and `ModelRuntime<M>` statically select the
[`ModelContract`](../helios/src/model.rs). Shared core code validates graph
geometry, constructs admissions, schedules work, and retires completions. The
contract supplies prepared-buffer declarations, model-specific constraints,
input encoding, and completion checks. Its associated output protocol composes
separate reasoning and tool parsers. The Python entrypoint selects
[`Qwen4Exp`](../helios/src/model/qwen4exp.rs) or
[`Glm5Next`](../helios/src/model/glm5next.rs) from the prepared input schema;
there is no runtime backend registry or dynamic model dispatch per token.
Unknown schema versions, fields, or capabilities fail before asset loading and
CUDA worker construction.

Captured operators use b12x's default numerical policy and tuned execution plans.
MoE reductions may use arrival-ordered atomic additions, so identical graph inputs
and RNG state do not guarantee identical logits or sampled tokens. State transfers
and request metadata remain exact; recomputed floating-point results use calibrated
component bounds.

The Python resource owner retains:

- Transformers modules and prepared b12x objects.
- PyTorch tensors and captured graphs.
- Mapped host allocations.
- Any captured library resources.

Rust retains a strong reference to this owner while borrowing its graph handles
and buffer addresses. The worker sets the same CUDA context used during
preparation and executes without acquiring the GIL. Native tokenizer, template,
processor, and parser instances are constructed during startup and used without
Python on every request.

The graph bundle is an in-process execution object. CUDA handles are not
serialized or passed between processes. Startup consumes the prepared model in
this order:

1. Validate bundle metadata, context, capacities, and native frontend assets;
   retain the strong resource owner.
2. Start the native workers and wait for their ready acknowledgement.
3. Bind the server socket and admit requests.

Any startup failure joins started workers and drains submitted work before
releasing the resource owner. A failed startup exposes no serving endpoint.
Shutdown stops admission, closes listeners, completes or cancels requests,
drains CUDA work and transfers, and joins native workers before releasing the
Python owner with the GIL. Interpreter finalization with live workers is an
error. Python can block in the native wait operation with the GIL released;
it has no request-processing or server-control callback loop.

PyTorch exposes the instantiated executable handle through
`raw_cuda_graph_exec()`. Re-instantiation can invalidate that handle, so
preparation must finish before Rust registration and no subsequent
re-instantiation is permitted.
[PyTorch CUDA graph API](https://docs.pytorch.org/docs/main/generated/torch.cuda.CUDAGraph.html)

Use direct CUDA bindings inside a small Rust module. The core does not require a
Rust tensor framework or LibTorch model execution.

### Public interface

The Python-facing interface contains only preparation and server lifetime:

```text
prepare_model(model_spec, capacities) -> PreparedModel
start_server(PreparedModel, ServerConfig) -> ServerHandle
ServerHandle.wait()
ServerHandle.close()
```

`start_server` consumes the prepared model; a second handoff is rejected.
`ServerConfig` declares native listener and request-processing capacities.
`ServerHandle` retains the runtime and resource owner until shutdown finishes.
Request submission, cancellation, event consumption, and stats are internal
Rust interfaces exposed through the native server, not Python methods.

A normalized native request contains token IDs, generation parameters,
positional/conditioning metadata, any preprocessed image inputs with their
position-embedding interpolation tables and per-image sequence offsets, and an
optional compiled grammar. It contains no executable Python callbacks.

Events carry stable request IDs, generated token ranges, completion reasons, and
errors. Rust incrementally decodes tokens, parses reasoning and tool calls, and
formats HTTP/SSE responses. CPU-heavy tokenization and image processing run on
bounded preparation workers so they cannot block socket service, health, or
cancellation. Streaming disconnects cancel the associated request; an event
channel closing without a terminal event reports an error.

The native model thread uses a private `Worker` for admission, completion,
publication, retirement, planning, and submission. Every completion is validated
before scheduler mutation. Exact instruction and prompt checkpoints become
visible atomically in scheduler completion; stopped requests retire before another
completion is processed. Submission reserves scheduler ownership before model
input packing and graph enqueue. The executor remains outside the worker's
unwind boundary, and reusable inputs remain owned until stream drain completes.

Native [`Error`](../helios/src/error.rs) distinguishes admission, execution,
and frontend failures. Cloneable failures retain their underlying source chain;
CUDA failures retain the operation, numeric code, name, and description. Dispatch
and error assertions use variants rather than message substrings. Request
validation maps to HTTP 400, model admission capacity or worker unavailability
to 503, and execution failure to 500. After SSE headers are sent, failure terminates the
stream with an error event. Workers retain the first failure, and a failed CUDA
drain retains the graph and buffer owners.

### Native frontend dependencies

Helios uses Axum/Tokio for HTTP, Hugging Face `tokenizers` for tokenization,
and MiniJinja with Python-method compatibility for chat templates. Dependency
versions and features are specified in [Cargo.toml](../helios/Cargo.toml).
The [template renderer](../helios/src/frontend/mod.rs) supplies Python-compatible
JSON formatting, whitespace handling, and template exceptions. Rendering must
match the checkpoint template's bytes and token IDs, including tool schemas and
image content.

The [native image processor](../helios/src/frontend/images/mod.rs) is checked
against the checkpoint's Transformers processor. Compare resized and normalized
pixels, patch order, image grids, placeholder expansion, and positions before
admitting image requests.

### Prepared graph bundle

The bundle exposes:

| Contract | Contents |
|---|---|
| Identity | ABI version, model fingerprint, device and CUDA context |
| Executables | Graph handles, operation kind, supported capacities |
| Buffers | Stable addresses, byte extents, types, strides, and roles |
| Resources | Page geometry, live-state and snapshot capacities |
| State operations | Initialize, restore, snapshot, and reset graph entries, each an indirect copy kernel driven by device-resident slot-index buffers |
| Execution ordering | Stream ownership and scratch-sharing constraints |
| Frontend assets | Fingerprinted tokenizer, chat template, processor configuration, generation defaults, and native output-parser selection |
| Lifetime | Strong owner reference and shutdown requirements |

Rust sees page IDs, state-slot IDs, capacities, and graph entries. Model-specific
state layout is encoded during preparation.

### Graph families

Prepare these operations:

- **Prefill:** metadata preparation, embedding, model forward, cache writes,
  state updates, and optional final-token sampling. Prompt penalty counts are
  initialized once at admission. The scheduler clips a chunk at a required
  instruction or prompt endpoint. GDN writes its final live recurrent state at
  that exact length; internal aligned checkpoint export is disabled.
- **Decode:** gather pending token from device request state, forward, sample,
  update request state including the penalty histogram and PLE n-gram history,
  retain the final hidden vector, and write completion records.
- **Resume sampling:** vocabulary projection and sampling from a checkpoint's
  saved final hidden vector.
- **Vision:** encode preprocessed patches and write fixed-capacity embedding
  storage.
- **State maintenance:** initialize or restore a live slot, or freeze its exact
  consumed state into an immutable snapshot. The model input schema
  `qwen4exp.graph-inputs.v2` carries the expected consumed count and physical-slot
  target/draft page tables for snapshot and restore.

Padding changes capacity, never logical sequence length. Every padded row must
be excluded from KV writes, convolution updates, recurrent updates, selector
updates, and sampling.

Vision graphs use bounded patch capacities declared at startup. The first
profile admits up to 16,384 patches per image, with capacity buckets of 256,
1,024, 4,096, and 16,384. Oversized processed images fail admission; the server
does not silently change processor settings.

### Native replay loop

Maintain persistent device storage indexed by request slot:

- Pending input token.
- Consumed-token count and position state.
- Generation count and completion flag.
- Page tables and state-slot references.
- Sampling parameters and RNG state.
- Penalty histogram over the vocabulary, about 1 MiB per slot.
- PLE n-gram history: the last two consumed tokens, gathered into packed rows
  each step because the b12x hash operator reads and never mutates it.

Rust uploads prompt chunks and scheduling changes through reusable pinned
buffers. Page tables remain resident; decode updates only changed entries.

The decode graph writes the sampled token back into the request's device state.
The following decode consumes it directly.

Use two completion frames and permit at most two decode steps in flight. A copy
stream transfers compact token/status records after their production event. A
frame is reused only after its transfer completes. Input staging buffers are
double-buffered so that a frame's uploads never overwrite a pending transfer.

A grammar-constrained request needs a vocabulary mask for step k+1 derived from
the token sampled at step k. While any constrained request is in the decode
batch, the loop runs one step in flight: the Rust-owned host grammar engine
advances on the transferred token, writes each constrained lane's mask into the
staging buffer, and the sampling operation applies the mask before filtering.
Unconstrained batches keep two steps in flight. Grammar compilation and
advancement do not call Python.

Device completion flags suppress further mutation for finished requests. Slot
retirement waits for every submitted operation that can reference the slot.

Prefill and vision initially execute on the same model stream as decode. This
allows b12x scratch and CSF buffers to remain shared safely. QSA's optional
selection stream and its readiness event are bound before capture; the captured
fork and join execute on replay.

### Sampling and RNG

The graph boundary includes sampling.

Temperature, top-k, top-p, min-p and penalty controls use one captured
probability-sampling path built from FlashInfer primitives. Captured operations
manage penalties, grammar masks and request-state updates. Temperature zero
constructs a point mass at the first maximum logit and uses the same sampler.
There is no separate greedy graph or runtime route.

Execution ABIs 2 and 3 retain `sampling_mode="general"` as their sole supported
manifest value. With batch capacities 1, 2 and 4, the ordinary serving catalog
contains `19 + len(prefill_capacities)` graphs: 28 with the 4,096-token default,
or 25 with a 512-token limit. The speculative catalog replaces three ordinary
decode graphs with 14 verification shapes, for 39 graphs at the default prefill
capacity. Bounded top-k optimization is deferred. Historical component
measurements of the general path were 120–170 microseconds per step; each
prepared implementation requires its own measurement. Threshold ties must never
be truncated to a fixed number of candidates.

Repetition, presence, and frequency penalties read a per-slot vocabulary
histogram. A captured operation builds it from the full prompt token buffer at
admission exactly once, including on a prefix hit. Prefill does not increment
those prompt counts again. Each sampled output token increments its count once
after sampling, including samples from prefill and resume-sampling graphs;
consuming that pending token on the next decode does not increment it again.
The histogram is request state outside every checkpoint.

Keep explicit per-request seeds and counters in server-owned device state.
Pass device seed and offset tensors to FlashInfer's sampler. Each request uses
a one-row draw with Philox subsequence zero and word offset `4 * counter`, so
batch packing cannot change its random stream. Counters advance once for each
successful live sample, including temperature-zero samples, and remain unchanged for
inactive, invalid or terminal requests.

Do not capture PyTorch generator-backed random operations in graphs launched
directly by Rust. PyTorch's own replay wrapper performs RNG bookkeeping before
launching its executable; a raw launch bypasses that bookkeeping.
[PyTorch replay implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGraph.cpp)

Generation defaults come from the checkpoint, with explicit request parameters
taking precedence. For this snapshot, the default stochastic configuration is
temperature 1.0, top-k 20, and top-p 0.95.

### Scheduling and output

Use a small scheduler:

- FIFO admission subject to resource reservations.
- A decode batch for runnable requests.
- At most one bounded prefill chunk between decode batches.
- Cancellation and completion processed at submission boundaries.
- Eviction of unused cached checkpoints before admitting additional work.

Reserve the private pages required for admitted prompt and output capacity.
Running requests therefore do not need emergency eviction of their own state.
Four requests with a 32,768-token output limit reserve about 3.2 GiB of output
pages at 24.75 KiB per token.

Rust token decoding and HTTP/SSE streaming consume native events asynchronously.
Bounded per-request output storage prevents an unresponsive consumer from
blocking unrelated requests. Native health reporting reflects worker readiness
and failure state without querying Python.

EOS and output-length enforcement belong to helios/device request state.
Token-sequence stopping also belongs to the core; presentation-level text
trimming must not mutate cached model history.

## 4. Prefix checkpoints and complete model state

### Immutable instruction, prompt, and terminal cache

The cache retains immutable, content-addressed checkpoints at the rendered
instruction boundary, final prompt boundary, and terminal consumed boundary.
Equal endpoints share one checkpoint. Parent-linked complete-page chains,
ancestor pinning, and eviction preserve ownership across active branches.

The [Rust prefix cache](../helios/src/cache.rs) manages ownership, readiness,
pinning, and eviction. Python allocates backing storage during preparation;
[captured maintenance operations](../flare/recipes/qwen38/maintenance.py)
reset, restore, and snapshot its contents.

Use one logical page allocation to identify the corresponding storage across all
sparse-attention layers:

- Main K pages.
- Main V pages.
- Compressed selector-key pages.

For 64 original tokens, QSA's compression ratio gives 16 selector rows. Live
raw selector rings are request-local; snapshots preserve the committed ring
entries needed to continue an incomplete four-token selector group.

### A checkpoint represents one exact consumed prefix

A Qwen checkpoint includes:

| State | Purpose |
|---|---|
| Immutable main KV pages | Attention history |
| Compressed selector-key pages | Sparse-attention selection history |
| Private partial main KV and selector pages | Exact endpoints within a page |
| Raw selector keys, logical tags, RoPE tags, and interval anchor | Open selector-group continuation |
| GDN recurrent matrices | Recurrent attention continuation |
| GDN convolution histories | Causal projection continuation |
| PLE convolution history | Dilated lexical-state continuation |
| PLE n-gram token history | Correct subsequent embedding lookup |
| Position continuation metadata | Text and multimodal RoPE continuity |
| Final hidden vector | Sampling after a complete prompt cache hit |
| Pre-mixer residual and its position, when speculation is prepared | Target feedback for learned MTP |
| Shifted draft pages and raw selector state, when learned MTP is prepared | Continuation at independently tracked draft length |

The PLE table and model weights are shared immutable model resources, not copied
into checkpoints.

For target consumed count `S`, snapshot filtering keeps raw entries with logical
positions in `[0,S)` and sets the restored interval anchor to `S−1`. Initialization
has zero consumed tokens, no valid raw entries, and anchor `−1`. Restore starts
with one accepted input; rejected speculative entries cannot become visible
through raw-ring tags.

A speculative target round writes eight recurrent checkpoint columns per request.
Snapshot selects column `previous_accepted−1`, copies the corresponding three-token
GDN and nine-token PLE convolution windows, and zeroes the unused speculative
history tails. Restore installs this normalized state in the live column and
sets its accepted count to one. PLE token history and hidden/residual vectors
already describe the committed target endpoint.

Learned MTP has consumed `max(S−1,0)` feedback pairs. Its restored raw anchor is
`max(S−1,0)−1`; its validity mask uses the draft count, independently of target
raw state. Draft logical page `j` uses physical allocation `j+1`. Write
`S=64N+r`: save draft page `N−1` when `N>0`, and page `N` when `r≥2`. Thus up to
two draft pages need private restoration even though the target saves at most
one partial page. `S=1` has no valid draft page or raw entry.

Distinguish:

- **Consumed tokens:** incorporated into every model state.
- **Pending token:** sampled but not yet consumed.
- **Visible output:** tokens exposed after presentation rules.

Only consumed history can define a checkpoint.

### Exact endpoint capture

The model-owned Qwen boundary helper renders leading system/developer instructions
with an empty user continuation, then requires the resulting token IDs to be an
exact prefix of the admitted prompt. Tools and reasoning defaults can synthesize
instructions without an explicit system-role message. Shared template framing is
part of the endpoint; a user header alone is not an instruction checkpoint. The
boundary is not rounded to a token group or KV page.

Admission reserves destinations for all distinct required instruction, prompt,
and terminal roles before model work. Completed cache hits satisfy already-covered
roles. Resource pressure triggers normal cache eviction or admission rejection;
it cannot silently omit a required checkpoint.

At each required prefill endpoint, the scheduler drains submitted work, snapshots
the exact live state, and publishes only after the copy fence completes. A newly
published endpoint pins the request's shared parent and transfers its completed
private pages into that chain without changing physical page order. Further
prefill or generation resumes afterward. Terminal capture uses the final consumed
count and excludes a sampled pending token. No rolling aligned candidate rows,
internal GDN checkpoint exports, or boundary-label fields participate in this path.

The scheduler and prefix cache use the model's declared physical token capacity
per page. Qwen uses 64 tokens; GLM uses 256. Prefix identity uses fixed 64-token
hash blocks independently of physical storage. CPU lifecycle tests cover both
page geometries, exact system/prompt/terminal checkpoints, and branching restores
with private partial pages.

Complete target pages are immutable and shared. When `S % page_tokens != 0`, snapshot
storage owns the whole physical tail page and restore copies it into a private
allocation; at an aligned endpoint, maintenance never writes a shared full page.
The copied unused KV rows and incomplete selector-group values do not define
committed state. Valid token ranges, completed selector groups, and raw tags do.

The model state pool uses row zero for the null state, rows1–4 for live requests,
and row `5 + snapshot_id` for immutable snapshots. Speculative recurrent columns
follow the immutable range. Metadata validation checks the exact source count,
epoch, transfer direction, needed page mappings, and cross-row aliases before
any state mutation.

Snapshot and restore preserve committed state exactly, protect immutable sources,
and allocate no storage during replay. Restored continuation must match direct
continuation with the same token partition and graph geometry. Comparing
uninterrupted prefill with split prefill also measures GDN partition sensitivity.

### Full prompt hits

Save the final hidden vector with each checkpoint: the combined stream after the
final hyperconnection mixer. Qwen4Exp sends that vector directly to `lm_head`;
there is no separate final normalization. The resume-sampling graph applies
vocabulary projection and sampling.

When the entire prompt matches a checkpoint, restore its state and run the
resume-sampling graph. This avoids replaying an already-consumed token or
rerunning the prefix merely to recover logits.

Sampling state starts from the new request's generation parameters and seed.
Prefix reuse does not inherit the previous request's RNG stream.

### Branching and lifetime

A branch pins its ancestor checkpoint, shares immutable pages, restores state
into a private live slot, and allocates private suffix pages including all
required target/draft boundary pages. A duplicate publication releases only its
unused snapshot reservation; it does not replace computed live pages with another
request's physical page chain. Admission reuse counts remain fixed even as the
request publishes additional shared ancestors.

Track logical references separately from outstanding GPU use. A page or slot
becomes reusable only after both have ended.

Under snapshot pressure, evict unpinned leaves first. If every leaf is pinned,
remove an ancestor checkpoint with exactly one child and no direct request pin:
transfer its ordered page segment and last-use fence to that child, bypass the
ancestor in the parent chain, and retire only the ancestor snapshot. Descendant
checkpoint IDs, snapshots, and complete page chains remain valid. The removed
prefix no longer matches lookups. This bounds snapshot consumption during long
appended conversations without discarding the newest reusable prefix. Branching
ancestors and directly pinned checkpoints retain their snapshots.

Maintain resource-specific padding sentinels in the prepared metadata contract.
Padding must never address another request's writable state.

### Multimodal identity

Extend the chained prefix digest with conditioning identities at the point where
those inputs enter the token stream.

Include image content identity, relevant processor settings, grid/position
information, and model/recipe identity. Appending another image must not alter
the digest of an earlier unchanged prefix.

A sequence of image-placeholder token IDs alone is insufficient to identify
model state.

### Memory significance

Per-slot state for this model:

| State | Size per slot |
|---|---:|
| FP32 GDN recurrent matrices, 36 layers | 108 MiB |
| GDN convolution histories, 36 layers × 10,240 channels × 3 | 2.1 MiB |
| PLE convolution history, 10,240 channels × 9 | 0.2 MiB |
| PLE n-gram history, position metadata, hidden vector | under 8 KiB |
| Target raw selector state and partial pages | about 1.6 MiB |
| **Ordinary snapshot row** | **117,291,776 bytes (111.86 MiB)** |
| **Seven-draft state row** | **122,932,224 bytes (117.24 MiB)** |

The default 4 GiB checkpoint-state budget holds 36 ordinary or 34 seven-draft
snapshot rows, beyond the null/live rows and speculative recurrent-column
storage. The speculative layout includes extended convolution histories, target
feedback, and two shifted draft boundary pages. Preparation validates that the
configured snapshot budget can hold at least one row of the selected layout.

BF16 main KV across the 12 attention layers consumes **24 KiB per cached token**,
with compressed selector keys adding **0.75 KiB per token**. Main KV stays BF16
because the checkpoint declares no KV-cache quantization; QSA's FP8 E4M3
main-cache mode with per-layer descale halves the page cost and is a separately
qualified option.

Report page pressure and snapshot pressure separately. Eviction may be necessary
to recover either resource.

## 5. Implementation sequence and acceptance

### Implementation sequence

1. **Establish the model reference.** Pin the checkpoint and source revisions
   in a locked environment. The pinned Transformers quantizer registry lacks
   `modelopt`, so its canonical architecture alone cannot load this mixed-format
   checkpoint through `from_pretrained`. Qualify an explicit reference
   decoding/loading adapter that preserves the checkpoint's CSF, NVFP4, MXFP8,
   and PLE semantics, including activation quantization, or a separately
   qualified runtime for this exact checkpoint. A full-BF16 dequantization is
   not automatically an equivalent reference. Generate golden outputs and store
   exact prompts, processor inputs, generation settings, and loader/runtime
   provenance. Do not substitute
   another model or precision variant. Transformers releases differ in rotary
   composition; the selected reference must preserve the pinned architecture
   and processor semantics.
   [Pinned quantizer registry](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/quantizers/auto.py)
2. **Build the Python Qwen recipe.** Load the exact mixed-precision checkpoint,
   add prepared GDN convolution to b12x, and validate text and vision execution
   with explicit state.
3. **Complete graph execution.** Capture metadata preparation, KV writes,
   sampling, state maintenance, arbitrary prefill tails, and complete-prompt
   resume sampling.
4. **Implement native replay.** Register the bundle through PyO3, retain its
   owners, and replay from a Rust worker with explicit device RNG and completion
   frames. Measure this path to prioritize optimization of staging, scheduling,
   and cache policy.
5. **Port scheduler and checkpoint policy.** Move request lifecycle, page
   allocation, prefix lookup, pinning, reservations, and eviction into Rust;
   retain the small policy unless measurements justify more complexity.
6. **Start the native server.** Make the Python CLI prepare and hand off the
   model. Rust starts HTTP/SSE after worker readiness and owns tokenization,
   templates, image preprocessing, grammar handling, output decoding,
   cancellation, and health. Validate shutdown and owner release in the same
   process with no Python request callbacks.
7. **Qualify agent workloads.** Validate repeated turns, tool-result append,
   shared-prefix branches, image conversations, cancellation, and cache pressure.
8. **Extend the qualified core.** Add tensor parallelism and SM121 qualification;
   implement native MTP with explicit verification and state commit/rollback.
   The b12x GDN decode operator already reads the state column selected by the
   accepted-token count and writes per-token checkpoint columns. Transformers
   drops the MTP tensors on load and has no MTP module. The
   [MTP design](mtp-design.md) specifies the feedback arithmetic,
   adaptive proposal budgets, draft-history caching, verification, and commit.

The model-independent bundle contract is exercised by small captured graphs
and the complete Qwen catalog. The [Rust scheduler](../helios/src/scheduler.rs)
owns request policy; the [native executor](../helios/src/execution/transport.rs)
owns metadata submission and replay. [Python graph preparation](../flare/preparation/graphs.py)
retains captured resources until native shutdown drains all work.

### Correctness acceptance

Use the exact checkpoint throughout integration qualification.

Required cases:

- Cold prefill, chunked prefill, decode, and arbitrary prefill tails.
- Batch sizes 1–4, including padded capacity and changing slot assignments.
- Boundaries around 4-token QSA groups, 64-token pages, every captured prefill
  capacity and the configured chunk limit, including 4,096-token chunks.
- EOS, output limits, cancellation, slot reuse, and resource exhaustion.
- Prefix lengths immediately before, at, and after page boundaries.
- Full prompt hits through saved-hidden-vector sampling.
- Parent/child branches continued independently after restore.
- Identical text with different images, plus repeated identical image inputs.
- PLE EOS handling and n-gram continuity across chunks.
- QSA raw-ring and compressed-cache continuity across prefill/decode handoff.
- RNG progression across repeated replay and invariance under batch reordering.
- Shutdown while requests and asynchronous transfers are active.
- Restore into a QSA raw-state slot previously used by a request at the same
  position.
- Penalty histograms after admission from a cold prompt and after a prefix hit.
- Exact-once prompt and generated-token penalty counts across prefill, decode,
  and full-prompt resume sampling.
- Grammar-constrained decode with one step in flight batched with unconstrained
  requests.
- Native template bytes and tokenizer IDs against the locked Transformers
  reference for tools, tool results, reasoning controls, assistant continuation,
  Unicode, and image-bearing messages.
- Native image processing against the exact processor reference: resized and
  normalized pixels, patch ordering, grids, prompt expansion, and positions.
- HTTP/SSE disconnect cancellation, slow consumers, terminal-event delivery,
  worker failure health, and startup failure before socket binding.

Compare Rust execution against Python replay of the same prepared graphs and
initial state. This isolates runtime correctness from kernel numerical
differences. Require bitwise equality for state transfers and metadata. Numerical
replay comparisons account for the selected kernels' reduction-order variation;
cross-capacity cold-versus-restored comparisons also account for changes in
reduction association.

Test each component on identical operands against its Transformers or b12x
operator reference, with tolerances derived from the required precision and
reduction arithmetic. Measure repeated execution when a kernel's reduction
order can vary. Generation tests cover chat, tool use, and long contexts with
explicit sampling settings and output budgets. Complete-model logit and encoder
relative L2 remain diagnostics; they do not isolate individual operator errors.

### Graph and lifetime acceptance

After readiness:

- No Python request, preprocessing, model-step, grammar, or output callbacks.
- No per-step tensor construction or CUDA allocation.
- No kernel compilation or recapture.
- No eager path for short prompts, tails, or changing batch occupancy.
- No mutation from inactive lanes.
- No reuse of resources still referenced by queued work.
- No concurrent execution over aliased b12x scratch.
- No dependency on HTTP/SSE consumers for device token feedback.
- No caching-allocator activity across a replay window.
- No graph-private pooled output retained across graph boundaries; test
  alternating graph families with asynchronous completion transfers.
- No resource-owner release before admission stops, CUDA work drains, and all
  native workers join.

Exercise changed inputs and state assignments on repeated replays. Replaying an
unchanged warmup input is insufficient.

### Performance acceptance

Established serving baselines set the minimum performance bar. Compare the
same checkpoint, GPU, precision, prompt corpus, sampling settings, output
budgets, cache conditions, and concurrency. Flare must match or improve
end-to-end TTFT, inter-token latency, and completed-work throughput across the
required local workloads at concurrency 1, 2, and 4. Report individual workloads
and tail latency; an aggregate win does not erase a workload regression. Repeat
noisy measurements before declaring parity. Primary serving measurements use
temperature 1, top-k 20, and top-p 0.95; temperature-zero diagnostics remain
separate. Correctness and request-completion gates apply to both engines.

Measure three paths with matching model, inputs, precision, and generation
behavior:

1. A serving baseline with the same workload and checkpoint.
2. The prepared Flare graphs under a minimal replay harness.
3. The complete Rust serving runtime.

Report:

- Time to first token and inter-token latency.
- Prefill and decode throughput.
- GPU execution time versus host submission gaps.
- Native scheduling/submission latency.
- Cache-hit restoration and suffix-prefill cost.
- Snapshot traffic, memory footprint, and capture/startup cost.
- Median and tail behavior under concurrency 1, 2, and 4.
- Per-kernel GPU timelines at batch one and four, attributed to named operators
  and layers; kernel count per layer and graph; summed kernel duration against
  total graph time.

Measure GDN recurrence splits, QSA partitions, MoE reduction policy and
hyperconnection projection precision individually against the same inputs and
timeline. Keep explicit diagnostic configurations for controlled comparisons.
Fusion work is guided by measured operator time and kernel count. Host graph
launch traces alone do not establish a GPU execution breakdown.

The replay harness establishes the achievable floor for the captured computation.
The native core must sustain decode within 5% of that floor for the same graph
sequence, with frontend delivery measured separately. This host-overhead limit
is an additional requirement; efficient replay of slow graphs does not satisfy
the serving-performance floor.

The weight-traffic bandwidth bound excludes kernel latency and other execution
costs. At decode batch 1 this model reads about
6.9 GB per token:

| Weights read per decoded token | GB |
|---|---:|
| Routed experts, 10 of 512 per layer × 48 layers | 1.26 |
| GDN projections, MXFP8, 36 layers | 2.1 |
| Sparse attention and indexer projections, MXFP8, 12 layers | 0.62 |
| Shared experts, MXFP8, 48 layers | 0.24 |
| Hyperconnection low-rank projections, BF16, 48 layers | 1.26 |
| Vocabulary head, BF16 | 1.27 |
| Routers, BF16 | 0.13 |

At the target GPU's 1.79 TB/s this gives a bandwidth bound near 3.9 ms per token.
Serving performance must meet the measured baseline, and native replay must meet
the measured graph-overhead limit.

Keep raw measurements and the configurations needed to interpret them.
Correctness failures block performance qualification.
