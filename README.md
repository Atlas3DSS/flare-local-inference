# Sparkgap

Sparkgap prepares Transformers models with FlashInfer kernels, including b12x,
in Python, then transfers captured CUDA graphs and their resource owners to a
Rust serving core. The execution contract targets NVIDIA SM120/121. See [the core design](docs/core-design.md)
for model, cache, runtime, and acceptance requirements.

The preparation path validates and loads the pinned Qwen3.8 checkpoint,
constructs its canonical module tree, and captures text, vision, sampling, and
state-maintenance graphs. Rust owns HTTP/SSE, tokenization, chat templates, image
preprocessing, grammar constraints, prefix caching, and request scheduling.

| Acceptance area | State | Evidence |
|---|---|---|
| Text numerics | All 48 layers traced; first cross-stack difference at HC normalization, reference variation at MoE; generation agreement pending | [Layer comparison](docs/validation/text-layer-comparison.md) |
| Vision numerics | Serving-path discrepancies require qualification | [Encoder comparisons](docs/validation/vision-serving-reference.md) |
| Runtime parity | Ordinary and copy-state executor parity pass; independent width-one comparison remains open | [Acceptance coverage](docs/validation/acceptance-gaps.md) |
| HTTP suites | Core text/image suites pass; grammar and tool API gaps remain | [Validation index](docs/validation/README.md) |
| Performance | Replay and both serving configurations measured; decode trails vLLM, prefix-hit latency is lower; numerical acceptance remains open | [Serving measurements](docs/validation/serving-performance.md) |
| Speculative execution | Copy speculation serves all supported shapes; transaction, agentic-output, and learned-MTP qualification remain open | [MTP design](docs/mtp-design.md), [agentic workload](docs/validation/speculative-agentic-serving.md) |

## Preparation environment

Install the locked Python 3.12 environment:

```sh
python scripts/setup_preparation.py
```

The setup script requires `uv`, Rust, and the sibling `../flashinfer` Git
checkout. FlashInfer is installed editable from that checkout. Setup requires
its HEAD to match `[tool.sparkgap.source-revisions].flashinfer` in `pyproject.toml`
and rejects tracked, staged, or untracked worktree changes. It prints the verified
checkout path, HEAD, and clean status before installing locked dependencies;
it never clones, checks out, or applies patches to FlashInfer.

`uv.lock` pins the remaining dependencies, including Transformers, CUDA 13 Torch,
and Cutlass DSL. FlashInfer's optional expert-parallel builds are disabled; its
required dependencies remain installed. The development environment uses the
editable main checkout at the committed revision pinned in `pyproject.toml`.
Historical patch and staged-source evidence is preserved in the
[source migration archive](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/manifest.json).

Set `CUDA_HOME` to the toolkit root when it is not `/usr/local/cuda`; this host
uses `CUDA_HOME=/opt/cuda`. GPU qualification commands require an explicit
`CUDA_VISIBLE_DEVICES` assignment and `SPARKGAP_RUN_CUDA_TESTS=1`.

Inspect a local checkpoint without initializing CUDA or reading tensor payloads:

```sh
.venv/bin/sparkgap-inspect /path/to/checkpoint/snapshot
```

Inspection checks the safetensors index and every tensor's name, shape, dtype,
and byte range against the pinned checkpoint contract. It also verifies model,
generation, tokenizer, template, processor, and shard-header fingerprints. The
JSON report separates GPU-bound storage, mapped-host PLE storage, and excluded
MTP weights. Payload values and inference correctness require separate checks.

Tensor transfer belongs to `b12x.loader.DirectWeightSession`; the manifest's
`shard_paths` and `index_path` supply its inputs. Generic Transformers
`from_pretrained` does not support this checkpoint's ModelOpt mixed format.

## Native handoff

`sparkgap.preparation.graphs.PreparedGraphBundle` retains named instantiated graphs,
persistent CUDA buffers, and additional owners such as modules, mapped memory,
and streams. Its raw `start_worker()` helper is restricted to native builds with
the `qualification` Cargo feature. Production hands the prepared model to
`Server`; Rust retains its owner until confirmed CUDA drain.

Raw graph replay bypasses PyTorch RNG bookkeeping. Capture must use explicit
device RNG state. After handoff, graph re-instantiation, storage resizing, and
concurrent external buffer mutation are forbidden. Values surviving graph
boundaries require persistent storage outside recycled graph scratch.

`sparkgap.preparation.execution.PreparedExecution` adds typed graph I/O and the
native execution manifest. Its executor uses two pinned input/completion frames,
a serialized compute stream, and an event-ordered copy stream. Grammar-constrained
work allows one submission in flight. Both streams must drain before resource
owners can be released. See [native/manifest.md](native/manifest.md) for the ABI
and [native/README.md](native/README.md) for qualification commands.

## Validation

```sh
.venv/bin/python -m pytest tests/preparation -q
cargo test --manifest-path native/Cargo.toml --features server --offline
```

Exact-checkpoint tests use the local Hugging Face cache. The checkpoint validator
also accepts `SPARKGAP_QWEN38_SNAPSHOT` for an explicitly selected snapshot path.
Missing cached weights cause those integration cases to skip.

CUDA component tests using `PreparedGraphBundle.start_worker()` require a native
extension built with `server,extension-module,qualification`. Normal package
builds omit raw replay bindings; the model's production handoff uses `Server`.

`tests/frontend/generate_fixtures.py` records rendered chat bytes, token IDs,
decoding, and template errors using the pinned Transformers revision and actual
checkpoint assets. `generate_image_fixtures.py` records exact pixel preprocessing,
image-token expansion, and multimodal positions. Native image admission supports
PNG and JPEG with EXIF orientation. JPEG decoding requires libturbojpeg >= 3.0;
see [native build requirements](native/README.md).

## Running the server

Use the locked environment and an explicitly assigned SM120/121 GPU:

```sh
CUDA_VISIBLE_DEVICES=<GPU-UUID> CUDA_HOME=/opt/cuda \
  .venv/bin/sparkgap --bind 127.0.0.1:8000 --memory-report /tmp/sparkgap-memory.json
```

The default snapshot is the pinned Qwen3.8 model in the Hugging Face cache;
`--snapshot` selects its local path. Preparation loads weights and mapped PLE
tables, primes default plans, and submits all declarations to one batched b12x
tuning pass. One `ScratchArena` supplies the maximum candidate workspace needed
by serialized dense, GDN, QSA, MoE, PLE, and vision operations. Its
`views(*shapes_and_dtypes)` method packs buffers used simultaneously into
nonoverlapping 256-byte-aligned regions; successive operators reuse the backing
storage. Preparation keeps the complete candidate extent stable during tuning,
synchronizes, and trims the arena to the largest selected-plan requirement.
It freezes that final allocation, refreshes selected bindings and aliases, and
releases trial inputs. Priming plans retain the arena owner rather than obsolete
scratch views, allowing previous allocations to be released after binding
refresh and trial cleanup. Preparation then warms and captures the graph
catalog, sizes KV storage, and freezes the execution bundle. Model state and live outputs
remain separately owned. GPU and full-model qualification is tracked in
[the scratch validation record](docs/validation/unified-scratch.md). The target
GPU defaults to eight compiler workers during the batched pass;
`B12X_COMPILE_WORKERS` overrides that count. Cancelled tuning or incomplete
selection prevents capture and startup.

`--prefill-chunk-size` defaults to 4,096 tokens. It accepts multiples of 64 from
64 through the configured context limit. Prefill capacities include 1, 16,
every power of two from 64 through the configured chunk size, and that exact
chunk size; tails use the smallest capacity that holds them. The default catalog has
43 graphs, covered by [configured-chunk HTTP checks](docs/validation/prefill-capacity.md).
Its [complete-catalog replay receipt](docs/validation/graph-execution-parity-prefill4096.md)
covers all 43 variants. `--prefill-chunk-size 512` selects the separately qualified
37-graph geometry.
Available GPU memory and kernel workspace constraints can reject an otherwise
valid chunk size during preparation. Startup fails rather than reducing the
declared capacity or selecting an execution fallback.

Searchable operators benchmark active fixtures with checkpoint weights. GDN
trials share bounded operands and borrow the hybrid state pool; preparation
restores that pool before capture.
BF16 projection backends and row tiles remain pinned, and packed MXFP8 inner
GEMM lowerings remain fixed. GDN prefill retains sequential recurrence with one
key split; QSA retains 16 attention partitions. See the
[preparation contract](docs/core-design.md#preparation-lifecycle) and
[qualification records](docs/validation/README.md#kernel-preparation).
`--prepare-only` performs preparation and releases resources without opening a
socket. The 96 GB target retains a 4 GiB immutable-state budget and 4 GiB GPU
headroom by default.

The native listener starts only after bundle validation and worker readiness.
Its routes are `/health`, `/v1/models`, `/v1/chat/completions`, and `/metrics`. Both complete
JSON and streaming SSE responses use native processing throughout. SIGINT or
SIGTERM stops admission and drains native CUDA work before Python releases the
prepared resources. See [native/model-contract.md](native/model-contract.md)
for supported request fields, image formats, and capacity limits.
