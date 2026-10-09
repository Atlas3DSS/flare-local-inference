# Exact-checkpoint autotuned bootstrap

The isolated implementation at `/tmp/sparkgap-autotune-integration` loaded `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, prepared and captured all **37 graphs**, validated the execution catalog, and exited with code **0** using `--prepare-only`. The CUDA device was `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`. HTTP requests and native model replay were outside this run.

The [receipt](autotune-bootstrap.json) records the exact command and hashes. The [evidence archive](/data/sparkgap-artifacts/validation-receipts/autotune-bootstrap.tar.gz) contains the complete log, memory report, install receipt, 155 verified snapshot source/test/script files, 533 installed b12x Python files, the 25-file FlashInfer patch, and the native/sampler binaries. Every archived source file matches its recorded hash; installed b12x sources match the pinned staging checkout. `MANIFEST.json` gives each archive member's SHA-256 and original path.

| Preparation result | Value |
|---|---:|
| Declarations / coalesced requests | 11,350 / 255 |
| Compiler workers | 8 |
| Measured candidates / cache-hit groups | 147 / 32 |
| Declaration selections: override / fixed / cached / tuned | 3,508 / 6,093 / 1,123 / 626 |
| Batched preparation elapsed time | 28.866 seconds |
| Parent-process CuTe / Triton compilations | 0 / 0 |

Compilation and selection caches were partially populated by the preserved [missing-producer failure](autotune-bootstrap-missing-producer.json) and [unsupported-route failure](autotune-bootstrap-moe-legality.json). Cache contents were not cleared or snapshotted before startup. The elapsed time does not characterize cold startup or fully cached startup.

| Device-memory result | Bytes |
|---|---:|
| Packed checkpoint weights | 72,667,278,336 |
| Hybrid state, including 37 snapshots | 5,319,819,776 |
| Final KV pools: 9,649 pages, 617,536 tokens | 15,650,832,384 |
| Allocated / reserved after final catalog capture | 95,759,251,456 / 96,569,655,296 |
| Free after final capture / required headroom | 4,335,927,296 / 4,294,967,296 |
| Allocated before / after trial-reference cleanup | 82,105,594,880 / 80,080,905,216 |
| Allocator-visible preparation storage released | 2,024,689,664 |
| Expert pool before / after selected-binding cleanup | 188,281,136 / 137,257,776 |
| Expert-pool ownership released: two unused keys | 51,023,360 |

The mapped PLE table occupies 28,800,138,240 host bytes. Recurrence workspace grew from 67,108,864 to 105,254,144 bytes to cover every legal candidate. Expert cleanup runs after all binding finalizers, retains exact selected scratch tuples and shared aliases, and drops unused pool keys before reference collection and allocator-cache release. Pool ownership counts and allocator-visible cleanup measure different points; their reductions must not be added as one measurement.

GDN selected sequential recurrence at every capacity. Prefill-16 selected `k_split=2`; prefill-1/64/128/256/512 selected `k_split=1`. MoE selected grouped dynamic M32 at prefill-16/512, grouped dynamic M16 at prefill-64/128/256, and micro kernels at prefill-1 and decode-1/2/4. The complete selected configurations and per-declaration coverage are in `evidence/memory.json` inside the archive. These choices require numerical and prefix-equivalence qualification.

BF16 projections remain explicitly pinned, and quantized MXFP8 wrappers do not tune their inner GEMM lowering. QSA keeps the fixed 16-partition arithmetic policy. Startup success does not establish model accuracy, tuned-catalog native replay parity, inference latency, or acceptance into the canonical serving source.

FlashInfer base revision is `98fb87a461183f98d6a7049ca947fb136cc6fb59`; declared patch SHA-256 is `c1293fbb4278b4ff779d37a73188fe5eaba93f21fb27f2b1555f664e88353059`. The launch sampler SHA-256 is `9dd90584c6792335f0ad45bb6c3c254e0bec70a25cff65ff1a535751f307bc99`, covered by the separate [sampler qualification](sampling-autotune-legality.md). The install receipt's `abf5c856…` binary predates that sampler rebuild and is preserved unchanged. Native extension SHA-256 is `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`. The execution tool observed exit 0 for session 26692, PID 1254299; no shell exit-marker file was produced.

Archive SHA-256: `f7a9ffbf349405836518e1f547bd6e0f2ea21be1155af528459c93e447a20592`.
