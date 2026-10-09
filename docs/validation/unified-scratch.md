# Shared operator scratch arena

The exact-checkpoint server retains 297,959,168 bytes of operator scratch and
captures 43 graphs. Text HTTP checks pass 98/98 and four-image aggregate checks
pass 30/30. Complete-catalog Python/Rust replay passes all 3,694 checks with
413,708 byte-identical tensor observations and zero mismatches.

## Conditions and storage contract

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The server uses the default
4,096-token prefill capacity, 37 snapshot slots, and 4 GiB GPU headroom on
SM120 GPU `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`. GDN uses BF16 intermediates
with sequential recurrence and one key split; QSA uses 16 attention partitions.
Kernels are installed implementations without isolated precision overlays.

`ScratchArena.views(*shapes_and_dtypes)` returns simultaneous typed views in
nonoverlapping 256-byte-aligned regions of one contiguous uint8 allocation.
Quantized projections, GDN, QSA, routed MoE, PLE, and vision attention reuse this
storage across serialized calls, layers, and graph capacities. Scratch contents
are volatile. Weights, CSF expansion storage, live activations/outputs, KV pages,
hybrid state, request metadata, snapshots, and completion frames have separate
owners.

Preparation reserves the maximum legal candidate requirement and keeps that
extent stable during measurement. After selection and synchronization, it trims
to the maximum selected public `memory_requirements()` value, freezes storage,
and refreshes all bindings and shared aliases through `arena.finalize()` before
capture. `prime_owners(arena, ...)` retains the arena owner instead of obsolete
priming views or bindings; recipe objects retain final bindings, and benchmark
calls retain trial resources until teardown. Replay cannot grow the arena.

GDN benchmark state separately borrows the unused hybrid-state pool. Teardown
drains trial writes, restores the pool, and synchronizes before capture. Native
execution serializes arena users and joins captured side streams; resource
owners remain alive until execution drains.

## Results

| Measurement | Result |
|---|---:|
| Default-priming arena | 321,562,624 bytes |
| Candidate requirement and reservation | 2,256,535,552 bytes |
| Selected requirement and retained arena | 297,959,168 bytes (284.156 MiB) |
| Model weights | 72,667,278,336 bytes |
| Mapped-host PLE table | 28,800,138,240 bytes |
| Hybrid state; preparation borrow adds no allocation | 5,319,819,776 bytes |
| KV pages / token capacity | 8,848 / 566,272 |
| KV storage | 14,351,597,568 bytes |
| Configured headroom / reported free memory | 4,294,967,296 / 4,319,150,080 bytes |
| Reported PyTorch allocated / reserved | 95,585,619,968 / 96,489,963,520 bytes |

These are recorded allocation states, not peak-memory measurements. The
[hyperconnection/KV-budget receipt](hyperconnection-priming.md) records 8,154
pages with the same headroom: the selected-arena server holds 694 additional
pages, or 44,416 tokens. Preparation processed 14,983 requests, with 51 selection
cache hits and no candidate measurements; reservation, trim, and final binding
refresh still ran.

| Qualification | Observed result |
|---|---|
| CPU ownership contracts | 80 passed in 4.88 seconds |
| Preparation/recipe CPU suites | 355 passed, 56 CUDA skips in 56.04 seconds; overlaps the ownership suite |
| Full test collection | Failed on legacy `tests/serve/test_api.py` (`fastapi` missing) and `tests/serve/test_serving_init.py` (`b12x.integration` missing) |
| MoE graph components | Nine passed with alternating `0xA5`/`0x5A` scratch poison; source precedes PLE arena wiring and final priming-owner handling |
| Text HTTP | 98/98 checks, 31 HTTP requests, zero case errors |
| HTTP before/after comparison | All 12 request bodies and semantic responses identical to the separate-workspace server |
| Four-image aggregate HTTP | 30/30 checks, 11 HTTP requests; cold JSON, repeated JSON and SSE, four full-capacity vision launches per model request |
| Complete-catalog Python/Rust replay | 3,694/3,694 checks; 43 graphs, 124 commands / 117 groups per executor; zero mismatches |
| Replay allocation and state isolation | 234 allocation checks and 234 unrelated-state checks pass |
| Scratch ownership audit | 24,972 views and 6,188 plan-owner checks fit the frozen arena; 1,735 distinct prepared payloads |
| Server shutdown | PID2184428 exited 0; recorded sources and native binary unchanged |
| Replay qualifier shutdown | PID2212255 exited 0; sources unchanged, process absent, GPU use 2 MiB |

## Evidence and limitations

The [compact JSON summary](unified-scratch.json) links the
[full receipt](/data/sparkgap-artifacts/unified-scratch/unified-scratch.json) and
[evidence archive](/data/sparkgap-artifacts/unified-scratch/unified-scratch.tar.gz).
All 1,426 provenance checks and 1,657 archived payload hashes verify. The archive
SHA256 is `f718597c484d2979fd3ea31485416bcd96d75f852e322c621974e15bde95449e`.
The artifact store also retains the failed collector output: three checks
misinterpreted source-archive metadata membership and whitespace in sampler
mapping records. Parsed mapping fields and both metadata payloads verify
against the original raw records.

The raw server evidence is `/tmp/sparkgap-owned-scratch-server-{start,exit,memory}.json`,
with HTTP receipts `/tmp/sparkgap-owned-scratch-{http,vision}.json`.
The memory report SHA256 is
`358aa04ecb414ed3d8ae85ca966298915a03bc0398e4825eca6e34a295e7acab`;
the native extension SHA256 is
`0dcc9ca245ed83bcbf7c0ea8147640a09c3ea2f7de6a8e6bc0559826058a0dca`.
CPU logs are `/tmp/sparkgap-prime-owner-arena-tests.log` and
`/tmp/sparkgap-owned-scratch-cpu.log`. The collection failure remains in
`/tmp/sparkgap-unified-scratch-all-tests.log`; the nine component results remain
in `/tmp/sparkgap-unified-scratch-moe-cuda.log`.

The component-only MoE results use an earlier source snapshot. The complete
model replay covers final scratch ownership, allocation stability, state
isolation, and bytewise Python/Rust equality across graph families. Independent
model accuracy and serving-performance acceptance remain open.
The [capacity qualification](prefill-capacity.md),
[prior catalog replay](graph-execution-parity-prefill4096.md), and
[prior strict prefix comparison](prefix-replay-prefill4096.md) retain their own
source fingerprints and results. Strict cross-capacity prefix equality is not
a shared-arena acceptance requirement.
