# Tiered prefix cache offload

Helios keeps immutable prefix checkpoints in GPU memory and evicts them under
pressure. The proposed host prefix store adds memory retention, disk retention,
and a composite of both. Disk records support staged transfers through
registered host memory and direct transfers through GPUDirect Storage.

Every tier stores the same checkpoint: immutable page chains and a complete
endpoint row across all ranks. Recipes define the bytes needed to resume;
storage moves those bytes without interpreting the model. A completed restore
publishes an ordinary GPU checkpoint, so branching, continuation, and retirement
use the existing cache semantics. Scheduler-owned transfer tickets retain
buffers independently of the requests waiting for them.

Host offload requires the implementation sequence in section 3. Derived
defaults become enabled only after correctness and the inference-latency
thresholds in section 4 pass. Preparation, captured graphs, and the runtime
package format retain their contracts. The
[core design](core-design.md#4-prefix-checkpoints-and-complete-model-state)
defines checkpoint contents and ownership; the [CLI design](cli-design.md)
defines configuration precedence.

## 1. Scope and configurations

- Published prefix checkpoints at instruction, prompt, and terminal endpoints
  are copied to the host store when they are created and restored to the GPU
  when a later request matches a checkpoint the GPU no longer holds.
- Live request state, in-flight speculative rounds, and anything not published
  as a checkpoint are never offloaded.
- Every model contract is covered through the package's runtime-sized
  allocation groups. The offload path contains no model-specific code.
- Offload work never blocks the model worker thread or the model stream.
  Section 2.10 states the isolation mechanisms and section 4 the measurements
  that gate the default.

One trait, `PrefixStore`, has three implementations, and the configuration
selects one of them or none:

| Memory size | Disk limit | Store | Behavior |
|---|---|---|---|
| nonzero | nonzero | `CompositeStore` | Records are retained in registered memory and written through to disk; disk reads warm the memory store. The default on discrete GPUs with ample host memory. |
| nonzero | zero | `MemoryStore` | Records are retained in registered memory only. |
| zero | nonzero | `DiskStore` | Records live on disk. The staged transport owns a small registered staging pool; the direct transport needs no host payload staging. The default on integrated-memory devices. |
| zero | zero | none | The worker holds no store. |

On an integrated-memory device, such as a DGX Spark, GPU memory and host
memory are one physical pool, and the GPU prefix cache already takes the
memory that `capacities` leaves after weights and live state. A retained
memory store there would duplicate bytes within the same pool, so the memory
size defaults to zero and the disk store is the only second tier. Under the
staged transport its staging pool exists because direct I/O needs a
CPU-mapped destination and the page planes are mapped only on the GPU side.
The direct transport removes payload staging; its indexes and I/O control
structures still occupy host memory.

## 2. Design

### 2.1 Checkpoint bytes

Each rank bundle declares runtime-sized allocations with a `Sizing` record in
`helios/proto/runtime_package.proto`: exactly one in the `STATE_ROWS` group and
one or more in the `KV_PAGES` group. Every `KV_PAGES` allocation on a rank is
indexed by the same physical page id, so one logical page occupies `unit_bytes`
at `base + id × unit_bytes` in each of them. These allocations are called page
planes below. The `STATE_ROWS` allocation holds one row per state slot. A GPU
checkpoint in `helios/src/cache.rs` owns its tail pages, meaning the complete
pages since its parent, and one snapshot row. The recipe stores everything else
that defines the endpoint inside the row: the partial tail page, recurrent
matrices, convolution histories, raw selector rings, sliding windows, the last
hidden vector, and the consumed count.

The complete bytes of a checkpoint are therefore:

- for each rank, for each tail page id, for each page plane, `unit_bytes` bytes
  at that id;
- for each rank, the row at the snapshot's state slot.

Neither contains physical ids. Restore and snapshot graphs take page tables and
slot ids from Rust uploads, and the row validates its own consumed count
against the expected count supplied at restore. Pages and rows can be restored
into any free ids. Qwen has 36 page planes (key, value, and selector for each
of 12 sparse-attention layers) plus draft planes; GLM has latent and index
planes per indexed-attention layer plus draft planes; DeepSeek has latent and
index planes per cache source layer. Draft planes at the first chain page hold
an unused draft page and are copied regardless. Allocations with `PEER` memory
belong to another rank and are excluded, so each byte is copied once under
tensor and context parallelism.

`CacheGeometry`, one per rank, records the row plane (name, address,
`unit_bytes`), the page planes in package order, the row count, and the page
count. `allocate_caches` in `helios/src/runtime_package/memory.rs` collects it
before consuming the sizing records, and `execution_manifest` in
`helios/src/runtime_package/model.rs` attaches it to the rank's
`ExecutionManifest` as `cache: Option<CacheGeometry>`.

A layout fingerprint is the 16-byte BLAKE2b digest of the rank count, the page
token count, and for each rank the ordered page plane names and unit sizes and
the row plane name and unit size. Together with the 32-byte model identity from
the server configuration it names the disk directory. The fingerprint is
required because model identity does not encode every row-layout parameter.
Preparing draft history can add row regions and page planes. Packages that
share a model identity can therefore have incompatible checkpoint bytes.
Unit sizes do not depend on the concurrency or verification width selected
at load time, so the fingerprint is stable across `flare run` options for
one package.

### 2.2 Store interface

The worker depends on one trait:

```rust
pub trait PrefixStore {
    fn lookup(&mut self, history: &PrefixHistory, resident: u64) -> Option<HostHit>;
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<bool>;
    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()>;
    fn poll(&mut self, events: &mut Vec<StoreEvent>);
    fn stats(&self) -> StoreStats;
    fn close(&mut self) -> Result<()>;
}
```

`store` copies a published GPU checkpoint out and returns whether it queued
the copy; `Ok(false)` means the key is already present or in flight or no
slabs could be freed, and no event follows. `fetch` copies a looked-up chain
into the ticket's page ids and state slot. Both enqueue work and return;
`poll` drains completions as `StoreEvent::Stored`, `StoreEvent::Fetched`, and
their failures. A `PublishedCheckpoint` names the GPU checkpoint, its key,
its parent's key, its bucket digest, every page of its chain from the root
with the count that belongs to the checkpoint itself, and the state slot of
its row. `MemoryStore`, `DiskStore`, and `CompositeStore` implement the
trait for serving, and a test double implements it for worker tests. The
worker holds `Option<Box<dyn PrefixStore>>`.

Three components are shared by the implementations:

- `Index`, the in-memory tree of section 2.3. Every store owns one.
- `SlabPool`, a registered pool of fixed-size page and row slabs, section 2.5.
  The memory store uses it with retention; the staged disk transport uses a
  small one as staging without retention.
- `Transfers`, the per-rank copy threads of section 2.5, constructed once and
  shared by handle.

### 2.3 Index and lookup

The index is a tree keyed by exact-endpoint `PrefixKey`, with the same shape
as `PrefixCheckpointCache`: each node has a parent key, a child count, a tail
page count, a location in its store, a transfer state, a pin count, and an
access tick. It is owned by the model worker thread, single-writer, with no
locks. A bucket map from the digest of the aligned 64-token boundary at or
below each endpoint to the node keys ending in that block supports lookup.

Lookup takes the request's `PrefixHistory` and the consumed count `a` of the
deepest ready GPU checkpoint. It walks the history's aligned boundaries from
the longest down to the block containing `a`, including the root bucket for
endpoints inside the first block, probing the bucket at each one. A candidate
node qualifies when its row exists, its consumed count `c` exceeds `a`, and
`history.key_at(c)` equals the node's key, which costs one hash of at most 64
tokens. The candidate is feasible when walking its parent links collects
every node whose consumed count exceeds `a` and stops at a parent at or below
`a`, or at the root. The result is that chain in ascending order; its tail
page counts sum to `c / page_tokens - p / page_tokens` for the stopping
parent's count `p`, and the restore skips the first `a / page_tokens - p /
page_tokens` of them, leaving exactly the pages `publish` requires. The first
feasible candidate wins. The deepest hit is usually within the last few
blocks because a new user message is short relative to the history, so the
probe count is about the new tokens divided by 64.

Page content is determined by the token prefix ending at the page, so a
suffix of a host chain's pages combines correctly with the GPU ancestor's
pages, in the same way a GPU checkpoint's pages combine with a child computed
later from a different request. A checkpoint published while its parent key
is absent from the host index is stored as a root record holding its complete
page chain, so a skipped or evicted ancestor never makes its descendants
unrestorable.

Eviction follows `PrefixCheckpointCache::evict`: remove the least recently
used unpinned leaf; when every leaf is pinned, compact the least recently used
unpinned single-child ancestor by dropping its row and keeping its pages. A
compacted node stays in the tree as page-only, no longer matches lookups, and
continues to supply chain pages. A node is evictable only when its store's
policy allows it: the memory store standalone evicts any unpinned node, the
disk store evicts any unpinned node, and the composite restricts the memory
store to durable nodes.

An outbound for a key already present or in flight is skipped. An inbound for
a key already in flight attaches the request as another waiter.

### 2.4 The three stores

`MemoryStore` owns an index, a slab pool with retention, and a transfer
handle. `store` allocates slabs, evicting under the tree policy when the pool
is full, and copies the checkpoint out; the record stays resident until
evicted. `fetch` copies resident slabs in. Lookup hits are always resident.

`DiskStore` owns an index rebuilt from its directory at startup, a file
backend, and a transport. The transport is a crate-private trait,
`DiskTransport`, with two implementations:

- `Staged` owns a staging slab pool without retention, a transfer handle, and
  the disk thread. `store` streams the checkpoint out through staging windows
  into a record file, and `fetch` streams record files through staging
  windows into the GPU. The staging pool is a detail of this transport: its
  size is derived from the package's record sizes, four row slabs plus 256
  page slabs, enough for two outbound and two inbound windows in flight, and
  it is not a setting. It also exposes `write_from(key, metadata, buffers)`
  and `read_into(key, buffers)`, which move a record between its file and
  caller-owned registered buffers, for the composite.
- `Direct` moves bytes between device memory and record files with the
  GPUDirect Storage API, section 2.6. It needs no host payload staging.

`CompositeStore` owns a `MemoryStore` and a `DiskStore` and implements the
trait by orchestrating them:

- `store` with the staged transport runs `memory.store`; when the copy-out
  lands, it runs `write_from` with the memory store's slabs for that record;
  when the write completes, it marks the memory record durable. With the
  direct transport it runs `memory.store` and `disk.store` concurrently, both
  reading the GPU, and marks the record durable when the disk write
  completes. The disk store indexes the record when the write starts, in a
  writing state, so lookups see it.
- `lookup` runs on the disk index, which is a superset of the memory index
  under write-through, including records whose writes are in flight.
- `fetch` walks the chain. A node resident in the memory store uploads from
  its slabs. With the staged transport, a node that is not resident gets slabs
  from the memory store, evicting durable unpinned residents if needed, is
  read with `read_into` into those slabs, and uploads from them; it is then a
  durable memory resident like any other. An ancestor whose endpoint shares
  its parent's last page owns no pages, so when it is not the endpoint nothing
  of it is read and an empty resident copy stands in for it. With the direct transport, a node
  that is not resident is read into the GPU directly, and the memory store
  warms through the ordinary publication path once the restored checkpoint is
  published, because the composite skips a memory copy-out only when the
  record is memory-resident, not when it is merely on disk.
- The memory store's eviction is restricted to durable records. Disk
  eviction removes the memory copy as well, so the memory contents remain a
  subset of the disk contents. A failed disk write removes the memory record
  and is counted.
- The staged transport's own staging pool is not constructed inside the
  composite; the memory store's slabs are its buffers.

### 2.5 Transfers, slabs, and record files

`Transfers` runs one thread per rank. Each thread makes its rank's context
current through `Stream::open`, owns one stream at the lowest priority and one
event, and receives jobs over a channel as page id lists, a state slot, and
slab addresses. The thread expands them into device and host addresses from
its rank's geometry, so descriptor construction never runs on the worker
thread. It issues batched copies through `cuMemcpyBatchAsync`, records its
event, created with blocking synchronization so the wait sleeps instead of
spinning a core, waits on it, and replies with the job id and rank. Each batch
call
carries at most 16 MiB and at most 4,096 entries. These bounds limit submitted
work; section 4 measures the resulting delays on shared copy engines. A
failure is reported only after the thread has synchronized its stream, so
destination ids are quiescent when the scheduler frees them. Host-observed
completion, through the event wait and the channel message, orders later
model-stream work after the copies; the model stream never waits on an
offload event.

Transfers move records in windows of at most 64 pages plus the row. A window
allocates its slabs, moves, and either retains or releases them according to
the owning pool's policy before the next window is issued. The memory store's
minimum size is two row slabs and 128 page slabs, one outbound and one inbound
window; a smaller explicit configuration fails at startup with the minimum for
the package.

`SlabPool` is one anonymous mapping (`MAP_PRIVATE`, `MAP_ANONYMOUS`,
`MAP_POPULATE`, advised for transparent huge pages), 4 KiB aligned, registered
once at startup as portable so every rank's context can use it, and split
into two sub-pools of fixed-size slabs: page slabs of the padded per-page
record size across ranks and row slabs of the padded per-row record size
across ranks, with rows taking one quarter of a retaining pool. Anonymous
memory is used rather than the driver's host allocator so that direct I/O
targets ordinary pages; the loader already uses this pattern for mapped host
regions in `helios/src/runtime_package/mod.rs`. Allocation never blocks: when
no slab is free and nothing is evictable, the requesting job fails, which
skips an outbound and fails an inbound.

A record file lives at
`<directory>/<identity prefix>-<layout fingerprint>/<consumed>-<digest>.ckpt`,
with the consumed count as 16 hexadecimal digits and the digest as 32. The
first 4 KiB is a little-endian header: magic `FLARECKP`, format version, flags,
the model identity, the layout fingerprint, the key (consumed count and
digest), the parent key, the bucket digest, the page token count, the tail
page count, the rank count, the per-rank plane sizes and row bytes, the page
segment offset, the row segment offset, the row checksum, and the creation
time. A checksum segment follows, one 64-bit xxh3 per tail page, padded to
4 KiB, so a restore that reads only a suffix of the pages verifies each page
it read. The page segment follows, page-major: page `i`, then rank `r`, then
planes in geometry order, with every plane padded to 4 KiB. The row segment
follows, rank-major, with every row padded to 4 KiB. Per-plane padding is
what lets the direct transport address each plane at an aligned file offset,
so one file format serves both transports; the cost is small because the
large planes are already multiples of 4 KiB and only small index planes pad.
The row exists when the file length covers the row segment; compaction
truncates the file at the row segment offset and changes nothing else. Writes
go to a temporary name in the same directory, use direct I/O, optionally
`fdatasync`, and rename into place. Deletion is `unlink`. Checksums are
64-bit xxh3 over each page record and over the row record to detect torn or
corrupted files; the staged transport computes and verifies them on the disk
thread, and the direct transport computes and verifies them on the GPU with a
prepared reduction over the planes, deferred until that kernel exists, so the
direct transport initially records and verifies a header-only checksum and
relies on the rename for atomicity.

The disk store's startup scan reads each file's header, validates the magic,
version, identity, fingerprint, and name, derives row presence from the size,
seeds the access order from the modification time, deletes temporary files,
and deletes any node whose parent is missing together with its descendants.
Leaf-first eviction never unlinks a parent before its children, so orphans
arise only from interrupted operations or external changes. The disk thread
owns an `io_uring` ring over files opened with `O_DIRECT`, chunks transfers at
8 MiB, and computes and verifies checksums.

Bindings added to `helios/src/cuda.rs`: `cuMemcpyBatchAsync` with the CUDA 13
signature (destinations, sources, sizes, count, attributes, attribute indices,
attribute count, stream) and one attribute whose source access order is
stream-ordered; `cuMemHostRegister_v2` with the portable flag;
`cuMemHostUnregister`; `cuEventSynchronize`; `cuStreamCreateWithPriority`;
and `cuDeviceGetAttribute` for the integrated-memory check, which the loader's
driver binding already has. The installed driver 610.57.04 exports the
batched copy, and `/opt/cuda/include/cuda.h` declares it.

### 2.6 GPUDirect Storage transport

The direct transport binds `libcufile.so` through `libloading`, the way
`helios/src/cuda.rs` binds the driver, in `helios/src/offload/cufile.rs`.
The b12x weight loader in the sibling FlashInfer checkout
(`flashinfer/experimental/b12x/loader/_gds_reader.c`) already uses the same
API on this class of host and is the reference for call patterns.

- Startup: `cuFileDriverOpen`, then `cuFileDriverGetProperties` to confirm
  the direct path. Compat mode, in which cuFile bounces through its own host
  pool, is refused for serving because it is an uncontrolled host copy; the
  store reports the reason and uses the staged transport instead. `direct`
  requires the `nvidia-fs` kernel module, a filesystem cuFile supports with
  `O_DIRECT`, and a PCIe path on which peer DMA between the NVMe devices and
  the GPUs is permitted.
- Registration: every page plane allocation and the row allocation on each
  rank is registered once with `cuFileBufRegister`, so transfers address the
  planes and rows in place with no copy. These are VMM allocations mapped
  with `cuMemMap`; cuFile support for them must be verified on the installed
  release, with `cuMemAlloc`-backed staging as the fallback if it is absent.
- Files: each record file is opened with `O_DIRECT` and registered with
  `cuFileHandleRegister` for the duration of a transfer. Plane offsets and
  sizes are 4 KiB aligned by the file format, so every I/O takes the direct
  path.
- Transfers: each rank's transfer thread submits one cuFile batch per window
  through `cuFileBatchIOSetUp`, `cuFileBatchIOSubmit`, and
  `cuFileBatchIOGetStatus`, with one entry per plane, and reports completion
  the way it reports a memcpy batch. Writes read the GPU pages directly, so
  the pinned GPU checkpoint is the only lifetime requirement. Reads land in
  the ticket's page ids and state slot directly, so the inbound completes when
  the batch completes on every rank. The stream-ordered `cuFileReadAsync` and
  `cuFileWriteAsync` variants are an alternative that records completion on
  the offload stream's event; the batch API is preferred because it reports
  per-entry failures.
- Isolation: the same bounds apply, at most 16 MiB per batch and the lowest
  stream priority, and cuFile's own worker threads are confined with the same
  scheduling priority as the disk thread. GPU memory bandwidth consumed by
  the NVMe DMA is measured by the section 4 interference measurements.
- Host configuration: the development host runs with the IOMMU off, clears
  every ACS control bit on every PCIe bridge at boot through
  `disable-acs.service`, loads `nvidia-fs` at boot, and serves `/data` as XFS
  on an `md` RAID 0 over eight PCIe 5 NVMe devices under root complexes
  separate from the GPUs. `gdscheck -p` reports the NVMe path as `nvfs`, and
  `gdsio` with compat mode forbidden completes direct transfers across that
  fabric. The transport probe repeats the same check at startup on any host
  before accepting the direct path.
- cuFile configuration: Flare supplies its own `cufile.json` through
  `CUFILE_ENV_PATH_JSON` before opening the driver, as fixed policy like the
  JIT and preload variables: compat mode disallowed, `max_direct_io_size_kb`
  raised so a 16 MiB entry is one I/O, and `max_batch_io_size` raised from
  its default of 128 entries within cuFile's limit, since one 64-page window
  of Qwen's 36 planes is 2,304 entries.

### 2.7 Scheduler integration

`helios/src/scheduler.rs` changes:

- `RequestPhase::Inbound`, which `next_step` never selects.
- Admission in two steps. `plan_admission(&spec)` validates the request,
  builds its history, performs the GPU lookup, and returns an `AdmissionPlan`
  with the history and the lookup without reserving anything. `admit_planned(
  spec, plan, inbound)` reserves and creates the request; `admit(spec)` is the
  composition with no inbound. With an inbound hit at consumed count `c` and
  `n` chain pages under GPU ancestor `A`, the request reserves pages and
  snapshots as if its prefix depth were `c`, and the scheduler creates or joins
  an inbound ticket for the key: `n` pages and one snapshot popped from the
  free lists through `ensure_resources`, a pin on `A`, the plan's history, and
  the waiting request handles. The request starts in `Inbound` with no prefix,
  `consumed_tokens` equal to `c`, and `admission_reused_tokens` equal to `c`.
- `finish_inbound(key, outcome)`. On success it publishes with parent `A`, the
  ticket pages as the tail, the ticket snapshot, and `ready_after` equal to the
  completed fence, so the checkpoint is ready immediately. A created checkpoint
  is pinned once per waiter, and each waiter receives the prefix, its page
  chain as shared pages, the snapshot, and phase `Restore`. An existing
  checkpoint returns the ticket's pages and snapshot to the free lists, which
  is safe because no graph has read them, and attaches the waiters to the
  existing id. The ticket's pin on `A` is released. On failure the ticket's
  resources return to the free lists and the waiters are returned to the
  caller for cancellation and re-admission.
- Cancelling a waiter removes it from the ticket. The ticket completes
  regardless, because the restored checkpoint is useful on its own.
- `take_publications()` returns every checkpoint that became GPU-ready since
  the previous call, each once, with its id, key, parent key, bucket digest,
  the page ids of its whole chain with the count belonging to the checkpoint,
  and the row's state slot from `snapshot_slot`. Checkpoints created by
  `publish_boundary`, `retire`, and `finish_inbound` enter the list when the
  completed fence makes them ready.
- `SchedulerStats` gains `host_lookups`, `host_hits`, `host_reused_tokens`,
  and `inbound_waiting`.
- `release_after_shutdown` drops tickets.

Ticket resources are owned by the scheduler rather than by any request, so
cancellation during inbound needs no special retirement rule, and several
requests sharing a prefix share one transfer.

### 2.8 Worker integration

`Worker` in `helios/src/server/runtime.rs` holds the optional store.
`ModelRuntime::start` constructs the configured store on the model thread
after the executor exists, from the per-rank contexts, devices, and cache
geometry in the local and peer manifests and from the pools and indexes the
loader created. Process-peer tensor parallelism rejects the option.

The worker loop order is: admit, poll and complete executor frames, publish
finished requests, drain publications into `store` calls, `poll` the store and
apply inbound and outbound results, plan, submit. Eviction happens only inside
admission, and outbound pins are taken before the next admission, so a
checkpoint cannot be evicted between publication and its copy-out. Every
store call from the worker returns without waiting.

Admission consults the store after `plan_admission`: `lookup` runs above the
GPU hit, the policy in section 2.10 decides whether to fetch, and
`admit_planned` receives the inbound request. A newly created ticket starts
the inbound job through `fetch`. On inbound failure the worker cancels each
waiter and pushes its `Incoming` to the front of the pending queue with host
lookup disabled for that attempt.

Shutdown closes the executor, then the store, which drains its streams, joins
its threads, and finishes pending disk writes within a bound, before
`release_after_shutdown`.

### 2.9 Loader integration

`NativeModel::load` in `helios/src/runtime_package/model.rs` constructs the
shared pieces after `allocate_caches`, under a progress phase named `Prepare
host prefix cache`: it parses the model identity from the package's serving
configuration, resolves the settings, maps and registers the slab pools with
rank zero's context current, opens the cuFile driver, registers buffers, and
probes the transport when the direct transport is selected, and scans the
record directory. `manifests` attaches a `Binding` to each rank's
`ExecutionManifest`; the binding shares one prepared bundle, holding every
rank's geometry and context, which the model worker takes once to construct
the store and its transfer threads.

### 2.10 Policy and isolation

Policy:

- Outbound: every publication, immediately, unless the key exists or no slabs
  can be freed for a window.
- Inbound: fetch when the estimated transfer time is below half the estimated
  prefill time of the tokens gained. Transfer time is record bytes divided by
  the tier rate; prefill time is tokens gained divided by prefill throughput.
  Initial constants are 20 GB/s for memory, 3 GB/s for staged disk, 10 GB/s
  for direct disk, and 8,000 tokens per second; the store replaces the
  transfer rates with measured values after its first transfers and the
  worker supplies measured prefill throughput from metrics.
- Eviction: the tree policy of section 2.3 under each store's evictability
  rule.

Offload limits interference through bounded work and explicit ownership:

- The model stream never waits on an offload event. Host-observed completion
  orders transfers and scheduler visibility. Shared bandwidth and copy engines
  still require the interference measurements in section 4.
- Copy engine queueing: offload batches are bounded to 16 MiB per call and the
  offload streams are created at the least stream priority. Every other Helios
  stream uses the default priority, which CUDA defines as the least priority
  as well, so priority separates offload work from model work only once the
  model and copy streams are created at a greater priority; the section 4
  measurements decide whether that change is worthwhile. Measure delays to
  input uploads, completion downloads, and Engram row copies on shared copy
  engines through per-token latency. Batch size and stream priority alone do
  not bound those delays.
- CPU: transfer threads sleep on event waits and channel receives; the disk
  thread computes checksums and submits I/O. Both use `SCHED_BATCH` with a
  positive nice value to reduce contention with the model worker, HTTP runtime,
  and frontend workers. Measure serving latency before adding CPU affinity.
- Page cache: all record I/O is direct, so disk writes neither grow the page
  cache nor evict the package file's mappings.
- Memory: every slab pool is fixed at startup and sized by the defaults in
  section 2.11. Descriptor arrays, rings, and events are preallocated.
- Backpressure: every queue is bounded. When the disk thread falls behind,
  records awaiting their write stay pinned; when a pool is exhausted by pinned
  records, copy-out is skipped and counted, never waited for. Inbound waits
  are per request and visible as the `Inbound` phase.
- Worker thread cost per publication is one job message carrying page ids;
  address expansion runs on the transfer threads.
- Direct NVMe DMA uses the same per-batch byte limit. Section 4 measures
  contention for GPU memory bandwidth and its effect on serving latency.

### 2.11 Configuration and defaults

- Memory size: `[prefix_cache] memory` in `config.toml` and
  `--prefix-cache-memory SIZE` on `flare run` set it; zero selects no memory
  store, and a nonzero size below the package's minimum fails startup. Until
  step 13 enables derived defaults, an unset size selects no memory store.
  The derived default is zero on an integrated-memory device, detected
  through `CU_DEVICE_ATTRIBUTE_INTEGRATED` on the selected devices, and
  otherwise one eighth of the host's available memory at startup, as
  `/proc/meminfo` reports it, clamped to the memory store's minimum and
  32 GiB. The attribute's value on DGX Spark hardware must be verified before
  the default is relied on there.
- Disk directory: `--prefix-cache-disk DIR` names the directory under which
  the model's record directory, `<identity prefix>-<layout fingerprint>`, is
  created; `ignition run` passes `[paths] prefix_cache` when it is set. An
  unwritable directory selects no disk store and says so on the startup line.
  Until step 13, an unset directory selects no disk store and the built-in
  value of `[paths] prefix_cache` is empty; the derived defaults are
  `~/.cache/flare/prefix-cache` for `ignition run` and, for standalone
  `flare run`, a directory beside the package file named after it with the
  `.flaretensors` extension replaced by `.prefix-cache`.
- Disk limit: one tenth of the free space on the directory's filesystem at
  startup. `[prefix_cache] disk_limit` and `--prefix-cache-disk-limit SIZE`
  override; zero selects no disk store.
- Disk transport: staged. `[prefix_cache] disk_transport` and
  `--prefix-cache-disk-transport` accept `staged` and `direct`; `direct` fails
  startup until the direct transport exists, and afterwards when the cuFile
  probe rejects the direct path. Once the probe exists, the default becomes
  direct where the probe accepts it.
- The staged transport's staging pool size is derived; it is not a setting.
- A memory store retains at most as many records as it has row slabs, and
  rows take one quarter of the pool, so on Qwen, whose rows are 112 MiB, a
  4 GiB pool holds eight records while the GPU cache holds dozens. A memory
  tier that cannot hold more records than the GPU cache rarely has a record
  the GPU has already lost; the derived default's lower clamp exists for the
  staging minimum, and the sizing guidance for operators is a pool whose row
  slabs exceed the GPU's state rows.
- `LoadOptions` gains `prefix_cache: PrefixCacheOffload` with the resolved
  memory size, directory, limit, and transport.
- The startup phase reports the selected store, transport, pool sizes, the
  directory, the limit, and the scanned record count.
- The GPU prefix cache remains derived, as the CLI design specifies. The host
  stores are derived capacities with overrides, for the reason
  `memory.reserve` exists: host memory and disk are shared with other
  software.

### 2.12 Invariants

- Only published, GPU-ready checkpoints are copied out; their pages and rows
  are immutable, so device-to-host copies need no model-stream ordering.
- A checkpoint is pinned in the GPU cache for the whole copy-out, which
  excludes both leaf eviction and ancestor compaction.
- Inbound destinations come from the scheduler's free lists, which hold only
  ids whose last GPU use has passed a completed fence.
- A node is pinned in its index while any transfer reads or writes its slabs
  or file.
- A transfer failure is reported only after the thread's stream or batch is
  drained.
- Ticket resources belong to the scheduler until `finish_inbound`.
- A record's row and pages are restored from one lineage: the host chain
  passes through the GPU ancestor's exact endpoint.
- Rows and pages are stored byte-exact; no compression or requantization.
- Record files are keyed by identity and layout fingerprint; a mismatched
  directory is never read. One file format serves both transports.
- In the composite, memory contents are a subset of disk contents, counting
  writes in flight, and memory evicts only durable records.
- Slab allocation never blocks; an unsatisfiable request fails its job.
- The model worker thread never waits for offload work, and the model stream
  never waits on an offload stream.
- Offload copies are bounded per issue, and offload threads run below the
  serving threads' priority.
- Steady-state serving allocates no device memory, no pinned memory, and no
  threads for offload; slabs, descriptor arrays, rings, events, and cuFile
  registrations are created at startup.

## 3. Implementation sequence

Each step is independently testable and leaves the build green. Steps 1
through 9 touch no scheduling behavior. Through step 12 the host store is
enabled only by explicit sizes; step 13 turns on the derived defaults after
the isolation measurements pass. Steps 15 through 17 add the direct transport
on the file format fixed in step 5.

### 3.1 Stores with the staged transport

1. **Cache geometry and layout fingerprint.** Add `CacheGeometry` and the
   fingerprint in `helios/src/offload/geometry.rs`. Collect it in
   `allocate_caches` before `sizing.take()`, store it on the rank's
   `NativeResources`, and attach it in `execution_manifest`. Add
   `cache: Option<CacheGeometry>` to `ExecutionManifest` and keep
   `same_layout` in `helios/src/execution/threaded.rs` comparing it.
   Acceptance: a unit test over a synthetic `Package` with one row allocation,
   several page planes, and a peer allocation yields the expected planes, and
   the fingerprint is unchanged across different `LoadOptions`.

2. **CUDA bindings.** Add the batched copy, host registration, unregistration,
   event synchronization, prioritized stream creation, and the device
   attribute query to `helios/src/cuda.rs`, with `Stream::copy_batch` taking
   slices of destinations, sources, and sizes and enforcing the per-call
   bounds. Add an ignored GPU test that allocates two device buffers,
   registers an anonymous mapping, round trips a scattered batch in both
   directions, and reads a file with `O_DIRECT` into the registered mapping.
   Acceptance: byte-exact round trips; the direct read succeeds into
   registered memory.

3. **Slab pool.** `helios/src/offload/slab.rs`: anonymous mapping, two
   fixed-size sub-pools, free lists, the retention flag, and separate
   `register` and `unregister` steps so CPU tests run without a driver.
   Acceptance: tests for exhaustion, reuse, alignment of every slab, release
   on completion without retention, and the minimum-size rule.

4. **Index.** `helios/src/offload/index.rs`: nodes, buckets, insert, lookup,
   pins, locations, leaf-first eviction with an evictability predicate,
   compaction, dedupe. Acceptance: CPU tests with 64, 128, and 256 token pages
   covering unaligned endpoints, feasibility through the resident ancestor, a
   chain broken by a page-only node still supplying pages, pinned and
   non-evictable nodes surviving eviction, compaction removing lookup matches
   but not chain pages, and LRU order.

5. **Record files.** `helios/src/offload/record.rs`: header encoding with
   per-plane sizes, file naming, per-plane 4 KiB padding, temporary write and
   rename, truncation, scan. Acceptance: CPU tests in a temporary directory
   for round trip, aligned plane offsets for every geometry, row presence by
   size, rejection of wrong identity or fingerprint, temporary file cleanup,
   orphan removal, and a rebuilt index equal to the one that wrote the files.

6. **Disk thread.** `helios/src/offload/disk.rs`: a thread owning an
   `io_uring` ring, with write, read, truncate, and unlink jobs over
   `O_DIRECT` files into caller-owned registered buffers, chunked at 8 MiB,
   with checksum computation and verification, running at reduced priority.
   Acceptance: CPU tests on a temporary directory that skip when the
   filesystem refuses direct I/O; a corrupted segment fails verification.

7. **Transfers.** `helios/src/offload/transfer.rs`: the per-rank threads,
   id-to-address expansion, windows, bounded batches, and the shared handle.
   Acceptance: an ignored GPU test, keyed by an environment variable naming
   any exported package with sized allocations, that loads the package, writes
   patterns into chosen page ids and a cache row, copies them to registered
   slabs, poisons the device storage, copies them back into different ids,
   and compares byte-exact.

8. **Memory store and disk store.** `helios/src/offload/memory.rs` and
   `helios/src/offload/disk_store.rs`, each implementing `PrefixStore` from
   `helios/src/offload/mod.rs` over the components above; the disk store
   behind the `DiskTransport` trait with the `Staged` implementation, its
   staging pool, `write_from`, and `read_into`. Acceptance: the step 7 GPU
   test repeated through each store's `store` and `fetch`, and CPU tests of
   each store's event sequencing with a transfer stub.

9. **Composite store.** `helios/src/offload/composite.rs`: write-through,
   read-through into the memory store, durability marking, subset
   maintenance, and failure handling. Acceptance: CPU tests with transfer and
   disk stubs covering a memory hit, a disk hit that becomes resident, a
   fetch of a record whose write is in flight, memory eviction restricted to
   durable records, and disk eviction removing the memory copy; the GPU test
   repeated through the composite.

10. **Scheduler changes.** The `Inbound` phase, two-step admission, tickets,
    `finish_inbound`, `take_publications`, and stats. Acceptance: CPU tests in
    `helios/src/scheduler/tests.rs` for: inbound requests are never scheduled;
    success attaches with the correct page chain, snapshot, consumed count,
    and `Restore` phase; a full-prompt hit resumes; an existing checkpoint
    frees the ticket; failure frees the ticket and reports waiters; two
    waiters share one ticket; cancellation during inbound leaves ticket
    resources intact; free page and snapshot counts balance across every path;
    publications report the exact tail pages and state slots.

11. **Worker, loader, options, flags, metrics.** Wire the store into `Worker`
    through the trait, the loop order, admission, requeue, and shutdown, with
    a test double covering requeue on failure. Add the loader phase,
    `LoadOptions` fields, and the `flare run` flags. Export store counters
    through `/metrics` and add one dashboard line in
    `helios/src/server/dashboard.rs`. Acceptance: Qwen TP1 with an explicit
    memory size and no disk: a conversation's checkpoints are evicted from the
    GPU by distinct prompts, its continuation reports the host hit, the reused
    token count equals the terminal endpoint, and generated tokens agree with
    the no-offload run under the retrieval check in `qualify/host_cache.py`.

12. **Disk configurations.** Directory resolution for `ignition run` and
    standalone `flare run`, the `[prefix_cache]` and `[paths] prefix_cache`
    settings in `ignition/cli/config.py`, and forwarding in
    `ignition/cli/serve.py`. Acceptance: restart the server with the same
    package and directory, replay the conversation, and observe a disk hit,
    prefill launches limited to the new tokens through `prefill_launches` in
    `qualify/server.py`, and the retrieval check, with the composite and with the
    disk store alone. A package exported with a different draft-token count
    must miss.

13. **Isolation measurement and default-on.** Run the measurements in
    section 4 with explicit sizes, baseline first. When every threshold holds,
    enable the derived defaults of section 2.11, including the
    integrated-memory check, replace the initial policy rates with measured
    values, and update `docs/core-design.md` section 4, `docs/cli-design.md`
    sections 3 and 4, and `docs/runtime-package-design.md` section 4. A
    threshold that fails is attributed to its channel in section 2.10 and
    fixed before the default changes.

14. **Parallel ranks and other models.** Qwen TP2, GLM TP4, and DeepSeek TP4
    through the same qualification scenarios and measurements, including
    DeepSeek with Engram tables in disk mode, which shares the NVMe and PCIe
    paths. Acceptance: identical results on every rank, and transfer byte
    counts matching the geometry.

### 3.2 Direct transport

15. **cuFile bindings and probe.** `helios/src/offload/cufile.rs`: driver
    open and close, properties, buffer and handle registration, and the batch
    API, loaded lazily so hosts without `libcufile.so` are unaffected. A
    startup probe reports whether the direct path is available and why not,
    refusing compat mode. Acceptance: an ignored GPU test that registers a
    `cuMemMap` allocation, writes it to a file with one batch, reads it back
    into a second allocation, and compares byte-exact; a CPU test of the
    probe's decision table over recorded property values.

16. **Direct transport.** `Direct` implementing `DiskTransport`: plane
    registration at startup, per-window batches from the transfer threads,
    per-entry failure handling, the composite's concurrent fan-out on
    `store`, and memory warming through the publication path on `fetch`.
    Acceptance: the step 8 and 9 GPU tests repeated with the direct transport;
    cuFile statistics confirm that every byte took the direct path.

17. **Direct transport measurement.** Run the section 4 measurements with
    the direct transport, including inbound latency against the staged
    transport on the same prefixes and the fabric bandwidth from each NVMe
    root complex to each GPU root complex. When the thresholds hold, the
    transport default becomes direct where the probe accepts it. Document the
    host requirements, `nvidia-fs`, ACS control cleared on the bridges, and
    an IOMMU that is off or in passthrough, in `docs/cli-design.md`.

## 4. Validation

CPU tests cover steps 3 through 6, 8 through 10, 15's decision table, and
the worker's use of the trait without a GPU. GPU tests are ignored by default
and keyed by environment variables that locate a package, following
`helios/src/runtime_package/event_tests.rs`. Each package-backed test loads
the whole package, so they run one at a time:

```sh
CUDA_VISIBLE_DEVICES=0 FLARE_OFFLOAD_TEST_PACKAGE=/path/to/package.flaretensors \
FLARE_OFFLOAD_TEST_DIR=/data/offload-tests \
cargo test --manifest-path helios/Cargo.toml --all-features -- --ignored --test-threads=1 offload::
```

`qualify/host_cache.py` launches Flare with and without a host store and runs
these scenarios, each with every store and transport that supports it:

- Memory hit after GPU eviction, with the host hit counter, reused tokens, and
  one restore and prefill launches limited to the new tokens.
- Disk hit after a server restart, with prefill launches limited to the new
  tokens.
- Branching: two continuations of one evicted prefix admitted together share
  one inbound ticket and both complete.

Every continuation asks for nine facts spread across the whole restored
context and must answer each one correctly, after GPU-cache restores and host
restores alike. Generated tokens are not compared verbatim: two GPU-cache
restores of one checkpoint diverge after about ten tokens at temperature zero
on Qwen, with or without speculation, so verbatim agreement does not
distinguish a faithful restore from an unfaithful one. The memory scenario
sizes its pool so that its row slabs outnumber the GPU's cached rows, because
the filler prompts that evict the GPU cache publish a checkpoint each.

Isolation measurements gate the defaults. Each is run without offload as the
baseline and then with the composite under the same package, flags, and GPU,
with raw numbers recorded and noisy results rerun:

| Measurement | Workload | Metric | Threshold |
|---|---|---|---|
| Decode latency under copy-out | concurrency 4, continuous short exchanges of 256-token prompts and 64-token answers for ten minutes, the maximum publication rate | per-token latency p50 and p99 from HTTP observation timestamps; tokens per second | within 2% of baseline |
| Decode latency under inbound | three requests decoding at 64K context while a fourth restores a 100K-token prefix from disk | the same | within 2% of baseline |
| Prefill throughput under copy-out | 32K-token prompts admitted back to back | prefill tokens per second | within 2% of baseline |
| Speculative round time | the first workload | round GPU time from `/metrics` | within noise |
| Inbound latency | 8K, 32K, and 100K token prefixes, through each store and transport | admission to `Restore` submission | recorded, compared with prefill time of the same prefix |
| Copy-out bandwidth | per rank, each transport | bytes per second from store counters | recorded |

A correctness failure in any scenario blocks performance claims.

## 5. Deferred

- A split operation in the GPU cache, the inverse of ancestor compaction, so a
  compacted ancestor's row can be restored onto pages its child still holds
  without copying pages.
- Page-granular matching in the GPU cache once rows are cheap to refetch.
- Rebalancing the GPU row budget in `capacities` toward pages when rows
  can be restored from the host.
- A prepared GPU checksum over record planes for the direct transport.
- Remote or shared stores, prefetch of likely continuations, and compression.
