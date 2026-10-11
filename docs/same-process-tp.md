# Tensor-parallel ranks in one process

Tensor-parallel serving from a runtime package runs every rank as a thread of
one Helios process: one CUDA context per GPU, one storage arena per rank at a
fixed base address, and peer buffers addressed as ordinary pointers. Package
export and restore for more than one rank are not implemented; this document
specifies the layout they use, the driver and hardware facts that make it
correct, how it differs from the rank-process layout that preparation uses,
and the measurement that supports it. The
[runtime package design](runtime-package-design.md) defines the package and
the single-rank loader; the [core design](core-design.md) defines the model
contracts and the rank coordinator that serves prepared models from Python.

## 1. The CUDA process model

A process has one virtual address space, and under unified addressing every
device allocation on every device, every mapped host allocation, and every
registered host region receives a distinct address in it. Both devices
measured in section 7 report `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING` as 1. An address
identifies memory; the device that owns the memory is a property of the
allocation, not of the pointer.

A context is a per-device container for streams, events, modules, functions,
graph executables, and `cuMemAlloc` storage. The current context is a
thread-local binding made by `cuCtxSetCurrent`; any thread can bind any
context, and the thread that created a context has no special relationship to
it. Driver entry points are thread-safe, so threads bound to different
contexts issue work concurrently. Modules and functions are per-context
handles: a kernel binary launched by every rank is loaded once per context. A
library loaded with `cuLibraryLoadData` is context-independent and
materializes its kernels in each context on first use.

A stream belongs to one context, but `cuStreamWaitEvent` accepts an event
recorded in another context or on another device, and a kernel graph node
names the context it runs in (`CUDA_KERNEL_NODE_PARAMS_v2.ctx`), so one graph
may span devices. Flare captures one graph per rank and keeps that structure.

A device fault is process-wide. `cuda.h` documents
`CUDA_ERROR_ILLEGAL_ADDRESS`, `CUDA_ERROR_LAUNCH_FAILED`, and
`CUDA_ERROR_LAUNCH_TIMEOUT` as leaving the process in an inconsistent state in
which every further CUDA call returns the same error, and requires the process
to be terminated and relaunched. The fault domain is the process whichever
device raised the fault.

## 2. Peer memory

When a kernel on device A dereferences an address backed by device B's
memory, A's memory management unit resolves it through a peer mapping and the
access crosses the PCIe fabric. The route depends only on where the two
devices sit in the fabric; `nvidia-smi topo -m` reports GPU 8 and GPU 10, and
GPU 3 and GPU 4, as pairs connected through a single PCIe bridge. Whether the address entered
the process through a local allocation or an interprocess import changes
nothing about the transfer.

Two driver mechanisms grant a device access to another device's memory:

- `cuCtxEnablePeerAccess(peer)` from the accessing context makes every
  `cuMemAlloc` allocation of the peer context, existing and future, accessible.
  The grant is unidirectional and bounded by hardware peer resources
  (`CUDA_ERROR_TOO_MANY_PEERS`).
- A virtual memory management (VMM) allocation is created on its owning device
  with `cuMemCreate`, mapped into the process range with `cuMemMap`, and made
  accessible by `cuMemSetAccess` with one `CUmemAccessDesc` per accessing
  device. The grant is per mapping and needs no context pairing. The package
  allocator in `heliopause/src/allocator.rs` and the loader in
  `helios/src/runtime_package/mod.rs` use this path.

Link attributes between GPU 8 and GPU 10, from
`~/projects/flare-research/scripts/p2p_attributes.py`:

| Attribute | Value |
|---|---:|
| `cuDeviceCanAccessPeer`, both directions | 1 |
| `CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED` | 1 |
| `CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED` | 0 |
| `CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED` | 1 |
| `CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK` | 0 |

Without native atomics over the link, cross-GPU synchronization is built from
system-scope stores into a peer's memory and system-scope loads from the
rank's own memory. The b12x PCIe barrier in
`flashinfer/experimental/b12x/comm/pcie/_cute_intrinsics.py` issues
`fence.sc.sys`, increments a counter in the rank's own signal buffer, stores it
into each peer's signal slot with `st.relaxed.sys.global.u32`, and spins on
`ld.relaxed.sys.global.u32` of its own slot until the peers' values arrive.
Its atomics (`atom.*.gpu`) touch only the rank's own memory. The TP2 peer-push
transport writes 32-byte lines in which 16 bytes of payload travel with their
generation word, so readiness arrives in the same PCIe transaction as the
data. Spin-waiting blocks require the whole worker grid to be resident on each
device, which b12x checks against the SM count before preparing a collective.
None of this depends on process boundaries: the kernels observe addresses.

Interprocess (IPC) handles are a naming mechanism, not a data path.
`cuIpcGetMemHandle` encodes the identity of a `cuMemAlloc` allocation so that
another process can map it with `cuIpcOpenMemHandle`; the importer receives an
address the driver chooses, and the mapping exists only in the importer's
address space. In the two-process probe under
`~/projects/flare-research/probe`, the child's import address differed on every
run. Inside one process there is nothing to import: the owner's address is
valid in every context as soon as access is granted.

## 3. The rank-process layout

Preparation runs one process per rank. `ignition/preparation/rank_group.py`
spawns ranks 1 through N-1 with `torch.multiprocessing`, connects them through
a Gloo group, and hands each a Unix socket to rank zero. Each rank needs its
own interpreter and PyTorch state, and the interpreter lock serializes Python
threads, so processes are the only workable unit for preparation.

b12x's PCIe collectives allocate their signal buffers (150,528 bytes per rank)
and payload slots with `cudaMalloc`, gather `cudaIpcGetMemHandle` handles
through the group, import the peers' handles with `cudaIpcOpenMemHandle`, and
write the resulting addresses into device-resident pointer tables of sixteen
eight-byte entries that the kernels index by rank. The TP2 peer-push transport
additionally passes peer slot addresses as kernel arguments. Every one of
these values is an import address chosen by the driver in the importing
process.

Serving a prepared model from Python keeps the processes. The coordinator in
`helios/src/execution/parallel.rs` runs on rank zero with one socket per peer.
A step sends PREPARE with the fence, graph index, grammar flag, and uploads;
every peer validates them against its own manifest and answers READY; rank
zero sends COMMIT and launches locally while each peer launches on its own
device. Each peer returns COMPLETE with its completion records, rank zero
checks that all records agree, and the fence retires. One step is outstanding
at a time, a completion deadline bounds a stranded collective, and a
supervisor over Linux pidfds terminates the whole group when a peer exits or a
drain deadline expires. Shutdown drains every rank before releasing graph
owners and IPC mappings.

## 4. The rank-thread layout

`flare run PACKAGE` is one process. The loader starts one thread per rank.
Each thread creates its context on its device with the recorded flags, binds
it once, reserves the rank's arena at the rank's recorded base, maps the
rank's allocations, loads its modules, rebuilds and instantiates its graphs,
and then owns every launch for that rank. Ranks load in parallel, so the
contents of N ranks move over N PCIe links at once, bounded by storage read
bandwidth.

**Addresses.** Each rank has its own device arena and its own host base, 1 TiB
apart from its neighbors, so the ranges are disjoint by construction inside
the one address space:

| Rank r | Device arena | Mapped host regions |
|---|---|---|
| 0 | `0x2000_0000_0000` | `0x1000_0000_0000` |
| r | `0x2000_0000_0000 + r × 0x0100_0000_0000` | `0x1000_0000_0000 + r × 0x0100_0000_0000` |

Each arena reservation is the first address-space operation its thread
performs and is larger than the driver's 64 MiB carve block, which is the
condition under which the driver honors a fixed base (section 3 of the runtime
package design). Every allocation record names its owning device and the set
of devices that access it. The loader maps each allocation under the owner's
context and grants access with one descriptor per accessing device. Model
packages record an address for every allocation, so their storage is VMM or
mapped host memory and `cuCtxEnablePeerAccess` is not involved.

**Peer pointers.** With every rank's storage at a process-wide, recorded
address, a pointer into a peer's buffer is an ordinary recorded value whether
it sits in a kernel parameter or in a device-resident pointer table. Replaying
the parameter bytes and storage contents verbatim reproduces it, exactly as for
every other pointer in the package; no relocation is needed.

**Address identity between preparation and serving.** The captured bytes hold
whatever addresses the preparation rank saw. For a peer buffer imported with
`cudaIpcOpenMemHandle`, that is an import address no loader can reproduce, so
the collective buffers need one of two treatments:

- Symmetric address spaces. Every preparation rank process reserves every
  rank's arena range at the shared bases; reservations cost no physical
  memory. Collective buffers are VMM allocations with POSIX file-descriptor
  shareable handles, exported by their owner. Each rank imports its peers'
  handles with `cuMemImportFromShareableHandle`, maps each at the owner's
  recorded address inside the owner's arena range, and grants its own device
  access. Every rank process then holds every collective buffer at the address
  the rebuilt process will use, and the captured bytes need no translation.
  This replaces b12x's `cudaMalloc` plus legacy IPC path with allocation from
  the package allocator and handle export.
- Relocation. The exporter records, for each imported address, the owning
  rank's allocation and offset; declares the pointer-table entries and the
  kernel-parameter fields that hold peer addresses as pointer relocations; and
  the loader rewrites them. The package format accepts these relocations, but
  identifying every parameter field that carries a peer address requires b12x
  to report them.

Symmetric address spaces are the chosen treatment: identical bytes in
preparation and serving, no inspection of parameter blobs, and the same
storage path as every other allocation. Mapping an imported handle at a fixed
address in a second process has not been measured; `cuMemMap` takes the
address, and fixed-base reservations above the carve block are honored.

**Host regions.** Each rank's mapped host tables are registered by the rank's
own context; the Qwen PLE table is sharded by rank, so no rank reads another's
copy. A host region that several ranks' kernels read must be registered with
`CU_MEMHOSTREGISTER_PORTABLE`; the loader registers with the device-map flag
only.

**Per-context costs.** Module loading and the sm_120 TMA stub's device
footprint of about 3.6 GiB apply per context, which is per rank in both
layouts.

## 5. Host coordination

The scheduler thread takes rank zero's role and rank threads replace peer
processes. For a step, the scheduler validates the uploads once against the
shared graph input schema, writes them into each rank's pinned input frame,
and signals each rank thread; each thread launches its graph on its stream and
records its production event; each rank's copy stream returns the completion
row; the scheduler compares the N records in memory and retires the fence when
they agree. Channels replace the socket protocol and its wire encoding. The
one-outstanding-step rule and the completion deadline remain. There is no
supervisor, parent-death signal, or socket poll.

One thread can drive every context by switching its binding, and the probe
below runs both ways; one thread per rank keeps each rank's launch latency
independent of N and lets each rank poll its own completion without a central
loop.

## 6. Faults, shutdown, and imaging

A device fault ends the process in either layout. With rank processes the
supervisor must kill the group without running CUDA destructors; with rank
threads the process exits. A stranded collective, where one rank's graph never
launches and the others spin at the barrier, is caught by the completion
deadline in both layouts. Shutdown stops admission, drains each rank's
streams, releases graph executables, and destroys the contexts.

A process image for validation (section 7 of the runtime package design)
requires every rank thread to be parked outside the driver during the
checkpoint window, since any thread touching GPU-side driver state during that
window corrupts the restore. No cuda-checkpoint job file is needed, because
the legacy IPC handles that require one are gone.

## 7. Measurement: two GPUs in one process

Conditions. `cr-probe p2p` from `~/projects/flare-research/probe`, driven by
`scripts/cr-single.sh` with `SUBCOMMAND=p2p`, `GPU` set to the UUIDs of GPU 8
and GPU 10, and `MODE=plugin` or `MODE=toggle`, once without and once with
`PROBE_ARGS=--threads`; results under `results/p2p-plugin-084920`,
`results/p2p-toggle-084927`, `results/p2p-threads-plugin-114431`, and
`results/p2p-threads-toggle-114435`. One process retains the primary context
of each device and enables legacy peer access in both directions. For each
rank it reserves 64 GiB at `0x7000_0000_0000` (rank 0) or `0x7100_0000_0000`
(rank 1) as the first address-space operation of that context, both honored;
maps a 16 MiB VMM buffer and a 32 KiB partial-sum buffer owned by the rank's
device with access descriptors for both devices; allocates a 16 MiB
`cuMemAlloc` buffer reachable through peer access; loads the probe kernels as
a library in that context; and creates a non-blocking stream and two timing
events. Each rank's graph is a stream capture of four `fill_pattern` launches
(1,024 blocks of 256 threads, 16 MiB each) into its own VMM buffer, its own
`cuMemAlloc` buffer, the other rank's VMM buffer, and the other rank's
`cuMemAlloc` buffer, each with a distinct seed. Twenty serialized rounds
launch rank 0's graph and then rank 1's, each timed with events and
synchronized, from one host thread that rebinds the context. Each of the four
buffers is then checksummed with `checksum_region` (4,096 blocks of 256
threads, order-independent 64-bit sum) once from the owning context and once
from the peer context, and the first 64 words of rank 0's VMM buffer are
compared against rank 1's seed, since rank 1's graph ran last. The process
then writes `ready`, is checkpointed and restored externally, reads `go`,
synchronizes both contexts, repeats the rounds and checksums, and finally
captures a new graph on rank 0 that fills a fresh `cuMemAlloc` buffer
allocated in rank 1's context, verifying its first 256 words.

With `--threads`, each rank also has its own thread. The thread binds the
rank's context once, launches the rank's graph twenty times concurrently with
the other rank's thread before the serialized rounds, blocks on a channel
receive through the checkpoint window, and after the restore launches twenty
more times without rebinding its context before the serialized rounds repeat.

Result. All four flows pass. In every run the VMM buffers sit at
`0x7000_0000_0000` and `0x7100_0000_0000`, the device footprint is 710 MiB,
the image is 1.5 GB, all eight checksums agree between the owning context and
the peer context, and all agree before and after the restore.

| Quantity | plugin | toggle | plugin, rank threads | toggle, rank threads |
|---|---:|---:|---:|---:|
| Serialized replay, median before → after | 0.6697 → 0.6708 ms | 0.6698 → 0.6700 ms | 0.6701 → 0.6695 ms | 0.6699 → 0.6699 ms |
| Concurrent replay per rank thread, before → after | | | 0.6796, 0.6796 → 0.6791, 0.6797 ms | 0.6796, 0.6796 → 0.6796, 0.6796 ms |
| First driver call after restore | 0.035 ms | 0.025 ms | 0.026 ms | 0.023 ms |
| New cross-device graph after restore | passes | passes | passes | passes |
| Checkpoint stages | dump 2.0 s, restore 2.7 s | lock 1.36, device-to-host 1.60, criu dump 0.17, criu restore 0.23, host-to-device 2.35, unlock 1.18 s | dump 1.08 s, restore 1.04 s | lock 0.47, device-to-host 0.63, criu dump 0.20, criu restore 0.16, host-to-device 1.09, unlock 0.35 s |

Reading a buffer from the peer context over PCIe yields the same checksum as
reading it from its owner, and a graph whose baked pointers cross the GPUs in
both directions replays unchanged in the same process and after an external
checkpoint and restore. Two rank threads launching concurrently, with both
graphs writing both GPUs, add about 1.5 percent to each graph's replay over
the serialized rounds. The threads' context bindings survive the checkpoint
and restore: the second concurrent pass launches without rebinding and reports
no error. Threads blocked on a channel receive during the window satisfy the
thread rule. No IPC handle, job file, or second process is involved.

The probe uses two GPUs and runs no spin-barrier collective. Four ranks, b12x
collectives inside the graphs, and fixed-address import of a shareable handle
in a second process are not covered by it.

## 8. Requirements on the package and loader

- Allocation records carry the owning device and the accessing devices;
  reservation records are per rank; the loader grants access with one
  descriptor per accessing device. The package allocator serves device zero
  only and must become one arena per rank.
- The loader creates one context per rank in its own thread, loads modules
  and instantiates graphs per context, and rebases one execution manifest per
  rank onto the live addresses.
- The export covers every rank's graphs and storage from the rank processes,
  with collective buffers at symmetric addresses.
- The server's rank coordination moves from sockets to in-process channels;
  validation, agreement, the outstanding-step rule, and the deadline keep
  their semantics.
- The TP1 package contract gains a rank count, and `flare` checks the
  architecture, address ranges, and free memory of every selected device.
