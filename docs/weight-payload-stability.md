# Stable weight payloads

This design is not implemented. [Runtime packages](runtime-package-design.md)
describes the implemented payload packing. Track implementation in
[TODO.txt](TODO.txt).

A published package reaches end users through Hugging Face revisions.
`flare download` and `flare run` reuse a cached payload blob whenever the Hub
ETag, which is the SHA-256 in the blob's name, matches
(`flare/src/hub.rs`). A rebuilt package therefore transfers exactly the blobs
whose names changed. This design makes a weight blob's name a function of three
inputs: the checkpoint bytes, the recipe's packing, and the rank's
tensor-parallel position. Graph catalogs, kernels, capacities, concurrency,
allocation addresses, tuning workspaces, and package identifiers do not change
it. A package rebuilt for a graph or kernel change transfers only its
executable and state blobs.

The loader's read pattern does not change: contiguous multi-MiB runs read with
direct I/O into pinned staging and uploaded asynchronously
(`helios/src/runtime_package/payload.rs`).

## 1. Causes of payload renaming in the implemented packing

- **The stored unit is an allocator segment.** Export captures each VMM
  mapping of the preparation arena as one allocation, named by its position in
  address order (`ignition/preparation/runtime_package.py`). Retained
  allocations are written in reachability discovery order
  (`heliopause/src/export/capture.rs`), and rank assembly sorts their names as
  strings (`heliopause/src/export/package.rs`). A segment holds whatever
  PyTorch placed in it: several weight tensors, pointer tables, tensor maps,
  counters, rotary tables, and warm-up residue.
- **Blobs are cut greedily.** Each rank's stored data fills one blob of up to
  64 GB in that order; a rank above the limit continues into the next logical
  data file (`heliopause/src/export/protection.rs`). One changed byte renames
  the blob containing it. One changed length moves every later boundary,
  including other ranks' boundaries when a rank exceeds the limit.
- **Contents embed addresses.** Allocations come from a first-fit arena
  (`heliopause/src/allocator.rs`), so the address of every weight depends on
  the sequence of allocations before it. Pointer tables and tensor maps stored
  in the same segments hold those addresses, and kernel parameter bytes are
  replayed verbatim rather than relocated.
- **Some prepared layouts follow tuning.** b12x prepares the scale layout that
  its selected MoE kernel consumes, so a different tuning outcome changes
  stored bytes without changing the checkpoint.
- **Executables change on every export.** The random package identifier salts
  executable encryption, so cubin and graph blobs always receive different names.
  They are small relative to weights and remain outside this design.

## 2. Storage unit: the weight owner

A weight owner is a retained tensor view named by the recipe's weight objects
and recorded in the prepared-weight catalog, `BundleIndex.rebuild_weights`,
whose `RebuildTensor` entries carry the view's name, dtype, shape, strides,
packing type, and allocation reference (`ignition/preparation/rebuild_manifest.py`).
The catalog is the complete list of owners; a tensor absent from it is not a
weight for storage purposes.

An owner run is the byte extent of the owner's view inside its allocation,
after 64 KiB zero-block elision computed relative to the run's start rather
than the allocation's. A residual run is any nonzero allocation byte outside
every owner run, elided relative to the allocation start as the implemented
exporter does. Overlapping views share storage; the first owner in declared
order owns the shared bytes, and a later owner's run excludes them, splitting
into several contiguous pieces when necessary. Each piece is one stored span.

Owner runs are packed into weight blobs. Residual runs are packed into a
per-rank state blob. Nothing else changes about what is stored: disposable
storage, allocator padding, and graph pools are cleared and omitted as before.

## 3. Ordering and grouping

Owners are ordered by their catalog names, which follow the recipe's weight
object structure and are independent of addresses. The order is deterministic
across exports of the same recipe version.

Each owner carries a group key declared by the recipe through its weight
objects: a layer or top-level component, a component family such as attention
or routed experts, and a sharding class. The sharding class is one of
replicated, output-sharded, input-sharded, or tensor-parallel-specific; it
exists so that [package resharding](package-resharding.md) can place owners
that every tensor-parallel size references directly in different blobs from
owners it must transform.

Each rank's owner runs of one group form one weight blob. A group smaller than
a merge threshold joins the following group of the same rank until the union
reaches the threshold. Merging is decided from group sizes alone. With group sizes
unchanged, a change inside one group renames only the blobs that contain it;
changed group sizes can also move later merge boundaries. Blobs never cross
rank boundaries, preserving per-rank transfer and signatures.

A large mixture-of-experts model has tens to a few hundred weight blobs per
rank. The loader pins every selected blob, so `flare run` checks the process
open-file limit against the package's blob count before loading and reports a
limit that is too low instead of failing midway.

## 4. Residual and derived storage

The per-rank state blob holds residual runs: pointer tables, tensor maps,
counters, rotary tables, n-gram mailboxes, and any retained bytes without an
owner. It changes on most exports and that is expected.

Buffers whose bytes are derived from weights by a tuning-dependent choice,
such as scale layouts prepared for a selected MoE kernel, form their own group
with a derived component family. A tuning change renames those blobs only.
Recipes may instead declare the layout and restrict tuning to kernels that
consume it, which removes the dependency entirely.

Executable and frontend resources keep their implemented packing.

## 5. Index schema

`Allocation.data_spans` entries describe destination offsets and
lengths, and the allocation's payload is their concatenation at
`Allocation.file`. Each data span gains an optional source location: a
logical filename and a source offset. A span with a source location is read
from that logical file at that offset. A span without one follows the implemented
rule, sequential placement within the allocation's resource span, so every
previously exported package loads unchanged. The unit that must lie within one
physical blob becomes the span rather than the allocation.

The signed `StoredData` parts of each rank enumerate its owner runs in blob
order, so the per-rank hash covers the same bytes under owner order and
`--verify-weights` keeps its meaning. Resource bounds, overlap checks, and
classification apply per span.

Helios handles the absent source location explicitly and keeps a regression
test with an older serialized index, as required by
[package compatibility](../AGENTS.md#model-package-compatibility). The loader
reads each span from its source at its recorded offset with the same direct
I/O, staging, and asynchronous upload it uses for every package; runs remain
contiguous and multi-MiB because owners are whole tensors.

## 6. Determinism requirements on preparation

An owner's bytes must be a function of the checkpoint bytes, the recipe's
packing version, and the rank's tensor-parallel position. Preparation code
that writes owner bytes must satisfy this:

- Quantization and packing kernels used during preparation must be
  deterministic. Block absolute-maximum reductions and element-wise
  conversions satisfy this; any step whose output varies between runs must be
  fixed or its owner moved to the derived group.
- Owner bytes must not depend on capture capacities, concurrency, or
  speculative configuration. Padding driven by those settings belongs in
  scratch or state storage, not in owner runs.
- Layouts selected by tuning belong to the derived group or to a recipe
  declaration, as in section 4.
- Quantization statistics computed over a rank's shard, such as the global
  NVFP4 scale that Step 5 derives from each expert shard's absolute maximum,
  are deterministic for one tensor-parallel size and therefore stable here.
  They prevent sharing across sizes; [package resharding](package-resharding.md)
  states the domain requirement.

## 7. Diagnostics

`flare inspect --memory-map` lists each blob with the owner ranges and group
keys it contains, so a renamed blob is attributable to a group in one command.
The export report prints reused and newly written bytes by group and names the
groups whose blobs changed.

## 8. Acceptance

- Exporting the same recipe and checkpoint twice produces an identical set of
  weight blob names.
- Exporting with a different maximum concurrency, and exporting after a
  change that affects only captured graphs, produce the same weight blob set
  as the first export.
- A package exported before this design loads and serves unchanged, through a
  regression test over its serialized index.
- Loading time for a package with per-group blobs matches the single-blob
  package within measurement noise on the same hardware.
- Re-downloading a rebuilt package from a Hub revision transfers only
  executable and state blobs, measured in bytes.

## 9. Limitations

- Executable and state blobs change on every export.
- A change to a recipe's packing legitimately renames every blob of the
  affected groups.
- Sharing between recipes, or between checkpoints with different bytes, is out
  of scope.
