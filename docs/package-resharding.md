# Tensor-parallel resharding of published packages

This design is not implemented. [Runtime packages](runtime-package-design.md)
describes the implemented packages, in which each tensor-parallel size is a
distinct export holding a complete copy of the weights. Track implementation in
[TODO.txt](TODO.txt).

The design shares one weight set per model and hardware revision across
exported tensor-parallel sizes that meet the derivation constraints below.
The publisher continues to export and validate one package per size. A combiner
folds those packages into one fat package in which the weights of every size
other than the smallest are described as byte-level derivations of the smallest size's
blobs. An end user selects a size at download or run time, and Flare
materializes that size's weight blobs locally from whatever blobs it already
holds, in one streaming pass. After materialization, loading proceeds exactly as
for a downloaded package. Flare never interprets dtypes, packing, or kernels: every
derivation is a copy of byte rectangles between files.

The scheme works on the implemented blob layout. Keeping the shared weight set
stable across re-exports of the canonical package additionally requires
[stable weight payloads](weight-payload-stability.md).

## 1. Terms

- **Canonical size.** The smallest tensor-parallel size the publisher exports
  for a model and hardware revision. It is 1 when the model fits one GPU for
  capture, and otherwise a supported size such as 2, 3, 4, 8, or 16. The
  canonical package's weight blobs are the canonical set.
- **Derived size.** Any larger exported size. Its package references canonical
  blobs directly where bytes coincide and through derivations elsewhere.
- No size smaller than the canonical one exists, and a size the publisher did
  not export cannot be derived: its graphs and executables do not exist.
  Derivation reduces weight storage and transfer; it never creates a serving
  configuration.
- **Owner.** A retained prepared tensor view named in the prepared-weight
  catalog `BundleIndex.rebuild_weights`.
- **Piece.** The part of an owner's bytes that one rank of a given size holds.
- **Descriptor.** An affine copy or fill instruction that produces a derived
  blob's bytes from canonical blobs.

## 2. Owner provenance records

Each `RebuildTensor` gains a provenance record describing how the owner's
prepared bytes were produced from logical checkpoint tensors. Ignition emits
it where `ignition/components/weights.py` builds the strided checkpoint views
from `TensorParallelShard.partition`, `fused_partitions`, `padded_partition`,
and `replicated_partition` in `ignition/components/parallel.py`, and it
travels with the tensor to owner registration, so every recipe receives it
without recipe-specific work. The record holds:

- **Sources.** One or more logical tensors, each identified by its checkpoint
  tensor name or by a derived identity such as the DeepSeek `q_weight` and
  `k_weight` product, with a half-open range per axis in unsharded
  coordinates and the concatenation axis and order for fused owners. A rank's
  routed-expert `w13` is two ranges, up then gate, in the global order the MoE
  kernel pairs.
- **Padding.** Destination regions with no logical source and their fill:
  zero, or a constant element pattern such as a neutral scale value.
- **Transform.** The packing type already recorded as `layout_type`, the
  parameters that fix the byte layout, such as quantization format, block or
  group size, scale atom geometry, tile order, and declared row rotation, and
  the locality tile: slicing a logical axis at multiples of the tile commutes
  with the transform.
- **Statistic domain.** For quantization performed during preparation, the
  domain of its statistics: tensor, expert, or shard.

Provenance is validated when written: ranges lie within source extents, and
destination regions do not overlap. Helios ignores it, as it ignores the rest
of the rebuild metadata.

## 3. Sharing classes

Every weight consumer the recipes use reads row-major weights with the input
dimension contiguous. A rank sharded along the output dimension holds whole
rows, so concatenating the unpadded rows of adjacent pieces is byte-identical
to the canonical rows. The F8_128x4 scale layout stores 128-row atoms with the
output-dimension tile outermost, so a piece whose row count is a multiple of
128 maps to a contiguous atom range. A rank sharded along the input dimension
holds a contiguous segment of every row, which is a two-dimensional rectangle
of the canonical bytes, and its scale atoms are contiguous per output tile
when the atom count divides.

| Class | Relation of a rank's piece to canonical bytes | Mechanism |
| --- | --- | --- |
| Replicated | Identical | Direct reference from every rank |
| Output-sharded weights | Contiguous row range | Direct reference |
| Output-sharded scales, piece rows a multiple of 128 | Contiguous atom range | Direct reference |
| Input-sharded weights | Column extraction | Two-dimensional copy |
| Input-sharded scales, atoms divisible | Column extraction per tile | Two-dimensional copy |
| Round-robin head columns | Strided head blocks | Three-dimensional copy |
| Any other relation | Not affine | Stored per size |

Known cases:

- Qwen 3.8 Flash routed experts have intermediate width 640, giving 320 rows
  at size 2 and 160 at size 4. Weight rows share; scale atoms are stored per
  size above 1.
- DeepSeek V4.1 Flash routed experts have intermediate width 2304, giving
  1152 rows at size 2 and 576 at size 4. Scale atoms share at size 2 and are
  stored per size at 4.
- Step 5 quantizes BF16 experts to NVFP4 during preparation using each
  shard's absolute maximum. Those owners are stored per size until the
  statistic domain is the whole expert.
- The W4A16 packed expert layout orders 16-row input tiles outermost, so an
  output-sharded piece is a two-dimensional rectangle rather than a row range.
- Replicated owners are stored once instead of once per rank.

Kernels and captured graphs are unchanged. Every size keeps the device layout
its kernels were tuned and captured with; derivation rearranges canonical
bytes into that layout. Piece-blocked kernel addressing, where a coarser size
would consume the finer size's pieces in place, is out of scope: the dense
GEMM builds a flat input-contiguous layout inside its JIT wrapper from raw
pointers, scale atoms are ordered input-atom-fastest over the full input
extent, the weight-only, GEMV, decode, and repack paths hard-code full-row
strides, the MoE kernel takes three-dimensional strided views, and ragged
pieces such as 576 and 160 would cost tensor-core work at every coarser size.

## 4. Descriptor format

A derivation describes one derived blob by its SHA-256 name and length and an
ordered list of entries:

- **Copy.** Source blob name, source offset, destination offset, up to three
  extents, source pitches, and destination pitches. A one-dimensional copy is
  a contiguous range; a two-dimensional copy extracts or interleaves columns;
  a three-dimensional copy covers per-expert allocations and strided head
  blocks in one entry.
- **Fill.** Destination offset, length, and a pattern of one or more bytes
  repeated over the range. A zero pattern is the zero padding the loader
  already produces for omitted regions.

Owners whose pieces are contiguous ranges of canonical bytes need no
descriptor: the derived package's layout maps their logical ranges straight
to canonical blobs at an offset. Derivations are carried in a second version
of the layout file beside the existing segment mapping. The layout supplies
physical addressing only, so every signed envelope stays exactly as exported:
signed resource bounds and `StoredData` hashes apply to the reconstructed
logical bytes, `--verify-weights` authenticates derived bytes as it
authenticates downloaded ones, and a derived blob's name is checked after
materialization. One entry covers a whole owner or a whole per-expert
allocation, so a large mixture-of-experts model needs thousands of entries,
well within the 128 MiB bound on metadata tensors.

## 5. The combiner

`ignition combine` takes independently exported, signed, and validated
packages for the sizes a publisher built and produces the fat package. Like
`ignition convert`, it needs no private key and no GPU; it verifies the
signed envelopes and reads stored data through the same validation. Its
inputs must agree on recipe, checkpoint repository and revision, and hardware
revision according to their rebuild manifests, and every size must be a
multiple of the canonical size.

For each owner of a derived package, the combiner:

1. Pairs the owner with the canonical owner of the same catalog name.
2. Intersects the rank's logical region, from provenance, with the canonical
   ranks' regions. Padded partitions need not nest, so one derived owner may
   require pieces from several canonical ranks.
3. Builds the candidate relation from the transform's locality tile: a direct
   reference, a copy entry, or none.
4. Reads both payloads, including elided zero spans, and compares the
   candidate's result against the derived package's actual bytes.
5. Emits the reference or descriptor on an exact match, and otherwise stores
   the owner per size and records the cause.

The outputs are the canonical blobs unchanged, each derived size's executable
and state blobs, per-owner fallback blobs cut so that a stored-per-size owner
costs only its own bytes, one layout per size, and the fat index. The
combiner prints a sharing report by class: bytes referenced directly, bytes
produced by descriptors, and bytes stored per size, with the cause of every
fallback. `--dry-run` produces only the report. When the canonical size is 2
and a derived size is 4 or 8, the same procedure applies; nothing refers to a
size 1 layout.

## 6. The fat package

A fat package is one Hub revision per hardware revision containing an index
file, one primary per tensor-parallel size, and the shared blobs. The index
lists each primary with its rank count. `flare download REPOSITORY --tp N`
selects by rank count, `flare run REPOSITORY --gpus LIST` selects by the
number of selected GPUs, and `--filename` overrides either. A revision holding
a single primary without an index remains valid.

Canonical blobs are cut by sharding class as well as by group, so blobs that
every size references directly, replicated and output-sharded owners, are
separate from blobs that exist only as derivation sources. A host that serves
one derived size can evict the sources after materialization and keep only
the blobs its primary references.

## 7. Materialization in Flare

`flare download` and `flare run` resolve the selected primary and then each
companion it names:

- A companion present in the blob cache is hardlinked into the snapshot.
- A companion absent from the cache and listed without a derivation is
  fetched.
- A companion absent from the cache and listed with a derivation is
  materialized. Each descriptor's source range is satisfied from any local
  blob whose own descriptors cover it, including a previously materialized
  blob of another size, by inverting and composing the affine maps; ranges no
  local blob covers come from the canonical blobs, fetched on demand.

Download and materialization run as one streaming pass: each arriving source
chunk is scattered to its destinations, the source's hash is checked as it
streams, and each derived blob is hashed after assembly, compared with its
declared name, and installed in the Hub blob cache under that name so later
revisions and sizes reuse it. Assembly reads the source ranges and writes the
derived bytes; verification reads each assembled blob once more. Materialization
appears as a loading stage with byte progress. By default the canonical sources are retained for later
resharding; `--evict-sources` removes sources that the selected primary does
not reference directly.

After materialization the package is an ordinary snapshot package. Loading
reads the blobs with direct I/O into pinned staging and uploads them
asynchronously, exactly as it does for a downloaded package, and
`--verify-weights` covers the derived bytes through the signed per-rank
hashes.

## 8. Acceptance

- A combiner dry run over a small model exported at sizes 1, 2, and 4 reports
  the sharing classes with the expected fallbacks named in section 3.
- Flare's materializer, given only the canonical blobs and a derived
  primary, reproduces the independently exported package's weight blobs byte
  for byte.
- A derived size serves through the existing qualification for that model.
- One-pass download and materialization produces the same blobs as
  downloading the canonical package and materializing afterwards.
- Materializing size 2 from a local size 4 copy, without canonical blobs,
  produces the same blobs.
- Loading time for a materialized package matches the independently exported
  package within measurement noise.

## 9. Limitations

- Derived rank counts must be multiples of the canonical rank count. Other
  tensor-parallel sizes remain separate packages.
- Owners stored per size duplicate their bytes for every size that needs
  them.
- Scale atoms for Qwen 3.8 Flash experts above size 1 and for DeepSeek V4.1
  Flash experts at size 4 are stored per size.
- Owners quantized with shard-local statistics cannot be derived until their
  recipes compute statistics over tensor-parallel-invariant domains.
- Piece-blocked kernel addressing, which would remove the materialization
  pass for input-sharded owners, requires b12x changes and is future work.
