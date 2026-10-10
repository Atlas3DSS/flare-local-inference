# TP-sliceable compressed checkpoint layout

## Goal

Store compressed weights and scale indexes in a form that b12x can consume at
TP=1, TP=2, and TP=4 without rebuilding exception indexes during model loading.
Use one compressed payload across these configurations. Loading should select
partition views and transfer their byte ranges into GPU storage, without CPU
scale conversion, GPU-to-CPU round trips, or a separate prepared-scale disk cache.

## Proposed storage contract

- Divide compressed scale data into independently addressable blocks with
  block-local exception positions and indexes prepared during checkpoint export.
- Store partition directories for TP=1, TP=2, and TP=4. Directories describe
  logical extents and payload ranges; changing TP selects metadata rather than
  rewriting exception records or duplicating weight payloads.
- Align logical block boundaries with the supported TP splits. Keep row bases
  separate where column partitions would otherwise duplicate them.
- Pair corresponding intermediate-channel partitions of gate and up with the
  matching column partitions of down. Specify projection order explicitly.
- Give the storage layout an explicit version independent of library source
  revisions. Specify alignment, padding, offset origins, and exception encoding.
- Adapt b12x scale readers to consume partition views while preserving inline
  decoding and expansion into caller-owned scratch.

## Native-layout constraint

Qwen3.8 Flash Next has expert intermediate width 640. TP=2 and TP=4 produce
320- and 160-channel partitions, which do not align with 128-row native scale
slabs. A TP=4 down projection has 10 scale columns, which do not align with
four-column native scale groups.

Avoiding exception reindexing therefore does not itself eliminate packing.
Determine whether readers can compose smaller blocks efficiently into native
tiles, or whether boundary tiles require a bounded GPU packing operation.
Account for padding, metadata traffic, and the layouts used by each supported
MoE execution path before fixing the on-disk block geometry.

## Required checks

Compare decoded scale bytes and model outputs across TP=1, TP=2, and TP=4,
including exceptions on partition boundaries. Measure checkpoint size, loading
time, peak preparation memory, and complete graph replay time. Preserve lossless
scale values, graph capture, caller-owned scratch, and allocation-stable replay.
