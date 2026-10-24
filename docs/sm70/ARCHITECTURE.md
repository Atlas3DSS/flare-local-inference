# SM70 architecture

This page explains why a V100 needs its own execution path, and the four
changes in the release tree on top of the SM70 base port. Each change section
gives the problem, the change, the tests that guard it, and its review history.
Commit messages hold the detailed validation records; they are cited by hash.

## Why Volta needs its own path

V100 tensor cores execute FP16 `mma.m8n8k4` only. Volta has no BF16, FP8 or FP4
tensor-core instructions and no FP8 conversion instructions. The
FlashInfer/b12x kernels Flare uses on SM120 do not target it.

The model makes that harder. Qwen3.8-27B has 48 Gated DeltaNet
(linear-attention) layers and 16 full-attention layers (24 query heads, 4 KV
heads, head dimension 256). Its MLP and vocabulary weights are NVFP4, its
attention projections tensor-scaled FP8. The checkpoint's MTP head drafts tokens
for speculative decoding.

The SM70 base port therefore:

- holds BF16 values in FP32 tensors and rounds them to BF16 with integer
  round-to-nearest-even, so the model's BF16 rounding points are kept without
  BF16 storage (`ignition/components/sm70_precision.py`);
- runs NVFP4 weight-only and tensor-FP8 projections from packed weights on FP16
  tensor-core tiles with FP32 accumulation. NVFP4 weights are E2M1 times the
  E4M3 block scale, then the checkpoint's global scale. Tensor-FP8 keeps the E4M3
  activation rounding and applies the combined scale after the dot. FP8 and FP4
  values are exact in FP16, so the tensor cores see exact operands
  (`sm70_projection*.py`, `sm70_dense.*`, `sm70_prefill.*`);
- runs prefill attention on a tensor-core kernel, and decode and verification
  attention on the S1L kernel (section 3);
- runs the Gated DeltaNet recurrence in Triton, token by token in FP32, with
  floating-point fusion disabled (`sm70_gdn.py`, `sm70_gdn_prefill*`).

**The exactness contract.** The port requires speculative decoding to be exact:
accepting k verification rows must leave the same bytes of committed state as
k single-token greedy decodes. The `forced_prefix_*` export gates check this
([EXPORT.md](EXPORT.md#export-gates)). Every design choice below is made to keep
that contract. Each kernel fixes its reduction order, its tile and its split, so
that a row's bits do not depend on batch size, verification width or the other
requests in the batch.

## 1. Scheduler hang fix

Commits `e81534a` and `718a187` (Helios, Rust). This change is not
SM70-specific. It was found while serving SM70 at 4 agents.

**Problem.** With 4 agents, a depth-4 pool of 760 pages and the host prefix
store on, the server could stay up with requests queued and make no further
progress. The host store pins each new checkpoint until its copy to host memory
completes. A request refused for capacity was retried only after the completed
fence, or the free page, snapshot or slot count, changed. Releasing a copy-out
pin makes a cached chain evictable without changing any of those. So when the
last running request retired while copy-outs still pinned the newest chains,
every waiting request was refused. The copies then finished, and nothing ran
whose completion could reopen admission (`e81534a`).

Two more ways to lose progress were found in review (`718a187`):

- A parked readmission picked for admission could keep its readmission pin when
  it was cancelled after the cancellation scan, or rejected with a non-capacity
  error. Nothing else released that pin.
- Restoring a GPU prefix hit retains its snapshot row, whose alias pages no
  request can use. A request whose pages fit the pool, but not beside that row,
  could never be admitted with the hit.

**Change.**

- `RequestScheduler::admission_key` now counts pins released through
  `unpin_checkpoint` and `release_readmission`, and the worker blocks a refused
  request on that key. A refused request is therefore retried when such a pin is
  released. Admission, eviction, chunking and restore points are unchanged.
- `take_eligible` drops a cancelled parked request through `Parked::abandon`,
  which releases its pin. The other rejection exits abandon the parked state
  too.
- `admit_planned` admits a hit that cannot fit beside its snapshot row as a
  miss, which prefills the same chunks from the start. On the served 4-agent
  geometry (760 pages, at most 257 pages per request) this never happens.

**Tests.** `helios/src/scheduler/progress_tests.rs` reduces the worker loop to
its scheduler calls on the served geometry: 760 pages of 128 tokens, 32 snapshot
rows aliased onto the top of the pool, depth-4 draft pages and width-5
verification, with a serial host-store copy-out that pins each publication. Both
hang regression tests stall without the released-pin count and pass with it; the
840-page depth-3 pool is the control (`e81534a`). `718a187` adds a
cancellation-between-checks test and an idle test with a 260-page pool, which
fail with the respective change disabled, plus a test of both restore outcomes.
Run them with `cargo test ... progress_tests` ([VALIDATE.md](VALIDATE.md#gate-1-unit-and-regression-tests)).

**Review history.** One independent review is recorded: `e81534a` was cleared
with required changes, and `718a187` applies them (its commit message).

**Not in this tree.** Two related fixes from another fork branch are not
included: greedy readmission that re-derives delivered tokens when the preempted
checkpoint cannot be kept, and a host-store admission deadlock in a one-context
pool ([LIMITATIONS.md](LIMITATIONS.md)).

## 2. Prefill projections on pgemm

Commit `8ad725c`; kernel `ignition/components/sm70_pgemm.cuh`.

**Problem.** On one V100, real coding-agent traffic is prefill-bound: prefill
took about 85 % of busy time (STATUS.md line 37). The previous prefill projection kernel
(`atlas_prefill`, from `a6712de`) staged the weight (B) tiles through shared
memory and processed every row of a graph's capacity, including padding.

**Change.** Prefill projections call `atlas_pgemm4`, a register-B kernel:

- Weights stay in the existing packed layout. Each lane of a warp reads its
  `mma.m8n8k4` B fragment straight from global memory into registers and decodes
  it there, so there is no shared-memory B tile and no B bank conflicts. Weight
  decode is pipelined.
- Activations (FP16 rows from the producer stage) use a double-buffered shared
  tile.
- A device live-row count comes from the embedding's live count. Rows at or past
  it are written as +0.0, and their activations and inverse scales are not read.
  No consumer reads a padded row into a live row.

Bits are unchanged. Each matrix shape keeps the split, producer, workspace and
epilogue pinned for the previous kernel, so every output element is computed by
the same sequence of `mma` operations in the same order. Tile sizes change which
warp computes an element, never its operand sequence. `TILES` picks one compiled
instantiation per capacity and shape, and `library()` refuses a served id that
is not compiled. Decode, verification and resume graphs are unchanged.

Volta's lack of BF16 tensor instructions is why this kernel exists at all. On
SM120, Flare runs these projections through b12x on native FP4/FP8 tensor
cores. On SM70 the FP4 and FP8 weights must be decoded to exact FP16 in
registers for FP16 tensor cores.

**Tests.**

- `tests/components/test_sm70_build.py` checks that every served tile is a
  compiled instantiation inside the reviewed template space, and that
  `sm70_pgemm.cuh` is byte-identical to the reviewed kernel (SHA-256
  `60a4ed9f...`).
- The commit message records the GPU evidence. The served instantiations
  compiled to SASS identical to the library that passed kernel identity (6,498
  cases, 0 mismatches). Captured-graph identity against the previous package
  found 0 mismatches over 136 prefill replays, 113,896 live-row hashes and 6,358
  committed tensors, and all 7,392 padded rows were +0.0. Strict validation
  passed 47/47 and invariance 28/28. KL and top-1 were bit for bit equal to the
  previous package.

**Review history.** The kernel was reviewed by two independent reviewers over
four rounds, and the integration by the same two. One cleared it with no
required changes. The other cleared it with gate-only required changes, applied
before any GPU run (`8ad725c`). The author did not review its own change.

**Not in this tree.** A later commit on the same fork branch (`7ffb3cf`) adds
a 256-row prefill capacity with deterministic chunking. It is not in the release
tree, which runs 128-token prefill chunks.

No end-to-end measurement of this change alone is recorded.

## 3. S1L decode, verification and draft attention

Commits `6b808ef` (row contract), `0aea71a` (draft attention), `ca16f00`
(r2); kernel `ignition/components/sm70_decode_attention.cu`.

**What S1L is.** S1L is the build name of the SM70 decode and verification
attention kernel: the tensor-core kernel (`v2b`), plus S1 (append-time exact
encoding of committed K/V blocks), plus L (a partition ladder). One CTA handles
one 512-token partition, agent and KV head, with 4 warps. Each warp loads its
32 x 64 block of BF16 K (then V) into registers. It finds the block's maximum
exponent and converts BF16 to power-of-two-scaled FP16 with integer operations,
which is exact for every admitted value. Q is split into hi and lo FP16 halves.
Softmax is FP32 online, and partitions merge in a fixed per-row order. Blocks that fail admission (values spanning more than 2^30 within a block, or
non-finite values) take an FP32 SIMT path for that block only.

The served package uses raw-BF16 mode with the ladder (variant `R | 128`,
metadata `NULL`). The ladder picks 128-, 256- or 512-token partitions per row
from that row's committed context. S1's encoded mode is compiled but not used.
K/V pages stay canonical BF16, so the prefill kernel, the KV writer,
prefix-cache offload and snapshot/restore read them unchanged. Encoded and raw
modes give identical bits.

**Decode path.** The 16 full-attention layers run decode and verification on
S1L. The kernel is fixed at four agents with up to five rows each (20 rows),
which is why serving is capped at 4 concurrent requests. It admits contexts up
to 32,768 tokens at partition 512; preparation refuses a wider page table, which
is why packages are prepared with `--context 32768`. Non-finite visible K/V
produce non-finite outputs, which the samplers turn into error code 1.

**Row contract (`6b808ef`).** Verification row j of an agent with R rows must
produce the same bits as a single-row launch at visible length ctx0 + j + 1.
That is the equality the `forced_prefix_*` gate needs. The earlier kernel failed
10 of 45 strict checks at `--context 32768`. The fix converts each boundary tile
per row and uses a per-row ladder rung; an agent whose rows straddle a rung
cutoff runs its rows at two rungs in separate CTAs. Row 0 and every single-row
launch are bit-identical to the earlier kernel.

**MTP interplay (`0aea71a`).** Each speculative round runs width - 1 draft
steps of the MTP head, each attending the draft K/V planes. They ran a scalar
Triton sweep, one program per token and head with no split-K: 0.20 ms per step
at short context and 1.16 ms at about 12k tokens. `MTPAttention` now packs each
draft step's metadata (one row per agent) into the same four-agent view and calls
the same S1L component as the target layers. Admission is the target decoder's:
at most 32,768 tokens, 4 agents and 5 rows per agent. This changes the draft
tokens' bits, not the output. Verification decides every emitted token, and the
coding-token goldens were regenerated for this change
([EXPORT.md](EXPORT.md#export-gates)).

**r2 (`ca16f00`).** The kernel's `load_block` loaded K/V positions at or past the
visible length and then replaced them with a select. Every warp waited for
memory at that select, so neither the V load nor the next tile's K load
overlapped compute. r2 leaves those positions as zeroed registers through a
predicated load: the same words and the same zeros. ABI, variants, workspace,
grid and `nvcc` flags are unchanged. The commit message records 7-12 % faster
per launch on every configuration (for example 9.95 to 9.08 ms per 16-layer set
at 4 agents x 4 rows and 12k context). It also records 0 mismatches in
captured-graph identity against the previous package (696 replay events, 81,960
row hashes, 5,895 committed tensors).

With replay-commit and the S1L draft attention, the release package's GPU step
per 1 x 5 round is 49.5-49.6 ms, against 55.1 ms for the replay-commit package
without the S1L draft attention (STATUS.md line 64).

**Tests.**

- `tests/components/test_sm70_decode_attention_cuda.py` (V100) covers
  four-agent packing, captured-graph replay identity, padded rows, an FP64
  reference, and the refusal of a 34,816-token page table at preparation.
- `tests/recipes/test_qwen38_dense_mtp.py` covers the draft attention's
  admission and its four-agent packing.
- The strict gates `forced_prefix_*` and `sm70_decode_attention_nonfinite_probe`
  cover the row contract and non-finite handling.

**Review history.**

- Row-contract kernel (`6b808ef`): one review chain is recorded in the commit
  message.
- Draft-attention routing (`0aea71a`): the commit message records no review.
- r2 (`ca16f00`): the kernel and the integration were each reviewed by two
  independent reviewers. The integration was cleared with gate-only required
  changes, applied before the package run.

## 4. GDN replay-commit

Commits `d3258bc` to `7cb1bc7`; kernels `_recurrence` (REPLAY mode) and
`_replay` in `ignition/components/sm70_gdn.py`.

**Gated DeltaNet.** Each linear-attention layer keeps, per request, a recurrent
state: 48 value heads of 128 x 128 FP32, plus a convolution history. For each
token the recurrence computes `state = state * decay`, then
`delta = (value - sum(state * k)) * beta`, then `state = state + delta * k`.
One request's continuation row, covering all 48 layers, the tail pages and the
last hidden state, is 166.1 MiB.

**Problem.** Speculative verification at width w used to write one candidate
state row per verified position, so that commit could copy the row of the last
accepted position. At 4 agents and width 5 that is 1 null row + 4 x 5 rows = 21
rows x 166.1 MiB = 3.41 GiB of the cache budget, which is memory that cannot
hold KV pages. The recurrence was bound by writing those candidate states, and
the commit copy cost another 1.29 ms per round (5.51 ms of recurrence and
1.29 ms of copy per 4 x 5 round, STATUS.md line 34). At `--memory-limit 31G`
the KV pool held 98,304 tokens, less than four long prompts need.

**Change.**

- Verification no longer writes candidate states. In REPLAY mode the decode
  recurrence stores, per token and layer, the FP32 operands each step consumed:
  the normalized k per key head, the value, decay and beta. That is 33,152 bytes
  per token per layer. The committed row stays the untouched initial state.
- A new `_replay` kernel replays each request's accepted prefix from the
  committed row, for all 48 layers in one launch, after the round. It replaces
  the row copy. It uses the same `[8 x 128]` tile, `num_warps=4`,
  `num_stages=1`, `enable_fp_fusion=False`, and the same update expressions in
  the same order.
- Each request now owns one committed row. Bootstrap rows drop from 1 + 4 x 5 =
  21 to 1 + 4 = 5. The contract declares `state_rows_per_request = 1` and the
  recipe `verification_rows = 0`, so Helios plans and packs a single row with
  no Rust and no Protobuf change.
- The replay operand buffer is 30.4 MiB (20 token slots x 48 layers), zeroed on
  export.

**Why it is exact.** The replayed row equals, byte for byte, the candidate
column the old path would have copied, and so the state after k greedy decodes:

1. same kernel arithmetic: the same Triton expressions in the same order, the
   same tile and warps, and no FMA contraction;
2. the stored operands are the FP32 values verification consumed, and FP32 store
   and load are lossless;
3. the same order and start: tokens replay in position order from the committed
   state, which verification no longer modifies;
4. the same masks for fresh, null and inactive rows;
5. layer independence: each layer replays only its own operands.

**The layout lesson.** Condition 1 failed on the first V100 run, although the
source expressions were identical. The bitwise test found the replayed rows 1-2
ulp off in about 6 % of elements (`a58e89e`; STATUS.md line 39). Every operand
and elementwise operation was exact. The difference was the association order of
the 128-key row sum:

- Triton specializes each launch on its arguments' alignment. The recurrence's
  state address had a 16-divisible stride, so its loads vectorized, and the
  compiler chose the blocked layout `sizePerThread [1,4]`,
  `threadsPerWarp [1,32]`, `warpsPerCTA [4,1]`. That is a four-term
  per-thread sum, then a 32-lane butterfly.
- `_replay`'s address added a per-layer offset loaded at run time, of unknown
  alignment. It could not vectorize and got `[1,1]`, `[1,32]`, `[1,4]`: a
  32-lane butterfly, then a four-warp combine.

Different summation trees round differently. An earlier compile-only check had
compiled the kernels without the launch-time attributes; both got the scalar
layout there, and the check passed. The fix passes the per-layer offsets in
16-float blocks and multiplies by the constant inside the kernel, so the base is
provably aligned and the compiler assigns the recurrence's layout.
`PreparedSM70ReplayCommit` rejects offsets, row strides or pool bases that are
not 16-float or 64-byte aligned. No arithmetic expression changed.

The lesson for any bit-exact reduction in Triton: identical source is not
identical arithmetic. Check the layout of the program that actually runs, with
its launch specialization, not a separate compile. The V100 test now reads the
TTGIR of the programs that run (the candidate recurrence, the REPLAY-mode
recurrence and the replay commit). It asserts that they share one layout equal
to `[1,4]`/`[1,32]`/`[4,1]` (`6a54f65`).

**No fused multiply-add.** The test also asserts that the replay commit's PTX
contains no floating-point `fma` or `mad`. The detector catches predicated forms
(`@%p1 fma.rn.f32`) and packed forms (`fma.rn.f16x2`). It ignores integer
`mad.lo` and `mad.wide`, which the aligned address arithmetic introduced
(`f17455a`, `f1d9117`).

**Why the tests compare bytes.** `torch.equal` on float tensors treats `+0.0`
and `-0.0` as equal, so it would hide a sign-of-zero difference between the
replayed state and the candidate column. Every comparison in the replay tests,
and the export gates' committed-state checks, compares `uint8` views of
same-shape, same-dtype tensors. Fixtures add signed zeros: `-0.0` and
alternating `±0.0` state tiles, zero k inputs, and a cancellation where the dot
product equals the value, so `value - sum` is exactly zero (`95b953f`).

**Tests.** `tests/components/test_sm70_gdn_replay.py`:

- a CPU model of both paths with the same expressions: replaying the stored
  operands equals every candidate column, for widths 1, 3 and 5, fresh and not
  fresh, with inactive, null and bystander rows untouched;
- on the V100: `test_replay_commit_matches_candidate_columns_byte_for_byte` for
  accepted counts 1, 4 and 5, with the actual-program layout check;
  `test_replay_commit_captures_without_allocation_and_skips_zero_counts`; and
  `test_replay_rejections`, including the three alignment rejections;
- `test_fp_fusion_pattern_cases`, which pins eight caught and six ignored
  instruction forms.

The export gates add the `forced_prefix_*` checks, which restore the committed
rows before each accepted count and drive the replay commit, and the
commit-growth gate's negative capacity round.

`7cb1bc7` changed a generic capacity check that still assumed one state column
per verified position, and so refused the replay round at bootstrap. It is a
validation change only.

**Effect.** At `--memory-limit 31G` with 4 agents, the KV pool grew from 98,304
to 131,584 tokens, +33,280 (+33.9 %, STATUS.md line 49). The 16 candidate rows
of 166 MiB each returned to the cache (line 59). The 131,584 figure is the
benchmark's `--kv-cache-limit concurrency` cap, not the memory limit
([BENCHMARK.md](BENCHMARK.md#what-was-run)). Four prompts of 22.4k to 30.2k
tokens (110,022 in total) stay resident together (line 65); the old pool could
not hold them.

**Review history.** The kernels were reviewed by two independent reviewers,
before the first GPU run and again for each delta. Round 1 was cleared by one
reviewer, and by the other with required changes: bitwise comparisons,
signed-zero and cancellation fixtures, and an int64 operand index (`95b953f`).
The layout fix and the test changes went through rounds 2 and 3, with required
changes applied in `6a54f65` and `f1d9117`. The capacity-check change `7cb1bc7`
was not independently reviewed (STATUS.md lines 44, 45, 59). The integrator's
commits record a self-review against the design principles in their trailers.
