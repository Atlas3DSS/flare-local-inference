# FlashInfer sampling contract for native graph replay

The prepared sampler uses FlashInfer revision
`98fb87a461183f98d6a7049ca947fb136cc6fb59` with device-resident parameters and
explicit RNG inputs. Python selects and captures the operations; native CUDA
graph replay executes them without Python callbacks. This document records CPU
source inspection. GPU correctness, allocation, and timing receipts are separate
acceptance evidence.

## Source identity

The installed distribution is `flashinfer-python 0.7.1`, built from
`.build-sources/flashinfer`. Its `direct_url.json` identifies that directory with
`editable: false`. The following hashes identify the upstream files inspected at
the clean checkout `/tmp/lunar-rebase-20261008/flashinfer-pinned`, whose HEAD is
the revision above. Installed and staged copies match these hashes except for
`air_top_p.cuh`, which carries the declared parallel-application change in
[`flashinfer-serving.patch`](/data/sparkgap-artifacts/flashinfer-source-migration/clone-and-patches-98fb87a/payload/patches/flashinfer-serving.patch). Its installed
SHA-256 is `6753f53c511ad0c64114d4817b50993880804a12664c7d7bfcb8053f3442a48b`.
Paths in the table are relative to a FlashInfer checkout. Installed Python files
reside under `.venv/lib/python3.12/site-packages/flashinfer`; installed C++/CUDA
files reside under that package's `data` directory.

| Source | SHA-256 |
| --- | --- |
| `flashinfer/sampling.py` | `ce794984221a9909499799d5723cd49867339ff7405bac9de98af0f70711c6d3` |
| `flashinfer/jit/sampling.py` | `82208d829c924ee74e4e4ddba06db9d97fd7f42fb753b42b3503f5d8e0a844a8` |
| `flashinfer/utils.py` | `abc97f7494cecc485395aeecd87d15e05ea518cf3d6ab1cc8c7e96fcdec3ad4d` |
| `include/flashinfer/sampling.cuh` | `e297a6a05581b7bb3328ca8b839ea480f9ff4c3463f86a824565655ed2cd8d6f` |
| `include/flashinfer/topk.cuh` | `7ca7db449d7d9560f04c34de60fe3e4567a7aca192ed8cced7dfbba39798d762` |
| `include/flashinfer/air_top_p.cuh` | `e41572fbb55a214fc0b4dfb077f8e8e127fe61bf69b4d4b53ac092f6b80e22bc` |
| `csrc/sampling.cu` | `41df79df3c8a3ed6a07101c599977d413e9c9e2df555d977ca83f714cc04361c` |
| `csrc/renorm.cu` | `06cc6aa68dbf102333dbc4995473a58468cf765ac5cde0834d481219d6d79340` |
| `flashinfer/cake_sampling.py` | `75f1a2fb6f84fe745b7cd78621ce8de4a89a0f9ef68deb79460c1a76b2d19c83` |

The inspected CUDA headers were `/opt/cuda/include/curand_kernel.h`, SHA-256
`4a37c07a1d77c9b5c8c627a4720733cee6b4da4200a844e8a49291858bc26adf`,
and `/opt/cuda/include/curand_uniform.h`, SHA-256
`81a6fb0c65c6710b1994e13ad8899f122c647af204dae3030c116b01ec6f1bb4`.

## Captured operations

For each packed row, apply the request's repetition, presence, and frequency
penalties, positive temperature scaling, and grammar mask to FP32 scores. Keep
an independent device validity flag for live epoch, request status, valid
allowed scores, and available probability mass. Invalid or inactive rows use a
safe dummy distribution inside the sampling kernels; completion logic decides
whether to report an error or leave the request unchanged.

The general path has a fixed captured composition:

1. `flashinfer.sampling.top_k_mask_logits(scores, device_top_k)`, with contiguous
   `int32[B]` thresholds. Translate disabled top-k (`0` or `-1`) to the vocabulary
   size before this call.
2. `flashinfer.sampling.softmax(masked_scores)` with FP32 input.
3. `flashinfer.sampling.top_p_renorm_probs(probabilities, device_top_p,
   is_deterministic=True)`, with contiguous `float32[B]` thresholds.
4. Captured tensor operations mask probabilities below
   `min_p * row_maximum` and normalize the positive remaining mass. Equality at
   the min-p threshold is retained.
5. One `flashinfer.sampling.sampling_from_probs` call per row, each receiving
   `float32[1,V]`, `uint64[1]` seed and offset views, `deterministic=True`,
   `check_nan=False`, and `return_valid=True`.
6. Completion logic writes persistent token, histogram, generated-count,
   RNG-counter, status, and completion-record fields once for each accepted
   sample. Disabled, stale, inactive, duplicate, and terminal rows follow the
   request-state contract independently of speculative sampler outputs.

Temperature-zero rows use `torch.max` on the masked scores and select the
lowest token ID among equal maxima. Top-k sampling with `k=1` cannot implement
that greedy rule because threshold ties may contain multiple tokens.

## RNG and endpoints

`include/flashinfer/sampling.cuh:794` implements `SamplingFromProbKernel`.
At line 805 it initializes `curand_init(seed, output_row, offset)` state; line 820
draws exactly one `curand_uniform`. All threads in the CTA initialize identical
state and obtain the same draw. Probability scanning adds no draws or retries.
Seed and offset tensors are read-only.

Each one-row invocation has output row `bx=0`. Set its device offset to
`4 * request_rng_counter`. CUDA's `/opt/cuda/include/curand_kernel.h:971`
interprets the offset as a count of 32-bit outputs: block index is `offset/4`
and word index is `offset%4`. Its initializer at line 1022 applies the seed,
subsequence, and offset. Each successful sample therefore takes the first word
of a different Philox block, independent of batch capacity, packed row, and
physical request slot. Counter multiplication remains injective below `2^62`;
admission limits a request's output budget to at most `2^31-1` tokens.

CUDA's `/opt/cuda/include/curand_uniform.h:69` converts the word to an FP32
uniform in `(0,1]`; the Philox wrapper is at line 255. This differs from the
24-bit `[0,1)` conversion used by the sorted-CDF sampler. Fixed seed/counter
values do not imply identical tokens across these sampling policies.

`SamplingFromProbKernel` only admits probabilities greater than zero
(`sampling.cuh:831`). Its endpoint/rounding fallback selects the last index
satisfying that same predicate (`sampling.cuh:645` and `:837`). A normalized,
finite positive-mass row therefore cannot select a zero-probability token even
when the uniform is exactly one. An all-zero row returns `valid=false`.
Positive infinity and malformed input distributions still require the
independent server validity guard.

The fused `min_p_sampling_from_probs` kernel uses `x >= max_probability * min_p`
(`sampling.cuh:1188`). At `min_p=0`, zero-probability tokens satisfy its predicate
and become fallback candidates; an all-zero row can return its last token with
`valid=true`. Explicit min-p masking followed by `sampling_from_probs` avoids
that source-derived edge without changing the requested min-p threshold.

## Filtering policy

Top-k retains every token equal to the kth score: the radix mask uses
`logit >= pivot` (`include/flashinfer/topk.cuh:1664`). Consequently `k=20` may
retain more than 20 tokens. The combined top-k/top-p API's scalar-k fast path
gathers exactly k entries when `k<=256` and `V>=65536`
(`flashinfer/sampling.py:1583` and `:1597`); it is not used by this composition.

AIR top-p selects a probability threshold and keeps every entry at or above
it (`include/flashinfer/air_top_p.cuh:366` and `:434`). Equal-probability entries
at the cutoff remain together. With probabilities `[0.5,0.25,0.25]` and
`top_p=0.6`, all three tokens survive. With `top_p=0.5`, the threshold is the
first probability. The sorted-CDF policy's stable token-ID prefix and
`cumulative_before <= top_p` rule selected two tokens in both examples.
The filtering policy and RNG differences require separate reference results;
sorted-CDF receipts remain evidence for their recorded implementation.

## Capture, storage, and execution

The public APIs allocate functional outputs and expose no `out=` argument.
Capture-time allocations are allowed inside the graph's private pool. Consume
those intermediates in the same graph and copy values that must survive another
graph into persistent storage. Retain graph, allocation-pool, sampler, and
workspace owners until the native worker confirms stream drain.

`flashinfer/sampling.py:2233` obtains a process-global 1 MiB top-k row-state
buffer through `flashinfer/utils.py:255`. It is zero-initialized on first
allocation and reset by the last departing CTA (`topk.cuh:1680`). Softmax also
uses a process-global 1 MiB workspace (`sampling.py:953`). Graphs sharing these
buffers must execute serially on one native stream; concurrent workers using
the same buffers would race. Warm up the largest required buffers before graph
capture. Changing k from disabled to enabled and back must preserve the radix
state's clean-launch invariant.

Deterministic AIR top-p owns a capture-time workspace, initialized inside each
invocation. At vocabulary size 248,320 its sizes are:

| Rows | Workspace bytes |
| ---: | ---: |
| 1 | 88,576 |
| 2 | 176,896 |
| 4 | 353,792 |

The calculation is in `sampling.py:2068`; the launcher captures initialization,
three radix passes, and normalization. Integer histogram accumulation is selected
explicitly by `is_deterministic=True`.

The declared patch distributes top-p application across three kernels: four
CTAs calculate the same 1,024 logical-thread partial sums in the same addition
order; the unchanged 1,024-thread CUB reduction computes normalization; parallel
token blocks apply the threshold and scale. Partial sums reuse histogram memory
after its final radix use, and normalization reuses the counter's completed
radix sum field. Public arguments, workspace size and threshold selection are
unchanged. [Operator qualification](sampling-air-parallel-qualification.json)
records 375 bitwise deterministic API comparisons and 300 bitwise apply
comparisons with identical thresholds on SM120. The original nondeterministic
radix implementation also exhibits support differences across repeated calls;
the prepared sampler explicitly selects deterministic histograms.

Prewarm module compilation, all selected shapes, and device-property queries
before capture. On SM120/121, the standard sampler uses 1,024-thread CUDA blocks
(`sampling.cuh:77`). The special Blackwell softmax route accepts only SM100/103
(`sampling.py:164`); SM120 uses the generic two-pass vocabulary-sliced path at
these batch/vocabulary sizes, with 31 slices (`sampling.cuh:1345`). Multi-CTA
top-k rejects devices with at most 16 SMs (`topk.cuh:1729`).

Supply both RNG tensors. Omitting either invokes Python generator state reads
and writes (`sampling.py:49` and `:278`). `check_nan=True` performs a tensor-to-host
truth test and is unsuitable during capture. With explicit tensors and
`check_nan=False`, the selected wrappers inspect tensor metadata, perform
device casts if needed, allocate outputs, and launch kernels; they do not read
request values on the CPU. Matching FP32 probabilities, int32 k, and FP32 p
avoids unnecessary dtype-copy kernels. Default API logging is disabled.

The pinned CAKE interface accepts only host-scalar Philox seed/offset and
reproduces captured draws on every raw replay
(`flashinfer/cake_sampling.py:263` and `:1547`). Its caller-provided output and
workspace support does not supply the dynamic device RNG contract required
here.

## Verification targets

FlashInfer's `tests/utils/test_sampling_seed_offset.py:145` captures all public
sampler variants with RNG tensors, changes seed and offset, and compares replay
with scalar calls. `tests/utils/test_sampling.py:950`, `:983`, and `:1144`
exercise deterministic renormalization and mixed enabled/disabled top-k.

Sparkgap's
[focused native replay tests](../../tests/recipes/test_qwen38_sampling_flashinfer.py)
cover full-vocabulary threshold ties, device parameter changes, capacity and
slot invariance, invalid/grammar-masked distributions, duplicate and stale
rows, greedy ties, EOS and budget termination, persistent output addresses, and
zero replay allocation-count growth. A successful CPU collection with CUDA
tests skipped is not GPU qualification.
