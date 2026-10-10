# Multi-token prediction and speculative execution

Flare's speculative engine targets the embedded MTP module in
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, on SM120/121. Python prepares
weights, kernels, storage and CUDA graphs; Rust schedules requests and launches
graphs. Acceptance and token feedback remain on the device. Speculative serving uses a versioned token-vector completion contract and an
explicit `--mtp` preparation option. Draft continuations from request history
are disabled by default and enabled with `--history-drafts`.

The design combines adaptive verification budgets, native MTP proposals and
continuations copied from request history. Compact recurrent commit and token
trees are additional capabilities with explicit operator requirements. A
separately trained parallel drafter is optional; it is not needed to use the
checkpoint's MTP weights.

The [core design](core-design.md) defines ownership, admission and graph
lifetime.

## Checkpoint model and reference

The exact checkpoint has one MTP decoder layer, 512 routed experts with top-10
routing, four 2,560-wide hidden streams, and a hyperconnection output mixer.
Its sparse-attention layer has 24 query heads and two KV heads of dimension 256.
The MTP decoder has no GDN or PLE layer. Target verification still executes all
36 GDN layers, 12 QSA layers and target PLE operations.

| MTP storage | Contract |
| --- | --- |
| 4,637 tensors, 1,596,720,640 bytes | About 1.487 GiB before preparation overhead |
| 1,536 packed expert weights | NVFP4 W4A16; E4M3 block scales and FP32 global scales |
| Dense projections, shared expert and norms | BF16 checkpoint tensors |
| Embedding and vocabulary head | Shared target weights; no separate stored MTP payload |
| Stored layer name | `mtp.layers.0`; the `mtp.layers.48` quantization alias is explicit metadata |

MTP expert preparation must preserve W4A16 arithmetic. The target expert loader
uses activation-quantized NVFP4 with CSF scales and cannot be applied unchanged.
The [checkpoint manifest](../flare/preparation/qwen38.py) validates MTP
headers; the [weight loader](../flare/recipes/qwen38/weights.py) excludes them.
Header validation does not verify tensor payload hashes or model execution.

Transformers `4cc2aa84301c9aa210b5513dab9fefae03981f6e` ignores `mtp.*`
and has no MTP module. The [MTP recipe](../flare/recipes/qwen38/mtp.py) composes
b12x operators for feedback, attention, expert execution, and output mixing.
Feedback arithmetic is checked against `b12x.sequence.mtp_feedback.reference`;
state alignment and graph replay are exercised by the recipe tests.

For a consumed target token `x[i]`, let `h[i]` be the target's 10,240-wide
residual **before its final mixer**. One teacher-forced MTP row receives
`(h[i], embedding(x[i+1]), position[i])`. Token IDs shift forward; hidden states
and RoPE positions do not. Prepared `b12x.sequence.mtp_feedback` normalizes and
projects those inputs into four streams. The MTP decoder and mixer produce both
a sampling vector and a multi-stream residual for recursive drafting. Repeating
the target's collapsed 2,560-wide sampling vector across streams is incorrect.

Recursive MTP states are proposal states. Even when their tokens are accepted,
their cached projections may depend on approximate MTP hidden states. Reusable
draft history must be absorbed from the **actual target** hidden states and
accepted next-token IDs.

## Available operator contracts

| Responsibility | Existing library interface | Bound or invariant |
| --- | --- | --- |
| MTP feedback | `b12x.sequence.mtp_feedback` prepared plan/bind/run | Four streams; caller-owned output and scratch |
| Target recurrent verification | `b12x.sequence.gdn_decode` | At most eight state-index columns; each token writes its post-token state |
| Target sparse attention | Ordinary prepared `b12x.attention.qsa` | Exact accepted-count, raw-tag and compression-group semantics |
| Draft selector reuse | QSA `DraftSelectionState`, `run(reuse=DraftSelectionReuse(...))` | Proposal-only; reset on round or request identity changes |
| Convolution history | `b12x.sequence.short_conv` and PLE operators | Speculative history extent and accepted-token selection |
| Stochastic chain verification | `flashinfer.sampling.chain_speculative_sampling` | Draft probabilities `[B,K,V]`, target probabilities `[B,K+1,V]`; explicit device RNG |

GDN's ordinary checkpoint path supports Qwen's 16 key heads and 48 value heads.
Its KDA recovery mode requires equal head counts. FlashInfer's separate
ReplaySSM cache/commit implementation is restricted to SM100/103. Neither is
a Qwen SM120 compact rollback implementation. Public QSA draft reuse omits
selection work and must never be used in target verification.

Library kernels remain the compute implementation. Server-owned captured
operations pack metadata, select committed state and update request records.
Missing tree or compact-state semantics require a public library contract;
there is no private eager model fallback.

## A speculative round is a state transaction

At its start, a request has `N` consumed tokens, one pending token `p`, an
accepted-prefix target state, and an MTP cache derived from committed target
history. A round offers `K` drafts after `p`.

1. **Draft:** complete the feedback pair using the saved target tail and `p`,
   then recursively propose up to `K` tokens. Copy candidates may replace or
   extend a proposal under the same verification budget. Proposal RNG and
   histories are separate from committed sampling state.
2. **Verify:** execute the target on `[p, d[0], ..., d[K-1]]`. Retain all
   `K+1` vocabulary distributions and relevant target residuals, not only the
   final row. Each row sees exactly its causal candidate prefix.
3. **Accept:** select the accepted draft prefix and the target correction or
   bonus token. With `a` accepted drafts, ordinary chain verification consumes
   `a+1` inputs and produces `a+1` newly generated tokens: `a` drafts and one
   pending target sample. The incoming `p` was generated previously.
4. **Commit:** clip at device-visible EOS or output budget; select the matching
   target state, advance committed histories and penalties once, repair draft
   history using actual target residuals, and write a bounded token-vector
   completion. Unaccepted rows remain private and unreachable.

Let `t` be the emitted prefix length after EOS and output-budget clipping,
`1 <= t <= a+1`. Advance consumed and generated counts by `t`, select the state
after verification input `t-1`, and retain emitted token `t-1` as pending.
For example, emitting only `d[0]` consumes `p`, not `d[0]`, even if later draft
rows were verified. A terminal token remains pending. Requests already at their
limit execute no state transition. Generated, consumed and visible token counts
are distinct. Device terminal flags suppress any later queued round for that
request. Resource retirement waits for every referencing graph and transfer.

### Target state and prefix publication

GDN verification reads the checkpoint column selected by the preceding accepted
count and writes one checkpoint per input row. Convolution and PLE buffers must
retain all candidate histories until acceptance. PLE hashing reads committed
history; a proposed token affects only the candidate prefix that contains it.
Inactive rows never update another request's state.

QSA may physically retain rejected KV and selector rows, but logical lengths,
raw tags and interval anchors must exclude them. Rejected writes cannot mutate
an immutable shared page. Reserve private suffix pages, including boundary
pages touched by a speculative window, before submission.

A nonempty prefix checkpoint at `P` consumed tokens retains target state through
`x[P-1]`, its final sampling vector and its pre-mixer residual `h[P-1]`. Reusable
MTP history contains pairs through `(h[P-2], x[P-1])`: **`P-1` draft-cache
entries**. The pair containing `x[P]` depends on the next continuation and
cannot belong to an immutable checkpoint of the shorter prefix.

This creates an alignment difference. At `P % 64 == 0`, target QSA has no open
four-token compression group, while the MTP cache has three entries in its open
group. Its checkpoint must retain raw selector keys, tags, positions and anchor
needed to complete that group. The target restore operation's blanket raw-ring
invalidation is not a valid MTP restore operation. Draft-only selection-reuse
anchors are reset separately.

A full prompt hit restores both model histories, samples `p` from the saved
target sampling vector, and completes the MTP feedback pair from the saved
pre-mixer tail. Request RNG and penalty state are initialized from the new
request. Branches share immutable target and draft pages and own private suffix
state. At a crossed 64-token boundary, publish only the state corresponding to
an actually accepted consumed prefix.

### Memory and compact recurrence

The target's FP32 recurrent matrices occupy **108 MiB per saved verification
position**. Four requests with seven additional positions require 2.953 GiB;
fifteen require 6.328 GiB, excluding histories, logits and weights. Recurrent
writes also cost 108 MiB per verified row: batch one at width eight writes
864 MiB, approximately 0.5 ms at 1.79 TB/s before other traffic. The excluded
batch-four width-eight shape would write 3.375 GiB, approximately 2 ms.
Measured round timings must include these writes. Existing GDN
supports a maximum of seven drafts plus the guaranteed input per request.

One additional BF16 QSA layer adds 2,112 bytes per cached token, including
compressed selector keys: 528 MiB for one 262,144-token context. One target
feedback tail occupies 20 KiB. A complete 4,096-row target residual occupies
80 MiB and must survive until MTP absorption; it cannot alias scratch overwritten
by the draft model. Preparation reports exact persistent, graph-pool and scratch
costs before allocating KV pages, preserving configured headroom and snapshot
budgets.

TensorFold's optional Flash Next Metal kernel demonstrates another representation:
one committed recurrent matrix plus compact per-row K/V, normalization and gate
records. The following round replays accepted updates before evaluating its
speculative rows. Two sixteen-row record banks across 36 layers occupy about
27.8 MiB, versus 1.688 GiB for sixteen full recurrent positions. These are
layout sizes, not measured Flare savings.

A Qwen-compatible b12x operation could implement this explicit deferred commit.
It must reproduce b12x's recurrence arithmetic and materialize all accepted
updates before snapshot publication or resumable terminal-state export. Full
checkpoint columns remain its correctness oracle. This is an extension of the
state representation, not permission to reconstruct an arbitrary checkpoint by
rerunning the model.

## Graph catalog, control and scheduling

The chain catalog has request capacities `B ∈ {1,2,4}` and integer verification
widths `1 <= W <= 8/B`, where `K=W-1`. Batch one captures every width from one
through eight, batch two captures one through four, and batch four captures one
and two. All default shapes stay within the eight-row MoE direct-micro band.
Grouped-path shapes require measured round costs before entering the catalog. Width one consumes the
pending input and emits one target token, with no draft proposals; MTP history
still absorbs the committed feedback pair. All shapes are prepared, tuned,
warmed and captured before handoff.
The Qwen metadata contract declares at least:

```text
operation: prefill+mtp_absorb | resume+mtp_seed | speculative_round | maintenance
request_capacity: B
verification_width: W
sampling_mode: general
proposal_source: mtp | copy
history_drafts: boolean
state_layout_version, arithmetic_policy, checkpoint_identity
```

Prefill-plus-absorb uses the configured prefill buckets. A speculative round
captures draft, target verification, acceptance and commit in one dependency
chain. A larger width cannot be selected until its kernels, history extent,
page reservations and completion storage exist. Padding changes capacity, not
causal history. All MTP variants retain the same seven-draft history horizon:
eight GDN state columns and matching QSA raw-ring/convolution capacities. The
preceding round's accepted count can exceed the following round's width. A
width-eight to width-one transition must preserve that count and state layout;
it cannot reuse an ordinary one-column, non-speculative binding by truncating
metadata. Width one also retains MTP absorption, so its measured cost differs
from a request with MTP disabled. Same-shaped graph variants share the existing
`ScratchArena.views(*shapes_and_dtypes)` storage only where lifetimes permit;
candidate logits, feedback, accepted IDs and state surviving another operator
have persistent storage. Final bindings refresh after tuning and freeze before
capture.

| Control | Owner |
| --- | --- |
| Graph handles, weights, plans, capacities, scratch envelope | Python preparation |
| Admission, page reservations, request grouping, graph choice | Rust |
| Acceptance, committed counts, next positions, pending tokens | Captured device operations |
| Acceptance/cost statistics and bounded exploration policy | Rust, using completed-round records |
| Grammar advancement, stop sequences, output presentation | Rust with ordered token-vector completions |

Rust adapts among prepared widths from completed-round acceptance and measured full-round
costs. The fixed-width catalog gives each active row the same maximum draft
count. Different logical counts under a shared row budget require matching
verification and acceptance contracts; the selected graph's real padded cost
is what the controller prices.
Masking later iterations of an unrolled draft graph does not automatically
eliminate their compute. The design requires neither runtime recapture nor
device-side graph construction.

Speculative execution permits one generation round in flight. Rust consumes its
completion before selecting the following width, copy continuation or grammar
mask. CUDA dependencies serialize model work and protect shared scratch. Rust
reserves the round's maximum possible page/output use before submission. ABI3
uses two transfer frames of four 24-field `int64` records; each record retains
the scalar ABI2 prefix and adds emitted/offered/accepted counts, copy-source
bits, candidate and draft-cache lengths, verification width, and eight padded
token IDs. A frame cannot be reused until its transfer finishes. Ordinary ABI2
decode retains its two-submission pipeline. A two-round speculative pipeline
requires a separate measurement and explicit configuration before deployment.

Copy-only grammar rounds clone tentative native grammar states and upload a
mask for each copied prefix, including the target bonus position. Rust advances
the committed grammar with only the accepted tokens. MTP proposals are generated
on the device; grammar-constrained MTP requests use width one until the runtime
can provide valid masks for those conditional prefixes. Host-evaluated custom
stops also use width one. Rust acknowledges each pending token before another
graph consumes it. Prefix publication cannot include unacknowledged continuation
tokens. One grammar mask reused across draft positions is invalid. Such rounds
remain single in-flight until the native grammar state and uploaded masks are ready.
Multi-token custom stops similarly require ordered matching across token and
round boundaries, retaining candidate checkpoints until the selected stopping
prefix is known. Tentative device progress beyond a host-selected stop remains
private, cannot publish snapshots, and retires only after all referencing work
drains. Host output truncation alone does not commit the corresponding model
state.

## Sampling correctness

All chains use FlashInfer's rejection sampler: accept
with `min(1, p(token)/q(token))`; on rejection, sample the normalized positive
residual `max(p-q,0)`, or sample the target bonus distribution after full
acceptance. `q` must be the distribution that actually generated the proposal.
Temperature, penalties, grammar and top-k/top-p/min-p filtering must be reflected
at each conditional prefix. A copied token has a point-mass proposal law:
`q(copy_token)=1`, with zero mass elsewhere. Copy substitutions use that law
at every temperature, including when replacing a native MTP proposal.

Temperature zero constructs a point mass at the first maximum logit and passes
it through the same FlashInfer probability sampler and chain verifier. There are
no separate greedy graphs, acceptance kernels or scheduler routes. Positive
temperatures use the checkpoint's sampling defaults unless a request overrides
them.

The chain sampler's counters accumulate in place. Reset them per round; use
**emitted draft count plus one**, not independent acceptance count, for output
length. Store returned token IDs in persistent completion storage before graph
pool reuse. No PyTorch generator-backed random operations are captured for raw
Rust replay.

The public chain API takes a fixed `K` and has no per-request valid-length
argument. Stochastic requests therefore use the captured width, or a declared
width-pattern graph with separate fixed-length one-row invocations. Arbitrary
per-round heterogeneous lengths require a qualified public length-aware sampler.
Masking target rows or inventing padded proposal probabilities is insufficient.

The existing sampler uses one-row FlashInfer calls so Philox subsequence zero
is independent of packed lane position. Preserve this property for speculative
requests: use per-request one-row chain invocations with explicit device seeds
and offsets, unless a library API supplies an equally qualified request-keyed
stream. Draft randomness and target acceptance/correction randomness have
separate declared domains. Their offset-consumption law must be verified from
the selected library kernel. The inspected chain kernel consumes `K+1` scalar
Philox draws when all drafts are accepted and `K+2` after a rejection: its
independent-acceptance statistic samples the rejection position again. Reserve
nonoverlapping ranges by converting that bound to the RNG API's offset units,
and qualify the compiled kernel's progression. Merely adding emitted count
to the ordinary single-token counter would reuse random draws.

Distribution-preserving verification does **not** imply the same tokens for one
fixed seed as ordinary sampling. TensorFold uses seed/absolute-position/token-ID
keyed draws and target-match verification to obtain a stronger coupling under
row-invariant arithmetic. Adopting that sampler would change Flare's random
contract; it is not required here. Numerical differences between verification
widths also remain subject to the measured reference floor, including changes
in the maximum-logit token near ties.

## Adaptive policy and operator extensions

The controller selects bounded captured widths from completed-round token counts
and measured round cost. Copied request continuations use the same verification
budget as MTP proposals. Compact recurrence is a separate state-representation
extension; token trees additionally require parent-indexed GDN, convolution,
PLE and QSA histories. The [research comparison](/data/sparkgap-artifacts/mtp-research/competitive-mechanisms.md)
records the external mechanisms, source revisions and their applicability to
this checkpoint.

## Implementation dependencies and acceptance

1. **Baseline and reference.** Record exact-checkpoint non-MTP serving at
   prefill capacities 512 and 4,096: cold TTFT, 256-token decode at concurrency
   1/2/4, 4K/64K prefill, 4,096-prefix-plus-11-tail TTFT, startup, and measured
   host submission gaps. Retain actual token IDs, EOS behavior and workload
   identities. CUDA completion events with timing disabled do not provide a GPU
   timeline; use explicit timing instrumentation or correlated profiler traces.
2. **Copy transaction and ABI.** Implement bounded vector completions,
   verification-row logits, state selection and persistent outputs using public
   operators and host-supplied copied continuations. Qualify every accepted
   count, prefix publication, and cloned per-candidate grammar states without
   loading MTP weights. Stochastic copy verification uses a point-mass proposal
   distribution at each copied token.
3. **MTP weights and arithmetic.** Load the exact W4A16 experts and BF16 tensors;
   qualify feedback, target-hidden alignment, draft attention, routed/shared
   experts and logits against component operator references. Match activation
   policy, KV dtype and draft-head policy explicitly. Add target-derived MTP absorption to prefill,
   accepted rounds and prefix checkpoints. Keep canonical eager comparisons
   diagnostic.
4. **Adaptive budget.** Measure fixed depths including zero first, with the
   checkpoint's stochastic defaults as the primary workload. Temperature-zero
   fixtures qualify probability semantics separately. Compare the controller
   against the same inputs and completed-token
   counts; record offered/accepted counts, draft/verify/commit timings,
   throughput, tail latency and memory. Preserve regressions and rejected cases.
5. **Compact commit and trees.** Require component state equivalence against
   full snapshots before replacing that representation. Add tree ancestry only
   with a corresponding target reference and measured benefit. Parallel-drafter
   training remains a separate model-development decision.

Correctness includes all-accept/all-reject and every intermediate prefix; QSA
groups of four and page boundaries of 64; all graph buckets and padded lanes;
width-eight to width-one transitions with eight preceding accepted inputs;
cold/prefix-hit/branch continuation; image-conditioned prompts; PLE history;
EOS, stop and budget clipping; penalties and RNG progression; slot generations;
cancellation and the single-round submission barrier; grammar barriers; and snapshot/page
pressure. Prompt counts are initialized once, generated counts advance once,
and rejected proposals never enter committed state or public output.

Compare Python and Rust replay of the same graphs bitwise, including same-shape
restore and every persistent state region. Cross-capacity and ordinary-versus-
verification arithmetic use tolerances established from measured same-input
variation. Teacher-forced diagnostics retain full-vocabulary logits,
relative L2 with the reference norm as denominator, top-10 overlap and FP32
`KL(reference || actual)`. Numerical acceptance requires component comparisons
on identical operands, tolerances derived from measured arithmetic differences,
and measured repeatability for kernels whose reduction order can vary.
Generation tests exercise the common probability sampler with explicit settings
and fixed output budgets. Whole-model logit relative L2 is diagnostic. A higher
speculative acceptance rate cannot close an unexplained component discrepancy.

Every admitted graph must replay without allocation, compilation, recapture or
Python callbacks. Poison shared scratch between calls; alternate graph families
and capacities; verify no escaping output aliases volatile storage; drain both
execution and transfer streams before releasing owners. Startup fails if weights,
verification state, graph pool and reserved pages cannot fit.
Qualify each supported GPU architecture independently; SM120 results do not
establish SM121 execution or performance.

### Workload and paired performance measurements

The [serving performance corpus](../qualify/performance.py) uses exact-length
numbered-list requests padded with repeated `apple` tokens. It measures serving
shapes, cache paths and replay overhead; it does not establish representative
MTP acceptance or speedup.

Freeze a separate MTP workload before measurement: agentic multi-turn exchanges
with recorded tool results, tool-call and JSON output, code edits with substantial
copiable context, non-copying reasoning and general chat, and long-context
continuations. Include both repeated prefixes with different tails and repeated
identical tails. Record source identities, exact messages/tools, rendered token
IDs, turn boundaries, tool-result fixtures, sampling settings and expected output
semantics. Preserve every category, including cases with little copy opportunity;
do not execute external tool side effects during the benchmark.

Pair no-MTP execution, fixed MTP depths, adaptive MTP with copying disabled, and
the same adaptive policy with copying enabled. Keep request contexts, concurrency,
prefix-cache treatment, EOS/custom-stop handling, output budgets and grammar
constraints identical across arms. Unsupported speculative grammar or custom-stop
cases exercise the declared one-token path and remain in the workload. Repeated
runs alternate arm order and retain individual observations and failed requests.

Report actual completed-output token counts, termination reasons, end-to-end
latency, TTFT, throughput and latency tails. Separate offered, verified and
emitted draft counts; condition acceptance on draft depth, proposal source and
workload category, and record copy eligibility. Compare draft, verification and
commit costs with the graph buckets actually executed. A longer output, early
EOS, repeated-text artifact or truncated response cannot be counted as equivalent
completed work. Synthetic serving throughput does not predict the result of
these workload measurements.

Performance compares ordinary decoding, minimal replay of the prepared
speculative graph sequence, and the complete native service with matched settings.
The native overhead target remains within 5% of that **measured replay floor**, separately
from frontend delivery. The approximate 3.9 ms single-token weight-traffic bound
is a non-speculative batch-one reference: amortized verification can reduce bytes
per emitted token, while extra expert routes and MTP work can increase round
cost. No fixed MTP speedup is assumed. Raw observations and large tensor/trace
data stays outside the repository.
