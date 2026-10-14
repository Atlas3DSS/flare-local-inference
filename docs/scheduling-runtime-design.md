# Prepared-graph scheduling in Helios

Helios selects prepared physical graphs by expected useful value per execution
time. Request-level token and delivery prices provide fairness. A decode-age
continuation term prices the delay caused by a nonpreemptible graph. The
implementation is in `helios/src/scheduler/{policy,service,utility,cost}.rs`;
[`scheduling-utility.md`](scheduling-utility.md) defines the utility and price
flow. Python experiments are in `scripts/scheduling_sim/`.

The native catalog supports separate prefill, decode, speculative verification,
state-management, and mixed prefill/decode graphs. Mixed shape selection, native
uploads, completion accounting, package selection, and rank-layout validation
are implemented. With Ignition's `--mixed-prefill` flag, the Qwen recipe captures
mixed target forwards alongside its separate inference graphs. Mixed capture is
disabled by default. Other recipes expose their declared graph catalogs.

## Useful service and selection

Prompt progress counts consumed uncached prompt tokens. Output progress counts
committed target tokens, including accepted speculative tokens and a replacement
or bonus token. Padding, rejected proposals, inactive rows, restored prompt
history, and state copies contribute no committed token service. A visible
speculative burst supplies one delivery regardless of its token count.

For a request with token rate `x` and delivery rate `z`, the utility gradient
prices are

```text
prefill token: 1 + f/x
output token: beta + f/x
output delivery: g/z
```

Only decoding requests have a delivery coordinate or output-age cost. Normalized
service prices follow the entropy-dual update in `scheduling-utility.md`. Actual
service updates history; expected acceptance scores choices. Conditional
acceptance is request-local and respects available proposals, grammar and stop
restrictions, output limits, and remaining context.

Let `T` be the graph duration in milliseconds and `A_i` the age of a decoding
request's last visible output. The reference interval `sigma_i` uses the fastest
batch-feasible delivery rate. For `K` delivery-bearing streams and a family of
`k` streams, choose the measured graph minimizing `T/min(batch_capacity,k)`;
then `sigma_i = K*T/min(batch_capacity,k)`. This corresponds to independent
uniform cohorts within each family and wall-time shares proportional to family
size. Prefill-only requests do not enlarge `K`. Multistage blocked streams use
an individual future-delivery reference.
For a ready inference graph, selection maximizes

```text
(sum of expected priced token and delivery service
 + sum over served decoders of c_i*sigma_i*(A_i+T)) / T
 - T/2 * sum over active decoders of c_i.
```

All values in one decision share a positive scaling factor to keep large prices
representable. `ScheduledPlan::selection_value_rate()` exposes the scaled score;
`selection_log_scale()` exposes the logarithm of that factor. An unknown duration
has an optimistic work score instead of a division by zero.

Every loaded physical batch capacity and selected verification width competes.
For a fixed graph, duration and excluded-stream waiting cost do not depend on
cohort membership, so the best cohort contains its highest additive lane values.
Selection uses a fixed 64-entry array. It includes larger padded physical batches
when those measurements are faster. It does not enumerate exponentially many
request subsets or restrict selection to heuristic acceptance groupings.

Each prefill capacity competes using the actual tokens it can consume. Atomic
image spans, system boundaries, and prompt length restrict logical work. A
4096-token graph processing a 64-token tail earns 64 prompt units and at most one
output token. There is no smallest-fitting-shape filter, duration ceiling,
forced oldest-request filter, or per-chunk prefill lateness cost.

With no active decoder, the age term vanishes. Fully occupied large prefill
graphs win when their useful value rate is highest. With decoding streams, a
smaller prefill can win near a throughput knee because it imposes less waiting.
Increasing a stream's output age raises its delivery value; it does not make
inefficient operations mandatory at a fixed deadline.

## Prerequisites and completion

Initialization, restore, vision execution, and intermediate snapshots enable
subsequent useful work. When a prerequisite immediately enables inference,
compare every pair of prepared prerequisite and following inference shapes.
Both cohorts are selected jointly. The first graph can also finish terminal
requests; the following graph can include already ready requests.

For fixed graph durations and capacities, lane values are additive. Enumerate
the number `k` of newly ready rows served by the following graph. Choose the
`k` largest such contributions, fill its remaining capacity with the largest
already ready contributions, and fill the prerequisite's remaining capacity with
the largest terminal-completion contributions. Sorted prefix sums give the best
choice over all `k` without enumerating request subsets. This solves the cohort
choice exactly for those two fixed shapes. At least one newly ready row must
participate in the following graph.

Score the sequence using total duration `T+F` and expected useful service,
with output ages resetting at the following graph's completion. Terminal
requests completed in the first graph stop accumulating age at time `T`.
Compare the productive first-stage prefix as well as the complete sequence.
Unused prerequisite capacity can advance additional rows at no extra physical
cost; equal value rates prefer advancing more requests.

The submitted plan retains the prerequisite's physical duration `T`; lookahead
time must not enter execution timing or cost learning. Only the first graph
executes before replanning. Setup never earns committed tokens.

For chains with more prerequisite stages before token production, an individual
continuation supplies value `V` in a further `F` milliseconds. Its cohort-ranking
contribution is `V*T/(T+F)`. Sum these contributions, divide by `T`, and apply
the cost of delaying active decoders. The continuation includes remaining image
operations. Unknown durations compare optimistically by useful value.

This is bounded lookahead, not an optimal schedule over all future execution
sequences. The fixed-pair cohort calculation is exact; longer multistage chains
use the individual continuation index. The stationary rate-utility result does
not establish optimality for those multistage decisions.

A terminal snapshot enables the visible response completion and frees the
request's execution resources. It contributes a delivery with zero output
tokens. Its delivery-rate reference comes from applicable snapshot graphs.
The completion event's price can grow while other requests run. Error and
cancellation paths preserve fenced resource ownership regardless of utility.

## Duration estimates

Each prepared graph descriptor has one `estimated_duration_ns` field. Native
estimates start at zero, including descriptors supplied with a nonzero value.
Zero means indeterminate. Physical graph kind, batch capacity, and token capacity
identify a cost entry; active row count does not change its identity.

The first positive device observation sets the estimate. Further valid samples
before expiry use an EMA with weight 1/8. An entry expires after 30 seconds without
a valid observation. Its next measurement replaces the expired estimate directly.
This lets an initially slow or noisy graph compete again.

The observation is the maximum positive rank-local CUDA replay interval. Host
queueing, input staging, polling, and publication delays do not replace GPU time.
Missing or zero intervals neither change a cost nor refresh its expiry. Expired
costs are also excluded from service-rate normalization.

An indeterminate candidate wins over a determinate candidate, then competes by
expected useful value. Equal values prefer shorter known time, less recently
used requests, and smaller physical padding. Unknown graphs require an empty
submission queue and completion feedback. A lone cold request consequently tries
its largest useful prefill or speculation extent without a solo-request rule.

## Service accounting and publication

`commit` records resident runnable handles and generations, with token and
delivery normalizers from the applicable measured catalog. Requests admitted
after dispatch and requests waiting for external restoration are outside that
interval's participant set. A cancelled handle cannot charge a later request
reusing its slot.

`complete` validates every lane before updating graph costs, acceptance, service
prices, or committed state. Useful service comes from differences against
committed counters, never counters saved before earlier queued work completed.
Each successful frame contributes one measured device interval to service
history. Unselected participants receive zero service over that interval.
Accounting changes do not invalidate unrelated immutable plans.

Completion initially records zero deliveries. The first successful visible
publication for the same fence replaces that interval's delivery update with one
served delivery; subsequent tokens in the burst do not change the count. Stop
buffering and output parsing can postpone visible output. A publication must be
reported before processing another completion, so the correction never rewinds
through later service observations. Output age uses the actual host publication
time independently of device service accounting.

A first published burst initializes a missing delivery history to one delivery
over its measured frame duration. Token and delivery prices are stored in
physical service units. Phase transitions, acceptance updates, and timing
normalizer changes preserve those prices until actual service changes them.

## Planning and the execution queue

```rust
fn advance_to(&mut self, now_ns: u64) -> Result<()>;
fn plan(&self) -> Option<ScheduledPlan>;
fn commit(&mut self, plan: &ScheduledPlan, fence: Fence) -> Result<()>;
fn complete(&mut self, fence: Fence, records: &[CompletionRecord],
            timing: ExecutionTiming) -> Result<Vec<GeneratedToken>>;
fn published_output(&mut self, request: RequestHandle, fence: Fence) -> Result<()>;
```

A plan freezes physical shape, logical lane work, request epochs, estimated
execution duration, and selection score. Commit checks ownership, epochs,
resource availability, and feedback dependencies without rerunning selection.
Cancellation stops further selection immediately; pages and state slots remain
owned through their last submitted fence.

The transport has two frame slots. Known nonterminal prefill and ordinary decode
can use both. Speculation, unknown timings, grammar, host stops, and operations
that change the request phase require completion feedback. A finishing prefill
therefore exposes its decoder and checkpoint operation before another decision
is committed.

There is no millisecond cap on either a frame or the sum of queued frames.
Expected finish times provide the planning start time of an appended frame. An
overrun suspends further queueing until completion supplies feedback; it does
not make that graph ineligible in later decisions.

Selection projects queued service into copies of request prices and projects
visible output ages through the queued interval. It does not credit committed
history speculatively. The completion measurement and validated counters remain
the only inputs to actual service history.

## Policy preferences

`PolicyConfig` can be supplied before admission through `configure_policy`.
Defaults specify these preferences:

| Field | Default | Meaning |
|---|---:|---|
| `output_token_value` | 64 | Prompt-value units per output token |
| `token_fairness` | 16 | Token-rate log-utility coefficient, value/ms |
| `delivery_fairness` | 4 | Delivery-rate log-utility coefficient, value/ms |
| `decode_age_price` | 0.1 | Linear output-age cost coefficient, value/ms² |
| `adaptation_ms` | 100 | Service-price adaptation time |
| `estimate_ttl_ns` | 30,000,000,000 | Cost expiry interval |

These weights express a throughput/cadence tradeoff. They are not graph timing
measurements or latency guarantees. Setting fairness or age coefficients to zero
removes the corresponding preference; intrinsic throughput alone does not ensure
starvation freedom. The native CLI uses these defaults.

## Mixed execution contract

A mixed graph declares `batch_capacity` total physical request rows,
`token_capacity` decode verification positions per row, and `prefill_capacity`
prompt tokens. The batch capacity is a power of two from 2 through 64; decode
width is 1 through 8; prompt capacity is positive and aligned to 64. Other graph
families set `prefill_capacity` to zero. Package selection retains the explicit
`speculation_enabled` variant matching the loaded decode mode, including width
one with draft-state maintenance enabled. Fixed-width limits apply to mixed and
separate speculative graphs alike.

At most one active row performs prefill. `prefill_row`, an `int32[1]` input,
identifies its index in the dense active lane list; -1 means no prompt row.
`ScheduledStep::mixed` carries that role and the selected decode mode. Every row
has a logical query length. Packed token offsets are prefix sums of those
lengths, independent of physical slot IDs. Prompt tokens, positions, and image
metadata are uploaded at the prompt row's packed offset; decoder pending tokens
and proposals follow the model's captured input contract. Inactive rows and
unused token capacity must perform no state mutation. The maximum packed extent
is `prefill_capacity + (batch_capacity - 1) * token_capacity`; prompt capacity
is at least decode width, so a decode-only population also fits.

For each shape, selection compares the best decode-only cohort with the best
prompt row plus the highest-value decoders fitting the remaining row capacity.
A prompt-only population is legal. Every comparison uses the same useful value
per full graph duration; padding earns no value. Duration estimates are keyed
by graph kind, physical batch, decode width, and prompt capacity. Mixed shapes
also participate in service-rate references and prerequisite lookahead. For a
prerequisite followed by mixed inference, the prompt can be absent, already
ready, or enabled by the prerequisite. Each case leaves an exact two-stage
capacity problem for the decode and terminal rows.

Completion validates and updates each row by its phase. A prompt row commits
its full scheduled prompt extent and can publish one first output token; a
speculative row commits the accepted prefix. Prompt checkpoints, acceptance
learning, publication prices, cancellation generations, and fenced page
retirement retain their ordinary contracts. Mixed submissions require completion
feedback before another graph is queued. State-row reservation covers the
maximum verification width across separate and mixed catalogs.

A captured mixed forward must share the packed target execution across phases
and include proposals, accepted-state commit, output projection, collectives,
and sampling. Width one uses decode attention and direct updates while retaining
the draft context required by its selected mode. Recurrent-state rollback and
prompt absorption belong in the prepared model implementation. Helios executes
only supplied captures; it does not synthesize a mixed graph from serial prefills
and decodes.

Qwen shares target projections, hyperconnections, expert execution, and the
vocabulary projection across packed prompt and verification tokens. QSA consumes
per-row phase metadata. GDN and PLE gather their projected operands into fixed
prompt and decode workspaces for phase-specific convolution and recurrent-state
updates, then return the results to the packed target layout. Prompt consumption,
accepted verification prefixes, and draft-context absorption retain separate
state updates. Completion publication combines their disjoint physical slots.
Prompt image features use the prompt's packed offset. Workspaces are shared
across serial graph variants and allocated before capture.

Qwen allocates sampling, target-distribution, proposal, verification, and MTP
feedback buffers at the largest prepared capacity. Each variant binds contiguous
views of those typed buffers before compilation. Sequential proposal positions
reuse one workspace after copying their probabilities into the round's draft
probability buffer. Target probabilities, saved draft probabilities, verification
operands, and prompt residuals remain disjoint while simultaneously live. MTP
feedback rows and raw-cache backups are shared across serialized calls. Persistent
request state and overlapping b12x operations retain separate storage.

`ignition/components/capacity.py::mixed_shapes` bounds the joint catalog before
tuning and compilation when `--mixed-prefill` is enabled. Each batch bucket of at
least two rows has every prompt bucket at verification width one and one prompt
bucket for each wider width.
The latter bucket minimizes `max(P / P_min, P_max / P)`, balancing the worst
multiplicative distance to short and long prompt chunks; ties choose the smaller
bucket. This coverage policy provides a mixed option at every fixed verification
depth without requiring every prompt/depth combination. Separate graphs retain
all batch/width combinations. Graph durations measured by Helios determine which
captured alternatives execute.

For `B` mixed batch buckets, `P` prompt buckets, and `W` verification widths,
speculative mixed capture adds `B * (P + W - 1)` graphs. Ordinary execution adds
`B * P` width-one graphs without draft-state updates. Qwen concurrency four,
4096-token chunks, and five draft tokens therefore add 30 speculative and 20
ordinary mixed graphs, compared with 140 for both complete products. Adaptive
speculative width one preserves draft state; loading with zero drafts selects
the ordinary catalog. Planning reports the same catalog used by preparation and
includes its maximum packed token extent in capture headroom.

## Verification

`helios/src/scheduler/policy_tests.rs` exercises the production selector with
measured synthetic catalogs: one, two, and sixteen long prefills; fitting and
padded tails; high- and low-acceptance speculation; heterogeneous cohorts;
padded batch capacities; long decode pairs; prefill/decode contention; graph
expiry and recovery; and output gaps. Mixed tests compare every legal cohort
against exhaustive useful-work/time enumeration, exercise fitting prompt tails,
ordinary and speculative decode rows, malformed completions, independent shape
expiry, and concurrent prompt/decode workloads. Native input tests check packed
prompt offsets, one query-length upload, row sampling masks, and role validation. Scheduler lifecycle tests separately
exercise prescribed cache and resource transitions without assuming those
operations always outrank another runnable request.

Five- and sixteen-request startup traces cover zero, three, and seven accepted
drafts, with both untrained and preseeded acceptance histories. On the measured
synthetic catalog, each 64-token prompt must deliver its first output within
100 ms, subsequent gaps stay below 150 ms, and another hundred rounds give every
stream progress with a spread of at most one eight-token burst. A separate test
varies snapshot cost to check that enabling a shared decode batch wins only
when the complete sequence has sufficient value per time.

The runtime phase tests cover actual worker completion, publication, stop
buffering, and input packing. `tests/server/test_scheduler.py` exercises an idle
native HTTP service with four request slots and a 4096-token prefill capacity.
It checks complete responses for concurrent long prompts, two and four decode
streams, a long prompt arriving during decode, short/long prompt contention,
and cancellation followed by slot reuse. Run it for ordinary, fixed-width, and
adaptive speculative execution:

```sh
FLARE_RUN_CUDA_TESTS=1 FLARE_SCHEDULER_TEST_URL=http://127.0.0.1:18131 \
  .venv/bin/python -m pytest -q -s tests/server/test_scheduler.py
```

The HTTP tests print actual prompt/output token counts, client SSE arrival
gaps, and server graph-counter deltas. SSE arrival gaps are not GPU execution
times or per-token latencies. Mixed capture correctness requires packages that
contain mixed model forwards. Synthetic scheduler traces do not establish GPU
throughput.
