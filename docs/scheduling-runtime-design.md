# Native interactive scheduling

## Design status and objective

The native scheduler in `helios/src/scheduler.rs` uses request-level
useful-progress deficits, joint cohort/chunk/width selection, a single-visit
latency check, and a duration-bounded submission window. Its policy and
accounting live in `helios/src/scheduler/policy.rs`; deterministic native traces
are in `helios/src/scheduler/policy_tests.rs`. The worker supplies monotonic
arrival, completion, and output-publication observations.

Each captured graph carries one `estimated_duration_ns` value; zero denotes an
indeterminate cost. Ignition leaves the field at zero. Native completions learn
each cost and refine it with an expiring EMA. All graph
kinds report GPU intervals, including each participating tensor-parallel rank.
The executor reports host elapsed time separately.

The Python simulator in [`scripts/scheduling_sim`](../scripts/scheduling_sim/)
explores mixed execution and policy alternatives with synthetic cost models.
Its definitions and results are in [the scheduling theory](scheduling-theory.md).
It does not model native prerequisites, output buffering, or multiple outstanding
submissions. Native mixed captures, residency rotation, and independent submission
past feedback barriers remain unimplemented; their required contracts appear below.

The scheduling entity is one request, with equal weight across requests. The
engine should provide prompt first service, regular streaming output, and
sustained useful progress. It optimizes execution efficiency subject to those
preferences. Prompt length and explicit output limits are known; EOS and future
speculative acceptance are unknown.

The policy is useful-progress deficit scheduling with an anticipatory
latency filter. Token-delivery slowdown is a measurement, not the primary
dispatch objective. There is no universal preferred prefill size, verification
depth, or cohort size.

## Ownership and interfaces

One native scheduler owns live requests, resource reservations,
the prepared action catalog, and dispatch decisions. The executor launches the
selected operation and reports its completion. It does not independently choose
speculative width, change batch membership, or initiate opportunistic GPU work.
Model contracts describe packing, state transitions, and legal shapes; recipes
prepare those implementations with Ignition.

The residency controller must separate logical request identity, continuation
residency, and execution batch rows:

- A logical request retains its identity, arrival time, accounting, output cursor,
  sampling state, grammar state, and cancellation state throughout its lifetime.
- A resident continuation owns cache pages and persistent model state. More
  continuations may reside on the GPU than fit into one execution batch.
- A batch row temporarily refers to a continuation through explicit indices.
  Excluding a request from a batch does not park it or copy its state.

All identities use generations to reject stale completions. Residency changes
do not create another request or reset its accounting.

The native planning interface is:

```rust
fn advance_to(&mut self, now_ns: u64) -> Result<()>;
fn plan(&self) -> Option<ScheduledPlan>;
fn commit(&mut self, plan: &ScheduledPlan, fence: Fence) -> Result<()>;
fn complete(&mut self, fence: Fence, records: &[CompletionRecord])
    -> Result<Vec<GeneratedToken>>;
fn published_output(&mut self, request: RequestHandle) -> Result<()>;
```

The worker advances the clock before lifecycle transitions and dispatch. `plan`
returns `None` while dependencies or the submission window prevent execution.
A plan freezes graph shape, lane work, request epochs, estimated duration, and
projected recovery lateness. Planning uses scratch reserved from the loaded
catalog and does not mutate committed token positions. Commit validates epochs,
resource availability, and the remaining submission window, reserves resources,
and records the fence without rerunning the selection policy.

Per-request planned positions and in-flight work are separate from committed
positions. Completion is the only operation that earns useful-work credit,
updates acceptance, and advances committed model state. Cancellation stops
further selection immediately; resources remain owned through their last fence.

## Prepared action contract

A candidate names a prepared operation and explicit work for each participating
request. Inference lanes contain a prefill segment or a decode round with a
chosen draft limit. Preparation, restore, vision, snapshot, and release
operations have explicit prerequisites and costs as well.

Ignition must prepare mixed execution. The mixed shape family contains
one prefill segment and a decode cohort. Its catalog key includes:

```text
prefill token capacity
decode lane capacity
verification width capacity
proposal/state-maintenance mode
```

Zero prefill and zero decode lanes describe the corresponding pure operations
where the prepared implementation supports them. Each descriptor declares its
physical padding, legal active lane counts, valid token extents, state indices,
scratch requirements, and completion layout. Unsupported combinations are not
candidates. Mixed graph capture includes proposal generation, target execution,
acceptance, state commit, output projection, collectives, and sampling.

Every selected decode lane has an explicit draft limit. A rectangular capture
may still pay the maximum width across its lane capacity. Smaller per-request
limits do not imply cheaper execution. Ragged execution earns its lower cost
only when the captured graph actually avoids the work. Width-one execution
retains any draft state needed to resume wider execution and uses the ordinary
decode attention/cache path.

The prefill catalog should expose a range of small and large capacities through
the configured maximum. Candidate active lengths respect atomic attention spans,
cache boundaries, and model-specific alignment. A 64-token logical segment in
a 128-token capture pays the measured cost of that physical execution. Choosing
64, 256, or 1024 tokens is a dispatch decision over measured shapes.

The capture catalog is finite and prepared before serving. A useful starting
catalog includes pure decode at supported batch/width combinations, pure
prefill capacities, and mixed variants around measured throughput knees and
latency limits. The capture-memory and preparation-time cost of a full Cartesian
product must be measured before selecting the mixed catalog. This design does
not require runtime graph capture, compilation, Python callbacks, or GPU
allocation during serving.

## Cost and completion observations

Each graph descriptor has one `estimated_duration_ns` field. Zero means
indeterminate, including the Protobuf default. Ignition exports zero and does
not collect replay timings during preparation. The native scheduler initializes
every graph to zero, including descriptors supplied with a nonzero field.
Physical batch and token capacities identify separate graphs; context length
and active lane count do not create separate estimates.

The first valid completion measurement sets a graph's estimate directly. Further
observations before expiry update it with weight 1/8:

```text
estimate_ns <- estimate_ns + (observed_graph_ns - estimate_ns) / 8
```

Each graph's estimate expires after `estimate_ttl_ns` without a valid observation,
returning zero to dispatch. The observation following expiry starts a fresh EMA;
it does not blend with the expired estimate. The default expiry is 30 seconds.

The observation is the maximum reported rank-local GPU replay interval. CUDA
events bracket graph execution, excluding device queue wait, input upload, and
completion readback. Missing or zero measurements neither change the estimate nor
refresh its observation time. Host dispatch and publication intervals remain
separate diagnostics.

Indeterminate candidates take precedence over determinate candidates within the
set retained by the service-age, projected-lateness, and deficit guards. They
compete by expected useful work rather than a duration ratio. A lone request
therefore chooses its largest useful prefill chunk or speculative depth when
all applicable costs are indeterminate. The same efficiency score favors more
useful work when known costs are equal. There is no single-request override.
Logical output limits, cache boundaries, and physical padding still constrain
useful work.

An indeterminate candidate uses zero in the optimistic latency projection but
requires an empty submission queue and completion feedback before another
submission. It cannot be appended as free work. First observation establishes
its execution cost, so genuinely slow shapes lose the comparison until expiry
permits another attempt. Shape applicability and request-service guards still
restrict selection; expiry is not a guarantee that every graph will run.

Request credit rates retain their normalization units independently of
prediction freshness. Expiry changes dispatch costs without erasing earned
credit or resetting deficits.

Completion reports, per request:

- Actual consumed prefill tokens and committed output positions.
- Accepted draft prefix, explicitly observed rejection if any, and censored
  termination at EOS, an output limit, or a host stop.
- The actual proposal source at each evaluated depth when sources are combined.
- Completion status and the fence protecting state and scratch ownership.

An offered-but-unaccepted position is not automatically a rejection: it may
never have been evaluated. Acceptance estimates belong to each request and
proposal source, with bounded conditional statistics by depth. A shared prior
can initialize a request; observations from unrelated requests do not replace
its own acceptance profile.

## Accounting and clocks

Reference rates freeze at admission when runtime measurements are available.
Output credit prefers the smallest observed batch and ordinary decode, followed
by the narrowest observed selected speculative width. Its per-row cost supplies
the reference unit. With only batched or speculative observations, this is an
observed normalization unit rather than an isolated ordinary-decode measurement.
Prefill credit uses the largest observed selected chunk repeatedly, then the
smallest observed capture covering the tail. Dividing that reference traversal
by uncached prompt length gives a fixed per-token rate. Atomic spans and system
boundaries constrain actual execution without changing the normalization.

If a reference rate is unknown at admission, it remains unset until an applicable
measurement arrives. Committed token counts accumulate in the meantime. Setting
the rate credits that accumulated progress exactly once, and the rate then stays
fixed for the request. Selection uses unit token progress for an unset rate,
allowing cold requests to compete without assigning them a fabricated graph
duration. Cache hits earn no credit for reused tokens; initialization, restore,
vision, and snapshots earn no committed credit.

For a GPU-runnable resident request, maintain a reflected deficit:

```text
L_i <- max(0, L_i + elapsed_occupied_time / runnable_count - committed_credit_i).
```

`advance_to` integrates time while any submission is outstanding, once across
overlapping frames. This is the host-observed occupied interval from commit to
completion consumption, including host delay while work is outstanding. It is
not the sum of GPU event intervals. The worker advances accounting before arrivals,
cancellation, completion, and host-store readiness change the runnable set.
Requests excluded from the selected batch remain entitled. Inbound dependencies,
finished requests, and terminal cleanup do not accrue entitlement. Actual
completion subtracts only committed prefill/output credit; failed work earns none.

Each request retains arrival time, latest useful model progress, and latest
published output. Prefill uses the progress clock; decode uses the publication
clock, or arrival before the first publication. Setup and snapshot operations do
not reset either clock. A prerequisite inherits the urgency of the progress it
enables. Stop-prefix buffering can delay publication after committed tokens have
already earned credit.

Publication means a token or protocol chunk successfully entered the native
streaming queue. Observation-only events and failed sends do not reset the clock.
Network delivery and client consumption occur later. A full output queue fails
and cancels the request through the existing bounded-output contract.

For a determinate cost, the native efficiency score is:

```text
sum_i (1 + max(0, L_i - pending_credit_i) / deficit_scale_ns)
      * expected_useful_credit_i / estimated_duration_ns.
```

Expected useful credit includes committed prefill/output credit and one extra
output-credit unit for a first output from prefill or resume. That bonus affects
selection only. Prerequisites score the useful progress they enable; terminal
snapshots score one eighth of an output-credit unit to release resources.
Already submitted progress reduces selection pressure but earns no actual credit
until completion. Explicit output limits cap speculative yield and pipelined
decode; EOS is not predicted.

The Python simulator uses a different first-output bonus: one deficit-scale unit
per completed prefill. Comparison of these first-output preferences, including
exact prefix hits, requires an ablation with equivalent action catalogs.

## Dispatch algorithm

At each dispatch opportunity:

1. Enumerate loaded legal graph shapes. Prefill considers every capture that
   can advance the request while respecting system boundaries and atomic spans.
   Cohorts of up to five eligible requests enumerate every subset. Larger groups
   include every singleton, the full group, and power-of-two prefixes ordered by
   service age, deficit, acceptance in both directions, and context position.
   Every cohort considers every selected verification width.
2. Filter by graph slice duration, submission-window duration, and feedback
   dependencies. If a request's cheapest atomic singleton exceeds a duration
   budget, allow it with an empty queue so a valid request can still progress.
3. Require the oldest overdue eligible request with no pending submission.
   Its next prerequisite qualifies as service for selection, but does not reset
   its useful-progress or publication clock.
4. Project the candidate followed by singleton rescues in service-deadline order.
   Include outstanding queued duration and mandatory stages through the next
   useful progress. Keep zero-lateness candidates when possible; otherwise keep
   candidates with minimum summed positive lateness. The plan exposes this sum
   as `recovery_lateness_ns`.
5. Within that set, require the largest deficit at or above `deficit_scale_ns`
   when possible. Prefer indeterminate candidates by expected useful work;
   otherwise maximize the efficiency score. Break ties by shorter duration,
   less recent service, smaller token capacity, and physical request-slot order.
6. Commit and enqueue the immutable selection. Completion updates committed
   progress, request-local acceptance, and the graph's duration estimate.

Conditional acceptance at each draft depth gives expected speculative output:

```text
E[output(k)] = 1 + sum_{j=1..k} product_{l=1..j} p_l.
```

High-acceptance cohorts compete with broad shallow cohorts using their actual
prepared graph costs. A high-yield pair can win when its efficiency is better,
and lose when excluding other requests would make their rescue late. There is
no independent width controller. Estimate expiry supplies opportunities to
rediscover neglected shapes through ordinary work selection.

## Committed execution window and latency

`PolicyConfig` supplies positive nanosecond values before admission:

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_slice_ns` | 20 ms | Maximum selected graph duration, except unavoidable singleton work |
| `submission_window_ns` | 20 ms | Maximum estimated outstanding execution |
| `service_gap_ns` | 40 ms | Target interval between prefill progress or published output |
| `deficit_scale_ns` | 20 ms | Deficit weight scale and forced-service threshold |
| `estimate_ttl_ns` | 30 s | Time without a valid observation before a graph cost becomes indeterminate |

These defaults are policy settings, not measurements. The worker uses the default
configuration. The Rust interface permits explicit settings before admission;
CLI/configuration-file overrides are not wired.

The scheduler tracks expected finish times of at most two submitted frames.
A second frame must fit behind the remaining estimated work. An uncompleted
oldest frame past its expected finish stops further submission until completion
is consumed. A selected plan retains its duration through cost updates, and
commit rechecks the window when reserving execution.

Ordinary unconstrained decode can pipeline within the window. Speculation,
grammar, and host-stop work use the executor's global feedback barrier: they
start with an empty queue and prevent further submission until completion.
Allowing independent requests past those barriers requires finer executor
readiness tracking. Prefill and mandatory state operations require each selected
request to have no pending work.

The deadline check is a single future visit with serial singleton rescue work.
It does not prove recurring feasibility and can reject a candidate that a batched
rescue could save. Batched rescue construction and a second visit are required
extensions for evaluating costly residency changes. Neither estimated graph
costs nor this finite search establish a hard latency bound. First-token latency
also includes the request's own remaining prompt work.

## Residency contract requiring implementation

Acceptance into the bounded logical queue is distinct from acquiring resident
resources. All accepted requests retain arrival age. A new request competes for
available continuation storage at dispatch boundaries even when every execution
batch row was occupied in the previous action. Idle batch rows are not required
for admission when separate resident continuation storage is available.

On genuine memory pressure, evict unpinned reusable prefix-cache entries first.
If that cannot make room, residency rotation must be available. Select the
oldest overdue residency waiter. Park only requests at committed boundaries,
with no pending use or copy, and with at least one useful service opportunity
since their last activation. Prefer victims with the smallest useful-work
deficit; among comparable victims prefer cheaper save/restore cost. A request
that has just become resident cannot be parked repeatedly without progress.
Reconsider victim choice against upcoming output deadlines and the complete
transfer cost. Conflicting overdue obligations enter the same explicit recovery
mode as conflicting compute deadlines.

A parked request preserves KV, recurrent state, draft context, pending target
input, RNG state, committed token positions, output delivery cursor, and grammar
state. Host-resident grammar and output state stay attached to the logical
request. Checkpoint backing and bounded transfer buffers must be reserved before
parking begins. Live state cannot be discarded as a substitute for making room.
Admission must respect both GPU resource capacity and available checkpoint
backing; each supported request must fit when given an isolated residency turn.

Retain full remaining-output page reservations for resident requests. This preserves an explicit progress invariant
while dispatch and residency ownership change. Parking returns its GPU
reservation after fenced save; restoring reacquires the remaining reservation.
Incremental KV growth would require a separate liveness argument and is not
necessary to implement the proposed dispatch policy.

The residency controller is an implementation requirement for fairness when the
active working set cannot remain on the GPU. Its transfer and admission policy
needs simulation with real state sizes and costs. A guarantee restricted to
resident requests cannot be presented as a bound for the accepted request queue.

## Background interference requiring integration

Initialization, resume, restore, and vision operations inherit the urgency of
the next useful progress or output they enable. Completion-time cleanup that
releases resources for an overdue request also has a named beneficiary. Account
for their time in the execution window and projected continuation.

Optional prefix-cache publication and background transfers run only with
available latency and bandwidth slack. Running them on another CUDA stream does
not make their interference free. Mandatory publication needed to preserve
state ownership is distinguished from optional cache retention. There is no
unconditional phase-priority ladder that can indefinitely postpone runnable
prefill or decode work.

## Rust and Ignition integration

The native integration points are:

- `helios/src/scheduler.rs`: logical request state, residency/resource ownership,
  plan/commit/complete, and request lifecycle.
- `helios/src/scheduler/policy.rs`: accounting, candidate generation, latency
  prediction, joint selection, and the submission window.
- `helios/src/scheduler/speculation.rs`: per-request/source acceptance models and
  expected-yield calculation used by joint action selection.
- `helios/src/execution/manifest.rs`: mixed graph descriptors, physical capacity
  semantics, legal lane operations, and state-indirection requirements.
- `helios/src/execution/transport.rs` and rank executors: timing for every action,
  committed-window observations, and explicit evaluated/censored draft outcomes.
- `helios/src/server/runtime.rs`: monotonic arrival and output clocks, bounded
  logical queue, completion publication, and passing measured state to planning.
- Ignition recipes and reusable preparation components: complete mixed forwards,
  bounded workspaces and capture catalog in the runtime package. Prepared package limits continue to bound native execution.

Native selection uses prepared pure graphs. Pure execution is also a useful measured candidate after mixed
graphs exist. No synthetic mixed cost is substituted for a missing native graph.

## Verification and remaining decisions

Native deterministic traces exercise the actual request scheduler and worker
publication path. A shared serialized trace format for Python/Rust differential
testing remains to be implemented. The tests
must include arrivals during execution, state-changing completion, cancellation
between plan and commit, EOS censoring, resource exhaustion, phase transitions,
and a losing latency continuation. Randomized workloads must preserve accounting
and fenced ownership, not merely finish without errors.

Extend simulation for the committed window, mandatory stages, separate output
and prefill clocks, residency rotation, and prediction error. Add ablations for
first-output treatment of prefix hits and batched rescue work. Those extensions
are required before attributing the simulator's resident-policy results to the
complete native design.

Validate mixed captures against equivalent separate execution for logits,
accepted tokens, rollback, recurrent state, inactive rows, cache-boundary
crossings, width-one transitions, and alternating graph variants. Repeated
native replay must preserve allocation stability and scratch/input lifetimes.

Through the actual service, measure queue delay, TTFT by prompt size, burst-gap
tails, first-token-correct delivery slowdown, useful-work deficits, target misses,
uncertified decisions, GPU step time, host dispatch time, transfer interference,
and aggregate output rate. Compare pure/mixed execution and shallow/deep
speculation at equivalent workload and resource settings. Synthetic results
cannot select numerical production budgets.

The dispatch ordering, accounting quantity, joint action selection, and ownership
contract are concrete implementation choices. Numerical latency budgets, capture
catalog density, prediction margins, and residency-rotation tuning require
measurements from prepared models. Hard recurring latency guarantees remain
unproved; neither a successful finite workload nor a two-visit rescue prediction
establishes them.
