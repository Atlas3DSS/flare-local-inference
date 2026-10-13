# Request scheduling by useful work and service value

## Execution model

An interactive inference scheduler chooses a prepared graph and the requests
that occupy its lanes. A graph is a nonpreemptible service quantum. Its prefill
capacity, decode row capacity, verification width, and duration are properties
of the prepared execution. The scheduler chooses among these quanta; it cannot
choose an arbitrary time slice independently of the work.

The fairness entity is one request. Requests have known remaining prompt work,
observable generated output, an explicit output limit, and unknown EOS.
Speculative acceptance is stochastic. A tensor-parallel GPU group is one
execution resource when its ranks execute a model step together.

The Python implementation in [`scripts/scheduling_sim`](../scripts/scheduling_sim/)
contains a fixed-value dispatch baseline, a discrete-event simulator, and small
mathematical references. The [ongoing-service experiment](scheduling-utility.md)
adds proportional-fair rate histories and a decode-age continuation value.
The [completion-episode oracle](scheduling-oracle.md)
also compares exact conditional schedules with a continuation-value prototype.
The models use positive, known graph durations and supplied
conditional acceptance probabilities. Actual output lengths are hidden from
the dispatcher. It has one graph in flight and publishes output at completion.
It does not model cache capacity, admission, restore, host prerequisites,
submission pipelining, timing discovery, or runtime estimation error. Its graph
timing table is synthetic unless supplied with `--catalog`.

The [native scheduler design](scheduling-runtime-design.md) applies the ongoing
service utility to Rust execution, including prerequisite operations, timing
discovery, and queued submissions. Python simulation results do not establish
native behavior or GPU performance.

## Connection to CPU scheduling

CPU fair scheduling distinguishes a request's entitlement from the granularity
at which it executes. CFS models an ideal processor shared among runnable tasks
and accounts for actual service with virtual runtime. EEVDF uses service lag
to determine eligibility and virtual deadlines to order eligible tasks. These
are useful foundations for accounting, rather than reasons to assign a real-time
deadline to each prefill chunk. See the Linux documentation for
[CFS](https://docs.kernel.org/scheduler/sched-design-CFS.html) and
[EEVDF](https://docs.kernel.org/scheduler/sched-eevdf.html).

For inference, the analogy has three qualifications:

- A decode stream remains runnable after producing output. Unlike an interactive
  CPU task blocked on input, the stream need not sleep between useful responses.
- Service cost is nonadditive. If two singleton decodes cost 15 ms each and their
  batch costs 17 ms, dividing those 17 ms between requests is an accounting
  convention, not a measured physical quantity.
- The quantum changes efficiency and yield. A wider speculative graph can take
  longer and produce disproportionately more output. A larger prefill can
  amortize graph overhead and use the hardware more efficiently.

A fair-service reference must therefore include the feasible batch rate vectors.
After defining that reference, a runtime policy can track deviation from it.
Copying CPU virtual-runtime arithmetic before defining service units leaves the
central allocation question unanswered.

Prefill has no externally visible per-chunk cadence. Its visible milestones are
first output and eventual completion. Decode has both useful token throughput
and a delivery cadence. Equal mean rates do not imply equal output gaps.

## One metric for graph selection

Let an action `a` take expected time `T(a) > 0`. For request `i`, let `P_i(a)`
be the actual prompt tokens consumed and `D_i(a)` the expected output tokens
emitted. A fixed-value baseline chooses the largest

```text
score(a) = sum_i w_i * (P_i(a) + beta * D_i(a)) / T(a).
```

`w_i` is a request weight, equal to one in the supplied workloads. One prefill
token has value one; `beta > 0` specifies the intrinsic value of an output token.
The CLI requires this value explicitly. The simulator does not infer a user
preference from hardware cost or select a production default.

All pure prefill, pure decode, and available mixed graphs compete on this score.
No duration ceiling, prefill size preference, or breadth-before-depth rule
precedes it. A graph consuming 64 useful prompt tokens earns 64 prefill units
whether its capacity is 64 or 4096. Inactive rows and rejected drafts earn zero.
Completing prefill emits one first token, counted once in `D_i`.

If conditional acceptance at draft position `j` is `q_ij`, a verification width
`k` has expected output, before EOS truncation,

```text
D_i(k) = 1 + sum_{j=1}^{k-1} product_{m=1}^{j} q_im.
```

The pending target token provides the first output; a draft contributes only
when the accepted prefix reaches it. The explicit remaining output budget caps
this expectation. A length-distribution model could additionally price hidden
EOS, but the fixed-value baseline does not have one.

For a fixed graph duration, fixed weights, and additive lane rewards, its best
decode cohort consists of the highest-valued rows up to capacity. Consequently,
ranking per-request contributions is exact for this baseline. A nonadditive
reward, context-dependent cohort timing, or an objective involving excluded
requests can invalidate that reduction.

This score gives the desired elementary behavior under appropriate timings:

- Fully occupied 4096-token prefills win when they have the highest useful
  prefill rate. Two long prefills do not need smaller chunks merely to alternate.
- For a 64-token tail, all sufficiently large graphs earn the same prefill work
  and first-token benefit. The fastest eligible graph wins.
- A wide speculative round wins when its additional expected accepted output
  outweighs its additional time. A 30 ms round is not excluded merely because
  a 17.3 ms alternative exists.
- A mixed graph with a smaller prefill can win on combined useful work per time.
  There is no universal crossover at 64, 256, 512, or 1024 tokens.

The invariants concern useful work and measured efficiency. They do not require
4096 tokens when a smaller graph has demonstrably better effective efficiency.
Likewise, a padded larger graph remains eligible if it is actually faster.

## What immediate value does not establish

Maximizing one action's expected value per time is not a proof of globally
optimal completion time, fairness, or interactivity.

Three executable counterexamples are in
[`tests/test_scheduling_sim.py`](../tests/test_scheduling_sim.py):

1. With prefill capacities 64, 128, and 256 costing 2, 3, and 4 ms, a 129-token
   prompt and `beta=1` select 128 tokens first. The remaining token needs another
   graph, taking 5 ms total. The padded 256 graph would finish in 4 ms.
2. A continuously backlogged decode with sufficiently high constant token value
   prevents a prefill from receiving any service over the tested horizon.
3. Five identical low-acceptance streams using batch four receive
   `[10, 10, 10, 5, 5]` tokens over ten rounds under oldest-service tie-breaking.
   Rotating which request was served most recently does not track accumulated
   entitlement.

A fixed set of completed requests earns a fixed total token reward. Maximizing
that reward divided by the time to finish all requests minimizes makespan;
request completion order and output cadence remain unspecified. Mean time to
first token and mean completion time are separate objectives.

For example, alternating full chunks between two equal long prefills and
finishing one before starting the other can have the same total prefill time.
Finishing one first improves mean time to first token if subsequent decode does
not alter the schedule. Alternation gives more even prompt progress. Choosing
between them requires a completion or fairness objective, not a different
hardware-throughput formula.

## A fair-allocation reference

For a frozen active set and unchanged execution conditions, express action `a`
as a vector of useful per-request rates:

```text
y_i(a) = E[useful tokens for request i in a] / T(a).
x_i = sum_a theta_a * y_i(a),   theta_a >= 0,   sum_a theta_a = 1.
```

`theta_a` is a fraction of wall time, not a fraction of graph launches. These
mixtures form the feasible rate region. The reference assumes indefinitely
backlogged requests with a fixed phase; finite prompt tails and phase changes
require a dynamic model.

A concrete family of allocation objectives is

```text
maximize F(x) = sum_i [v_i * x_i + f_i * log(x_i / x_ref_i)].
```

Here `v_i >= 0` is intrinsic token value, `f_i > 0` is a fairness coefficient,
and `x_ref_i > 0` fixes the units of the logarithm. All coefficients are supplied
explicitly to the reference. With `v_i=0`, this is weighted proportional
fairness. Concave rate utility and proportional fairness have established
foundations in [Kelly, Maulloo, and Tan's rate-control formulation](https://www.statslab.cam.ac.uk/~frank/rate.pdf).

The graph selector associated with the objective's gradient has the same
work-over-time structure:

```text
q_i = dF/dx_i = v_i + f_i / x_i
best action = argmax_a sum_i q_i * E[useful tokens_i(a)] / T(a).
```

This is a supporting-direction calculation for the allocation problem. It is
not yet an online algorithm: deciding how to estimate `x_i`, initialize arriving
requests, or retain service history changes transient behavior.

The reference has several useful properties:

- On a CPU-like resource that runs one request at a time at rate `r_i`, pure
  proportional fairness gives time share `theta_i=f_i/sum_j f_j`, independent
  of `r_i`. It recovers weighted processor sharing without equating token rates.
- Batch economies appear directly in `y(a)`. They require no invented division
  of a graph's physical time between its rows.
- Positive fairness coefficients keep every feasible request's reference rate
  positive. They establish no finite output-gap bound.
- The objective is increasing in every request's rate. A feasible allocation
  that is worse for every request cannot be optimal. This excludes wasted
  efficiency without requiring the largest physical graph on every dispatch.
- The intrinsic value term and fairness term have distinct meanings. Replacing
  `log(x_i)` with `log(beta_i*x_i)` adds a constant and cannot make decode more
  valuable. A unit conversion inside the logarithm is not a phase preference.

[`analysis.py`](../scripts/scheduling_sim/analysis.py) implements `fair_allocation`
with pairwise conditional gradients and scalar line search. It returns time
shares, rates, utility, and a concavity gap that bounds utility suboptimality.
It fails explicitly if it cannot meet its requested numerical tolerance. This
is an offline reference over caller-supplied action vectors, not the dispatcher.

For five equal streams, suppose a four-row graph takes 21 ms and a graph holding
all five takes 29 ms. Uniformly rotating the five possible four-row cohorts
achieves `4/(5*21)` tokens/ms per request, versus `1/29` for the larger graph:
38.10 versus 34.48 tokens/s each. The reference selects the efficient smaller
batch and equal shares. It does not require sacrificing either fairness or
mean rate in this case. A delivery-gap objective could still distinguish these
allocations.

A runtime lag can eventually be defined as reference progress minus delivered
progress, integrated over the intervals when a request is runnable. Defining
how that reference changes on arrivals, departures, phase transitions, and
cost revisions is necessary before turning it into a dispatch weight.

## Stream age and nonpreemptible execution

An intrinsic token value rewards useful volume. Ending an output stall is an
additional benefit associated with a delivery event. Producing eight tokens at
one instant does not end eight distinct waiting intervals.

Let `A_i(t)` be time since the last published output for a decoding request.
A family of explicit interactivity costs is

```text
integral sum_{i in decode} c_i(A_i(t)) dt,
```

where `c_i` is increasing. A linear age cost accumulates during a graph as

```text
integral_0^T (A_i + u) du = A_i*T + T*T/2.
```

This applies to selected and unselected streams: both wait until graph
completion. At completion, the age of a served stream resets once, regardless
of burst size. Requests that arrive during execution also accumulate waiting
from their arrival. Prefill requires a first-token waiting cost if such a cost
is part of the objective; its internal chunk completions do not reset a stream
age clock.

The identity is tested by `output_age_area`. The fixed-value simulator records
output gaps and integrated linear decode age without optimizing this cost.
The completion-episode oracle optimizes explicit time and waiting prices.
An increasing token weight based solely on time since last service is a
possible approximation, not a derivation of
this objective. In particular, multiplying the entire speculative burst by a
stall-clearing bonus can overvalue deep speculation for a neglected stream.

Average rate, accumulated service lag, and output age are separate state
variables. A burst can repay useful-progress lag while still leaving a poor
history of pauses. An age-only rule can repeatedly favor an inefficient stream
without a specified throughput tradeoff. Both effects belong in an explicit
objective before selecting a formula.

## Remaining work and completion value

The prefill token count is known, but its minimum remaining execution time
need not equal remaining tokens divided by one rate. The reference
`prefill_completion_ms` solves the isolated constant-cost case exactly:

```text
H(0) = 0
H(p) = min_{prefill graph g} [T(g) + H(max(0, p - capacity(g)))].
```

It ignores decode competition and assumes prefill timing is independent of
context. Its purpose is to expose greedy chunking errors, not to price an
entire mixed workload as isolated work.

For generated output `g` and a length prior `L`, expected remaining length is

```text
E[L - g | L > g].
```

`conditional_remaining` evaluates this expression for an explicit discrete
prior. A known fixed length gives decreasing remaining work. A geometric prior
has a memoryless residual. A heavy-tailed prior can predict more remaining
work after observing a longer prefix. For a continuous Pareto survival law
with exponent `alpha > 1`, above its minimum length,

```text
E[L - g | L > g] = g / (alpha - 1).
```

Thus an estimate of twice the attained output corresponds to one particular
Pareto exponent, `alpha=1.5`. It is not a distribution-free consequence of
observing that a request is still decoding. Explicit output limits truncate
any applicable prior. Learning a prior must also handle unfinished requests
as censored observations rather than completed short outputs.

Remaining length matters when the objective rewards completion or the future
capacity released by completion. It is not needed to evaluate the immediate
acceptance yield of a graph. Accurate graph timing does not remove this
uncertainty about the request's total duration.

## Dynamic objective and lookahead

Variable-duration decisions with stochastic next states fit a semi-Markov
control model. State includes remaining prompt work, attained output and any
length belief, acceptance information, service accounting, and stream ages.
Actions are complete graph executions. Expected transition reward can include
useful work and integrated waiting costs. This modeling framework and its
average-reward equations are described by
[Baykal-Gursoy](https://gursoy.rutgers.edu/papers/eorms0757-final11.pdf).

Under the conditions for an average-reward optimality equation, a continuing
workload has a relative value `h` and optimal reward rate `rho` satisfying

```text
h(s) = max_a { r(s,a) - rho*T(s,a) + E[h(s')] }.
```

Equivalently, optimal actions maximize

```text
[r(s,a) + E[h(s') - h(s)]] / T(s,a).
```

Dropping the future-state term produces the immediate value baseline. The
remaining term accounts for such effects as unlocking decode after a prompt
tail, ending a stall, completing a request, and changing future batch options.
A practical approximation needs tests against small solvable instances of a
chosen objective. The equation alone supplies neither its utility parameters
nor a latency guarantee. Average reward on an infinite horizon is also
uninformative for a finite batch followed by permanent idleness; finite-horizon
or completion-cost objectives apply to that experiment.

The [completion-episode oracle](scheduling-oracle.md) supplies those small
instances. It evaluates hidden-length beliefs, all legal cohorts, and acceptance
outcomes through completion. Its policy-improvement prototype keeps expected
net work plus continuation-value change over graph time as the dispatch score.
Normalizing negative net episode rewards by episode duration can reward wasted
execution time; that objective has an explicit counterexample. A fixed positive
time price supplies a separate reference for completion and waiting tradeoffs.

For a stationary sequence of actions with fixed rewards, a cycle's value rate
is a duration-weighted average of its action rates:

```text
(sum_j reward_j) / (sum_j time_j)
    = sum_j [time_j / sum_k time_k] * [reward_j / time_j].
```

It cannot exceed its best component. Looking ahead is useful because state,
future value, or fairness changes, not because averaging unchanged rates fixes
a starvation-prone linear objective.

## Comparison with the cycle-based simulator

The separate worktree `.claude/worktrees/scheduling-sim/` contains
`scripts/schedsim/policies.py::LeximinPolicy`. It projects in-flight work,
enumerates prefill, decode, and mixed choices, and compares short action cycles
through phase completion. Its primary ordering is projected request slowdown;
throughput breaks ties after feasibility filters that include a stream-gap
limit.

Its useful contributions are explicit future phase transitions, conditional
speculative yield, and comparisons extending beyond each graph's own completion
instant. Its lexicographic slowdown objective and stream-gap filter do not
implement the value-rate objective specified here. Copying those filters would
reintroduce a priority stronger than measured efficiency.

Cycle projection and an isolated-completion reference can test dynamic value
approximations. Neither establishes a universal breadth-before-depth rule,
a preferred prefill capacity, or a fixed graph-duration ceiling.

## Running the experiments

From the repository root:

```sh
python -m pytest tests/test_scheduling_sim.py -q
python -m scripts.scheduling_sim --decode-values 1,16,64,256 \
  --output /tmp/flare-value-separate
python -m scripts.scheduling_sim --decode-values 1,16,64,256 --mixed \
  --output /tmp/flare-value-mixed
python -m scripts.scheduling_sim --workload prefill_five_decodes \
  --decode-values 64 --catalog /path/to/graph-times.json
```

A catalog is a JSON array. Durations are milliseconds for complete graph
execution. Unused capacities remain part of its measured cost:

```json
[
  {"name": "prefill-64", "duration_ms": 3, "prefill_capacity": 64},
  {"name": "prefill-4096", "duration_ms": 66, "prefill_capacity": 4096},
  {"name": "decode-b1-w8", "duration_ms": 30, "decode_capacity": 1, "width": 8}
]
```

The output directory contains aggregate measurements in `summary.csv`, individual
request measurements in `requests.csv`, selected actions and their scores in
`trace.jsonl`, and the timing input in `catalog.json`. The decode value and
random seed accompany each case. A graph started before the requested horizon
finishes in full; elapsed time can exceed that horizon by one graph.

The sixteen workloads cover solo and concurrent prefills, prompt tails,
low/high/heterogeneous acceptance, up to sixteen requests, mixed phases, delayed
arrivals, and different output lengths. Tests include exhaustive cohort scoring,
conditional acceptance sampling, padding, explicit output limits, uniform time
scaling, starvation counterexamples, exact prefill completion, and analytic
fair-allocation cases. Metrics include incomplete requests, integrated linear
decode age in milliseconds squared, and open output stalls at the measurement
boundary. A request already decoding at arrival has zero initial age; its wait
until the first simulated delivery counts. Prefill contributes no decode-age
cost until its first output. Busy time means execution-resource use,
not GPU occupancy or hardware efficiency.

## Runtime policy choices

The ongoing policy uses separate intrinsic output value, token fairness,
delivery fairness, and decode-age coefficients. Their concrete defaults and
units are specified in the [native design](scheduling-runtime-design.md).
First-token waiting has no independent age penalty: prompt work competes by
its useful progress and token-service price. A burst earns its accepted token
volume and one delivery. The delivery reference accounts for feasible batching;
changing a rate normalizer preserves marginal service prices across phases.

The runtime uses no EOS-length prior. Exact conditional completion schedules
remain an offline reference. Native prerequisite selection uses bounded
lookahead to include a following shared inference graph, with individual
continuation values for longer chains. The stationary allocation results do not
prove that approximation optimal. The runtime also projects two queued frames
into service history and pauses dispatch at feedback dependencies.

Unknown duration cannot be divided into reward as a numerical zero. The native
contract uses zero to mean indeterminate, requiring an explicit exploration
choice. Expiration makes a stale estimate eligible for rediscovery; it does not
prove efficient or fair behavior during exploration. Those effects require
tests beyond known-cost allocation. Native tests cover zero-cost startup,
expiry, recovery, and phase transitions; real-device execution is still required
to measure their throughput and latency cost.
