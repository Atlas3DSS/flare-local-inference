# Ongoing service utility and decode-age continuation

[`scripts/scheduling_sim/utility.py`](../scripts/scheduling_sim/utility.py)
implements a known-cost scheduling experiment for ongoing requests. It combines
intrinsic useful work, proportional-fair service prices, and a continuation
value for output age. The corresponding Rust arithmetic and fixed-shape cohort
selection are in
[`helios/src/scheduler/utility.rs`](../helios/src/scheduler/utility.rs).
Native completion and publication update these service prices through
[`helios/src/scheduler/service.rs`](../helios/src/scheduler/service.rs).
Native dispatch uses these prices and the decode-age score in
[`helios/src/scheduler/policy.rs`](../helios/src/scheduler/policy.rs).
The prepared catalog supports separate prefill and decode execution.

The model has one nonpreemptible graph in flight. Prepared shape costs are fixed,
positive, and supplied. Acceptance probabilities are conditional on accepting
the preceding prefix. Output limits are observable; actual EOS is hidden.
Arrival, departure, and phase transitions are simulated, but the stationary
allocation results below assume a frozen set of backlogged requests.

## Progress and delivery are different services

Let `x_i` be request `i`'s useful-token rate, counting consumed prompt tokens and
emitted output tokens. Let `z_i` be its output-delivery rate while decoding.
A burst of eight accepted tokens earns eight output units and one delivery.
Rejected drafts, padding, and inactive rows earn neither service. An unfinished
prefill chunk earns prompt progress and no delivery.

The stationary rate utility is

```text
F = intrinsic useful work per millisecond
    + sum_i f_i log(x_i / m_i)
    + sum_{i in decode} g_i log(z_i / d_i).
```

Intrinsic work is `sum_i w_i(P_i + beta D_i)`. The constants `m_i` and `d_i`
are maximum individually feasible token and delivery rates from the catalog.
They normalize the logarithms; they do not set intrinsic token values. Holding
the catalog fixed, these normalizations only add constants to `F`.

`beta` prices an output token relative to a prompt token. `f_i` and `g_i` price
token-rate fairness and delivery-rate fairness in value units per millisecond.
The experiment uses `f_i = w_i * token_fairness` and
`g_i = w_i * delivery_fairness`. A zero coefficient disables its utility term.

The gradient prices are

```text
prompt price_i = w_i + f_i / x_i
output price_i = w_i * beta + f_i / x_i
delivery price_i = g_i / z_i.
```

Summing these prices times expected service and dividing by expected graph time
gives the supporting-direction calculation for `F`. Decode value remains in
the calculation; it does not cancel as it does in a finite episode where every
request must finish and collect a fixed total reward.

Gradient scheduling with averaged rates has an asymptotic foundation for fixed
service slots and sufficiently slow adaptation. See
[Stolyar's gradient scheduling analysis](https://stolyar.ise.illinois.edu/pfgen_or.pdf).
The simulator uses variable-duration graph rounds. Its finite adaptation window,
phase transitions, and age continuation require additional validation; the
fixed-slot result is not a proof for the combined dispatch policy.

## Exponential and cumulative rate references

For an execution lasting `T`, update the normalized token-rate history with
actual consumed and emitted tokens `Y_i`:

```text
r_i <- exp(-T/tau) * r_i
       + (1 - exp(-T/tau)) * Y_i / (m_i * T).
```

Delivery history uses the same update with `Y_i` equal to zero or one and
normalizer `d_i`. This estimator spreads service over the occupied interval
for rate accounting. Output-age accounting still places delivery at completion.
Equal-rate intervals compose consistently when split into different launch
durations. A launch-count EMA would not have that property.

New token histories start at `1 / active_request_count`. Changing the normalizer
from `m` to `m'` rescales a stored fraction by `m/m'`, preserving the physical
service rate and its marginal price. This includes the prefill-to-decode
transition, changes in speculative acceptance, and timing updates. Changing a
normalization constant inside a logarithm must not itself reprice a token.

The first visible output initializes delivery history to one delivery over its
measured execution interval. A request already decoding without an observed
delivery starts from the batch-feasible delivery reference defined below.
Token and delivery histories then persist until the request departs.

The implementation stores logarithms of rate fractions. An unserved interval
decreases log rate by `T/tau`, so long stalls remain representable. All candidate
values share a positive scaling factor to avoid overflowing their reciprocal
rate prices. Traces record the scaled `selection_score` and its
`selection_log_scale`; comparisons use a common scale for the whole decision.

`tau` controls history, not graph eligibility. Every legal graph remains a
candidate regardless of whether its duration exceeds `tau`.

`--rate-history cumulative` instead keeps the time-weighted mean of all observed
normalized service, initialized with `tau` milliseconds of prior service at
fraction `1/n`. Its reciprocal grows linearly during a long unserved interval.
It retains lifetime history, including across changes in the active set.

## Service-weighted prices

`--rate-history service` implements entropy mirror descent on the rate-utility
dual. It is also the price update implemented by Rust `ServicePrice`.
Let `u_i` be normalized service rate and `q_i` its price. Suppressing constants
from logarithm normalization, the dual objective for a frozen service region is

```text
G(q) = max_{feasible u} [intrinsic value rate(u) + sum_i q_i*u_i]
       + sum_i f_i * (log(f_i/q_i) - 1).
```

The maximizing service vector gives a subgradient
`g_i = u_i - f_i/q_i`. Coordinates with zero fairness coefficient are omitted.
Using the entropy mirror map gives the continuous price flow

```text
dq_i/dt = -q_i*g_i/tau = (f_i - q_i*u_i)/tau.
```

For any dual optimum `q*`, the nonnegative relative entropy

```text
B(q*,q) = sum_i [q*_i log(q*_i/q_i) - q*_i + q_i]
```

satisfies `dB/dt = -(q-q*) dot g / tau <= -(G(q)-G(q*))/tau` along the flow.
This provides a common optimization basis for the price update and graph
selection; an arbitrary time-since-service multiplier is unnecessary.

A nonpreemptible graph holds its selected service rates through its duration.
For observed normalized service rate `u`, the price update integrates the flow
for that held action exactly:

```text
q' = q*exp(-u*T/tau) + (f/u)*(1-exp(-u*T/tau)),   u > 0
q' = q + f*T/tau,                              u = 0.
```

The scheduler uses actual service for this update. Selection uses expected
accepted tokens. Output volume and delivery count remain separate coordinates.
The initial token price is `f*n`, corresponding to normalized rate `1/n`.
Rust stores the physical marginal price `p=q/m`, divided by its coefficient,
as `log(p/f)`. For physical service rate `y`, its equivalent update is

```text
dp/dt = (f - p*y)/(m*tau).
```

The normalizer controls adaptation speed but cannot change `p` without an
observed service interval. Python stores the normalized fraction and rebases it
when `m` changes, preserving the same physical price.

An unserved interval increases price linearly. Received service turns over its
history. Unlike an exponential rate estimator, this rule does not exponentially
inflate prices merely because a full service round takes longer than `tau`.
For sixteen requests, fifteen 66 ms prefills, and `tau=100 ms`, an unserved
request's price changes from `16f` to `25.9f`. The exponential rate rule changes
it from `16f` to `16f*exp(9.9)`.

During a graph, the action that maximizes value under the changing prices may
differ from the already running action. That difference is a nonnegative
selection error in the dual calculation. The continuous-flow inequality does
not establish an exact bound for finite graph durations, sampled acceptance,
phase transitions, or age-aware dispatch. Stationary tests compare both
exponential-rate and service-weighted controllers with the same offline rate
optimum; neither changes the supplied utility coefficients or graph timings.

## Pricing the delay caused by a graph

Rate utility distinguishes token volume from delivery frequency, but an average
delivery rate alone does not specify individual gaps. Let `A_i` be elapsed time
since request `i`'s last output and `c_i` a price per millisecond squared. Its
linear age cost during a nonpreemptible graph is

```text
c_i * (A_i*T + T*T/2).
```

Selected and excluded streams both wait through the graph. A selected stream's
age resets once at completion. Prefill has no decode-age clock.

Consider a reference that independently chooses an action each round from a
fixed distribution. Let `p_i` be its probability of serving stream `i`, and
`mu` its mean round duration. Then

```text
sigma_i = mu / p_i
h(A) = -sum_i c_i * sigma_i * A_i
```

is the age-dependent part of its average-reward continuation value. To verify
it, substitute the transition `A_i' = 0` on service and `A_i' = A_i + T`
otherwise. The coefficient of each `A_i` in
`E[reward + h(A') - h(A)]` is
`c_i * (-mu + sigma_i*p_i) = 0`.
The remaining constant is the reference's net value per time multiplied by
`mu`. This identity holds when graph durations and service indicators are
correlated; the derivation does not replace their joint moments with independent
means.

For `K` decoding requests sharing a graph family, the reference chooses the
graph minimizing `T/min(batch_capacity,K)`. Each round independently selects a
uniform cohort of that many requests. Every stream then has
`sigma = K*T/min(batch_capacity,K)`. Partially occupied graphs use their actual
occupied row count. Speculation yields one delivery per served stream, so
accepted token count does not multiply this reference rate.

The native selector also has resume and terminal-snapshot families. It assigns
each family's wall-time fraction in proportion to its stream count. Within a
family of `k` streams, uniformly sampled cohorts of `b=min(capacity,k)` give
`sigma_i = K*T_family/b`, where `K` counts all delivery-bearing streams.
The action probabilities are the wall-time fractions divided by their durations,
then normalized. This defines a realizable independent reference on frozen
ready families. Prefilling requests do not acquire a delivery clock or enlarge
`K`; their token service retains its own fairness price. Streams blocked by
additional prerequisite stages use an individual future-delivery reference.
That multistage extension is a scheduling approximation.

Using `h` for one-step policy improvement gives

```text
score(a) = [expected priced token and delivery service
            + sum_{served decode i} c_i*sigma_i*(A_i+T)] / T
           - T/2 * sum_{active decode i} c_i.
```

Terms independent of the chosen action have been removed. The stall-clearing
term is paid once per stream, irrespective of speculative yield. Expected work
and waiting are expressed in the same value units before division by time.

For frozen prices, phases, and a fixed reference, this is an average-reward
policy-improvement calculation. A long-run improvement statement additionally
requires the continuation term to vanish after division by elapsed time.
Updating prices, changing the active set, and substituting finite prompt tails
change the reference; there is no global optimality claim for those transitions.
Variable-duration reward and continuation comparisons are instances of the
[renewal optimization framework](https://ee.usc.edu/stochastic-nets/docs/renewal-optimization.pdf).

With no active decoder, the age terms vanish. Full prefills compete by useful
value per time. With active decoders, a smaller prefill can win when the loss in
prompt throughput is smaller than the reduction in priced waiting. No duration
filter or preferred chunk size is involved.

For a fixed physical graph, the excluded-stream cost depends on its duration,
not cohort membership. Its best cohort therefore contains the largest additive
lane values. Rust uses a bounded array to select these lanes; it does not
enumerate joint speculative outcomes or every subset.

## Native completion accounting

Each committed submission records the resident runnable request handles,
including their generations, and their individually feasible token and delivery
rates at dispatch. Only positive, unexpired graph estimates supply rate
normalizers. Rates include logical prompt lengths, remaining output budgets,
conditional speculative acceptance, and physical graph padding. An unknown
normalizer leaves that price coordinate uninitialized; it does not invent a
duration. Token prices persist across the prefill/decode transition. The first
published burst initializes a missing delivery history from its measured frame;
an already decoding request without that observation uses the feasible reference.

`RequestScheduler::complete` validates all records before learning graph cost or
updating service prices. It uses the maximum positive rank-local GPU interval
and the change from committed request counters. Two queued decode frames each
credit their own completion delta. Host polling, input staging, queue wait, and
publication delay do not add service intervals. A missing or zero device interval
leaves prices and graph costs unchanged.

An unselected participant receives zero service over the measured interval.
Requests admitted after dispatch and requests waiting for an external restore
are outside that interval's participant set. Generation checks prevent an
unselected request's cancelled handle from charging a replacement in its slot.
Accounting changes do not invalidate immutable plans for unrelated requests.

Completion initially credits zero deliveries. The first visible publication for
its fence replaces that interval's delivery update with one served delivery;
further tokens in the same burst leave it unchanged. Buffered stop-sequence
tokens earn no delivery until the worker releases them. Publication must occur
before the next completion is processed, as enforced by `published_output`.
This permits an exact correction of the held-action price update without
rewinding through subsequent frames. Output age uses the actual publication
timestamp independently of GPU service time.

## Measurements and limitations

The tests compare three stationary service regions against the offline concave
allocation solver: unequal CPU-like rates with weighted fairness, the same
rates with intrinsic preferences, and a batched service region. With a 1000 ms
history and action durations of 1–13 ms, the measured utility gaps after 25,000
warmup rounds and 25,000 measurement rounds are below 0.0001.

The finite serving sweep uses the synthetic catalog in
`scripts/scheduling_sim/workloads.py`. With `beta=64`, token fairness 16,
delivery fairness 4, `tau=100 ms`, service-weighted history, and age price 0.1,
all requests in these finite workloads complete:

| Workload | Separate graphs: elapsed / maximum gap, ms | Mixed catalog: elapsed / maximum gap, ms |
|---|---:|---:|
| Two 32,768-token prompts | 1568 / 98 | 1568 / 98 |
| Sixteen 32,768-token prompts | 9966 / 440 | 9513.2 / 279.4 |
| Long prefill plus low-acceptance decode | 2680.5 / 147 | 2351.8 / 32 |
| Long prefill plus high-acceptance decode | 1208 / 162 | 1187.2 / 32 |
| Long prefill plus four decodes | 1588 / 70 | 1461 / 44.2 |
| Long prefill plus five decodes | 1716 / 78 | 1605.5 / 88.4 |

Each request emits 128 tokens. Mixed graphs use the supplied synthetic durations;
these values are not GPU measurements. The sixteen-prompt gap and the worse
five-decoder gap with mixed graphs show that enabling more choices does not
establish a latency bound for the combined online policy.

For simultaneous 64-token prompts with long outputs, the same preferences give
all five streams their first output by 15 ms and all sixteen by 48 ms. Testing
zero, three, and seven accepted drafts gives maximum subsequent gaps below
150 ms. With token fairness reduced to 4 and three accepted drafts, the last
of sixteen streams receives its first output at about 2697 ms. Fairness weights
specify a substantive startup/throughput preference; there is no universal
weight implied by the stationary utility derivation.

The dispatcher also retains the immediate-value baseline's finite-tail
limitation: a 129-token prompt may use 128 plus 1 tokens when a padded 256-token
graph would finish faster. The completion oracle remains the reference for
testing that effect. Neither an EOS prior nor a finite-tail continuation is
part of this ongoing-service policy.

Native dispatch compares unknown costs optimistically and projects queued work
into service prices before selecting a second frame. For prerequisites that
immediately enable token production, it compares complete two-graph sequences,
including already ready streams in the following batch. The simulator's
`UtilityScheduler.score(..., preceding_ms=...)` evaluates the corresponding
non-delivering prerequisite plus delivery graph with frozen prices and ages.

For each pair of prerequisite and following inference shapes, native dispatch
jointly selects both cohorts. It enumerates how many newly ready streams the
following graph serves, then uses sorted prefix sums for those streams, already
ready streams, and first-stage terminal completions. Exhaustive subset tests
check the resulting maximum. Longer prerequisite chains retain an individual
continuation index. Exact fixed-pair cohort choice does not establish a globally
optimal schedule over arbitrary future sequences. Native mixed inference also
considers an absent, already ready, or newly enabled prompt row alongside the
decode cohort. The native shape and row contract is implemented; Ignition mixed
model captures and GPU replay validation remain outstanding.

Run the simulator with every preference supplied explicitly:

```sh
python -m scripts.scheduling_sim --policy utility --decode-values 64 \
  --rate-history service \
  --token-fairness 16 --delivery-fairness 4 --adaptation-ms 100 \
  --age-price 0.1 --horizon-ms 15000 --output /tmp/flare-utility
python -m pytest tests/test_scheduling_utility.py -q
```

Add `--mixed` for the synthetic mixed catalog or `--catalog PATH` for supplied
complete-graph durations. `summary.csv`, `requests.csv`, and `trace.jsonl` retain
the preference values needed to interpret each measurement.
