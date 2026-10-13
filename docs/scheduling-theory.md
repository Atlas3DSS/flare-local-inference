# Interactive inference scheduling

The [native scheduler design](scheduling-runtime-design.md) proposes the Rust
execution contract, mixed graph captures, timing, admission, and residency
machinery that apply these scheduling principles to the serving runtime.

## Objective and execution model

An interactive inference engine should bound interference between requests,
give small requests prompt access, and preserve sustained progress for large
requests. Throughput is optimized within those requirements. The fairness
entity is an individual request. No accounting is shared between requests
from the same user or agent.

The Python simulator in [`scripts/scheduling_sim`](../scripts/scheduling_sim/)
models one execution resource. A tensor-parallel GPU group can be treated as
one such resource when a model step occupies the group together. An action is
a nonempty collection of request work:

- A prefill segment of a known remaining prompt.
- An ordinary decode step.
- A speculative round with a chosen number of draft tokens.
- One prefill segment combined with decode or speculative rows.

Actions have duration `T(a, x)`, conditional on execution state `x`. They are
nonpreemptive. Arrivals become visible during execution but can affect the next
dispatch only. One action is in flight; no future actions are queued ahead.
Every resident request has at least one legal action producing positive
committed progress. Its minimum action must fit the configured maximum duration
`Q`. The simulator fails explicitly if this condition is false.

Prompt length is visible to the scheduler. Output length and future acceptance
outcomes are hidden. Cost prediction is exact within the synthetic model.
Real execution would require an error allowance or a conservative duration
estimate for latency claims.

Three distinct properties require different analyses:

1. **Dispatch responsiveness:** how soon a scheduler can reconsider its choice.
2. **Service regularity:** how long a resident request can go without progress.
3. **Progress fairness:** how much useful work a request receives over time.

A small graph duration establishes only the first property. Emitting a large
burst of tokens can improve average output rate without improving the second.

## Useful work and resource accounting

GPU time is shared nonadditively. If each of eight decodes takes 10 ms alone
and their batch takes 15 ms, there is no unique per-request physical charge.
Equal division, standalone-cost proportional division, and marginal cost are
different allocation conventions. Marginal costs need not sum to batch cost.

Useful-progress accounting avoids requiring that split. Associate request
`i` with a fixed monotone reference-work function `W_i(z)`, in milliseconds,
where `z` describes committed progress. An action earns

```text
g_i(a) = W_i(z_i after a) - W_i(z_i before a).
```

Reference work is a normalization, not a measurement of physical occupancy.
It is independent of the scheduler's choice of action boundaries. Rejected
drafts add execution cost and earn no useful-work credit. A completed prefill
includes its first output token; that token receives no second decode credit.
Accounting continues across the prefill/decode transition.

The simulator freezes two coefficients before dispatch:

- Prefill credit per token is the total cost of an isolated, cost-efficient
  greedy traversal of the prepared chunk catalog, divided by prompt length.
  The traversal respects `Q` and evolving prompt context. It is a specified
  reference algorithm, not a claim of globally optimal chunking.
- Ordinary decode credit is a singleton, zero-draft round at the initial decode
  context. The optional speculative reference is the smallest singleton
  `T(k) / E[tokens(k)]` across configured depths at that context.

Both coefficients stay fixed. Chunking into smaller segments cannot inflate
credit. Later context growth is reflected in action costs but does not reprice
the reference. Consequently `reference_slowdown` is completion time divided by
this reference demand; it is not a measured counterfactual isolated runtime.

The ordinary reference requires no acceptance oracle. The speculative reference
uses the workload's true acceptance profile to isolate the effects of that
fairness definition. It is an oracle experiment even when the scheduling policy
learns acceptance online. An implementation with measured isolated rates would
need to specify how those rates are estimated and when they may change.

An efficient request can earn more reference work than elapsed wall time through
batching. That is a benefit, not an accounting violation. Surplus progress is
not banked indefinitely to justify future stalls.

## Token-delivery fairness

Completion slowdown cannot distinguish steady output from a long stall followed
by a large speculative burst. Let `F_i` be the frozen first-token reference:
total prefill credit for a request with a prompt, or one decode credit for a
request already in decode. Let `d_i` be its frozen decode credit. Define

```text
R_i(k) = F_i + (k - 1) d_i,              k >= 1
S_i = max_k (delivery_time_i(k) - arrival_i) / R_i(k).
```

`max_delivery_slowdown` measures `S_i`. Queue time is included. No latency floor
or clamp to one is applied. A ratio below one represents delivery ahead of the
specified reference, rather than an invalid observation. `R_i` is a reference
curve, not a replay of the request running alone. In particular, a speculative
decode reference expresses expected work per token, not isolated burst timing.

If an action emits positions `k` through `k+m-1` together at time `t`, the
largest ratio in that burst is `(t-arrival_i)/R_i(k)`. The denominator belongs
to the **first** token of the burst. Using its last token would let speculation
erase delay that the receiver already experienced. For example, with a 10 ms
first-token reference and 10 ms per following token, token one at 10 ms and
tokens two through seven at 110 ms give a worst delivery slowdown of `110/20 =
5.5`, despite completion slowdown of `110/70 = 1.57`. Later output cannot reduce
the historical maximum. The trace records each burst's first output position
and slowdown; a test expands bursts into individual tokens to check the metric.

Delivery slowdown and absolute output gaps remain separate objectives. At a
late output position, an added delay `delta` changes the normalized ratio by
only `delta/R_i(k)`. Long streams can therefore accumulate substantial stalls
with little effect on their lifetime ratio. An interactive engine needs an
absolute service-gap control as well as normalized progress and delivery metrics.

## Stationary capacity and fair allocation

Freeze contexts, acceptance distributions, and the backlogged request set. For
each legal action define its expected reference-progress rate vector:

```text
r_i(a) = E[g_i(a)] / T(a).
```

If `alpha_a` is the fraction of wall time devoted to action `a`, attainable
long-run mean rates are in the convex hull of these vectors:

```text
r_i = sum_a alpha_a r_i(a),    alpha_a >= 0,    sum_a alpha_a = 1.
```

These are time fractions, not fractions of dispatches. If an action takes twice
as long, assigning it half the rounds gives it two thirds of execution time.
The convex-hull description assumes no switching cost beyond `T`, no state
change, and no response to acceptance that changes the action distribution.

Possible objectives include max-min normalized progress, proportional fairness
(`sum_i log r_i`), and maximum aggregate progress subject to per-request floors.
They are different policies. Equal accepted token rates are also a different
policy: a request with a much better isolated speculative rate would receive a
smaller fraction of its isolated performance under equal token rates.

The intended interactive policy uses useful-work deficits and service regularity
to constrain an efficiency preference. It does not claim to solve the stationary
max-min problem. The exact two-request solver in
[`frontier.py`](../scripts/scheduling_sim/frontier.py) supplies a separate
reference: it enumerates every absent/present/depth assignment, constructs rate
vectors, and finds the best minimum rate among vertices and intersections of
pairwise segments with `r_1 = r_2`. In two dimensions an optimum occurs on the
upper boundary of the convex hull, so those candidates suffice. A throughput
frontier does not determine a low-jitter order for executing its mixture.

A requested floor must lie in the downward closure of the achievable rate
region. Equal time sharing supports the `1/n` reference floor if every resident
has a legal isolated policy with reference rate at least one. Frozen credit at
the initial context need not retain that property as context grows. An infeasible
floor cannot be repaired by a better priority rule.

Stationary efficiency also differs from finite-workload completion. A greedy
ratio can spend extra time accelerating easy requests, finish them early, and
leave a poorly batched tail of difficult requests. In a deterministic synthetic
case with four always-accepting and four always-rejecting equal-length decodes,
greedy speculative progress per round improves while workload makespan worsens
relative to ordinary decode. A test encodes this counterexample. An acceptance
oracle alone does not supply the hidden remaining output lengths needed to
reason perfectly about that tail.

## Speculation as optional work

Let `k` be the number of drafts, and let `p_j` be the conditional probability of
accepting draft `j` given that all earlier drafts were accepted. Before output
termination, a linear speculative round produces `1 + A` tokens, where `A` is
the accepted prefix length. The tail-sum identity gives

```text
E[tokens(k)] = 1 + sum_{j=1..k} product_{l=1..j} p_l.
```

This does not require independence of acceptance positions: the probabilities
are conditional. A single average acceptance probability loses information.
For example, profiles `(1, 0)` and `(0, 1)` have the same average but expected
outputs of two and one tokens, respectively.

The marginal expected output from one more draft is its prefix-survival
probability. Compare its benefit with the added full-action duration, including
draft generation, target verification, and commit work. Also compare against
adding another request, servicing a prefill, or ending the action sooner.

For `n` equal requests rotated in cohorts of size `b`, a fixed action taking
`T(b,k)` gives each request approximately

```text
E[tokens(k)] / (ceil(n/b) * T(b,k))
```

tokens per unit time when cohorts and requests are otherwise identical. The
ceiling expression assumes a cycle of equally costly cohorts, including any
padding in the final cohort. Comparing only one selected cohort ignores the
delay paid by excluded requests.

Grouping similar acceptance profiles can reduce wasted verification when a
batch requires uniform or padded depths. It can also repeat weight reads,
shrink batches, and lengthen revisit intervals. With efficient ragged execution,
different depths in one batch may be better. Grouping is justified by the
complete cycle's progress and duration, not by acceptance similarity alone.

The learned controller estimates conditional acceptance separately by depth.
It updates only observed successes and the first observed rejection. Deeper
positions after rejection and positions after EOS are censored. Each observed
depth decays its previous success/failure counts by `0.95`, then adds its
observation; both counts begin at one. A periodic deep probe is permitted only
when there is no forced request and it fits before the oldest request's urgency
threshold. This does not guarantee every request receives enough exploration.

### Expected speed is not a deterministic progress guarantee

Suppose the first draft is accepted with probability `p < 1`. A sequence of `m`
first-draft rejections has positive probability `(1-p)^m` under independent
rounds. Every such round produces one token, regardless of a much higher
expected output. Therefore a bound on useful-progress lag relative to expected
speculative speed cannot hold on every sample path with a fixed constant.

Probability bounds require assumptions about dependence, drift, and estimation.
A deterministic guarantee can instead refer to actual execution service or a
worst-case progress rate. The prototype guarantees regular opportunities for
positive progress under its execution assumptions; its useful-progress fairness
is a heuristic to measure, not a proved bounded-lag property.

## Nonpreemption and service-gap bounds

With one action in flight and `T <= Q`, an arrival waits at most `Q` before its
presence can affect a dispatch. This does not guarantee admission or first
output within `Q`. A large prompt still requires its own prefill work.

For arrivals independent of a stationary busy renewal process, expected residual
action time is `E[T^2] / (2 E[T])`. Thus a policy with rare long actions can have
worse arrival delay than another policy with the same mean duration. The formula
does not apply unchanged to correlated arrivals or arbitrary nonstationary runs.

Both `interactive` and `delivery` apply an oldest-service guard:

1. Each resident request records completion time of its most recent service;
   admission initializes this time.
2. At a dispatch boundary, if the oldest service age is at least `D`, every
   eligible candidate must include that oldest request.
3. Each selected action finishes within `Q` and produces positive progress for
   every included request.

**Proposition.** With at most `N` resident requests, every resident request's
service gap is at most `D + (N + 1) Q`.

**Proof.** A request becomes due `D` after its previous service. The first
dispatch boundary after that instant is at most `Q` later. There are at most
`N - 1` requests with older service timestamps. While the request remains due,
the guard must service one of those older requests or the request itself. Each
older request receives a timestamp later than the waiting request's and cannot
overtake it again. Newly admitted requests also cannot overtake it. At most `N`
further actions, each bounded by `Q`, finish service for the request. Adding the
threshold and the initial residual gives the bound.

This bound is conservative. `D` is an urgency trigger, not a promised maximum
gap. For decode, service produces at least one output token or termination.
For prefill, service may produce no visible output until the prompt completes.
Queued requests outside the resident set are not covered. Prediction overruns,
multiple queued actions, nonprogressing actions, or eviction invalidate the
stated proof and require a revised model.

### Checking a service-gap target before dispatch

`service_gap_target_ms = G` enables a separate lookahead filter for the two
protected policies. At a boundary, request `i` has a next-service deadline
`last_service_i + G`. For every candidate action:

1. Predict completion of the included requests at `now + T`.
2. Schedule one minimum-duration singleton action for each excluded resident,
   ordered by deadline.
3. Compute the maximum lateness across those completions, including the first
   candidate's participants.

All singleton rescue jobs are ready at the candidate's completion. Their costs
are fixed while other requests run in this model. Earliest-deadline ordering
minimizes maximum lateness for these jobs: swapping an inverted adjacent pair
cannot increase the maximum of their latenesses, and leaves all other completion
times unchanged. The implementation checks this property against all orders of
a small job set. This result concerns fixed singleton jobs, not optimal LLM
batching, admission, or speculative-width selection.

A nonpositive maximum lateness certifies a concrete continuation that serves
every resident's pending visit on time. Only certified candidates remain when
any exist. Otherwise the filter retains candidates with the smallest sum of
positive projected latenesses and increments `uncertified_decisions`. The requested target
stays unchanged. `service_gap_violations` counts actual completed service
intervals exceeding `G`; `max_service_gap_excess_ms` measures their largest
overrun. Targets are also measured for baseline policies without enforcing the
filter. `latency_guard_enabled` distinguishes enforcement from measurement.

The recovery score differs from the feasibility score. Once a deadline is
already missed, serving only its request may reduce the worst lateness by a
fraction of a millisecond while delaying every other request by a full action.
Repeated minimization of that maximum can keep the engine in small batches.
Minimizing total positive lateness favors recovering the overdue group; it can
increase the single largest gap. A deterministic test compares these scores
for three overdue decodes. The continuation order still minimizes maximum
lateness; it does not optimize total tardiness over all possible rescue orders.

The certificate is sufficient, not necessary. Eight requests served two at a
time by 3 ms actions can meet a 12 ms gap. A candidate plus six serial singleton
rescues takes 21 ms and cannot certify that schedule. A test checks that this
case has uncertified decisions and no actual gap violations. Batching the
rescues would strengthen the check at additional search cost.

The certificate covers one pending visit per resident. It does not cover new
arrivals, subsequent visits, or cost growth after progress, and does not prove
recursive feasibility. The filter recomputes at each dispatch; the
`D + (N + 1) Q` bound remains the unconditional resident bound under the stated
model assumptions. Harder guarantees need admission against a feasible recurring
schedule, duration bounds over evolving context, and reserved execution slack.
The target currently controls prefill progress opportunities and decode output
opportunities. It does not promise first output before a long prompt finishes.

## Useful-progress deficits and batch choice

For resident request `i`, maintain a reflected deficit `L_i >= 0`. During an
action of duration `T` with `n` residents, update

```text
L_i <- max(0, L_i + T/n - g_i).
```

Requests not included in the action have `g_i = 0`. Reflection prevents surplus
batching gains from buying arbitrarily long future exclusion. The entitlement
clock begins at admission and does not include time before arrival or queued
time without resident resources. Queue delay is reported separately.

The implemented `interactive` selection rule is:

1. Enforce the oldest-service guard when due.
2. Apply the optional service-gap continuation filter.
3. Otherwise, force the largest-deficit request if its deficit reaches the
   configured `lag_scale_ms` and a remaining candidate includes it.
4. Otherwise, permit a periodic deep probe within service-age slack.
5. Among remaining candidates maximize

```text
[sum_i (1 + L_i/lag_scale_ms) E[g_i]
 + lag_scale_ms * number_of_prefills_completed] / T.
```

The first-output bonus favors completing a known prefill stage. There is no
decode-completion bonus because remaining output length is hidden. The lag
scale controls a preference and forced-service trigger; it is not a hard bound
on lag or borrowing. Fairness and regularity can conflict, and the oldest-service
guard takes precedence. No general stability or bounded-deficit theorem is
claimed for this candidate-restricted heuristic.

All requests retain singleton candidates. Decode subsets are exhaustively
enumerated through five decode requests. Larger sets use prefixes ordered by
age, deficit, acceptance in both directions, and context, plus every singleton.
Each cohort tries every homogeneous depth. Ragged mode also tries per-request
singleton-optimal depths together. Mixed candidates include the shortest,
oldest, and largest-deficit prefills. This bounds search while retaining a legal
action for the guard; it is not exhaustive joint optimization.

### Delivery-pressure policy

The `delivery` policy compares a different priority rule using exactly the same
engine, action catalog, reference curves, acceptance estimator, oldest-service
guard, and optional gap filter. It observes committed output count, but not
remaining output length. Its pressure for a request with unfinished prefill is

```text
P_i = (elapsed_since_arrival + remaining_prefill * prefill_credit_i) / F_i.
```

For a decoding request with `z` committed tokens it is

```text
P_i = (elapsed_since_arrival + ordinary_singleton_cost_i) / R_i(z + 1).
```

After mandatory guard filtering, the policy requires progress on a request
with the highest pressure at the dispatch boundary when a remaining candidate
can include it. It then projects every resident's pressure after each candidate
and minimizes the descending vector lexicographically. Expected useful work
per time breaks ties. Periodic deep probes obey the same age slack and gap
filter, and retain the pressure anchor.

For speculative work, the projected pressure averages over every possible
accepted-prefix length using the conditional acceptance model. The ordinary
singleton continuation is costed at the corresponding post-action context.
Computing pressure at the mean emitted count is not equivalent to this
expectation. Even with a constant continuation cost, it underestimates pressure
because the reciprocal reference denominator is convex.
EOS is unknown, so projections assume another token remains. Prefill completion
projects to the second output token; partial prefill retains the first-token
reference. The frozen remaining-prefill credit approximates remaining work and
need not equal an actual continuation's cost.

The pressure anchor is necessary to prevent an unrelated short action from
repeatedly winning a local ratio comparison. With two equally delayed prefills,
servicing either one still leaves the other as the worst projected ratio. An
unrelated, slightly shorter decode can then improve that maximum solely by
advancing less wall time. Repeating this choice postpones both prefills. Requiring
service of a pressure bottleneck excludes that choice; it is not a proof of
optimal delivery fairness.

Projected next-token pressure is a forward priority heuristic. The measured
`S_i` is a historical maximum over actual delivered tokens. The two quantities
differ, and already incurred delay cannot be undone. Neither this policy nor
the deficit policy solves finite-horizon minimax token-delivery slowdown.

## Mixed prefill and decode

For identical committed prefill work `P` and decode/verification work `B`,

```text
T_mixed(P,B) < T_prefill(P) + T_decode(B)
```

is an execution-efficiency opportunity. It does not establish a latency win:
the first decode result may arrive later than under decode-first ordering.
The engine should choose among prefill capacities in time units, accounting
for included and excluded requests, rather than enforce a universal token size.

The synthetic model is

```text
T = launch + max(weight_cost, target_token_cost * padded_target_rows)
    + attention_coefficient * sum_i q_i * (context_i + q_i/2) / 1024
    + draft_step_cost * max_draft_depth
    + draft_token_cost * draft_rows
    + mixed_penalty_if_applicable.
```

Prefill projection rows round up to a multiple of 64. Rectangular decode
projection and draft rows use the next power-of-two cohort capacity and maximum
depth. Ragged mode uses active row counts. Attention uses active query lengths.
Mixed execution shares the target weight term; an explicit mixed penalty can
reverse its benefit. The full formulas and default coefficients are in
[`model.py`](../scripts/scheduling_sim/model.py).

These coefficients are synthetic. The model does not simulate MoE routing,
offloaded tables, kernel selection discontinuities beyond padding, physical KV
allocation, restore/eviction, rank skew, CPU overhead, or changing GPU clocks.
Resident slots model admission interference only. They do not model a KV-byte
budget. No simulation rate is a claim about Flare or a particular GPU.

## Experiments and interpretation

The simulator is standard-library Python and does not import Ignition or CUDA.
Run from the repository root using Python 3.12 or later:

```sh
python -m scripts.scheduling_sim --seeds 0,1,2 --output /tmp/scheduling-baselines
python -m pytest -q tests/test_scheduling_sim.py
```

The compared policies are FIFO request completion, strict decode-first,
oldest-service round-robin with zero drafts, greedy normalized useful-work
throughput, the interactive deficit policy, and the delivery-pressure policy.
All use the same duration cap, legal
cost model, and candidate catalog before their policy-specific filtering.
The throughput baseline optimizes reference work per time, not a sum of raw
input and output token counts.

Built-in workloads cover large-prefill interference, eight concurrent decodes,
heterogeneous acceptance, short arrivals during long decodes, heterogeneous
contexts, and resident-slot exhaustion. The slot-pressure workload defaults
to two residents. Explicit CLI or configuration values override that default.

Explore batch size, speculation, and padding:

```sh
python -m scripts.scheduling_sim --workload acceptance_groups \
  --policies interactive,throughput --batch-sizes 2,8 --draft-depths 0,2,5 \
  --shapes rectangular,ragged --acceptance oracle --seeds 0,1,2 \
  --output /tmp/scheduling-speculation
```

Explore prefill capacity and duration constraints by changing the available
catalog. A small cap can be infeasible at long contexts; infeasibility is an
error, not permission to silently enlarge the cap:

```sh
python -m scripts.scheduling_sim --workload prefill_burst \
  --policies interactive,throughput --quantums 8,12,20 \
  --chunks 64,128,256,512,1024 --seeds 0,1,2 \
  --output /tmp/scheduling-prefill --trace
```

Use `--no-mixing` for a matched mixing ablation. `--config PATH` accepts a JSON
object containing `config` and `model` field overrides. `--workload-file PATH`
accepts a JSON list of request objects, for example:

```json
[
  {"name": "large", "arrival_ms": 0, "prompt_tokens": 8192, "output_tokens": 128},
  {"name": "small", "arrival_ms": 10, "prompt_tokens": 64, "output_tokens": 8,
   "acceptance": [0.95, 0.9, 0.8, 0.6, 0.4]}
]
```

Compare deficit and delivery priorities with matched service-gap controls:

```sh
python -m scripts.scheduling_sim --policies interactive,delivery \
  --gap-targets none,8,12,20,40 --seeds 0,1,2 \
  --output /tmp/scheduling-delivery
```

`none` disables the lookahead filter. A numeric target overrides
`service_gap_target_ms` in the configuration file. `urgent_after_ms` continues
to control the oldest-service guard independently. No gap target is enabled by
default; choosing a deployable target requires measured action costs and an
admission contract.

`summary.csv` records parameters and aggregate metrics. `requests.csv` records
per-request parameters and latencies. Optional `trace.jsonl` records action
timing, participants, draft depths, acceptance observations, and emitted counts.
It also records the selected action's rescue lateness when the filter is enabled.
The maximum resident service gap includes admission-to-first-service. Output
gaps are intervals between nonempty emission bursts; simultaneous tokens do not
insert artificial zero gaps. TTFT includes admission waiting. Percentiles are
linearly interpolated empirical quantiles within each finite workload.
`p95_delivery_slowdown` is the percentile of per-request worst token-delivery
ratios. It is not a percentile across tokens. The reference coefficients and
worst token position are recorded per request. Gap-violation counts are zero
when no target is configured, and have meaning only with a numeric target.
Changing the reference catalog or quantum can change the prefill normalization;
reference-lag values should be compared only under the same normalization.

Plot individual seeds on the throughput/latency plane with the optional
Matplotlib utility:

```sh
uv run --no-project --with matplotlib python -m scripts.scheduling_sim.plot \
  /tmp/scheduling-baselines/summary.csv --workload prefill_burst \
  --output /tmp/scheduling-prefill.svg
```

Random acceptance draws are indexed by seed, input request index, committed
output position, and draft depth. Reordering other requests' execution does not
advance a shared random stream. Different policies can still encounter different
positions and depths because their accepted prefixes differ. The scheduler
never receives those draws or remaining output length.
Draws are allocated through each request's full supplied acceptance profile, so
changing the configured draft-depth limit also preserves the draw at each
shared output position and depth.

The stationary two-request calculation is independent of the online heuristic:

```sh
python -m scripts.scheduling_sim.frontier --acceptance 0.97,0.1
python -m scripts.scheduling_sim.frontier --acceptance 0.97,0.1 --shape ragged
python -m scripts.scheduling_sim.frontier --acceptance 0.97,0.1 --reference speculative
```

It reports exact max-min normalized rate and action time fractions for fixed
contexts and acceptance. Comparing ordinary and speculative normalization
demonstrates how the fairness definition itself changes the preferred schedule.

Tests cover the acceptance identity, Monte Carlo agreement, censoring at EOS,
arrival boundaries, prefill completion, admission queues, fragmentation-invariant
credit, impossible quanta, mixed-mode counterexamples, padding, oldest-service
protection, randomized finite-workload service bounds, and the exact two-request
frontier against a brute-force mixture grid. They also encode a counterexample
to a deterministic useful-progress guarantee based on expected speculation.
Delivery tests cover burst boundaries, retained historical maxima, queue delay,
and speculative expectations. Latency tests cover anticipatory filtering,
reported misses, conservative certificate failures for feasible batch schedules,
and recovery from overdue service without singleton lock-in.

Experiments should compare latency and throughput together, retain per-request
outliers, and compare identical workload seeds and cost coefficients. Finite
workloads can reveal unfairness; finishing them is not a proof against starvation
under unbounded arrivals. Useful-progress stability, prediction-error robustness,
and memory preemption require additional models or proofs.

### Synthetic workload results

With the default cost coefficients, `Q = 20 ms`, `D = 40 ms`, batch capacity
eight, at most five drafts, learned acceptance, and seeds `0,1,2`, the
`prefill_burst` workload gives the following means of per-seed metrics. Its
small requests contain 64 prompt tokens and 16 output tokens; one competing
request contains an 8192-token prompt and 64 output tokens.

| Policy | Output tokens/s, higher is better | P95 small-request TTFT, ms, lower is better | Maximum resident service gap, ms, lower is better |
| --- | ---: | ---: | ---: |
| Round-robin, zero drafts | 310.9 | 20.6 | 18.4 |
| Greedy useful-work throughput | 392.6 | 111.7 | 116.4 |
| Interactive | 386.0 | 7.9 | 17.4 |

The interactive policy exchanges about 1.7% of aggregate throughput for much
shorter small-request TTFT and service gaps in this model. It does not dominate
the other policies across workloads. On `acceptance_groups`, ordinary
round-robin delivers 2538.1 tokens/s while interactive delivers 2213.4 tokens/s.
On `long_context`, round-robin also has lower P95 TTFT than interactive
(52.0 versus 122.2 ms), while interactive has higher throughput
(1519.2 versus 1431.8 tokens/s). The interactive policy's mean per-seed maximum
service gap is 69.3 ms there, above the 40 ms urgency trigger but below the
proved bound.

For `prefill_burst`, disabling mixing with other parameters fixed reduces
interactive throughput from 386.0 to 337.5 tokens/s and increases its mean
maximum service gap from 17.4 to 49.5 ms. In seed zero, the interactive mixed
actions use seven 64-token chunks and twenty 128-token chunks at each tested
quantum of 8, 12, and 20 ms. The cost model's projection knee is near
`weight_ms / target_token_ms = 136` rows. This result reflects those coefficients
and the candidate scoring rule; it does not establish 128 as a hardware default.

With oracle acceptance on `acceptance_groups`, rectangular batches of two with
up to five drafts deliver 975.3 tokens/s under interactive scheduling, versus
2303.7 tokens/s for batches of eight with up to two drafts. Eight-request
ordinary decode delivers 2538.1 tokens/s. These are adaptive depth limits, not
fixed-depth executions. The three configurations differ in both throughput and
service frequency, illustrating why speculative width cannot be selected in
isolation from cohort size and finite-workload tails.

### Delivery priorities and service-gap controls

With the same default cost model, learned acceptance, and seeds `0,1,2`, the
following values are means of per-seed metrics. `G` is the optional resident
service-gap target. A missing target leaves the oldest-service guard enabled.
Lower gap and slowdown values are better; higher output throughput is better.

| Workload | Policy | G, ms | Output tokens/s | Maximum service gap, ms | Worst token-delivery slowdown |
| --- | --- | ---: | ---: | ---: | ---: |
| Long contexts | Interactive | Disabled | 1519.2 | 69.33 | 6.42 |
| Long contexts | Interactive | 40 | 1796.5 | 39.99 | 4.07 |
| Long contexts | Delivery | 40 | 1561.5 | 39.16 | 17.57 |
| Short arrivals | Interactive | 40 | 1815.7 | 9.55 | 3.00 |
| Short arrivals | Delivery | 40 | 1319.5 | 24.91 | 5.97 |
| Acceptance groups | Interactive | 40 | 2213.3 | 5.91 | 1.44 |
| Acceptance groups | Delivery | 40 | 2393.0 | 18.91 | 1.17 |
| Acceptance groups | Round-robin | 40, measured only | 2538.1 | 3.15 | 1.00 |

The 40 ms target produces no service-gap violations or uncertified decisions
for either protected policy across all six built-in workloads and these seeds.
On long contexts, the interactive filter improves throughput by 18.3% while
reducing maximum service gaps by 42.3%. Its P95 TTFT falls from 122.2 to 76.8 ms.
The other five interactive workloads retain their schedules at that target.
These observations do not establish a hard 40 ms guarantee.

Resident guarantees also exclude admission queues. In the slot-pressure workload
at seed zero, the interactive policy has no 40 ms gap violations, yet one
queued request waits 307.3 ms for admission and 310.5 ms for its first token.
Its worst delivery slowdown is 98.31. Admission delay remains visible in the
delivery metric even when every resident service interval satisfies the target.

The short-arrival workload contains three long requests with 128-token prompts
and 200 output tokens, followed by ten requests with 64-token prompts and four
output tokens. At `G = 40`, the interactive policy has P95 TTFT of 9.1 ms versus
17.1 ms for delivery pressure. Delivery pressure improves worst delivery
slowdown for heterogeneous acceptance, but gives up absolute service regularity.
With oracle acceptance on that workload, it selects ordinary batched decode
and matches round-robin's 2538.1 tokens/s. The interactive policy instead
achieves 2194.1 tokens/s with a worst delivery slowdown of 1.87. Speculative
useful-work efficiency alone does not resolve finite-workload tails or first
delivery delay.

Tighter targets can be impossible at admission. In `long_context`, all six
requests arrive with unfinished prompts. The catalog serves at most one
prefill request per action. Their minimum first-service costs are 11.350,
11.350, and four times 3.286 ms, totaling 35.844 ms. No policy over that catalog
can give all six an initial service opportunity within 20 ms. With a 20 ms
target, both protected policies record four gap violations and twelve
uncertified decisions per seed, with a maximum service gap of 49.09 ms. At an
8 ms target, they record 27 violations and reach 64.89 ms. Reducing an
infeasible target need not reduce observed gaps; recovery optimizes total
overdue time after the feasibility filter fails.

The deficit rule with a feasible gap target is the preferred prototype for
interactive mixed workloads in these experiments. Delivery pressure remains a
comparison policy. A production design needs admission that accounts for
first-service demand, a recurring service plan rather than a single-visit
certificate, and measured costs for the prepared action catalog.

The comparison matrix can be run with:

```sh
python -m scripts.scheduling_sim \
  --policies round_robin,throughput,interactive,delivery \
  --gap-targets none,40 --seeds 0,1,2 --output /tmp/scheduling-controls
python -m scripts.scheduling_sim --policies interactive,delivery \
  --gap-targets 8,12,20 --seeds 0,1,2 --output /tmp/scheduling-gap-targets
python -m scripts.scheduling_sim --policies interactive,delivery \
  --gap-targets none,40 --acceptance oracle --seeds 0,1,2 \
  --output /tmp/scheduling-delivery-oracle
```

## Related scheduling theory

- [EEVDF](https://docs.kernel.org/scheduler/sched-eevdf.html) separates service
  eligibility from virtual deadlines. Its CPU accounting and preemption model
  cannot be transferred unchanged to nonadditive batch execution.
- [Fair Service Curve scheduling](https://www.cs.cmu.edu/~hzhang/papers/SIGCOM97.pdf)
  separates delay and sustained allocation and exposes conflicts between them.
- [Fair Sojourn Protocol](https://www1.ece.neu.edu/~ningfang/SimPaper/p229-friedman.pdf)
  improves completion order against a processor-sharing reference under known
  job sizes. Unknown output lengths and batched service change those assumptions.
- [Proportional fairness](https://www.statslab.cam.ac.uk/~fpk1/rate.html) supplies
  a utility-based allocation objective over heterogeneous achievable rates.
- [VTC](https://www.usenix.org/conference/osdi24/presentation/sheng) establishes
  LLM fairness under a weighted input/output token service definition. The
  useful-work normalization here defines a different service quantity.
- [SmartSpec](https://arxiv.org/abs/2406.14066) connects speculation depth, batch
  size, and accepted progress per execution time.
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  demonstrates chunked prefill and decode co-scheduling under latency constraints.
- [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)
  applies feedback queues to partially known request sizes.
