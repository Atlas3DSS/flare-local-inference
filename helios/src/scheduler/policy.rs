//! Expected useful value per time over prepared physical execution shapes.

use super::{
    utility::{LaneValue, ServicePrice, ValueRate, best_cohort, best_two_stage_cohort},
    *,
};

#[derive(Clone, Copy, Debug)]
pub struct PolicyConfig {
    pub output_token_value: f64,
    /// Prompt-value units per consumed prompt token.
    pub prefill_token_value: f64,
    pub token_fairness: f64,
    pub delivery_fairness: f64,
    pub decode_age_price: f64,
    pub adaptation_ms: f64,
    pub estimate_ttl_ns: u64,
    /// Keep a second unconstrained speculative round in flight. Captured graphs own
    /// positions, pending tokens and acceptance, so the next round needs no host
    /// feedback; the host then hides graph submission behind the running round.
    pub pipelined_speculation: bool,
    /// Opt-in shortest-remaining-prompt-first prefill (default off: prefill chunks follow the token-service price,
    /// which interleaves concurrent prompts chunk by chunk). When on, each one-request prefill chunk goes to the
    /// first prompt in `srpt_key` order: the prompt already mid-prefill (no mid-prompt preemption), then prompts that
    /// waited `prefill_srpt_max_wait_chunks` other chunks (FIFO), then effective remaining prompt tokens with linear
    /// aging, then arrival and slot. The step keeps the value of the best-priced prefill lane, so prefill-versus-decode
    /// choices are unchanged; prompt chunk cuts, graphs and arithmetic are unchanged.
    pub prefill_srpt: bool,
    /// SRPT starvation bound: a prompt that waited this many other prompts' prefill chunks is served FIFO (0 = no cap).
    pub prefill_srpt_max_wait_chunks: u64,
    /// SRPT aging: remaining-token credit per other prompt's prefill chunk waited.
    pub prefill_srpt_aging_tokens_per_chunk: u64,
    /// With `prefill_srpt`: a prompt with at most this many uncached prompt tokens left (and no chunk in flight) takes
    /// one prefill chunk ahead of the active leader when the leader's chunk was the last prefill committed, and its
    /// Init/Restore step is not held behind prompt chunks. Short prompts and the leader alternate chunk by chunk, so the
    /// leader receives at least every other prefill chunk (finishing within twice its own chunk count) and keeps its
    /// place; a prompt with more tokens left never preempts it. 0 = off (the v3 order).
    pub prefill_srpt_short_tokens: u64,
    /// With `prefill_srpt`: bounded decode priority. While a request that has emitted fewer than this many output tokens
    /// is decoding (or in the boundary snapshot that resumes its decoding) and has a round in flight or runnable, no
    /// prompt chunk is offered; each request therefore delays waiting prompts by at most one boundary snapshot plus this
    /// many output tokens. A non-zero value also values each SRPT prefill step by the prompt that runs (v3: the
    /// best-priced waiting prompt, whose unserved price keeps rising), so decoders past the budget compete with the
    /// running prompt. 0 = off (v3).
    pub prefill_srpt_decode_priority_tokens: u64,
    /// Runtime admission: a request whose prefix the host store holds, but whose restore lacks pages or a snapshot row
    /// that a plain admission would not need, waits for the restore while other requests run, at most as long as
    /// recomputing the restorable tokens takes (their count over the measured prefill rate), instead of being admitted
    /// at once to prefill them again. With nothing else running, or past that bound, it is admitted as before. Off: the
    /// restore falls back to the plain admission immediately (the default).
    pub host_restore_wait: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            output_token_value: 64.0,
            prefill_token_value: 1.0,
            token_fairness: 16.0,
            delivery_fairness: 4.0,
            decode_age_price: 0.1,
            adaptation_ms: 100.0,
            estimate_ttl_ns: 30_000_000_000,
            pipelined_speculation: false,
            prefill_srpt: false,
            prefill_srpt_max_wait_chunks: 256,
            prefill_srpt_aging_tokens_per_chunk: 32,
            prefill_srpt_short_tokens: 0,
            prefill_srpt_decode_priority_tokens: 0,
            host_restore_wait: false,
        }
    }
}

impl PolicyConfig {
    /// Prefer token throughput with token-service fairness, without pricing
    /// the frequency of output bursts or their age separately.
    pub fn throughput() -> Self {
        Self { delivery_fairness: 0.0, decode_age_price: 0.0, ..Self::default() }
    }

    /// Agents block on every prompt (tool output) before producing more tokens, so a
    /// consumed prompt token is worth an output token: a waiting prompt is served
    /// before other requests drain their decoding, and then decodes with them. Agents
    /// read whole turns, so output cadence is not priced: like `throughput`, wider
    /// speculative rounds are chosen whenever they deliver more tokens per GPU second.
    pub fn agent() -> Self {
        let throughput = Self::throughput();
        Self { prefill_token_value: throughput.output_token_value, ..throughput }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RequestAccounting {
    pub arrival_ns: u64,
    pub last_progress_ns: u64,
    pub last_output_ns: Option<u64>,
    pub last_output_fence: Option<Fence>,
    pub published_deliveries: u64,
    pub committed_prefill_tokens: u64,
    pub committed_output_tokens: u64,
    /// Scheduler prefill-chunk count when the request was admitted (SRPT aging).
    pub arrival_prefill_chunk: u64,
    /// Prefill chunks scheduled for this request (SRPT aging).
    pub served_prefill_chunks: u64,
    pub(super) pending_publication_fence: Option<Fence>,
    pub(super) service: service::ServiceAccounting,
}

impl RequestAccounting {
    pub(super) fn credit(&mut self, prefill: u64, output: u32, now: u64) {
        self.committed_prefill_tokens += prefill;
        self.committed_output_tokens += u64::from(output);
        if prefill != 0 || output != 0 {
            self.last_progress_ns = now;
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Candidate {
    pub kind: GraphKind,
    pub mask: u64,
    pub batch: usize,
    pub capacity: usize,
    pub packed_token_capacity: usize,
    pub prefill_capacity: usize,
    pub prefill_slot: Option<usize>,
    pub work: usize,
    pub duration_ns: u64,
    pub rate: ValueRate,
    pub feedback: bool,
    pub verification_budget: Option<usize>,
}

impl Candidate {
    pub(super) fn cost_key(self) -> cost::Key {
        cost::Key {
            kind: self.kind,
            batch: self.batch,
            tokens: self.capacity,
            prefill: self.prefill_capacity,
            packed: self.packed_token_capacity,
        }
    }
}

impl RequestScheduler {
    pub fn policy(&self) -> &PolicyConfig {
        &self.policy
    }

    pub fn configure_policy(&mut self, policy: PolicyConfig) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("scheduling policy must be configured before admission".into());
        }
        if !policy.output_token_value.is_finite()
            || policy.output_token_value <= 0.0
            || !policy.prefill_token_value.is_finite()
            || policy.prefill_token_value <= 0.0
            || !policy.adaptation_ms.is_finite()
            || policy.adaptation_ms <= 0.0
            || policy.estimate_ttl_ns == 0
            || [
                policy.token_fairness,
                policy.delivery_fairness,
                policy.decode_age_price,
            ]
            .into_iter()
            .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err("invalid scheduler utility coefficients or estimation intervals".into());
        }
        self.policy = policy;
        self.costs.advance_to(self.now_ns, policy.estimate_ttl_ns);
        Ok(())
    }

    pub fn advance_to(&mut self, now_ns: u64) -> Result<()> {
        if now_ns < self.now_ns {
            return Err("scheduler clock must be monotonic".into());
        }
        self.now_ns = now_ns;
        self.costs.advance_to(now_ns, self.policy.estimate_ttl_ns);
        Ok(())
    }

    pub fn set_arrival(&mut self, handle: RequestHandle, arrival_ns: u64) -> Result<()> {
        let request = self.request(handle)?;
        if arrival_ns > self.now_ns || request.last_use != Fence(0) {
            return Err("arrival must precede admission and be assigned before dispatch".into());
        }
        let request = self.slots[handle.slot as usize].request.as_mut().unwrap();
        request.accounting.arrival_ns = arrival_ns;
        request.accounting.last_progress_ns = arrival_ns;
        Ok(())
    }

    pub fn published_output(&mut self, handle: RequestHandle, fence: Fence) -> Result<()> {
        if fence != self.completed {
            return Err("output publication must be reported before the next completion".into());
        }
        if self.request(handle)?.accounting.pending_publication_fence != Some(fence) {
            return Err(
                "output publication must name the request's latest output completion".into(),
            );
        }
        let accounting = &mut self.slots[handle.slot as usize]
            .request
            .as_mut()
            .unwrap()
            .accounting;
        if accounting.last_output_fence != Some(fence) {
            accounting.published_deliveries += 1;
            accounting.last_output_fence = Some(fence);
            accounting.service.publish(fence, self.policy.adaptation_ms);
        }
        accounting.last_output_ns = Some(self.now_ns);
        Ok(())
    }

    pub(super) fn accounting(&self) -> RequestAccounting {
        RequestAccounting {
            arrival_ns: self.now_ns,
            last_progress_ns: self.now_ns,
            arrival_prefill_chunk: self.prefill_chunks_scheduled,
            ..Default::default()
        }
    }

    /// SRPT order of a prompt in its prefill phase (smaller first); see `PolicyConfig::prefill_srpt`.
    pub(super) fn srpt_key(&self, slot: usize, r: &LiveRequest) -> (bool, bool, bool, u64, u64, usize) {
        let remaining = (r.prompt_tokens as u64).saturating_sub(r.consumed_tokens);
        let waited = self
            .prefill_chunks_scheduled
            .saturating_sub(r.accounting.arrival_prefill_chunk)
            .saturating_sub(r.accounting.served_prefill_chunks);
        let aged_out = self.policy.prefill_srpt_max_wait_chunks > 0
            && waited >= self.policy.prefill_srpt_max_wait_chunks;
        let effective = if aged_out {
            0
        } else {
            remaining.saturating_sub(waited.saturating_mul(self.policy.prefill_srpt_aging_tokens_per_chunk))
        };
        let leader = self.active_prefill == Some(r.handle) && r.phase == RequestPhase::Prefill;
        let preempt = !leader && r.pending == 0 && self.short_prompt_may_preempt(r.handle, remaining);
        (!preempt, !leader, !aged_out, effective, r.accounting.arrival_ns, slot)
    }

    /// Whether a prompt with `remaining` uncached tokens left may take the next chunk ahead of a live leader: it is
    /// short, and the leader's chunk was the last prefill committed (leader and short prompts alternate).
    pub(super) fn short_prompt_may_preempt(&self, handle: RequestHandle, remaining: u64) -> bool {
        let limit = self.policy.prefill_srpt_short_tokens;
        limit > 0 && remaining <= limit && self.live_leader_other_than(handle)
            && self.last_prefill == self.active_prefill
    }

    /// The active prefill leader exists, is not `handle`, and still has prompt tokens to schedule (a stale handle of a
    /// finished request never matches: handles carry the slot generation).
    pub(super) fn live_leader_other_than(&self, handle: RequestHandle) -> bool {
        self.active_prefill.is_some_and(|leader| {
            leader != handle
                && self.request(leader).is_ok_and(|l| {
                    l.phase == RequestPhase::Prefill && l.scheduled_consumed < l.prompt_tokens as u64
                })
        })
    }

    /// A short prompt that may take the next chunk ahead of the leader is in its Init/Restore step: prompt chunks are
    /// held so that step runs next (it needs completion feedback, so it runs once the window drains).
    pub(super) fn short_prompt_preparing(&self) -> bool {
        self.policy.prefill_srpt
            && self.policy.prefill_srpt_short_tokens > 0
            && self.slots.iter().filter_map(|e| e.request.as_ref()).any(|r| {
                matches!(r.phase, RequestPhase::Initialize | RequestPhase::Restore)
                    && (r.pending > 0 || self.eligible(r))
                    && self.short_prompt_may_preempt(
                        r.handle,
                        (r.prompt_tokens as u64).saturating_sub(r.consumed_tokens),
                    )
            })
    }

    /// Bounded decode priority: a decoding request within its budget is runnable or has a round in flight.
    pub(super) fn decode_priority_holds_prefill(&self) -> bool {
        let budget = self.policy.prefill_srpt_decode_priority_tokens;
        self.policy.prefill_srpt
            && budget > 0
            && self.slots.iter().filter_map(|e| e.request.as_ref()).any(|r| {
                // A boundary snapshot that resumes decoding counts as decoding (it runs before the next round).
                let decoding = matches!(r.phase, RequestPhase::Decode | RequestPhase::Resume)
                    || (r.phase == RequestPhase::Snapshot
                        && r.boundary_capture.is_some_and(|c| {
                            matches!(c.resume, RequestPhase::Decode | RequestPhase::Resume)
                        }));
                decoding
                    && u64::from(r.generated_tokens) < budget
                    && (r.pending > 0 || self.eligible(r))
            })
    }

    /// The first prompt with prompt tokens left in SRPT order, in flight or not.
    pub(super) fn srpt_first(&self) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(s, e)| e.request.as_ref().map(|r| (s, r)))
            .filter(|(_, r)| r.phase == RequestPhase::Prefill && r.consumed_tokens < r.prompt_tokens as u64)
            .min_by_key(|&(s, r)| self.srpt_key(s, r))
            .map(|(s, _)| s)
    }
    fn request_kind(&self, request: &LiveRequest) -> Option<GraphKind> {
        Some(match request.phase {
            RequestPhase::Initialize => GraphKind::Init,
            RequestPhase::Restore => GraphKind::Restore,
            RequestPhase::Vision => GraphKind::Vision,
            RequestPhase::Prefill => GraphKind::Prefill,
            RequestPhase::Resume => GraphKind::Resume,
            RequestPhase::Decode if self.speculative_widths.is_empty() => GraphKind::Decode,
            RequestPhase::Decode if self.split.is_some() => {
                if self.retained_proposal(request).is_some() {
                    GraphKind::Verification
                } else {
                    GraphKind::Proposal
                }
            }
            RequestPhase::Decode => GraphKind::Speculative,
            RequestPhase::Snapshot => GraphKind::Snapshot,
            _ => return None,
        })
    }

    pub(super) fn eligible(&self, request: &LiveRequest) -> bool {
        if self.request_kind(request).is_none()
            || (request.phase == RequestPhase::Vision && !request.conditioned_inputs_ready)
        {
            return false;
        }
        if request.pending > 0
            && !(request.phase == RequestPhase::Decode
                && (self.speculative_widths.is_empty()
                    || (self.policy.pipelined_speculation && self.split.is_none()))
                && self.capabilities.completion_inputs.is_empty()
                && request.pending < 2
                && !request.grammar_constrained
                && !request.host_stops)
        {
            return false;
        }
        if request.phase == RequestPhase::Decode {
            request.scheduled_consumed < self.limits.max_sequence_tokens as u64
                && request.generated_tokens as usize + request.pending
                    < request.max_new_tokens as usize
                && self.fits_next_rounds(request)
        } else {
            true
        }
    }

    /// Whether a speculative round queued behind `request`'s in-flight rounds has the width it
    /// would have after their completion: no output or context limit can clip it even if every
    /// in-flight round emits its maximum. Otherwise the round waits for feedback.
    fn queued_round_is_exact(&self, request: &LiveRequest, width: usize) -> bool {
        let ahead = request.scheduled_consumed - request.consumed_tokens;
        request.pending == 0
            || (u64::from(request.max_new_tokens - request.generated_tokens) >= ahead + width as u64
                && self.limits.max_sequence_tokens as u64 >= request.scheduled_consumed + width as u64)
    }

    pub(super) fn prefill_work(&self, request: &LiveRequest, capacity: usize) -> usize {
        let start = request.consumed_tokens as usize;
        let mut end = request
            .message_boundaries
            .iter()
            .copied()
            .find(|n| *n > start)
            .unwrap_or(request.prompt_tokens)
            .min(start.saturating_add(capacity));
        if let Some(span) = request
            .prefill_spans
            .iter()
            .find(|s| s.start < end && end < s.end)
        {
            end = span.start;
        }
        end - start
    }

    pub fn submission_window_available(&self, duration_ns: u64) -> bool {
        self.submissions.len() < 2
            && !self.submissions.values().any(|s| s.requires_feedback)
            && (self.submissions.is_empty() || duration_ns > 0)
            && self.window_remaining().is_some()
    }

    pub(super) fn window_remaining(&self) -> Option<u64> {
        let Some((_, first)) = self.submissions.first_key_value() else {
            return Some(0);
        };
        if self.now_ns >= first.expected_finish_ns {
            return None;
        }
        Some(
            self.submissions
                .last_key_value()
                .unwrap()
                .1
                .expected_finish_ns
                .saturating_sub(self.now_ns),
        )
    }

    pub(super) fn lane_values(&self, window_ns: u64) -> ([LaneValue; REQUEST_LANES], f64, f64) {
        let mut values = [LaneValue::default(); REQUEST_LANES];
        let count = self
            .slots
            .iter()
            .filter_map(|s| s.request.as_ref())
            .filter(|r| self.request_kind(r).is_some())
            .count()
            .max(1);
        let mut token_log = [f64::NEG_INFINITY; REQUEST_LANES];
        let mut delivery_log = token_log;
        let rates: [_; REQUEST_LANES] = std::array::from_fn(|slot| {
            self.slots
                .get(slot)
                .and_then(|s| s.request.as_ref())
                .filter(|r| self.request_kind(r).is_some())
                .map(|r| self.service_rates(r))
                .unwrap_or_default()
        });
        let references = self.delivery_reference_intervals(&rates.map(|(_, d)| d));
        let mut scale = 0.0f64
            .max(self.policy.output_token_value.ln())
            .max(self.policy.decode_age_price.ln());
        for (slot, entry) in self.slots.iter().enumerate() {
            let Some(r) = entry
                .request
                .as_ref()
                .filter(|r| self.request_kind(r).is_some())
            else {
                continue;
            };
            let (tokens, deliveries) = rates[slot];
            let service = self.projected_service(r);
            if let Some(rate) = tokens {
                token_log[slot] = service
                    .tokens
                    .unwrap_or_else(|| ServicePrice::initial(rate / count as f64))
                    .log_price(self.policy.token_fairness);
            }
            if let Some(reference) = references[slot].filter(|_| deliveries.is_some()) {
                delivery_log[slot] = service
                    .deliveries
                    .unwrap_or_else(|| ServicePrice::initial(1.0 / reference))
                    .log_price(self.policy.delivery_fairness);
                values[slot].reference_delivery_ms = reference;
                values[slot].age_price = self.policy.decode_age_price;
                let pending_delivery = self.submissions.values().any(|s| {
                    matches!(
                        s.kind,
                        GraphKind::Decode | GraphKind::Speculative | GraphKind::Resume
                    ) && s.lanes.iter().any(|l| l.request == r.handle)
                });
                values[slot].age_ms = if pending_delivery {
                    0.0
                } else {
                    self.now_ns.saturating_add(window_ns).saturating_sub(
                        r.accounting
                            .last_output_ns
                            .unwrap_or(r.accounting.last_progress_ns),
                    ) as f64
                        / 1e6
                };
            }
            scale = scale.max(token_log[slot]).max(delivery_log[slot]);
        }
        let unit = (-scale).exp();
        let mut age_price = 0.0;
        for (slot, value) in values.iter_mut().enumerate() {
            value.prefill_token =
                self.policy.prefill_token_value * unit + (token_log[slot] - scale).exp();
            value.output_token =
                self.policy.output_token_value * unit + (token_log[slot] - scale).exp();
            value.delivery = (delivery_log[slot] - scale).exp();
            value.age_price *= unit;
            age_price += value.age_price;
        }
        (values, age_price, scale)
    }

    fn next_progress(&self, r: &LiveRequest, value: LaneValue) -> Option<(u64, f64)> {
        let mut best: Option<(ValueRate, f64)> = None;
        let mut consider = |ns, work| {
            let rate = ValueRate::new(ns, work, 0.0);
            if best.is_none_or(|(old, _)| rate.compare(old).is_gt()) {
                best = Some((rate, work));
            }
        };
        if r.consumed_tokens < r.prompt_tokens as u64 {
            for &capacity in &self.prefill_catalog {
                let work = self.prefill_work(r, capacity);
                if work == 0 {
                    continue;
                }
                if let Some(ns) = self.costs.duration(GraphKind::Prefill, 1, capacity) {
                    consider(
                        ns,
                        value.prefill(
                            work,
                            r.consumed_tokens + work as u64 == r.prompt_tokens as u64,
                        ),
                    );
                }
            }
        } else {
            for batch in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
            {
                if r.generated_tokens == 0 || self.speculative_widths.is_empty() {
                    let kind = if r.generated_tokens == 0 {
                        GraphKind::Resume
                    } else {
                        GraphKind::Decode
                    };
                    if let Some(ns) = self.costs.duration(kind, batch, 1) {
                        consider(ns, value.decode(1.0, ns as f64 / 1e6));
                    }
                } else {
                    for &width in &self.speculative_catalog[batch.ilog2() as usize] {
                        if let Some(ns) = self.speculative_duration(r, batch, width)
                        {
                            let expected = self.expected_speculative_tokens(r, self.speculative_tokens(r, width));
                            consider(ns, value.decode(expected, ns as f64 / 1e6));
                        }
                    }
                }
            }
        }
        for &(batch, width, prefill) in &self.mixed_catalog {
            let Some(ns) = self
                .costs
                .shape_duration(GraphKind::Mixed, batch, width, prefill)
            else {
                continue;
            };
            if r.consumed_tokens < r.prompt_tokens as u64 {
                let work = self.prefill_work(r, prefill);
                if work > 0 {
                    consider(
                        ns,
                        value.prefill(
                            work,
                            r.consumed_tokens + work as u64 == r.prompt_tokens as u64,
                        ),
                    );
                }
            } else if r.generated_tokens > 0 {
                let tokens = self.speculative_tokens(r, width);
                consider(
                    ns,
                    value.decode(
                        self.expected_speculative_tokens(r, tokens),
                        ns as f64 / 1e6,
                    ),
                );
            }
        }
        best.map(|(rate, value)| (rate.duration_ns, value))
    }

    fn prerequisite_value(
        &self,
        r: &LiveRequest,
        kind: GraphKind,
        duration_ns: u64,
        value: LaneValue,
    ) -> f64 {
        if terminal_cleanup(r) {
            return value.decode(0.0, duration_ns as f64 / 1e6);
        }
        let Some((mut following, work)) = self.next_progress(r, value) else {
            return 0.0;
        };
        if duration_ns == 0 || following == 0 {
            return work;
        }
        for patches in r
            .vision_patches
            .iter()
            .skip(usize::from(kind == GraphKind::Vision))
        {
            let cost = self
                .capabilities
                .vision
                .patch_capacities
                .iter()
                .filter(|n| *n >= patches)
                .filter_map(|n| self.costs.duration(GraphKind::Vision, 1, *n))
                .min()
                .unwrap_or(0);
            if cost == 0 {
                return work;
            }
            following = following.saturating_add(cost);
        }
        work * duration_ns as f64 / duration_ns.saturating_add(following) as f64
    }

    fn following_kind(&self, r: &LiveRequest, kind: GraphKind) -> Option<GraphKind> {
        let phase = match kind {
            GraphKind::Snapshot => r.boundary_capture?.resume,
            GraphKind::Init | GraphKind::Restore | GraphKind::Vision => {
                if r.vision_patches.len() > usize::from(kind == GraphKind::Vision) {
                    return None;
                }
                if r.consumed_tokens == r.prompt_tokens as u64 {
                    RequestPhase::Resume
                } else {
                    RequestPhase::Prefill
                }
            }
            _ => return None,
        };
        match phase {
            RequestPhase::Prefill => Some(GraphKind::Prefill),
            RequestPhase::Resume => Some(GraphKind::Resume),
            RequestPhase::Decode if self.speculative_widths.is_empty() => Some(GraphKind::Decode),
            RequestPhase::Decode => Some(GraphKind::Speculative),
            _ => None,
        }
    }

    fn continuation_candidate(
        &self,
        kind: GraphKind,
        capacity: usize,
        duration_ns: u64,
        first_offers: &[(usize, f64)],
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) -> Option<(u64, ValueRate)> {
        if !matches!(
            kind,
            GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot | GraphKind::Vision
        ) {
            return None;
        }
        let mask = first_offers
            .iter()
            .fold(0, |mask, (slot, _)| mask | (1 << slot));
        let fill = |selected: u64| {
            let count = selected.count_ones() as usize;
            if count == capacity {
                return selected;
            }
            let mut remaining = [(0, 0.0); REQUEST_LANES];
            let mut n = 0;
            for &offer in first_offers {
                if selected & (1 << offer.0) == 0 {
                    remaining[n] = offer;
                    n += 1;
                }
            }
            selected | best_cohort::<REQUEST_LANES>(&remaining[..n], capacity - count)
        };
        let mut best: Option<(u64, ValueRate)> = None;
        let mut consider = |following, batch, width, prefill| {
            let ns = if following == GraphKind::Speculative {
                self.proposal_round_duration(batch, width)
            } else { self.costs.shape_duration(following, batch, width, prefill) };
            let Some(ns) = ns else {
                return;
            };
            let horizon = if ns == 0 || duration_ns == 0 {
                0
            } else {
                ns.saturating_add(duration_ns)
            };
            let total_ms = horizon as f64 / 1e6;
            let first_ms = duration_ns as f64 / 1e6;
            let mut preparing = [(0, 0.0); REQUEST_LANES];
            let mut ready = preparing;
            let mut finishing = preparing;
            let mut prompts: [Option<(usize, f64)>; 2] = [None, None];
            let (mut p, mut r, mut f) = (0, 0, 0);
            let mixed = following == GraphKind::Mixed;
            let decode = if self.speculative_widths.is_empty() {
                GraphKind::Decode
            } else {
                GraphKind::Speculative
            };
            for (slot, entry) in self.slots.iter().enumerate() {
                let Some(request) = entry.request.as_ref() else {
                    continue;
                };
                let offered = mask & (1 << slot) != 0;
                if offered && terminal_cleanup(request) {
                    finishing[f] = (slot, values[slot].finish(first_ms, total_ms));
                    f += 1;
                    continue;
                }
                let promoted_kind = offered
                    .then(|| self.following_kind(request, kind))
                    .flatten();
                let ready_kind = self
                    .eligible(request)
                    .then(|| self.request_kind(request))
                    .flatten()
                    .map(|k| if k == GraphKind::Proposal { GraphKind::Speculative } else { k });
                let accepts = |k: GraphKind| {
                    if mixed {
                        k == GraphKind::Prefill || k == decode
                    } else {
                        k == following
                    }
                };
                let (role, promoted) = if let Some(k) = promoted_kind.filter(|k| accepts(*k)) {
                    (k, true)
                } else if let Some(k) = ready_kind.filter(|k| accepts(*k)) {
                    (k, false)
                } else {
                    continue;
                };
                let extent = if mixed && role == GraphKind::Prefill {
                    prefill
                } else {
                    width
                };
                if role == GraphKind::Prefill && self.prefill_work(request, extent) == 0 {
                    continue;
                }
                let offer = (
                    slot,
                    self.lane_value(request, role, extent, horizon, values[slot]),
                );
                if mixed && role == GraphKind::Prefill {
                    let best = &mut prompts[usize::from(promoted)];
                    if best.is_none_or(|(_, value)| offer.1 > value) {
                        *best = Some(offer);
                    }
                    continue;
                }
                if promoted {
                    preparing[p] = offer;
                    p += 1;
                } else {
                    ready[r] = offer;
                    r += 1;
                }
            }
            for prompt in std::iter::once(None).chain(prompts.into_iter().enumerate().filter_map(
                |(promoted, offer)| offer.map(|(slot, value)| Some((slot, value, promoted == 1))),
            )) {
                let preparing_prompt = usize::from(prompt.is_some_and(|(_, _, promoted)| promoted));
                let Some((selected, _, mut value)) = best_two_stage_cohort::<REQUEST_LANES>(
                    &preparing[..p],
                    &ready[..r],
                    &finishing[..f],
                    capacity - preparing_prompt,
                    batch - usize::from(prompt.is_some()),
                    1 - preparing_prompt,
                ) else {
                    continue;
                };
                let mut selected = selected;
                if let Some((slot, work, promoted)) = prompt {
                    value += work;
                    if promoted {
                        selected |= 1 << slot;
                    }
                }
                let rate = ValueRate::new(horizon, value, age_price);
                if best.is_none_or(|(_, old)| rate.compare(old).is_gt()) {
                    best = Some((fill(selected), rate));
                }
            }
        };
        for &width in &self.prefill_catalog {
            consider(GraphKind::Prefill, 1, width, 0);
        }
        for batch in crate::execution::BATCH_CAPACITIES
            .into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
        {
            consider(GraphKind::Resume, batch, 1, 0);
            if self.speculative_widths.is_empty() {
                consider(GraphKind::Decode, batch, 1, 0);
            } else {
                for &width in &self.speculative_catalog[batch.ilog2() as usize] {
                    consider(GraphKind::Speculative, batch, width, 0);
                }
            }
        }
        for &(batch, width, prefill) in &self.mixed_catalog {
            consider(GraphKind::Mixed, batch, width, prefill);
        }
        let mut terminal = [(0, 0.0); REQUEST_LANES];
        let mut count = 0;
        for slot in
            bits(mask).filter(|s| terminal_cleanup(self.slots[*s].request.as_ref().unwrap()))
        {
            terminal[count] = (slot, values[slot].decode(0.0, duration_ns as f64 / 1e6));
            count += 1;
        }
        if count > 0 {
            let selected = best_cohort::<REQUEST_LANES>(&terminal[..count], capacity);
            let immediate = terminal[..count]
                .iter()
                .filter(|(s, _)| selected & (1 << s) != 0)
                .map(|(_, v)| v)
                .sum();
            let rate = ValueRate::new(duration_ns, immediate, age_price);
            if best.is_none_or(|(_, old)| rate.compare(old).is_gt()) {
                best = Some((fill(selected), rate));
            }
        }
        best
    }

    fn lane_tokens(&self, r: &LiveRequest, kind: GraphKind, capacity: usize) -> usize {
        match kind {
            GraphKind::Prefill => self.prefill_work(r, capacity),
            GraphKind::Decode => 1,
            GraphKind::Verification => self.retained_proposal(r)
                .map_or_else(|| self.speculative_tokens(r, capacity), |p| p.width),
            GraphKind::Speculative | GraphKind::Proposal => {
                self.speculative_tokens(r, capacity)
            }
            _ => 0,
        }
    }

    fn lane_value(
        &self,
        r: &LiveRequest,
        kind: GraphKind,
        capacity: usize,
        duration_ns: u64,
        value: LaneValue,
    ) -> f64 {
        let tokens = self.lane_tokens(r, kind, capacity);
        match kind {
            GraphKind::Prefill => value.prefill(
                tokens,
                r.consumed_tokens + tokens as u64 == r.prompt_tokens as u64,
            ),
            GraphKind::Decode
            | GraphKind::Speculative
            | GraphKind::Proposal
            | GraphKind::Verification => value.decode(
                self.expected_speculative_tokens(r, tokens),
                duration_ns as f64 / 1e6,
            ),
            GraphKind::Resume => value.decode(1.0, duration_ns as f64 / 1e6),
            _ => self.prerequisite_value(r, kind, duration_ns, value),
        }
    }

    fn add_candidate(
        &self,
        output: &mut Vec<Candidate>,
        kind: GraphKind,
        batch: usize,
        capacity: usize,
        mask: u64,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) {
        let Some(duration_ns) = self.costs.duration(kind, batch, capacity) else {
            return;
        };
        let mut offers = [(0usize, 0.0); REQUEST_LANES];
        let mut count = 0;
        for slot in bits(mask) {
            let r = self.slots[slot].request.as_ref().unwrap();
            if kind == GraphKind::Prefill {
                let work = self.prefill_work(r, capacity);
                // Explore larger graphs only when they can consume additional tokens.
                if work == 0
                    || (duration_ns == 0 && self.prefill_catalog.iter()
                        .take_while(|&&smaller| smaller < capacity)
                        .any(|&smaller| self.prefill_work(r, smaller) == work))
                {
                    continue;
                }
            }
            if kind == GraphKind::Vision && capacity < r.vision_patches[0] {
                continue;
            }
            if kind == GraphKind::Verification
                && self.retained_proposal(r).is_none_or(|p| p.width > capacity)
            {
                continue;
            }
            offers[count] = (
                slot,
                self.lane_value(r, kind, capacity, duration_ns, values[slot]),
            );
            count += 1;
        }
        if count == 0 {
            return;
        }
        let (selected, expected) = if kind == GraphKind::Prefill && batch == 1 && self.policy.prefill_srpt {
            let offered = |slot: usize| offers[..count].iter().any(|(s, _)| *s == slot);
            let slot = match self.srpt_first() {
                Some(first) if offered(first) => first,
                // The first prompt's chunk is in flight: its next chunk waits for that feedback, and no other
                // prompt's chunk is offered meanwhile (prompts finish one after another).
                Some(first) if self.slots[first].request.as_ref().is_some_and(|r| r.pending > 0) => return,
                // Not offered for another reason (e.g. another capacity serves it): the best offered prompt in
                // SRPT order runs, so prefill never stalls on an absent leader.
                _ => offers[..count]
                    .iter()
                    .map(|(s, _)| *s)
                    .min_by_key(|&s| self.srpt_key(s, self.slots[s].request.as_ref().unwrap()))
                    .unwrap(),
            };
            let value = if self.policy.prefill_srpt_decode_priority_tokens > 0 {
                // Bounded decode priority: value the step by the prompt that runs.
                offers[..count].iter().find(|(s, _)| *s == slot).map_or(0.0, |(_, v)| *v)
            } else {
                offers[..count].iter().map(|(_, v)| *v).fold(0.0f64, f64::max)
            };
            (1u64 << slot, value)
        } else {
            let selected = best_cohort::<REQUEST_LANES>(&offers[..count], batch);
            let expected = offers[..count]
                .iter()
                .filter(|(s, _)| selected & (1 << s) != 0)
                .map(|(_, v)| v)
                .sum::<f64>();
            (selected, expected)
        };
        let feedback = !self.costs.is_fresh(cost::Key {
            kind, batch, tokens: capacity, prefill: 0, packed: 0,
        })
            || !(matches!(kind, GraphKind::Decode | GraphKind::Prefill)
                || (kind == GraphKind::Speculative && self.policy.pipelined_speculation))
            || bits(selected).any(|slot| {
                let r = self.slots[slot].request.as_ref().unwrap();
                r.grammar_constrained
                    || r.host_stops
                    || (kind == GraphKind::Speculative && !self.queued_round_is_exact(r, capacity))
                    || (kind == GraphKind::Prefill
                        && (r.consumed_tokens + self.prefill_work(r, capacity) as u64
                            == r.prompt_tokens as u64
                            || r.captures_message(
                                r.consumed_tokens as usize + self.prefill_work(r, capacity),
                            )))
            });
        // A step that needs completion feedback (a new request's Init, a prompt's final chunk, a
        // stale cost) stays a candidate behind in-flight work; `select_plan` drains the window
        // when it is the most valuable step.
        let (selected, rate) = self
            .continuation_candidate(
                kind,
                batch,
                duration_ns,
                &offers[..count],
                values,
                age_price,
            )
            .unwrap_or_else(|| (selected, ValueRate::new(duration_ns, expected, age_price)));
        output.push(Candidate { packed_token_capacity: 0, verification_budget: None,
            kind,
            mask: selected,
            batch,
            capacity,
            prefill_capacity: 0,
            prefill_slot: None,
            work: if kind == GraphKind::Prefill {
                self.prefill_work(
                    self.slots[selected.trailing_zeros() as usize]
                        .request
                        .as_ref()
                        .unwrap(),
                    capacity,
                )
            } else {
                0
            },
            duration_ns,
            rate,
            feedback,
        });
    }

    fn candidates(
        &self,
        output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) {
        let hold_prefill = self.decode_priority_holds_prefill() || self.short_prompt_preparing();
        self.candidates_with(output, values, age_price, hold_prefill);
        // A hold never withholds every plan from an idle GPU: if nothing else can run, prompt chunks are offered.
        if hold_prefill && output.is_empty() && self.submissions.is_empty() {
            self.candidates_with(output, values, age_price, false);
        }
    }

    /// Candidates; `hold_prefill` removes prompt lanes (standalone prefill and the prompt of mixed graphs). Split
    /// speculation candidates are Proposal/Verification only and are not affected.
    fn candidates_with(
        &self,
        output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
        hold_prefill: bool,
    ) {
        output.clear();
        self.mixed_candidates(output, values, age_price, hold_prefill);
        if self.split.is_some() {
            self.split_candidates(output, values, age_price);
        }
        for kind in [
            GraphKind::Prefill,
            GraphKind::Vision,
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Resume,
            GraphKind::Decode,
            GraphKind::Speculative,
        ] {
            let mask = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, entry)| {
                    entry
                        .request
                        .as_ref()
                        .filter(|r| self.eligible(r) && self.request_kind(r) == Some(kind))
                        .map(|_| 1u64 << slot)
                })
                .fold(0, |a, b| a | b);
            if mask == 0 || (kind == GraphKind::Prefill && hold_prefill) {
                continue;
            }
            if kind == GraphKind::Prefill {
                for &capacity in &self.prefill_catalog {
                    self.add_candidate(output, kind, 1, capacity, mask, values, age_price);
                }
            } else if kind == GraphKind::Vision {
                for &capacity in &self.capabilities.vision.patch_capacities {
                    self.add_candidate(output, kind, 1, capacity, mask, values, age_price);
                }
            } else {
                for batch in crate::execution::BATCH_CAPACITIES
                    .into_iter()
                    .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
                {
                    if kind == GraphKind::Speculative {
                        for &capacity in &self.speculative_catalog[batch.ilog2() as usize] {
                            self.add_candidate(
                                output, kind, batch, capacity, mask, values, age_price,
                            );
                        }
                    } else {
                        self.add_candidate(output, kind, batch, 1, mask, values, age_price);
                    }
                }
            }
        }
    }

    fn mixed_candidates(
        &self,
        output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
        hold_prefill: bool,
    ) {
        if !self.submissions.is_empty() {
            return;
        }
        let decode = if self.speculative_widths.is_empty() {
            GraphKind::Decode
        } else {
            GraphKind::Speculative
        };
        for &(batch, width, prefill) in &self.mixed_catalog {
            let Some(duration_ns) =
                self.costs
                    .shape_duration(GraphKind::Mixed, batch, width, prefill)
            else {
                continue;
            };
            let mut decoders = [(0usize, 0.0); REQUEST_LANES];
            let mut count = 0;
            let mut prompt: Option<(usize, f64)> = None;
            for (slot, entry) in self.slots.iter().enumerate() {
                let Some(r) = entry.request.as_ref().filter(|r| self.eligible(r)) else {
                    continue;
                };
                match self.request_kind(r) {
                    Some(GraphKind::Prefill) if !hold_prefill && self.prefill_work(r, prefill) > 0 => {
                        if duration_ns == 0 && self.mixed_catalog.iter().any(|&(b, w, smaller)| {
                            b == batch && w == width && smaller < prefill
                                && self.prefill_work(r, smaller) == self.prefill_work(r, prefill)
                        }) {
                            continue;
                        }
                        let value = self.lane_value(
                            r,
                            GraphKind::Prefill,
                            prefill,
                            duration_ns,
                            values[slot],
                        );
                        let better = match prompt {
                            None => true,
                            Some((other, _)) if self.policy.prefill_srpt => {
                                self.srpt_key(slot, r)
                                    < self.srpt_key(other, self.slots[other].request.as_ref().unwrap())
                            }
                            Some((_, best)) => value > best,
                        };
                        if better {
                            prompt = Some((slot, value));
                        }
                    }
                    Some(kind) if kind == decode => {
                        decoders[count] = (
                            slot,
                            self.lane_value(r, decode, width, duration_ns, values[slot]),
                        );
                        count += 1;
                    }
                    _ => {}
                }
            }
            for prompt in std::iter::once(None).chain(prompt.map(Some)) {
                if prompt.is_none() && duration_ns == 0
                    && self.mixed_catalog.iter().any(|&(b, w, smaller)| {
                        b == batch && w == width && smaller < prefill
                    })
                {
                    continue;
                }
                let mut mask = best_cohort::<REQUEST_LANES>(
                    &decoders[..count],
                    batch - usize::from(prompt.is_some()),
                );
                let mut expected = decoders[..count]
                    .iter()
                    .filter(|(s, _)| mask & (1 << s) != 0)
                    .map(|(_, v)| v)
                    .sum::<f64>();
                if let Some((slot, value)) = prompt {
                    mask |= 1 << slot;
                    expected += value;
                }
                if mask == 0 {
                    continue;
                }
                output.push(Candidate { packed_token_capacity: 0, verification_budget: None,
                    kind: GraphKind::Mixed,
                    mask,
                    batch,
                    capacity: width,
                    prefill_capacity: prefill,
                    prefill_slot: prompt.map(|(s, _)| s),
                    work: 0,
                    duration_ns,
                    rate: ValueRate::new(duration_ns, expected, age_price),
                    feedback: true,
                });
            }
        }
    }

    pub(super) fn select_plan(&self) -> Option<ScheduledPlan> {
        if self.submissions.len() >= 2 || self.submissions.values().any(|s| s.requires_feedback) {
            return None;
        }
        let window = self.window_remaining()?;
        let (values, age_price, log_scale) = self.lane_values(window);
        let mut candidates = self.candidates.borrow_mut();
        self.candidates(&mut candidates, &values, age_price);
        let compare = |a: &Candidate, b: &Candidate| {
            let last = |c: &Candidate| {
                bits(c.mask)
                    .map(|s| self.slots[s].request.as_ref().unwrap().last_use.0 as u128)
                    .sum::<u128>()
            };
            a.rate
                .compare(b.rate)
                .then(a.mask.count_ones().cmp(&b.mask.count_ones()))
                .then(last(b).cmp(&last(a)))
                .then(b.batch.cmp(&a.batch))
                .then(b.capacity.cmp(&a.capacity))
                .then(b.prefill_capacity.cmp(&a.prefill_capacity))
                .then(b.mask.cmp(&a.mask))
        };
        let ordinary = *candidates.iter().max_by(|a, b| compare(a, b))?;
        if let Some(episode) = self.refresh_episode() {
            let mut continuations = Vec::new();
            self.refresh_verification_candidates(
                &mut continuations, &values, age_price, episode.owned_mask(),
                (episode.target.kind == GraphKind::Verification).then_some(episode.target),
            );
            let candidate = *continuations.iter().max_by(|a, b| compare(a, b))?;
            return Some(self.materialize(candidate, log_scale,
                PlanOrigin::RefreshContinuation(episode.id)));
        }
        // Pipelined decoding keeps work in flight, so a step that needs completion feedback could
        // never be queued and a new request would wait until every decoder finished. When such a
        // step is the most valuable, queue nothing: the window drains and it runs next.
        if ordinary.feedback && !self.submissions.is_empty() {
            return None;
        }
        if ordinary.rate.duration_ns == 0 {
            return Some(self.materialize(ordinary, log_scale, PlanOrigin::Discovery));
        }
        if self.refresh_ready() && self.submissions.is_empty() {
            let mut targets: Vec<_> = self.costs.stale_keys().filter(|key| *key != ordinary.cost_key())
                .map(|key| (self.costs.refresh_order(key), key)).collect();
            targets.sort_by_key(|(order, _)| *order);
            let mut best: Option<(u64, cost::Key, Candidate)> = None;
            let mut choices = Vec::new();
            let mut budget_cache = split::RefreshBudgetCache::default();
            for (order, target) in targets {
                if best.is_some_and(|(best_order, _, _)| best_order < order) { break; }
                choices.clear();
                choices.extend(candidates.iter().copied().filter(|candidate| candidate.cost_key() == target));
                if self.split.is_some() {
                    self.split_refresh_candidates(&mut choices, target, &values, age_price, Some(&mut budget_cache));
                }
                if let Some(candidate) = choices.iter().copied().max_by(compare) {
                    if best.is_none_or(|(_, _, previous)| compare(&candidate, &previous).is_gt()) {
                        best = Some((order, target, candidate));
                    }
                }
            }
            if let Some((_, target, candidate)) = best {
                return Some(self.materialize(candidate, log_scale, PlanOrigin::RefreshStart(target)));
            }
        }
        Some(self.materialize(ordinary, log_scale, PlanOrigin::Ordinary))
    }

    fn materialize(&self, candidate: Candidate, log_scale: f64, origin: PlanOrigin) -> ScheduledPlan {
        let mut lanes = Vec::with_capacity(candidate.mask.count_ones() as usize);
        let mut epochs = Vec::with_capacity(lanes.capacity());
        for slot in bits(candidate.mask) {
            let r = self.slots[slot].request.as_ref().unwrap();
            let kind = if candidate.kind != GraphKind::Mixed {
                candidate.kind
            } else if candidate.prefill_slot == Some(slot) {
                GraphKind::Prefill
            } else if self.speculative_widths.is_empty() {
                GraphKind::Decode
            } else {
                GraphKind::Speculative
            };
            epochs.push(r.epoch);
            lanes.push(ScheduledLane {
                request: r.handle,
                start_position: r.scheduled_consumed,
                tokens: if kind == GraphKind::Prefill {
                    if candidate.kind == GraphKind::Mixed {
                        self.prefill_work(r, candidate.prefill_capacity)
                    } else {
                        candidate.work
                    }
                } else {
                    self.lane_tokens(r, kind, candidate.capacity)
                },
            });
        }
        ScheduledPlan {
            origin,
            selected_cost: self.selected_cost(candidate.cost_key()),
            owner: self.plan_owner.clone(),
            epochs,
            estimated_duration_ns: candidate.duration_ns,
            selection_value_rate: candidate.rate.score,
            selection_log_scale: log_scale,
            verification_budget: candidate.verification_budget,
            step: ScheduledStep { packed_token_capacity: candidate.packed_token_capacity,
                kind: candidate.kind,
                batch_capacity: candidate.batch,
                token_capacity: candidate.capacity,
                mixed: (candidate.kind == GraphKind::Mixed).then(|| MixedStep {
                    prefill_capacity: candidate.prefill_capacity,
                    prefill_row: candidate
                        .prefill_slot
                        .map(|s| (candidate.mask & ((1u64 << s) - 1)).count_ones() as usize),
                    speculative: !self.speculative_widths.is_empty(),
                }),
                lanes,
                requires_feedback: candidate.feedback,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn phase_plan(
        &self,
        kind: GraphKind,
        request: Option<RequestHandle>,
    ) -> ScheduledPlan {
        let (values, age_price, scale) = self.lane_values(0);
        let mask = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                entry
                    .request
                    .as_ref()
                    .filter(|r| {
                        self.eligible(r)
                            && self.request_kind(r) == Some(kind)
                            && request.is_none_or(|h| h == r.handle)
                    })
                    .map(|_| 1u64 << slot)
            })
            .fold(0, |a, b| a | b);
        let mut candidates = Vec::new();
        if kind == GraphKind::Prefill {
            for &capacity in &self.prefill_catalog {
                self.add_candidate(&mut candidates, kind, 1, capacity, mask, &values, age_price);
            }
        } else {
            for batch in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
            {
                self.add_candidate(&mut candidates, kind, batch, 1, mask, &values, age_price);
            }
        }
        let candidate = *candidates
            .iter()
            .max_by(|a, b| {
                a.rate
                    .compare(b.rate)
                    .then(a.mask.count_ones().cmp(&b.mask.count_ones()))
                    .then(b.batch.cmp(&a.batch))
            })
            .expect("requested lifecycle operation must be feasible");
        let origin = if candidate.rate.duration_ns == 0 { PlanOrigin::Discovery } else { PlanOrigin::Ordinary };
        self.materialize(candidate, scale, origin)
    }
}

pub(super) fn terminal_cleanup(request: &LiveRequest) -> bool {
    request.phase == RequestPhase::Snapshot
        && request.boundary_capture.is_none_or(|capture| {
            matches!(
                capture.resume,
                RequestPhase::Snapshot | RequestPhase::Finishing
            )
        })
}

pub(super) fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let slot = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(slot)
    })
}

pub(super) fn scratch() -> std::cell::RefCell<Vec<Candidate>> {
    std::cell::RefCell::new(Vec::new())
}

#[cfg(test)]
#[path = "prefill_srpt_tests.rs"]
mod prefill_srpt_tests;

#[cfg(test)]
#[path = "prefill_srpt_refinement_tests.rs"]
mod prefill_srpt_refinement_tests;
