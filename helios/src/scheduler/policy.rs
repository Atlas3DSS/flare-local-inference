//! Request-level useful-progress accounting and selection over prepared graphs.

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct PolicyConfig {
    pub max_slice_ns: u64,
    pub submission_window_ns: u64,
    pub service_gap_ns: u64,
    pub deficit_scale_ns: u64,
    pub estimate_ttl_ns: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            max_slice_ns: 20_000_000,
            submission_window_ns: 20_000_000,
            service_gap_ns: 40_000_000,
            deficit_scale_ns: 20_000_000,
            estimate_ttl_ns: 30_000_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RequestAccounting {
    pub arrival_ns: u64,
    pub last_progress_ns: u64,
    pub last_output_ns: Option<u64>,
    pub deficit_ns: f64,
    pub entitlement_ns: f64,
    pub committed_credit_ns: f64,
    pub prefill_credit_ns: f64,
    pub decode_credit_ns: f64,
    committed_prefill_tokens: u64,
    committed_output_tokens: u64,
}

impl RequestAccounting {
    pub(super) fn credit(&mut self, prefill: u64, output: u32, now: u64) {
        self.committed_prefill_tokens += prefill;
        self.committed_output_tokens += u64::from(output);
        let credit =
            prefill as f64 * self.prefill_credit_ns + output as f64 * self.decode_credit_ns;
        self.committed_credit_ns += credit;
        self.deficit_ns = (self.deficit_ns - credit).max(0.0);
        if prefill != 0 || output != 0 {
            self.last_progress_ns = now;
        }
    }

    fn initialize_rates(&mut self, prefill: f64, decode: f64) {
        let mut credit = 0.0;
        if self.prefill_credit_ns == 0.0 && prefill > 0.0 {
            self.prefill_credit_ns = prefill;
            credit += self.committed_prefill_tokens as f64 * prefill;
        }
        if self.decode_credit_ns == 0.0 && decode > 0.0 {
            self.decode_credit_ns = decode;
            credit += self.committed_output_tokens as f64 * decode;
        }
        self.committed_credit_ns += credit;
        self.deficit_ns = (self.deficit_ns - credit).max(0.0);
    }
}

#[derive(Clone, Copy)]
pub(super) struct Candidate {
    kind: GraphKind,
    mask: u64,
    capacity: usize,
    work: usize,
    duration_ns: u64,
    feedback: bool,
    lateness: f64,
    score: f64,
}

impl RequestScheduler {
    pub fn configure_policy(&mut self, policy: PolicyConfig) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("scheduling policy must be configured before admission".into());
        }
        if policy.max_slice_ns == 0
            || policy.submission_window_ns == 0
            || policy.service_gap_ns == 0
            || policy.deficit_scale_ns == 0
            || policy.estimate_ttl_ns == 0
        {
            return Err("scheduling time budgets must be positive".into());
        }
        self.policy = policy;
        self.costs.advance_to(self.now_ns, policy.estimate_ttl_ns);
        Ok(())
    }

    /// Integrates occupied execution time once, before changing the runnable set.
    pub fn advance_to(&mut self, now_ns: u64) -> Result<()> {
        if now_ns < self.now_ns {
            return Err("scheduler clock must be monotonic".into());
        }
        let elapsed = now_ns - self.now_ns;
        self.now_ns = now_ns;
        self.costs.advance_to(now_ns, self.policy.estimate_ttl_ns);
        if elapsed == 0 || self.submissions.is_empty() {
            return Ok(());
        }
        let count = self
            .slots
            .iter()
            .filter_map(|s| s.request.as_ref())
            .filter(|r| gpu_runnable(r))
            .count();
        if count != 0 {
            let share = elapsed as f64 / count as f64;
            for request in self
                .slots
                .iter_mut()
                .filter_map(|s| s.request.as_mut())
                .filter(|r| gpu_runnable(r))
            {
                request.accounting.entitlement_ns += share;
                request.accounting.deficit_ns += share;
            }
        }
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

    pub fn published_output(&mut self, handle: RequestHandle) -> Result<()> {
        self.request(handle)?;
        self.slots[handle.slot as usize]
            .request
            .as_mut()
            .unwrap()
            .accounting
            .last_output_ns = Some(self.now_ns);
        Ok(())
    }

    pub(super) fn accounting(&self, remaining_prefill: usize) -> RequestAccounting {
        let (prefill_credit_ns, decode_credit_ns) = self.reference_rates(remaining_prefill);
        RequestAccounting {
            arrival_ns: self.now_ns,
            last_progress_ns: self.now_ns,
            prefill_credit_ns,
            decode_credit_ns,
            ..Default::default()
        }
    }

    fn reference_rates(&self, remaining_prefill: usize) -> (f64, f64) {
        let decode = crate::execution::BATCH_CAPACITIES
            .into_iter()
            .find_map(|batch| {
                self.costs
                    .reference_duration(GraphKind::Decode, batch, 1)
                    .or_else(|| {
                        self.speculative_catalog[batch.ilog2() as usize]
                            .iter()
                            .find_map(|width| {
                                self.costs
                                    .reference_duration(GraphKind::Speculative, batch, *width)
                            })
                    })
                    .map(|n| n as f64 / batch as f64)
            })
            .unwrap_or(0.0);
        let prefill = if remaining_prefill == 0 {
            0.0
        } else {
            self.prefill_catalog
                .iter()
                .rev()
                .find_map(|capacity| {
                    let cost = self
                        .costs
                        .reference_duration(GraphKind::Prefill, 1, *capacity)?;
                    let full = remaining_prefill / capacity;
                    let tail = remaining_prefill % capacity;
                    let tail_cost = if tail == 0 {
                        0
                    } else {
                        self.prefill_catalog
                            .iter()
                            .filter(|n| **n >= tail)
                            .find_map(|n| self.costs.reference_duration(GraphKind::Prefill, 1, *n))
                            .unwrap_or(cost)
                    };
                    Some((full as f64 * cost as f64 + tail_cost as f64) / remaining_prefill as f64)
                })
                .unwrap_or(0.0)
        };
        (prefill, decode)
    }

    pub(super) fn initialize_credit_rates(&mut self) {
        for slot in 0..self.slots.len() {
            let Some(request) = self.slots[slot].request.as_ref() else {
                continue;
            };
            if request.accounting.prefill_credit_ns != 0.0
                && request.accounting.decode_credit_ns != 0.0
            {
                continue;
            }
            let remaining = request.prompt_tokens - request.admission_reused_tokens as usize;
            let (prefill, decode) = self.reference_rates(remaining);
            self.slots[slot]
                .request
                .as_mut()
                .unwrap()
                .accounting
                .initialize_rates(prefill, decode);
        }
    }

    fn request_kind(&self, request: &LiveRequest) -> Option<GraphKind> {
        Some(match request.phase {
            RequestPhase::Initialize => GraphKind::Init,
            RequestPhase::Restore => GraphKind::Restore,
            RequestPhase::Vision => GraphKind::Vision,
            RequestPhase::Prefill => GraphKind::Prefill,
            RequestPhase::Resume => GraphKind::Resume,
            RequestPhase::Decode if self.speculative_widths.is_empty() => GraphKind::Decode,
            RequestPhase::Decode => GraphKind::Speculative,
            RequestPhase::Snapshot => GraphKind::Snapshot,
            _ => return None,
        })
    }

    fn eligible(&self, request: &LiveRequest) -> bool {
        if self.request_kind(request).is_none() {
            return false;
        }
        if request.pending > 0
            && !(request.phase == RequestPhase::Decode
                && self.speculative_widths.is_empty()
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
        } else {
            true
        }
    }

    fn prefill_work(&self, request: &LiveRequest, capacity: usize) -> usize {
        let start = request.consumed_tokens as usize;
        let mut end = request
            .system_boundary
            .filter(|n| *n > start)
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

    fn service_clock(&self, request: &LiveRequest) -> u64 {
        let accounting = &request.accounting;
        if request.consumed_tokens < request.prompt_tokens as u64 || terminal_cleanup(request) {
            accounting.last_progress_ns
        } else {
            accounting.last_output_ns.unwrap_or(accounting.arrival_ns)
        }
    }

    pub fn submission_window_available(&self, duration_ns: u64) -> bool {
        self.submissions.len() < 2
            && self.window_remaining().is_some_and(|remaining| {
                self.submissions.is_empty()
                    || (duration_ns != 0
                        && duration_ns <= self.policy.max_slice_ns
                        && remaining.saturating_add(duration_ns)
                            <= self.policy.submission_window_ns)
            })
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

    fn add_candidate(
        &self,
        output: &mut Vec<Candidate>,
        kind: GraphKind,
        mask: u64,
        capacity: usize,
    ) {
        if mask == 0 {
            return;
        }
        let batch = (mask.count_ones() as usize).next_power_of_two();
        let Some(duration_ns) = self.costs.duration(kind, batch, capacity) else {
            return;
        };
        let first = self.slots[mask.trailing_zeros() as usize]
            .request
            .as_ref()
            .unwrap();
        let work = if kind == GraphKind::Prefill {
            self.prefill_work(first, capacity)
        } else {
            0
        };
        if kind == GraphKind::Prefill
            && (work == 0
                || self.prefill_catalog.iter().copied().find(|n| *n >= work) != Some(capacity))
        {
            return;
        }
        let feedback = duration_ns == 0
            || kind == GraphKind::Speculative
            || bits(mask).any(|slot| {
                let r = self.slots[slot].request.as_ref().unwrap();
                r.grammar_constrained || r.host_stops
            });
        output.push(Candidate {
            kind,
            mask,
            capacity,
            work,
            duration_ns,
            feedback,
            lateness: 0.0,
            score: 0.0,
        });
    }

    fn add_cohort(&self, output: &mut Vec<Candidate>, kind: GraphKind, mask: u64) {
        if kind == GraphKind::Speculative {
            let batch = (mask.count_ones() as usize).next_power_of_two();
            for width in &self.speculative_catalog[batch.ilog2() as usize] {
                self.add_candidate(output, kind, mask, *width);
            }
        } else {
            self.add_candidate(output, kind, mask, 1);
        }
    }

    fn candidates(&self, output: &mut Vec<Candidate>) {
        output.clear();
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
            if mask == 0 {
                continue;
            }
            if kind == GraphKind::Prefill {
                for slot in bits(mask) {
                    for capacity in &self.prefill_catalog {
                        self.add_candidate(output, kind, 1 << slot, *capacity);
                    }
                }
                continue;
            }
            if kind == GraphKind::Vision {
                for slot in bits(mask) {
                    let patches = self.slots[slot].request.as_ref().unwrap().vision_patches[0];
                    if let Some(capacity) = self
                        .capabilities
                        .vision
                        .patch_capacities
                        .iter()
                        .find(|n| **n >= patches)
                    {
                        self.add_candidate(output, kind, 1 << slot, *capacity);
                    }
                }
                continue;
            }
            let count = mask.count_ones() as usize;
            if count <= 5 {
                let mut subset = mask;
                while subset != 0 {
                    self.add_cohort(output, kind, subset);
                    subset = subset.wrapping_sub(1) & mask;
                }
            } else {
                for slot in bits(mask) {
                    self.add_cohort(output, kind, 1 << slot);
                }
                self.add_cohort(output, kind, mask);
                let mut order = [0usize; REQUEST_LANES];
                for (i, slot) in bits(mask).enumerate() {
                    order[i] = slot;
                }
                for ordering in 0..5 {
                    order[..count].sort_unstable_by(|a, b| {
                        let a = self.slots[*a].request.as_ref().unwrap();
                        let b = self.slots[*b].request.as_ref().unwrap();
                        let acceptance = |r: &LiveRequest| {
                            r.acceptance.expected_tokens(
                                self.proposal_source,
                                self.speculative_tokens(r, 8),
                            )
                        };
                        match ordering {
                            0 => self.service_clock(a).cmp(&self.service_clock(b)),
                            1 => b.accounting.deficit_ns.total_cmp(&a.accounting.deficit_ns),
                            2 => acceptance(b).total_cmp(&acceptance(a)),
                            3 => acceptance(a).total_cmp(&acceptance(b)),
                            _ => a.scheduled_consumed.cmp(&b.scheduled_consumed),
                        }
                        .then(a.last_use.cmp(&b.last_use))
                        .then(a.handle.slot.cmp(&b.handle.slot))
                    });
                    let mut subset = 0;
                    for (i, slot) in order[..count].iter().enumerate() {
                        subset |= 1 << slot;
                        if i > 0 && (i + 1).is_power_of_two() && i + 1 < count {
                            self.add_cohort(output, kind, subset);
                        }
                    }
                }
            }
        }
    }

    fn next_credit(&self, request: &LiveRequest) -> f64 {
        let remaining = request
            .prompt_tokens
            .saturating_sub(request.consumed_tokens as usize);
        if remaining > 0 {
            request.accounting.prefill_credit_ns.max(1.0)
                * remaining.min(self.limits.prefill_chunk_size) as f64
        } else {
            request.accounting.decode_credit_ns.max(1.0)
        }
    }

    fn useful_credit(
        &self,
        request: &LiveRequest,
        kind: GraphKind,
        tokens: usize,
        score: bool,
    ) -> f64 {
        let prefill_credit = request.accounting.prefill_credit_ns.max(f64::from(score));
        let decode_credit = request.accounting.decode_credit_ns.max(f64::from(score));
        match kind {
            GraphKind::Prefill => {
                prefill_credit * tokens as f64
                    + if request.consumed_tokens + tokens as u64 == request.prompt_tokens as u64 {
                        decode_credit
                            * if score && request.generated_tokens == 0 {
                                2.0
                            } else {
                                1.0
                            }
                    } else {
                        0.0
                    }
            }
            GraphKind::Decode | GraphKind::Speculative => {
                decode_credit
                    * request
                        .acceptance
                        .expected_tokens(self.proposal_source, tokens)
            }
            GraphKind::Resume => {
                decode_credit
                    * if score && request.generated_tokens == 0 {
                        2.0
                    } else {
                        1.0
                    }
            }
            _ if !score => 0.0,
            _ if terminal_cleanup(request) => decode_credit / 8.0,
            _ => self.next_credit(request),
        }
    }

    fn lane_tokens(&self, request: &LiveRequest, candidate: &Candidate) -> usize {
        match candidate.kind {
            GraphKind::Prefill => candidate.work,
            GraphKind::Decode => 1,
            GraphKind::Speculative => self.speculative_tokens(request, candidate.capacity),
            _ => 0,
        }
    }

    fn following_rescue(&self, request: &LiveRequest, kind: GraphKind) -> u64 {
        if matches!(
            kind,
            GraphKind::Prefill | GraphKind::Decode | GraphKind::Speculative | GraphKind::Resume
        ) || terminal_cleanup(request)
        {
            return 0;
        }
        let mut total: u64 = 0;
        let skip = usize::from(kind == GraphKind::Vision);
        for patches in request.vision_patches.iter().skip(skip) {
            if let Some(capacity) = self
                .capabilities
                .vision
                .patch_capacities
                .iter()
                .find(|n| *n >= patches)
            {
                total = total.saturating_add(
                    self.costs
                        .duration(GraphKind::Vision, 1, *capacity)
                        .unwrap_or(0),
                );
            }
        }
        if request.consumed_tokens < request.prompt_tokens as u64 {
            let minimum = self
                .prefill_catalog
                .iter()
                .filter(|n| self.prefill_work(request, **n) > 0)
                .filter_map(|n| self.costs.duration(GraphKind::Prefill, 1, *n))
                .min()
                .unwrap_or(0);
            total.saturating_add(minimum)
        } else if request.generated_tokens == 0 {
            total.saturating_add(self.costs.duration(GraphKind::Resume, 1, 1).unwrap_or(0))
        } else {
            let minimum = if self.speculative_widths.is_empty() {
                self.costs.duration(GraphKind::Decode, 1, 1)
            } else {
                self.speculative_catalog[0]
                    .iter()
                    .filter_map(|w| self.costs.duration(GraphKind::Speculative, 1, *w))
                    .min()
            };
            total.saturating_add(minimum.unwrap_or(0))
        }
    }

    pub(super) fn select_plan(&self) -> Option<ScheduledPlan> {
        if self.submissions.len() >= 2 || self.submissions.values().any(|s| s.requires_feedback) {
            return None;
        }
        let window = self.window_remaining()?;
        let mut candidates = self.candidates.borrow_mut();
        self.candidates(&mut candidates);
        let mut minimum = [u64::MAX; REQUEST_LANES];
        for c in candidates.iter().filter(|c| c.mask.count_ones() == 1) {
            let slot = c.mask.trailing_zeros() as usize;
            minimum[slot] = minimum[slot].min(c.duration_ns);
        }
        candidates.retain(|c| {
            let unavoidable = c.mask.count_ones() == 1
                && c.duration_ns == minimum[c.mask.trailing_zeros() as usize];
            (!c.feedback || self.submissions.is_empty())
                && (c.duration_ns <= self.policy.max_slice_ns
                    || (self.submissions.is_empty() && unavoidable))
                && (window.saturating_add(c.duration_ns) <= self.policy.submission_window_ns
                    || (self.submissions.is_empty() && unavoidable))
        });
        if candidates.is_empty() {
            return None;
        }
        let mut order = [0usize; REQUEST_LANES];
        let mut count = 0;
        for (slot, entry) in self.slots.iter().enumerate() {
            if entry
                .request
                .as_ref()
                .is_some_and(|r| self.eligible(r) && r.pending == 0)
                && minimum[slot] != u64::MAX
            {
                order[count] = slot;
                count += 1;
            }
        }
        order[..count].sort_unstable_by_key(|slot| {
            let r = self.slots[*slot].request.as_ref().unwrap();
            (
                self.service_clock(r),
                r.last_use,
                r.accounting.arrival_ns,
                *slot,
            )
        });
        let overdue = order[..count].first().copied().filter(|slot| {
            self.now_ns
                .saturating_sub(self.service_clock(self.slots[*slot].request.as_ref().unwrap()))
                >= self.policy.service_gap_ns
        });
        if let Some(slot) = overdue {
            candidates.retain(|c| c.mask & (1 << slot) != 0);
        }
        if candidates.is_empty() {
            return None;
        }
        let mut pending_credit = [0.0f64; REQUEST_LANES];
        let mut following = [0u64; REQUEST_LANES];
        for slot in &order[..count] {
            let r = self.slots[*slot].request.as_ref().unwrap();
            following[*slot] = self.following_rescue(r, self.request_kind(r).unwrap());
        }
        for submission in self.submissions.values() {
            for lane in &submission.lanes {
                if let Ok(r) = self.request(lane.request) {
                    pending_credit[lane.request.slot as usize] +=
                        self.useful_credit(r, submission.kind, lane.tokens, false);
                }
            }
        }
        for c in candidates.iter_mut() {
            let end = self
                .now_ns
                .saturating_add(window)
                .saturating_add(c.duration_ns);
            let mut rescue_end = end;
            let mut lateness = 0.0;
            for slot in &order[..count] {
                let r = self.slots[*slot].request.as_ref().unwrap();
                let included = c.mask & (1 << slot) != 0;
                let completion = if included && following[*slot] == 0 {
                    end
                } else {
                    rescue_end = rescue_end
                        .saturating_add(following[*slot])
                        .saturating_add(if included { 0 } else { minimum[*slot] });
                    rescue_end
                };
                lateness += completion.saturating_sub(
                    self.service_clock(r)
                        .saturating_add(self.policy.service_gap_ns),
                ) as f64;
            }
            c.lateness = lateness;
            c.score = bits(c.mask)
                .map(|slot| {
                    let r = self.slots[slot].request.as_ref().unwrap();
                    let weight = 1.0
                        + (r.accounting.deficit_ns - pending_credit[slot]).max(0.0)
                            / self.policy.deficit_scale_ns as f64;
                    weight * self.useful_credit(r, c.kind, self.lane_tokens(r, c), true)
                })
                .sum::<f64>()
                / c.duration_ns.max(1) as f64;
        }
        let best_lateness = candidates
            .iter()
            .map(|c| c.lateness)
            .min_by(f64::total_cmp)?;
        candidates.retain(|c| c.lateness <= best_lateness);
        if overdue.is_none() {
            if let Some(slot) = order[..count]
                .iter()
                .copied()
                .max_by(|a, b| {
                    self.slots[*a]
                        .request
                        .as_ref()
                        .unwrap()
                        .accounting
                        .deficit_ns
                        .total_cmp(
                            &self.slots[*b]
                                .request
                                .as_ref()
                                .unwrap()
                                .accounting
                                .deficit_ns,
                        )
                        .then(b.cmp(a))
                })
                .filter(|slot| {
                    self.slots[*slot]
                        .request
                        .as_ref()
                        .unwrap()
                        .accounting
                        .deficit_ns
                        >= self.policy.deficit_scale_ns as f64
                })
            {
                if candidates.iter().any(|c| c.mask & (1 << slot) != 0) {
                    candidates.retain(|c| c.mask & (1 << slot) != 0);
                }
            }
        }
        let candidate = *candidates.iter().max_by(|a, b| {
            let last = |c: &Candidate| {
                bits(c.mask)
                    .map(|s| self.slots[s].request.as_ref().unwrap().last_use.0 as u128)
                    .sum::<u128>()
            };
            (a.duration_ns == 0)
                .cmp(&(b.duration_ns == 0))
                .then(a.score.total_cmp(&b.score))
                .then(b.duration_ns.cmp(&a.duration_ns))
                .then(last(b).cmp(&last(a)))
                .then(b.capacity.cmp(&a.capacity))
                .then(b.mask.cmp(&a.mask))
        })?;
        let mut lanes = Vec::with_capacity(candidate.mask.count_ones() as usize);
        let mut epochs = Vec::with_capacity(lanes.capacity());
        for slot in bits(candidate.mask) {
            let r = self.slots[slot].request.as_ref().unwrap();
            epochs.push(r.epoch);
            lanes.push(ScheduledLane {
                request: r.handle,
                start_position: r.scheduled_consumed,
                tokens: self.lane_tokens(r, &candidate),
            });
        }
        Some(ScheduledPlan {
            owner: self.plan_owner.clone(),
            epochs,
            estimated_duration_ns: candidate.duration_ns,
            recovery_lateness_ns: candidate.lateness,
            step: ScheduledStep {
                kind: candidate.kind,
                batch_capacity: lanes.len().next_power_of_two(),
                token_capacity: candidate.capacity,
                lanes,
                requires_feedback: candidate.feedback,
            },
        })
    }
}

fn gpu_runnable(request: &LiveRequest) -> bool {
    !matches!(
        request.phase,
        RequestPhase::Inbound | RequestPhase::Finishing
    ) && !terminal_cleanup(request)
}

fn terminal_cleanup(request: &LiveRequest) -> bool {
    request.phase == RequestPhase::Snapshot
        && request.boundary_capture.is_none_or(|capture| {
            matches!(
                capture.resume,
                RequestPhase::Snapshot | RequestPhase::Finishing
            )
        })
}

fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
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
