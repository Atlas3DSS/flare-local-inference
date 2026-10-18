//! Bounded retained proposals, causal verification widths, and shared execution costs.

use super::{policy::{Candidate, bits}, utility::{LaneValue, ValueRate}, *};

mod packed;

pub(super) struct SplitSpeculation {
    pub width: usize,
    pub retained: [Option<RetainedProposal>; REQUEST_LANES],
    // Forecasts average prefix-survival probabilities, preserving correlation
    // between the conditional probabilities in each observed draft block.
    forecast: [Option<Forecast>; REQUEST_LANES],
    population: [f32; 7],
    pub(super) pending_budget: Option<(Fence, usize)>,
}

#[derive(Clone, Copy)]
struct Forecast {
    request: RequestHandle,
    prefixes: [f32; 7],
    observed_at: [u32; 7],
}

impl Forecast {
    fn estimate(&self, prior: &[f32; 7], generated: u32) -> [f32; 7] {
        std::array::from_fn(|i| {
            let age = generated.saturating_sub(self.observed_at[i]);
            let keep = 0.5f64.powf(f64::from(age) / FORECAST_HALF_LIFE_TOKENS) as f32;
            prior[i] + (self.prefixes[i] - prior[i]) * keep
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct RetainedProposal {
    pub request: RequestHandle,
    pub fence: Fence,
    pub consumed: u64,
    pub width: usize,
    limit: usize,
    confidence: [f32; 7],
}

#[derive(Clone, Copy, Default)]
struct PackedCall {
    mask: u64,
    batch: usize,
    width: usize,
    nanos: u64,
}

struct Packing {
    nanos: u64,
    calls: Vec<PackedCall>,
}

fn horizon(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a.saturating_add(b) }
}

const FORECAST_PRIOR: f32 = 0.5;
const FORECAST_WEIGHT: f32 = 0.25;
const POPULATION_WEIGHT: f32 = 1.0 / 64.0;
const FORECAST_HALF_LIFE_TOKENS: f64 = 32.0;
const FORECAST_REFRESH_TOKENS: u32 = 32;

fn expected(confidence: &[f32; 7], width: usize) -> f64 {
    let mut survival = 1.0;
    let mut tokens = 1.0;
    for &p in &confidence[..width.saturating_sub(1)] {
        survival *= f64::from(p);
        tokens += survival;
    }
    tokens
}

fn survival(confidence: &[f32; 7]) -> [f32; 7] {
    let mut prefix = 1.0;
    confidence.map(|p| { prefix *= p; prefix })
}

fn forecast_tokens(prefixes: &[f32; 7], width: usize) -> f64 {
    let mut bound = 1.0f32;
    1.0 + prefixes[..width.saturating_sub(1)].iter().map(|&p| {
        bound = bound.min(p);
        f64::from(bound)
    }).sum::<f64>()
}

// Sorting widths permits optimal contiguous groups with O(requests squared) transitions.
fn pack(needs: &[(usize, usize)], shapes: &[[Option<PackedCall>; 9]; REQUEST_LANES + 1]) -> Option<Packing> {
    if needs.is_empty() { return Some(Packing { nanos: 0, calls: vec![] }); }
    let mut rows = needs.to_vec();
    rows.sort_unstable_by_key(|&(slot, width)| (std::cmp::Reverse(width), slot));
    let mut masks = [0; REQUEST_LANES + 1];
    for (index, &(slot, _)) in rows.iter().enumerate() { masks[index + 1] = masks[index] | (1 << slot); }
    let mut best = [None; REQUEST_LANES + 1];
    best[0] = Some((0u64, 0usize, 0usize, PackedCall::default(), false));
    for end in 1..=rows.len() {
        for start in 0..end {
            let Some((previous, count, _, _, unknown)) = best[start] else { continue; };
            let Some(mut call) = shapes[end - start][rows[start].1] else { continue; };
            call.mask = masks[end] ^ masks[start];
            let unknown = unknown || call.nanos == 0;
            let total = previous.saturating_add(call.nanos);
            let key = (!unknown, total, count + 1, call.batch, call.width);
            if best[end].is_none_or(|(old, n, _, c, u)| key < (!u, old, n, c.batch, c.width)) {
                best[end] = Some((total, count + 1, start, call, unknown));
            }
        }
    }
    let (total, _, _, _, unknown) = best[rows.len()]?;
    let mut calls = Vec::new();
    let mut end = rows.len();
    while end > 0 {
        let (_, _, start, call, _) = best[end]?;
        calls.push(call);
        end = start;
    }
    Some(Packing { nanos: if unknown { 0 } else { total }, calls })
}

impl RequestScheduler {
    pub fn split_speculation(&mut self, width: usize) -> Result<()> {
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || !(2..=8).contains(&width)
            || self.speculative_widths != (1..=width).collect::<Vec<_>>()
            || !self.mixed_catalog.is_empty()
            || self.proposal_source != speculation::ProposalSource::Mtp
        {
            return Err("split speculation requires learned proposals and contiguous logical prefix widths before admission".into());
        }
        self.split = Some(SplitSpeculation {
            width,
            retained: [None; REQUEST_LANES],
            forecast: [None; REQUEST_LANES],
            population: survival(&[FORECAST_PRIOR; 7]),
            pending_budget: None,
        });
        Ok(())
    }

    pub(super) fn owns_proposal(&self, request: &LiveRequest) -> bool {
        self.split.as_ref().is_some_and(|s| s.retained[request.handle.slot as usize]
            .is_some_and(|p| p.request == request.handle))
    }

    pub(super) fn retained_proposal(&self, request: &LiveRequest) -> Option<RetainedProposal> {
        self.split.as_ref()?.retained[request.handle.slot as usize].filter(|p| {
            p.request == request.handle
                && p.fence <= self.completed
                && request.phase == RequestPhase::Decode
                && p.consumed == request.consumed_tokens
        })
    }

    pub(super) fn expected_speculative_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        if let Some(proposal) = self.retained_proposal(request) {
            return expected(&proposal.confidence, width.min(proposal.width));
        }
        if let Some(split) = &self.split {
            let prefixes = split.forecast[request.handle.slot as usize]
                .filter(|f| f.request == request.handle)
                .map_or(split.population, |f| f.estimate(&split.population, request.generated_tokens));
            forecast_tokens(&prefixes, width)
        } else {
            request.acceptance.expected_tokens(self.proposal_source, width)
        }
    }

    fn proposal_forecast_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        let split = self.split.as_ref().unwrap();
        let forecast = split.forecast[request.handle.slot as usize]
            .filter(|f| f.request == request.handle);
        let prefixes = forecast.map_or(split.population, |f| f.estimate(&split.population, request.generated_tokens));
        let mut bound = 1.0f32;
        let mut total = 1.0;
        for (i, &prefix) in prefixes.iter().take(width - 1).enumerate() {
            let at = forecast.map_or(0, |f| f.observed_at[i]);
            // A short proposal cannot refresh the omitted suffix. Price stale
            // positions optimistically, bounded by the preceding prefix, so
            // deeper proposals remain discoverable after repeated short rounds.
            if request.generated_tokens.saturating_sub(at) < FORECAST_REFRESH_TOKENS {
                bound = bound.min(prefix);
            }
            total += f64::from(bound);
        }
        total
    }

    fn proposal_cost(&self, batch: usize, width: usize) -> Option<u64> {
        (width..=self.split.as_ref()?.width)
            .filter_map(|w| self.costs.duration(GraphKind::Proposal, batch, w)).min()
    }

    pub(super) fn proposal_round_duration(&self, batch: usize, width: usize) -> Option<u64> {
        match &self.split {
            Some(_) if width > 1 => Some(horizon(self.proposal_cost(batch, width)?,
                self.costs.duration(GraphKind::Verification, batch, width)?)),
            _ => self.costs.duration(GraphKind::Speculative, batch, width),
        }
    }

    pub(super) fn speculative_duration(&self, request: &LiveRequest, batch: usize, width: usize) -> Option<u64> {
        if self.split.is_some() {
            if let Some(p) = self.retained_proposal(request) {
                if width < p.width { return None; }
                self.costs.duration(GraphKind::Verification, batch, width)
            } else if width == 1 {
                self.costs.duration(GraphKind::Speculative, batch, 1)
            } else {
                Some(horizon(self.proposal_cost(batch, width)?,
                    self.costs.duration(GraphKind::Verification, batch, width)?))
            }
        } else {
            self.costs.duration(GraphKind::Speculative, batch, width)
        }
    }

    fn packing_shapes(&self, kind: GraphKind) -> [[Option<PackedCall>; 9]; REQUEST_LANES + 1] {
        let mut choices = [[None; 9]; REQUEST_LANES + 1];
        for batch in crate::execution::BATCH_CAPACITIES.into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let widths = if kind == GraphKind::Proposal { &self.speculative_widths[..] }
                else { &self.speculative_catalog[batch.ilog2() as usize][..] };
            for &width in widths {
                let Some(nanos) = self.costs.duration(kind, batch, width) else { continue; };
                let call = PackedCall { mask: 0, batch, width, nanos };
                for row in choices.iter_mut().take(batch + 1).skip(1) {
                    for choice in row.iter_mut().take(width + 1).skip(1) {
                        if choice.is_none_or(|c: PackedCall| (nanos, batch, width) < (c.nanos, c.batch, c.width)) {
                            *choice = Some(call);
                        }
                    }
                }
            }
        }
        choices
    }

    #[cfg(test)]
    fn pack(&self, kind: GraphKind, needs: &[(usize, usize)]) -> Option<Packing> {
        pack(needs, &self.packing_shapes(kind))
    }

    fn retained_needs(&self) -> Vec<(usize, usize)> {
        self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let r = entry.request.as_ref()?;
            self.retained_proposal(r).map(|p| (slot, p.width))
        }).collect()
    }

    // Neither the participant set nor measured costs change while fresh confidence is read.
    fn finalize_proposals(&mut self, fresh: u64) -> Result<()> {
        let mut needs = self.retained_needs();
        let split = self.split.as_ref().unwrap();
        let maximum = needs.iter().filter(|(s, _)| fresh & (1 << s) != 0)
            .map(|(s, _)| split.retained[*s].unwrap().limit).max().unwrap_or(1);
        let trial = crate::execution::BATCH_CAPACITIES.into_iter()
            .filter(|b| *b <= needs.len().max(1).next_power_of_two())
            .rev().find_map(|batch| self.speculative_catalog[batch.ilog2() as usize].iter().copied()
                .find(|w| *w <= maximum && self.costs.duration(GraphKind::Verification, batch, *w) == Some(0)));
        if let Some(width) = self.speculative_fixed_width.or(trial) {
            for (slot, selected) in &mut needs {
                if fresh & (1 << *slot) != 0 { *selected = width.min(split.retained[*slot].unwrap().limit); }
            }
        } else {
            let (values, age_price, _) = self.lane_values(0);
            let shapes = self.packing_shapes(GraphKind::Verification);
            let score = |needs: &[(usize, usize)]| -> Result<ValueRate> {
                let packing = pack(needs, &shapes).ok_or("no verification packing for retained proposals")?;
                let value = needs.iter().map(|&(s, w)| values[s].decode(
                    expected(&split.retained[s].unwrap().confidence, w), packing.nanos as f64 / 1e6)).sum();
                Ok(ValueRate::new(packing.nanos, value, age_price))
            };
            let mut rate = score(&needs)?;
            loop {
                let mut best = None;
                for width in 1..maximum {
                    let mut frontier: Vec<_> = needs.iter().enumerate().filter_map(|(index, &(slot, selected))| {
                        let proposal = split.retained[slot].unwrap();
                        (selected == width && fresh & (1 << slot) != 0 && width < proposal.limit).then(|| {
                            let gain = values[slot].output_token * (expected(&proposal.confidence, width + 1)
                                - expected(&proposal.confidence, width));
                            (index, gain)
                        })
                    }).collect();
                    // Equal-width extensions share the same packing cost at each
                    // cohort size. Rank their marginal token value, then price
                    // every prefix together so one row need not pay the whole
                    // increase in graph duration. Only immediate frontiers enter.
                    frontier.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    let mut extended = 0u64;
                    for &(index, _) in &frontier {
                        needs[index].1 += 1;
                        extended |= 1 << index;
                        let candidate = score(&needs)?;
                        if candidate.compare(rate).is_gt()
                            && best.is_none_or(|(_, old)| candidate.compare(old).is_gt()) {
                            best = Some((extended, candidate));
                        }
                    }
                    for &(index, _) in &frontier {
                        needs[index].1 -= 1;
                    }
                }
                // Requests at different widths can also share a graph-cost
                // increase. Price their joint immediate frontier separately.
                let joint: Vec<_> = needs.iter().enumerate().filter_map(|(index, &(slot, width))|
                    (fresh & (1 << slot) != 0 && width < split.retained[slot].unwrap().limit)
                        .then_some(index)).collect();
                if joint.len() > 1 && joint.iter().any(|&i| needs[i].1 != needs[joint[0]].1) {
                    let mut extended = 0u64;
                    for &index in &joint {
                        needs[index].1 += 1;
                        extended |= 1 << index;
                    }
                    let candidate = score(&needs)?;
                    for &index in &joint { needs[index].1 -= 1; }
                    if candidate.compare(rate).is_gt()
                        && best.is_none_or(|(_, old)| candidate.compare(old).is_gt()) {
                        best = Some((extended, candidate));
                    }
                }
                let Some((extended, candidate)) = best else { break; };
                for index in bits(extended) { needs[index].1 += 1; }
                rate = candidate;
            }
        }
        self.finalize_selected(fresh, needs);
        Ok(())
    }

    fn finalize_selected(&mut self, fresh: u64, needs: Vec<(usize, usize)>) {
        let generated: Vec<_> = needs.iter()
            .map(|(slot, _)| self.slots[*slot].request.as_ref().map_or(0, |r| r.generated_tokens)).collect();
        let split = self.split.as_mut().unwrap();
        for ((slot, width), generated) in needs.into_iter().zip(generated) {
            let proposal = split.retained[slot].as_mut().unwrap();
            proposal.width = width;
            if fresh & (1 << slot) != 0 {
                let observation = survival(&proposal.confidence);
                let mut forecast = split.forecast[slot].filter(|f| f.request == proposal.request)
                    .unwrap_or(Forecast { request: proposal.request, prefixes: split.population, observed_at: [0; 7] });
                let previous = forecast.estimate(&split.population, generated);
                for i in 0..proposal.limit.saturating_sub(1) {
                    forecast.prefixes[i] = previous[i] + FORECAST_WEIGHT * (observation[i] - previous[i]);
                    forecast.observed_at[i] = generated;
                    split.population[i] += POPULATION_WEIGHT * (observation[i] - split.population[i]);
                }
                split.forecast[slot] = Some(forecast);
            }
        }
    }

    pub(super) fn split_candidates(&self, output: &mut Vec<Candidate>, values: &[LaneValue; REQUEST_LANES], age_price: f64) {
        if !self.submissions.is_empty() { return; }
        if self.costs.packed_shapes().next().is_some() {
            self.packed_candidates(output, values, age_price);
            return;
        }
        let split = self.split.as_ref().unwrap();
        let retained = self.retained_needs();
        let proposal_shapes = self.packing_shapes(GraphKind::Proposal);
        let verification_shapes = self.packing_shapes(GraphKind::Verification);
        let make = |kind, call: PackedCall, rate| Candidate { packed_token_capacity: 0, verification_budget: None,
            kind, mask: call.mask, batch: call.batch, capacity: call.width,
            prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: call.nanos, rate, feedback: true,
        };
        if let Some(packing) = pack(&retained, &verification_shapes) {
            for call in packing.calls {
                let value = bits(call.mask).map(|s| {
                    let p = split.retained[s].unwrap();
                    values[s].decode(expected(&p.confidence, p.width), call.nanos as f64 / 1e6)
                }).sum();
                output.push(make(GraphKind::Verification, call, ValueRate::new(call.nanos, value, age_price)));
            }
        }
        let available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let r = entry.request.as_ref()?;
            (r.phase == RequestPhase::Decode && self.eligible(r) && !self.owns_proposal(r)).then_some(slot)
        }).collect();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let Some(ns) = self.costs.duration(GraphKind::Speculative, batch, 1) else { continue; };
            let full = self.speculative_fixed_width.is_some_and(|w| self.speculative_catalog[batch.ilog2() as usize].contains(&w));
            let offers: Vec<_> = available.iter().filter(|&&s| !full
                || self.speculative_tokens(self.slots[s].request.as_ref().unwrap(), split.width) == 1)
                .map(|&s| (s, values[s].decode(1.0, ns as f64 / 1e6))).collect();
            let mask = utility::best_cohort::<REQUEST_LANES>(&offers, batch);
            if mask != 0 {
                let value = offers.iter().filter(|(s, _)| mask & (1 << s) != 0).map(|(_, v)| v).sum();
                output.push(make(GraphKind::Speculative, PackedCall { mask, batch, width: 1, nanos: ns }, ValueRate::new(ns, value, age_price)));
            }
        }
        for &width in self.speculative_widths.iter().filter(|w| **w > 1
            && self.speculative_fixed_width.is_none_or(|fixed| fixed == **w)) {
            let mut offers: Vec<_> = available.iter().copied().filter(|s| {
                self.speculative_tokens(self.slots[*s].request.as_ref().unwrap(), split.width) > 1
            }).collect();
            let value = |slot: usize| {
                let r = self.slots[slot].request.as_ref().unwrap();
                values[slot].decode(self.proposal_forecast_tokens(r, self.speculative_tokens(r, width)), 0.0)
            };
            offers.sort_by(|a, b| value(*b).total_cmp(&value(*a)).then(a.cmp(b)));
            for count in 1..=offers.len() {
                let propose: Vec<_> = offers[..count].iter().map(|&s|
                    (s, self.speculative_tokens(self.slots[s].request.as_ref().unwrap(), width))).collect();
                let Some(proposal) = pack(&propose, &proposal_shapes) else { continue; };
                let mut verify = retained.clone();
                verify.extend(offers[..count].iter().map(|&s| (s, self.speculative_tokens(self.slots[s].request.as_ref().unwrap(), width))));
                let Some(verification) = pack(&verify, &verification_shapes) else { continue; };
                let ns = horizon(proposal.nanos, verification.nanos);
                let expected = verify.iter().map(|&(s, w)| {
                    let tokens = split.retained[s].map_or_else(|| self.proposal_forecast_tokens(self.slots[s].request.as_ref().unwrap(), w),
                        |p| expected(&p.confidence, p.width));
                    values[s].decode(tokens, ns as f64 / 1e6)
                }).sum();
                if let Some(call) = proposal.calls.into_iter().max_by(|a, b| {
                    let value = |c: &PackedCall| bits(c.mask).map(value).sum::<f64>();
                    value(a).total_cmp(&value(b)).then(b.mask.cmp(&a.mask))
                }) {
                    output.push(make(GraphKind::Proposal, call, ValueRate::new(ns, expected, age_price)));
                }
            }
        }
    }

    pub fn complete_proposal(&mut self, fence: Fence, records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming, confidence: &[[f32; 7]]) -> Result<Vec<GeneratedToken>> {
        let split = self.split.as_ref().ok_or("proposal completion without a split catalog")?;
        let step = self.submitted_step(fence)?.clone();
        if step.kind != GraphKind::Proposal || confidence.len() < step.batch_capacity
            || !(2..=split.width).contains(&step.token_capacity) {
            return Err("proposal completion does not match its batch and confidence extent".into());
        }
        for row in confidence.iter().take(step.lanes.len()) {
            if row[..step.token_capacity - 1].iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err("proposal confidence must be finite and in [0, 1]".into());
            }
        }
        for (lane, &confidence) in step.lanes.iter().zip(confidence) {
            self.split.as_mut().unwrap().retained[lane.request.slot as usize] = Some(RetainedProposal {
                request: lane.request, fence, consumed: lane.start_position, width: 1, limit: lane.tokens, confidence,
            });
        }
        let generated = match self.complete(fence, records, timing) {
            Ok(generated) => generated,
            Err(error) => {
                for lane in &step.lanes { self.split.as_mut().unwrap().retained[lane.request.slot as usize] = None; }
                return Err(error);
            }
        };
        let mut fresh = 0;
        for lane in &step.lanes {
            if self.request(lane.request).is_ok_and(|r| r.phase == RequestPhase::Decode && r.error_code.is_none()) {
                fresh |= 1 << lane.request.slot;
            } else {
                self.split.as_mut().unwrap().retained[lane.request.slot as usize] = None;
            }
        }
        let budget = self.split.as_mut().unwrap().pending_budget.take();
        if let Some((owner, tokens)) = budget {
            if owner != fence { return Err("packed verification budget belongs to another proposal".into()); }
            self.finalize_packed(fresh, tokens)?;
        } else {
            self.finalize_proposals(fresh)?;
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_budget_survives_cost_updates_before_commit_and_redistributes_across_requests() {
        let mut s = engine(8);
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(8);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        for batch in [1, 2, 4, 8] {
            for (kind,width,budget) in [(GraphKind::Speculative,1,0),(GraphKind::Proposal,8,0)].into_iter()
                .chain((1..=8).map(|width| (GraphKind::Verification,8,batch*width))) {
                graphs.push(GraphDescriptor { name: format!("{kind:?}-{batch}-{width}-{budget}"),kind,
                    sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch, token_capacity: width,
                    packed_token_capacity: budget, prefill_capacity: 0, estimated_duration_ns: 1, input_bindings: vec![] });
            }
        }
        s.graph_durations(&graphs).unwrap();
        let mut measurements = Vec::new();
        for graph in &graphs {
            let duration = match graph.kind {
                GraphKind::Proposal => 2_000_000,
                GraphKind::Verification => 8_000_000 + graph.packed_token_capacity as u64 * 375_000,
                _ => 11_000_000,
            };
            let step = ScheduledStep { kind: graph.kind, batch_capacity: graph.batch_capacity,
                token_capacity: graph.token_capacity, packed_token_capacity: graph.packed_token_capacity,
                mixed: None, lanes: vec![], requires_feedback: true };
            s.costs.set_measured(&step, duration);
            measurements.push((step, duration));
        }
        s.split.as_mut().unwrap().population = survival(&[0.7,0.5,0.1,0.1,0.1,0.1,0.1]);
        let handles: Vec<_> = (0..8).map(|_|ready(&mut s)).collect();
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind,GraphKind::Proposal);
        assert_eq!(plan.batch_capacity,8);
        assert_eq!(plan.verification_budget, Some(24));
        for (step, _) in &measurements {
            if step.kind == GraphKind::Verification {
                s.costs.set_measured(step, if step.packed_token_capacity == 8 { 1 } else { 100_000_000 });
            }
        }
        assert_eq!(s.plan().unwrap().verification_budget, Some(8));
        let fence = Fence(1);
        s.commit(&plan,fence).unwrap();
        let budget = s.split.as_ref().unwrap().pending_budget.unwrap().1;
        assert_eq!(budget,24);
        for (step, duration) in &measurements {
            s.costs.set_measured(step, *duration);
        }
        s.split.as_mut().unwrap().population = [1.;7];
        let mut records = vec![CompletionRecord::default();s.limits.request_capacity()];
        for &h in &handles {
            records[h.slot as usize] = CompletionRecord { generation: h.generation as i64,
                submission_sequence: fence.0 as i64, consumed_tokens: 65, generated_tokens: 1,
                token_id: -1, status: 1, vector: Some(TokenVector::default()), ..Default::default() };
        }
        let confidence: Vec<_> = (0..8).map(|row|if row<2 {[0.999;7]} else {[0.3;7]}).collect();
        assert!(s.complete_proposal(fence,&records,Default::default(),&confidence).unwrap().is_empty());
        let prefixes: Vec<_> = handles.iter().map(|&h| s.retained_proposal(s.request(h).unwrap()).unwrap().width).collect();
        assert_eq!(prefixes,[8,8,2,2,1,1,1,1]);
        assert_eq!(prefixes.iter().sum::<usize>(),budget);
        let verify = s.plan().unwrap();
        assert_eq!(verify.kind,GraphKind::Verification);
        assert_eq!(verify.batch_capacity,8);
        assert_eq!(verify.packed_token_capacity,24);
        assert_eq!(verify.token_capacity,8);
        s.commit(&verify,Fence(2)).unwrap();
    }
    use crate::execution::{GraphDescriptor, SamplingMode, TokenVector};

    #[test]
    fn profitable_proposals_compete_with_full_cohort_decode() {
        let mut s = measured_split_engine(8);
        let confidence = [0.73, 0.59, 0.49, 0.43, 0.38, 0.0, 0.0];
        for (batch, plain, draft) in [(1, 7.5, 1.215), (2, 8.5, 1.563), (4, 9.5, 1.935), (8, 10.99, 2.529)] {
            for (kind, width, ms) in [(GraphKind::Speculative, 1, plain), (GraphKind::Proposal, 6, draft)] {
                s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind, batch_capacity: batch, token_capacity: width,
                    mixed: None, lanes: vec![], requires_feedback: true }, (ms * 1e6) as u64);
            }
        }
        costs(&mut s, 8, &[10.99, 14.609, 17.357, 19.252, 21.921, 24.046].map(|x| (x * 1e6) as u64));
        for _ in 0..8 {
            let h = ready(&mut s);
            let r = s.request_mut(h).unwrap();
            r.accounting.service.tokens = Some(utility::ServicePrice::initial(0.09));
            r.accounting.service.deliveries = Some(utility::ServicePrice::initial(0.09));
        }
        s.split.as_mut().unwrap().population = survival(&confidence);
        let (values, age, _) = s.lane_values(0);
        let mut candidates = vec![];
        s.split_candidates(&mut candidates, &values, age);
        candidates.sort_by(|a, b| b.rate.compare(a.rate));
        assert_eq!(candidates[0].kind, GraphKind::Proposal);
        assert_eq!(candidates[0].mask.count_ones(), 8);
    }

    #[test]
    fn throughput_preference_can_choose_more_tokens_in_a_longer_burst() {
        for (policy, wanted) in [(PolicyConfig::default(), 2), (PolicyConfig::throughput(), 3)] {
            let mut s = measured_split_engine(8);
            s.configure_policy(policy).unwrap();
            costs(&mut s, 8, &[10.8, 14.5, 16.6, 22.0, 25.0, 28.0].map(|x| (x * 1e6) as u64));
            let mut fresh = 0;
            for _ in 0..8 {
                let h = ready(&mut s);
                let request = s.request_mut(h).unwrap();
                request.accounting.service.tokens = Some(utility::ServicePrice::initial(0.09));
                request.accounting.service.deliveries = Some(utility::ServicePrice::initial(0.065));
                retain(&mut s, h, [0.67, 0.38, 0.01, 0.01, 0.01, 0.0, 0.0]);
                s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 6;
                fresh |= 1 << h.slot;
            }
            s.finalize_proposals(fresh).unwrap();
            assert!(s.split.as_ref().unwrap().retained.iter().flatten().all(|p| p.width == wanted));
        }
    }

    fn engine(active: usize) -> RequestScheduler {
        let mut s = super::super::tests::scheduler(active, 512, 64);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(active);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        let mut shapes = Vec::new();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= active.next_power_of_two()) {
            for (kind, widths) in [
                (GraphKind::Proposal, vec![8]), (GraphKind::Speculative, vec![1]),
                (GraphKind::Verification, (1..=8).filter(|w| *w == 1 || batch * w <= 8).collect()),
            ] {
                for width in widths {
                    if kind == GraphKind::Verification { shapes.push((batch, width)); }
                    graphs.push(GraphDescriptor { packed_token_capacity: 0, name: format!("{kind:?}-{batch}-{width}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: width, prefill_capacity: 0,
                        estimated_duration_ns: if kind == GraphKind::Speculative { 100_000 } else { 1000 },
                        input_bindings: vec![] });
                }
            }
        }
        s.speculative_graphs(&shapes).unwrap();
        s.measured_graph_durations(&graphs).unwrap();
        s.split_speculation(8).unwrap();
        s
    }

    fn ready(s: &mut RequestScheduler) -> RequestHandle {
        let handle = s.admit(super::super::tests::spec(65, 64)).unwrap();
        let r = s.request_mut(handle).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 65;
        r.scheduled_consumed = 65;
        r.generated_tokens = 1;
        handle
    }

    fn proposal(s: &mut RequestScheduler, handles: &[RequestHandle], confidence: &[[f32; 7]]) -> Fence {
        let fence = Fence(s.submitted.0 + 1);
        let step = ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: handles.len().next_power_of_two(),
            token_capacity: 8, mixed: None, requires_feedback: true,
            lanes: handles.iter().map(|&h| ScheduledLane { request: h, start_position: 65, tokens: 8 }).collect() };
        let plan = ScheduledPlan { owner: s.plan_owner.clone(), epochs: handles.iter().map(|h| s.request(*h).unwrap().epoch).collect(),
            estimated_duration_ns: 1000, selection_value_rate: 0.0, selection_log_scale: 0.0, verification_budget: None, step };
        s.commit(&plan, fence).unwrap();
        let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
        for &h in handles { records[h.slot as usize] = CompletionRecord {
            generation: h.generation as i64, submission_sequence: fence.0 as i64,
            consumed_tokens: 65, generated_tokens: 1, token_id: -1, status: 1,
            vector: Some(TokenVector::default()), ..Default::default()
        }; }
        let output = s.complete_proposal(fence, &records, Default::default(), confidence).unwrap();
        assert!(output.is_empty());
        fence
    }

    fn retain(s: &mut RequestScheduler, handle: RequestHandle, confidence: [f32; 7]) {
        s.split.as_mut().unwrap().retained[handle.slot as usize] = Some(RetainedProposal {
            request: handle, fence: s.completed, consumed: 65, width: 1, limit: 8, confidence,
        });
    }

    fn costs(s: &mut RequestScheduler, batch: usize, times: &[u64]) {
        for (i, &ns) in times.iter().enumerate() {
            s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Verification, batch_capacity: batch,
                token_capacity: i + 1, mixed: None, lanes: vec![], requires_feedback: true }, ns);
        }
    }

    fn measured_split_engine(active: usize) -> RequestScheduler {
        measured_split_engine_depths(active, &[6])
    }

    fn measured_split_engine_depths(active: usize, depths: &[usize]) -> RequestScheduler {
        let mut s = super::super::tests::scheduler(active, 512, 64);
        s.enable_speculation(&(1..=6).collect::<Vec<_>>()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(active);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        let mut shapes = Vec::new();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= active.next_power_of_two()) {
            for (kind, widths) in [(GraphKind::Proposal, depths.to_vec()), (GraphKind::Speculative, vec![1]),
                                   (GraphKind::Verification, (1..=6).collect())] {
                for width in widths {
                    if kind == GraphKind::Verification { shapes.push((batch, width)); }
                    graphs.push(GraphDescriptor { packed_token_capacity: 0, name: format!("{kind:?}-{batch}-{width}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: width, prefill_capacity: 0, estimated_duration_ns: 1000, input_bindings: vec![] });
                }
            }
        }
        s.speculative_graphs(&shapes).unwrap();
        s.measured_graph_durations(&graphs).unwrap();
        s.split_speculation(6).unwrap();
        let ms = |v: f64| (v * 1e6) as u64;
        for (batch, row) in [(1, [7.3, 7.8, 8.7, 9.0, 9.6, 11.0]), (2, [8.4, 9.6, 10.6, 11.6, 12.6, 13.8]),
                             (4, [9.8, 11.9, 13.4, 15.0, 16.6, 18.4]), (8, [11.6, 15.4, 17.5, 20.0, 22.6, 25.4])] {
            if batch > active.next_power_of_two() { continue; }
            costs(&mut s, batch, &row.map(ms));
        }
        s
    }

    #[test]
    fn shared_verification_calls_widen_together_at_measured_costs() {
        let confidence = [0.70, 0.62, 0.54, 0.52, 0.51, 0.5, 0.5];
        for active in [1, 8] {
            let mut s = measured_split_engine(active);
            let handles: Vec<_> = (0..active).map(|_| ready(&mut s)).collect();
            let mut fresh = 0;
            for &h in &handles {
                retain(&mut s, h, confidence);
                s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 6;
                fresh |= 1 << h.slot;
            }
            s.finalize_proposals(fresh).unwrap();
            let widths: Vec<_> = handles.iter().map(|h| s.retained_proposal(s.request(*h).unwrap()).unwrap().width).collect();
            assert!(widths.iter().all(|w| *w == widths[0] && *w >= 3), "active {active}: {widths:?}");
        }
    }

    #[test]
    fn proposal_depth_follows_the_cost_and_expected_verification_prefix() {
        let mut s = measured_split_engine_depths(8, &[2, 3, 4, 5, 6]);
        for batch in [1, 2, 4, 8] {
            for width in 2..=6 {
                s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: batch,
                    token_capacity: width, mixed: None, lanes: vec![], requires_feedback: true },
                    (width as u64 - 1) * 500_000);
            }
            s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: batch,
                token_capacity: 1, mixed: None, lanes: vec![], requires_feedback: true }, 11_000_000);
        }
        for _ in 0..8 { ready(&mut s); }
        for (confidence, depth) in [([0.9, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0], 2), ([0.99; 7], 6)] {
            s.split.as_mut().unwrap().population = survival(&confidence);
            let plan = s.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Proposal);
            assert_eq!(plan.token_capacity, depth);
            assert_eq!(plan.lanes.len(), 8);
            assert!(plan.lanes.iter().all(|lane| lane.tokens == depth));
        }
    }

    #[test]
    fn short_proposals_refresh_only_the_observed_prefix() {
        let mut s = measured_split_engine_depths(1, &[2, 6]);
        let h = ready(&mut s);
        let fence = Fence(1);
        let step = ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: 1, token_capacity: 2,
            mixed: None, requires_feedback: true,
            lanes: vec![ScheduledLane { request: h, start_position: 65, tokens: 2 }] };
        let plan = ScheduledPlan { owner: s.plan_owner.clone(), epochs: vec![s.request(h).unwrap().epoch],
            estimated_duration_ns: 1_000_000, selection_value_rate: 0.0, selection_log_scale: 0.0, verification_budget: None, step };
        s.commit(&plan, fence).unwrap();
        let record = CompletionRecord { generation: h.generation as i64, submission_sequence: 1,
            consumed_tokens: 65, generated_tokens: 1, token_id: -1, status: 1,
            vector: Some(TokenVector::default()), ..Default::default() };
        let before = s.split.as_ref().unwrap().population;
        s.complete_proposal(fence, &[record], Default::default(), &[[0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).unwrap();
        let f = s.split.as_ref().unwrap().forecast[h.slot as usize].unwrap();
        assert_eq!(f.observed_at, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&s.split.as_ref().unwrap().population[1..], &before[1..]);
        assert!(s.retained_proposal(s.request(h).unwrap()).unwrap().width <= 2);
        s.split.as_mut().unwrap().retained[h.slot as usize] = None;
        s.request_mut(h).unwrap().generated_tokens = FORECAST_REFRESH_TOKENS + 1;
        s.split.as_mut().unwrap().forecast[h.slot as usize].as_mut().unwrap().observed_at[0] = FORECAST_REFRESH_TOKENS + 1;
        assert!(s.proposal_forecast_tokens(s.request(h).unwrap(), 6)
            > s.expected_speculative_tokens(s.request(h).unwrap(), 6));
    }

    #[test]
    fn a_low_forecast_relaxes_until_proposals_resume() {
        let ms = |v: f64| (v * 1e6) as u64;
        let mut s = measured_split_engine(1);
        let h = s.admit(super::super::tests::spec(65, 1024)).unwrap();
        let r = s.request_mut(h).unwrap();
        (r.phase, r.consumed_tokens, r.scheduled_consumed, r.generated_tokens) = (RequestPhase::Decode, 65, 65, 1);
        s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: 1, token_capacity: 1,
            mixed: None, lanes: vec![], requires_feedback: true }, ms(7.3));
        s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: 1, token_capacity: 6,
            mixed: None, lanes: vec![], requires_feedback: true }, ms(1.34));
        let generated = s.request(h).unwrap().generated_tokens;
        s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast { request: h, prefixes: [0.0; 7], observed_at: [generated; 7] });
        assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
        s.request_mut(h).unwrap().generated_tokens += 8 * FORECAST_HALF_LIFE_TOKENS as u32;
        assert_eq!(s.plan().unwrap().kind, GraphKind::Proposal);
    }

    #[test]
    fn forecasts_average_proposals_from_the_prior() {
        let mut s = engine(1);
        let h = ready(&mut s);
        for _ in 0..2 {
            retain(&mut s, h, [1.0; 7]);
            s.finalize_proposals(1 << h.slot).unwrap();
        }
        let forecast = s.split.as_ref().unwrap().forecast[h.slot as usize].unwrap().prefixes;
        assert!((forecast[0] - (0.5 + 0.5 * (1.0 - 0.75 * 0.75))).abs() < 1e-6, "{forecast:?}");
    }

    #[test]
    fn forecasting_averages_joint_prefixes_before_summing_token_value() {
        let mut s = engine(1);
        let h = ready(&mut s);
        let prior = survival(&[FORECAST_PRIOR; 7]);
        let mut reference = prior;
        for confidence in [[1.0; 7], [0.0; 7], [1.0; 7]] {
            retain(&mut s, h, confidence);
            s.finalize_proposals(1 << h.slot).unwrap();
            let observation = survival(&confidence);
            reference = std::array::from_fn(|i| reference[i] + FORECAST_WEIGHT * (observation[i] - reference[i]));
        }
        s.split.as_mut().unwrap().retained[h.slot as usize] = None;
        for width in 1..=8 {
            assert!((s.expected_speculative_tokens(s.request(h).unwrap(), width)
                - forecast_tokens(&reference, width)).abs() < 1e-6);
        }
        // Two perfectly correlated Bernoulli positions have E[p1*p2] = 0.5;
        // multiplying their separate means would incorrectly predict 0.25.
        assert_eq!(forecast_tokens(&[0.5; 7], 3), 2.0);
        assert_eq!(expected(&[0.5; 7], 3), 1.75);
    }

    #[test]
    fn an_unprofitable_population_does_not_lock_out_confidence_refresh() {
        let mut s = measured_split_engine(8);
        for batch in [1, 2, 4, 8] {
            for (kind, width, nanos) in [(GraphKind::Speculative, 1, 10_000_000),
                                        (GraphKind::Proposal, 6, 2_000_000)] {
                s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind, batch_capacity: batch,
                    token_capacity: width, mixed: None, lanes: vec![], requires_feedback: true }, nanos);
            }
        }
        let handles: Vec<_> = (0..8).map(|_| ready(&mut s)).collect();
        s.split.as_mut().unwrap().population = [0.0; 7];
        for &h in &handles {
            s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast { request: h, prefixes: [0.0; 7], observed_at: [1; 7] });
        }
        assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
        for &h in &handles { s.request_mut(h).unwrap().generated_tokens += FORECAST_REFRESH_TOKENS; }
        let refresh = s.plan().unwrap();
        assert_eq!(refresh.kind, GraphKind::Proposal);
        assert_eq!(refresh.lanes.len(), 8);
        let mut fresh = 0;
        for h in handles {
            retain(&mut s, h, [0.0; 7]);
            s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 6;
            fresh |= 1 << h.slot;
        }
        s.finalize_proposals(fresh).unwrap();
        assert!(s.split.as_ref().unwrap().retained.iter().flatten().all(|p| p.width == 1));
    }

    #[test]
    fn requests_without_recent_proposals_are_priced_like_their_peers() {
        let mut s = engine(2);
        let proposer = ready(&mut s);
        let quiet = ready(&mut s);
        for _ in 0..256 {
            retain(&mut s, proposer, [0.9; 7]);
            s.finalize_proposals(1 << proposer.slot).unwrap();
        }
        let tokens = s.expected_speculative_tokens(s.request(quiet).unwrap(), 2);
        assert!((tokens - 1.9).abs() < 0.01, "{tokens}");
        let generated = s.request(quiet).unwrap().generated_tokens;
        s.split.as_mut().unwrap().forecast[quiet.slot as usize] = Some(Forecast { request: quiet, prefixes: [0.0; 7], observed_at: [generated; 7] });
        s.request_mut(quiet).unwrap().generated_tokens += 8 * FORECAST_HALF_LIFE_TOKENS as u32;
        let tokens = s.expected_speculative_tokens(s.request(quiet).unwrap(), 2);
        assert!(tokens > 1.85, "{tokens}");
    }

    #[test]
    fn delivery_references_do_not_batch_incompatible_retained_widths() {
        let mut s = engine(2);
        let narrow = ready(&mut s);
        let wide = ready(&mut s);
        for h in [narrow, wide] { retain(&mut s, h, [0.5; 7]); }
        s.split.as_mut().unwrap().retained[wide.slot as usize].as_mut().unwrap().width = 8;
        costs(&mut s, 1, &[100, 200, 300, 400, 500, 600, 700, 800]);
        costs(&mut s, 2, &[100, 200, 300, 400]);
        let mut rates = [None; REQUEST_LANES];
        rates[narrow.slot as usize] = s.service_rates(s.request(narrow).unwrap()).1;
        rates[wide.slot as usize] = s.service_rates(s.request(wide).unwrap()).1;
        let references = s.delivery_reference_intervals(&rates);
        assert_eq!(references[narrow.slot as usize], Some(0.0002));
        assert_eq!(references[wide.slot as usize], Some(0.0016));
    }

    #[test]
    fn causal_frontier_never_uses_a_suffix_to_cross_an_unprofitable_extension() {
        for tail in [0.0, 1.0] {
            let mut s = engine(1);
            let h = ready(&mut s);
            costs(&mut s, 1, &[100, 300, 210, 210, 210, 210, 210, 210]);
            retain(&mut s, h, [1.0, tail, tail, tail, tail, tail, tail]);
            s.finalize_proposals(1 << h.slot).unwrap();
            assert_eq!(s.retained_proposal(s.request(h).unwrap()).unwrap().width, 1);
        }
    }

    #[test]
    fn shared_verification_cost_can_make_a_cohort_extension_profitable() {
        let mut s = engine(2);
        let first = ready(&mut s);
        let second = ready(&mut s);
        for handle in [first, second] {
            retain(&mut s, handle, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        costs(&mut s, 1, &[10_000_000, 14_000_000, 100_000_000,
            100_000_000, 100_000_000, 100_000_000, 100_000_000, 100_000_000]);
        costs(&mut s, 2, &[10_000_000, 14_000_000, 100_000_000, 100_000_000]);
        let (values, age_price, _) = s.lane_values(0);
        let score = |widths: [usize; 2]| {
            let needs = [(first.slot as usize, widths[0]), (second.slot as usize, widths[1])];
            let packed = s.pack(GraphKind::Verification, &needs).unwrap();
            let value = needs.iter().map(|&(slot, width)| {
                values[slot].decode(expected(&[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], width),
                    packed.nanos as f64 / 1e6)
            }).sum();
            ValueRate::new(packed.nanos, value, age_price)
        };
        assert!(score([2, 2]).compare(score([1, 1])).is_gt());
        assert!(score([2, 1]).compare(score([1, 1])).is_lt());
        s.finalize_proposals((1 << first.slot) | (1 << second.slot)).unwrap();
        let selected: Vec<_> = [first, second].into_iter().map(|h|
            s.retained_proposal(s.request(h).unwrap()).unwrap().width).collect();
        assert_eq!(selected, [2, 2]);
    }

    #[test]
    fn conditional_confidence_is_multiplied_and_replaces_acceptance_history() {
        let mut s = engine(1);
        let h = ready(&mut s);
        retain(&mut s, h, [0.8, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0]);
        s.finalize_proposals(1 << h.slot).unwrap();
        assert_eq!(s.retained_proposal(s.request(h).unwrap()).unwrap().width, 3);
        assert!((s.expected_speculative_tokens(s.request(h).unwrap(), 8) - 2.2).abs() < 1e-6);
    }

    #[test]
    fn unmeasured_trial_width_does_not_depend_on_fresh_confidence() {
        let mut chosen = Vec::new();
        for p in [0.0, 1.0] {
            let mut s = engine(1);
            let h = ready(&mut s);
            costs(&mut s, 1, &[100, 100, 0, 100, 100, 100, 100, 100]);
            retain(&mut s, h, [p; 7]);
            s.finalize_proposals(1 << h.slot).unwrap();
            chosen.push(s.retained_proposal(s.request(h).unwrap()).unwrap().width);
        }
        assert_eq!(chosen, [3, 3]);
    }

    #[test]
    fn multiple_proposals_regroup_without_changing_finalized_prefixes() {
        let mut s = engine(4);
        let handles: Vec<_> = (0..4).map(|_| ready(&mut s)).collect();
        proposal(&mut s, &handles[..2], &[[1.0; 7], [0.0; 7]]);
        assert_eq!(s.request(handles[0]).unwrap().consumed_tokens, 65);
        proposal(&mut s, &handles[2..], &[[0.0; 7], [1.0; 7]]);
        let needs = s.retained_needs();
        assert_eq!(needs.iter().map(|r| r.1).collect::<Vec<_>>(), [4, 1, 1, 4]);
        let packing = s.pack(GraphKind::Verification, &needs).unwrap();
        assert_eq!(packing.calls.len(), 2);
        assert!(packing.calls.iter().any(|c| c.mask == (1 << handles[1].slot) | (1 << handles[2].slot)));
        let mut seen = 0;
        for call in packing.calls {
            assert_eq!(seen & call.mask, 0);
            seen |= call.mask;
            for slot in bits(call.mask) { assert!(needs[slot].1 <= call.width); }
        }
        assert_eq!(seen, 15);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Verification);
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        assert!(s.commit(&plan, Fence(fence.0 + 1)).is_err());
        for lane in &plan.lanes { assert!(s.split.as_ref().unwrap().retained[lane.request.slot as usize].is_none()); }
    }

    #[test]
    fn proposal_is_a_prerequisite_with_no_progress_credit() {
        let mut s = engine(2);
        let h = ready(&mut s);
        let before = s.request(h).unwrap().accounting;
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Proposal);
        proposal(&mut s, &[h], &[[0.9; 7]]);
        let after = s.request(h).unwrap().accounting;
        assert_eq!(before.committed_output_tokens, after.committed_output_tokens);
        assert_eq!(before.published_deliveries, after.published_deliveries);
        assert!(s.plan().unwrap().kind == GraphKind::Verification);
    }

    #[test]
    fn proposal_storage_is_bounded_and_cancellation_does_not_follow_slot_reuse() {
        let mut s = engine(1);
        let h = ready(&mut s);
        proposal(&mut s, &[h], &[[1.0; 7]]);
        s.cancel(h).unwrap();
        let replacement = ready(&mut s);
        assert_eq!(replacement.slot, h.slot);
        assert_ne!(replacement.generation, h.generation);
        assert!(!s.owns_proposal(s.request(replacement).unwrap()));
        let prior = forecast_tokens(&s.split.as_ref().unwrap().population, 8);
        assert!((s.expected_speculative_tokens(s.request(replacement).unwrap(), 8) - prior).abs() < 1e-10);
        assert!(prior < 2.1, "one proposal moves the population prior only slightly: {prior}");
    }

    #[test]
    fn retained_proposal_cannot_be_preempted_for_another_request() {
        let mut s = engine(2);
        let first = ready(&mut s);
        let last = ready(&mut s);
        s.request_mut(first).unwrap().growable_output = true;
        s.request_mut(last).unwrap().growable_output = true;
        proposal(&mut s, &[last], &[[1.0; 7]]);
        s.preempt_newest().unwrap();
        assert!(s.request(first).unwrap().preempted);
        assert!(!s.request(last).unwrap().preempted);
        assert!(s.owns_proposal(s.request(last).unwrap()));
    }
}
