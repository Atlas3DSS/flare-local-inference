//! Select a target-row budget before sampling, then allocate causal prefixes within it.

use super::*;
use crate::scheduler::cost::PackedShape;

impl RequestScheduler {
    fn budget_choice(&self, offers: &[(usize, usize)], batch: usize,
        values: &[LaneValue; REQUEST_LANES], age_price: f64) -> Option<(PackedShape, ValueRate)> {
        let width = self.split.as_ref()?.width;
        let proposal_ns = self.costs.duration(GraphKind::Proposal, batch, width)?;
        let mut scores = Vec::new();
        for &(slot, limit) in offers {
            let request = self.slots[slot].request.as_ref()?;
            let mut before = 1.0;
            for position in 0..limit.saturating_sub(1) {
                let total = self.proposal_forecast_tokens(request, position + 2);
                scores.push((total - before, position, slot));
                before = total;
            }
        }
        scores.sort_unstable_by(|a,b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        self.costs.packed_shapes().filter(|shape| shape.batch == batch && shape.width >= width
            && shape.tokens >= offers.len()).map(|shape| {
            let nanos = horizon(proposal_ns, shape.nanos);
            let mut expected = [0.0; REQUEST_LANES];
            for &(slot, _) in offers { expected[slot] = 1.0; }
            for &(score, _, slot) in scores.iter().take(shape.tokens - offers.len()) { expected[slot] += score; }
            let value = offers.iter().map(|&(slot,_)| values[slot].decode(expected[slot], nanos as f64 / 1e6)).sum();
            (shape, ValueRate::new(nanos, value, age_price))
        }).max_by(|(a,ar),(b,br)| ar.compare(*br).then(b.tokens.cmp(&a.tokens)))
    }

    pub(super) fn finalize_packed(&mut self, fresh: u64, budget: usize) -> Result<()> {
        let mut needs = self.retained_needs();
        let count = fresh.count_ones() as usize;
        if count > budget { return Err("proposal row budget cannot cover its pending inputs".into()); }
        let mut survival = [1.0f64; REQUEST_LANES];
        for (slot,width) in &mut needs {
            if fresh & (1 << *slot) != 0 { *width = 1; }
        }
        for _ in 0..budget - count {
            // Only a lane's immediate frontier is visible. A later confidence
            // cannot influence admission of the token on which it depends.
            let best = needs.iter().enumerate().filter_map(|(index,&(slot,width))| {
                let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                (fresh & (1 << slot) != 0 && width < proposal.limit).then(||
                    (index, survival[slot] * f64::from(proposal.confidence[width-1]), width, slot))
            }).max_by(|a,b| a.1.total_cmp(&b.1).then(b.2.cmp(&a.2)).then(b.3.cmp(&a.3)));
            let Some((index,probability,_,slot)) = best else { break; };
            survival[slot] = probability;
            needs[index].1 += 1;
        }
        self.finalize_selected(fresh, needs);
        Ok(())
    }

    pub(super) fn packed_candidates(&self, output: &mut Vec<Candidate>, values: &[LaneValue; REQUEST_LANES], age: f64) {
        let retained = self.retained_needs();
        for shape in self.costs.packed_shapes() {
            let mut offers: Vec<_> = retained.iter().copied().filter(|(_,width)| *width <= shape.width).collect();
            offers.sort_unstable_by(|&(a,aw),&(b,bw)| {
                let value = |slot: usize, width: usize| {
                    let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                    values[slot].decode(expected(&proposal.confidence,width), shape.nanos as f64 / 1e6)
                };
                (value(b,bw)/bw as f64).total_cmp(&(value(a,aw)/aw as f64)).then(a.cmp(&b))
            });
            let (mut mask,mut tokens,mut count,mut value) = (0u64,0usize,0usize,0.0);
            for (slot,width) in offers {
                if count == shape.batch || tokens + width > shape.tokens { continue; }
                mask |= 1 << slot;tokens += width;count += 1;
                let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                value += values[slot].decode(expected(&proposal.confidence,width), shape.nanos as f64 / 1e6);
            }
            if mask != 0 {
                output.push(Candidate { kind: GraphKind::Verification, mask, batch: shape.batch, verification_budget: None,
                    capacity: shape.width, packed_token_capacity: shape.tokens, prefill_capacity: 0, prefill_slot: None,
                    work: 0, duration_ns: shape.nanos, rate: ValueRate::new(shape.nanos,value,age), feedback: true });
            }
        }
        let mut available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot,entry)| {
            let request = entry.request.as_ref()?;
            (request.phase == RequestPhase::Decode && self.eligible(request) && !self.owns_proposal(request)).then_some(slot)
        }).collect();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let Some(nanos) = self.costs.duration(GraphKind::Speculative,batch,1) else { continue; };
            let offers: Vec<_> = available.iter().map(|&slot| (slot,values[slot].decode(1.0,nanos as f64/1e6))).collect();
            let mask = utility::best_cohort::<REQUEST_LANES>(&offers,batch);
            if mask != 0 {
                let value = offers.iter().filter(|(slot,_)| mask & (1 << slot) != 0).map(|(_,v)| v).sum();
                output.push(Candidate { kind: GraphKind::Speculative, mask, batch, capacity: 1, packed_token_capacity: 0, verification_budget: None,
                    prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: nanos,
                    rate: ValueRate::new(nanos,value,age), feedback: true });
            }
        }
        let width = self.split.as_ref().unwrap().width;
        available.retain(|&slot| self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(),width)>1);
        available.sort_unstable_by(|&a,&b| values[b].output_token.total_cmp(&values[a].output_token).then(a.cmp(&b)));
        for count in 1..=available.len() {
            let offers: Vec<_> = available[..count].iter().map(|&slot|
                (slot,self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(),width))).collect();
            let batch = count.next_power_of_two();
            if let Some((shape,rate)) = self.budget_choice(&offers,batch,values,age) {
                output.push(Candidate { kind: GraphKind::Proposal, mask: offers.iter().fold(0,|mask,(slot,_)|mask|(1<<slot)),
                    batch, capacity: width, packed_token_capacity: 0, prefill_capacity: 0, prefill_slot: None,
                    work: 0, duration_ns: rate.duration_ns, rate, feedback: true, verification_budget: Some(shape.tokens) });
            }
        }
    }
}
