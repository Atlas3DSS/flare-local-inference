//! Expiring duration estimates; zero denotes an indeterminate graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

pub(super) fn observed_duration_ns(timing: ExecutionTiming) -> Option<u64> {
    timing
        .gpu_ns
        .into_iter()
        .flatten()
        .max()
        .filter(|ns| *ns > 0)
        .map(|ns| if timing.queued { ns } else { ns.max(timing.wall_ns) })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    kind: GraphKind,
    batch: usize,
    tokens: usize,
    prefill: usize,
    packed: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PackedShape {
    pub batch: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

struct Estimate {
    nanos: u64,
    handoff_ns: u64,
    observed_at_ns: u64,
}

impl Estimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

#[derive(Default)]
pub(super) struct ActionCosts {
    entries: Vec<(Key, Estimate)>,
    now_ns: u64,
    ttl_ns: u64,
}

impl ActionCosts {
    #[cfg(test)]
    pub fn set_measured(&mut self, step: &ScheduledStep, nanos: u64) {
        let key = Key {
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        let estimate = &mut self.entries.iter_mut().find(|(k, _)| *k == key).unwrap().1;
        estimate.nanos = nanos;
        estimate.handoff_ns = 0;
        estimate.observed_at_ns = self.now_ns;
    }

    pub fn prepared(graphs: &[GraphDescriptor]) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
                prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity,
            };
            if entries.iter().any(|(k, _)| *k == key) {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
                    handoff_ns: 0,
                    observed_at_ns: 0,
                },
            ));
        }
        Ok(Self {
            entries,
            now_ns: 0,
            ttl_ns: super::PolicyConfig::default().estimate_ttl_ns,
        })
    }

    pub fn advance_to(&mut self, now_ns: u64, ttl_ns: u64) {
        self.now_ns = now_ns;
        self.ttl_ns = ttl_ns;
    }

    pub fn duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.shape_duration(kind, batch, tokens, 0).or_else(|| {
            (kind == GraphKind::Verification).then(|| self.packed_shapes()
                .filter(|shape| shape.batch == batch && shape.width >= tokens && shape.tokens >= batch * tokens)
                .map(|shape| shape.nanos).min()).flatten()
        })
    }

    pub fn packed_shapes(&self) -> impl Iterator<Item = PackedShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::Verification && key.packed != 0)
            .map(|(key, estimate)| PackedShape { batch: key.batch, width: key.tokens, tokens: key.packed,
                nanos: if estimate.fresh(self.now_ns, self.ttl_ns) {
                    estimate.nanos.saturating_add(estimate.handoff_ns)
                } else { 0 } })
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.exact_duration(kind, batch, tokens, prefill, 0)
    }

    fn exact_duration(&self, kind: GraphKind, batch: usize, tokens: usize, prefill: usize, packed: usize) -> Option<u64> {
        self.entry(Key {
            kind,
            batch,
            tokens,
            prefill,
            packed,
        })
        .map(|estimate| {
            if estimate.fresh(self.now_ns, self.ttl_ns) {
                estimate.nanos.saturating_add(estimate.handoff_ns)
            } else {
                0
            }
        })
    }

    fn entry(&self, key: Key) -> Option<&Estimate> {
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, estimate)| estimate)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.exact_duration(
            step.kind,
            step.batch_capacity,
            step.token_capacity,
            step.prefill_capacity(),
            step.packed_token_capacity,
        )
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(sample) = timing.gpu_ns.into_iter().flatten().max().filter(|n| *n > 0) else {
            return;
        };
        let key = Key {
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        if let Some((_, estimate)) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            let fresh = estimate.fresh(self.now_ns, self.ttl_ns);
            if !timing.queued {
                let handoff = timing.wall_ns.saturating_sub(sample);
                estimate.handoff_ns = if fresh {
                    ((7 * u128::from(estimate.handoff_ns) + u128::from(handoff) + 4) / 8) as u64
                } else { handoff };
            } else if !fresh {
                estimate.handoff_ns = 0;
            }
            estimate.nanos = if estimate.fresh(self.now_ns, self.ttl_ns) {
                ((7 * u128::from(estimate.nanos) + u128::from(sample) + 4) / 8) as u64
            } else {
                sample
            };
            estimate.observed_at_ns = self.now_ns;
        }
    }
}
