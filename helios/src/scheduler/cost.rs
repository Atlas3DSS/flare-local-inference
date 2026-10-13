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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    kind: GraphKind,
    batch: usize,
    tokens: usize,
    prefill: usize,
}

struct Estimate {
    nanos: u64,
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
        };
        let estimate = &mut self.entries.iter_mut().find(|(k, _)| *k == key).unwrap().1;
        estimate.nanos = nanos;
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
            };
            if entries.iter().any(|(k, _)| *k == key) {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
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
        self.shape_duration(kind, batch, tokens, 0)
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.entry(Key {
            kind,
            batch,
            tokens,
            prefill,
        })
        .map(|estimate| {
            if estimate.fresh(self.now_ns, self.ttl_ns) {
                estimate.nanos
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
        self.shape_duration(
            step.kind,
            step.batch_capacity,
            step.token_capacity,
            step.prefill_capacity(),
        )
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(sample) = observed_duration_ns(timing) else {
            return;
        };
        let key = Key {
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
        };
        if let Some((_, estimate)) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            estimate.nanos = if estimate.fresh(self.now_ns, self.ttl_ns) {
                ((7 * u128::from(estimate.nanos) + u128::from(sample) + 4) / 8) as u64
            } else {
                sample
            };
            estimate.observed_at_ns = self.now_ns;
        }
    }
}
