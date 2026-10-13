//! Expiring duration estimates; zero denotes an indeterminate graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    kind: GraphKind,
    batch: usize,
    tokens: usize,
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
    pub fn prepared(graphs: &[GraphDescriptor]) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
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
        self.entry(kind, batch, tokens).map(|estimate| {
            if estimate.fresh(self.now_ns, self.ttl_ns) {
                estimate.nanos
            } else {
                0
            }
        })
    }

    // Normalization units retain their last estimate independently of prediction expiry.
    pub fn reference_duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.entry(kind, batch, tokens)
            .map(|estimate| estimate.nanos)
            .filter(|n| *n != 0)
    }

    fn entry(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<&Estimate> {
        let key = Key {
            kind,
            batch,
            tokens,
        };
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, estimate)| estimate)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.duration(step.kind, step.batch_capacity, step.token_capacity)
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(sample) = timing
            .gpu_ns
            .into_iter()
            .flatten()
            .max()
            .filter(|ns| *ns > 0)
        else {
            return;
        };
        let key = Key {
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
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
