//! Bounded width selection from completed graph costs and emitted progress.

use crate::{Result, execution::BATCH_CAPACITIES};

pub const WIDTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProposalSource {
    #[default]
    Mtp,
    Copy,
}

impl ProposalSource {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "mtp" => Ok(Self::Mtp),
            "copy" => Ok(Self::Copy),
            _ => Err("speculative_proposal_source must be mtp or copy".into()),
        }
    }
}

/// Host-validated continuation and masks for the corresponding committed prefix.
#[derive(Clone, Debug)]
pub struct CopyProposal {
    pub consumed: u64,
    pub tokens: [i64; 7],
    pub count: usize,
    pub grammar_masks: Vec<i32>,
}

pub fn graph_width_masks(
    widths: &[usize],
    fixed: Option<usize>,
    budget: usize,
    capacity: usize,
) -> [u8; 7] {
    let batches = capacity.ilog2() as usize + 1;
    let count = if fixed.is_some() { 1 } else { widths.len() };
    let mut remaining = budget;
    let mut exhausted = false;
    std::array::from_fn(|index| {
        if count == 0 || index >= batches {
            return 0;
        }
        let take = if exhausted {
            1
        } else {
            count.min(remaining.saturating_sub(batches - 1 - index))
        };
        let mask = if exhausted {
            1
        } else if let Some(width) = fixed {
            1 << (width - 1)
        } else {
            widths
                .iter()
                .rev()
                .take(take)
                .fold(0, |mask, width| mask | (1 << (width - 1)))
        };
        exhausted |= take < count;
        remaining = remaining.saturating_sub(take);
        mask
    })
}

pub fn graph_catalog(
    widths: &[usize],
    fixed: Option<usize>,
    budget: usize,
    capacity: usize,
) -> [Vec<usize>; 7] {
    graph_width_masks(widths, fixed, budget, capacity)
        .map(|mask| (1..=8).filter(|w| mask & (1 << (w - 1)) != 0).collect())
}

pub fn validate_widths(widths: &[usize]) -> Result<()> {
    if !widths.is_empty()
        && (widths[0] != 1
            || widths.windows(2).any(|w| w[0] >= w[1])
            || widths.iter().any(|w| !WIDTHS.contains(w)))
    {
        return Err(
            "speculative widths must be sorted unique integers in 1..=8 including 1".into(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Cost {
    rounds: u64,
    nanos: f64,
    emitted: f64,
}

/// Conditional acceptance of one draft depth given that every shallower draft was accepted.
/// Starts from a 0.5 prior worth two observations and averages at most the last 16.
#[derive(Clone, Copy)]
struct DepthAcceptance {
    probability: f64,
    weight: f64,
}
impl Default for DepthAcceptance {
    fn default() -> Self {
        Self {
            probability: 0.5,
            weight: 2.0,
        }
    }
}
impl DepthAcceptance {
    fn observe(&mut self, accepted: bool) {
        self.weight = (self.weight + 1.0).min(16.0);
        self.probability += (if accepted { 1.0 } else { 0.0 } - self.probability) / self.weight;
    }
}
/// Completed rounds observed before measured acceptance replaces whole-round sampling.
const MODEL_ROUNDS: u64 = 8;
/// Every this many completed rounds, the widest legal width refreshes deep acceptance.
const MODEL_EXPLORATION: u64 = 16;

#[derive(Default)]
pub struct WidthController {
    costs: [[[Cost; 8]; 3]; 7],
    completed: u64,
    depths: [DepthAcceptance; 7],
    acceptance_rounds: u64,
}
impl WidthController {
    fn coordinates(batch: usize, context: u64) -> (usize, usize) {
        (
            batch.trailing_zeros() as usize,
            if context <= 2048 {
                0
            } else if context <= 32768 {
                1
            } else {
                2
            },
        )
    }
    pub fn choose(
        &self,
        widths: &[usize],
        batch: usize,
        context: u64,
        maximum: usize,
    ) -> Option<usize> {
        if !BATCH_CAPACITIES.contains(&batch) {
            return None;
        }
        let (b, c) = Self::coordinates(batch, context);
        let legal = || widths.iter().copied().filter(|w| *w <= maximum);
        if let Some(width) = legal().find(|w| self.costs[b][c][*w - 1].rounds == 0) {
            return Some(width);
        }
        if self.acceptance_rounds >= MODEL_ROUNDS {
            // Expected tokens per round follow from per-depth acceptance measured in every
            // round; only round time comes from each width's own measurements.
            if self.completed.is_multiple_of(MODEL_EXPLORATION) {
                return legal().max();
            }
            return legal().max_by(|left, right| {
                let score = |width: usize| {
                    self.expected_tokens(width) / self.costs[b][c][width - 1].nanos
                };
                score(*left)
                    .total_cmp(&score(*right))
                    .then_with(|| right.cmp(left))
            });
        }
        if self.completed > 0 && self.completed.is_multiple_of(64) {
            let count = legal().count();
            if count > 0 {
                return legal().nth((self.completed as usize / 64) % count);
            }
        }
        legal().max_by(|left, right| {
            let score = |width: usize| {
                let cost = self.costs[b][c][width - 1];
                cost.emitted / cost.nanos
            };
            score(*left)
                .total_cmp(&score(*right))
                .then_with(|| right.cmp(left))
        })
    }
    /// Expected tokens one row emits at `width`: the pending token plus each draft prefix
    /// whose drafts are all accepted.
    fn expected_tokens(&self, width: usize) -> f64 {
        let mut chain = 1.0;
        let mut total = 1.0;
        for depth in &self.depths[..width.saturating_sub(1).min(7)] {
            chain *= depth.probability;
            total += chain;
        }
        total
    }
    /// Records each row's chain outcome: drafts 1..=accepted were accepted and, when a draft
    /// remained, the next one was rejected. Deeper drafts were not evaluated.
    pub fn observe_acceptance(&mut self, rows: &[(usize, usize)]) {
        let mut observed = false;
        for &(offered, accepted) in rows {
            let offered = offered.min(7);
            if offered == 0 {
                continue;
            }
            let accepted = accepted.min(offered);
            for depth in 0..accepted {
                self.depths[depth].observe(true);
            }
            if accepted < offered {
                self.depths[accepted].observe(false);
            }
            observed = true;
        }
        if observed {
            self.acceptance_rounds += 1;
        }
    }
    pub fn observe(
        &mut self,
        batch: usize,
        context: u64,
        width: usize,
        nanos: u64,
        emitted: usize,
    ) -> Result<()> {
        if !BATCH_CAPACITIES.contains(&batch)
            || !WIDTHS.contains(&width)
            || nanos == 0
            || emitted > batch * width
        {
            return Err("invalid measured speculative round".into());
        }
        if emitted == 0 {
            return Ok(());
        }
        let (b, c) = Self::coordinates(batch, context);
        let cost = &mut self.costs[b][c][width - 1];
        let alpha = if cost.rounds == 0 { 1.0 } else { 0.125 };
        cost.nanos += alpha * (nanos as f64 - cost.nanos);
        cost.emitted += alpha * (emitted as f64 - cost.emitted);
        cost.rounds += 1;
        self.completed += 1;
        Ok(())
    }
}

/// Copies only observed continuation tokens; the device guards its consumed-position epoch.
pub fn copy_proposal(history: &[u32], maximum: usize) -> ([i64; 7], usize) {
    let mut output = [-1; 7];
    let maximum = maximum.min(7);
    for length in [8, 4] {
        if history.len() <= length || maximum == 0 {
            continue;
        }
        let suffix = history.len() - length;
        for position in (suffix.saturating_sub(4096)..suffix).rev() {
            if history[position..position + length] != history[suffix..] {
                continue;
            }
            let continuation = &history[position + length..];
            let count = maximum.min(continuation.len());
            for (target, token) in output.iter_mut().zip(continuation.iter()).take(count) {
                *target = i64::from(*token);
            }
            if count > 0 {
                return (output, count);
            }
        }
    }
    (output, 0)
}
