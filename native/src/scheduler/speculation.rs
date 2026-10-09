//! Bounded width selection from completed graph costs and emitted progress.

use crate::Result;

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

#[derive(Default)]
pub struct WidthController {
    costs: [[[Cost; 8]; 3]; 3],
    completed: u64,
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
        if ![1, 2, 4].contains(&batch) {
            return None;
        }
        let (b, c) = Self::coordinates(batch, context);
        let maximum = maximum.min(8 / batch);
        let legal = || widths.iter().copied().filter(|w| *w <= maximum);
        if let Some(width) = legal().find(|w| self.costs[b][c][*w - 1].rounds == 0) {
            return Some(width);
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
    pub fn observe(
        &mut self,
        batch: usize,
        context: u64,
        width: usize,
        nanos: u64,
        emitted: usize,
    ) -> Result<()> {
        if ![1, 2, 4].contains(&batch)
            || !WIDTHS.contains(&width)
            || batch * width > 8
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
