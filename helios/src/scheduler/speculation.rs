//! Request-local speculative acceptance and observed copy proposals.

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

pub fn graph_width_masks(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [u8; 7] {
    let mask = fixed.map_or_else(
        || {
            widths
                .iter()
                .fold(0, |mask, width| mask | (1 << (width - 1)))
        },
        |width| 1 << (width - 1),
    );
    BATCH_CAPACITIES.map(|batch| if batch <= capacity { mask } else { 0 })
}

pub fn graph_catalog(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [Vec<usize>; 7] {
    graph_width_masks(widths, fixed, capacity)
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

/// Conditional acceptance of one draft depth given that every shallower draft was accepted.
/// Request-local observations are weighed against a prior worth two observations; after
/// saturation each outcome has weight `1 / DEPTH_MEMORY`.
#[derive(Clone, Copy, Default)]
struct DepthAcceptance {
    mean: f64,
    weight: f64,
    /// Generated tokens of the request when this depth was last observed.
    observed_at: Option<u32>,
}
impl DepthAcceptance {
    fn observe(&mut self, accepted: bool, generated: u32) {
        self.weight = (self.weight + 1.0).min(DEPTH_MEMORY - PRIOR_WEIGHT);
        self.mean += (if accepted { 1.0 } else { 0.0 } - self.mean) / self.weight;
        self.observed_at = Some(generated);
    }

    fn probability(&self, prior: f64) -> f64 {
        (self.weight * self.mean + PRIOR_WEIGHT * prior) / (self.weight + PRIOR_WEIGHT)
    }
}
const DEPTH_MEMORY: f64 = 128.0;
const PRIOR_WEIGHT: f64 = 2.0;
const INITIAL_PRIOR: f64 = 0.5;
/// Output tokens after which an unobserved depth is priced at the bound of its shallower prefix.
pub const STALE_DEPTH_TOKENS: u32 = 32;
/// Saturated weight of one observation in the model-local population estimate is 1/64.
const POPULATION_MEMORY: f64 = 64.0;

/// Conditional acceptance belongs to one logical request and proposal source.
#[derive(Clone, Copy, Default)]
pub struct AcceptanceProfile {
    depths: [[DepthAcceptance; 7]; 2],
    rounds: u64,
}

impl AcceptanceProfile {
    /// Request-local estimate from the initial prior; depth zero is the first draft after
    /// the pending target token.
    pub fn probability(&self, source: ProposalSource, depth: usize) -> Option<f64> {
        self.depths[source as usize]
            .get(depth)
            .map(|d| d.probability(INITIAL_PRIOR))
    }

    pub fn expected_tokens(&self, source: ProposalSource, width: usize) -> f64 {
        let mut chain = 1.0;
        let mut total = f64::from(width != 0);
        for depth in &self.depths[source as usize][..width.saturating_sub(1).min(7)] {
            chain *= depth.probability(INITIAL_PRIOR);
            total += chain;
        }
        total
    }

    /// Expected output of a width for scheduling. Request observations are weighed against
    /// the model-local population instead of the initial prior, so a new request starts from
    /// what earlier requests measured. A depth unobserved for `STALE_DEPTH_TOKENS` outputs is
    /// priced at the bound of its shallower prefix, so a wider round becomes worth probing.
    pub fn forecast_tokens(
        &self,
        source: ProposalSource,
        width: usize,
        population: &AcceptancePopulation,
        generated: u32,
    ) -> f64 {
        let mut chain = 1.0;
        let mut total = f64::from(width != 0);
        for (index, depth) in self.depths[source as usize][..width.saturating_sub(1).min(7)]
            .iter()
            .enumerate()
        {
            if generated.saturating_sub(depth.observed_at.unwrap_or(0)) < STALE_DEPTH_TOKENS {
                chain *= depth.probability(population.probability(source, index));
            }
            total += chain;
        }
        total
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Only the accepted prefix and an explicitly observed rejection are samples.
    /// `copy_mask` identifies the proposal source at each evaluated draft depth;
    /// `generated` counts the request's outputs before this round.
    pub(super) fn observe(
        &mut self,
        offered: usize,
        accepted: usize,
        rejected: bool,
        copy_mask: u8,
        generated: u32,
    ) {
        debug_assert!(accepted <= offered && offered <= 7);
        if offered == 0 {
            return;
        }
        for depth in 0..accepted {
            self.depths[usize::from(copy_mask & (1 << depth) != 0)][depth].observe(true, generated);
        }
        if rejected && accepted < offered {
            self.depths[usize::from(copy_mask & (1 << accepted) != 0)][accepted]
                .observe(false, generated);
        }
        self.rounds = self.rounds.saturating_add(1);
    }
}

/// Model-local conditional acceptance by proposal source and depth, shared by every request.
/// Starts at the initial prior with weight two; once saturated, each observation of any
/// request contributes `1 / POPULATION_MEMORY`.
#[derive(Clone, Copy)]
pub struct AcceptancePopulation {
    depths: [[(f64, f64); 7]; 2],
}

impl Default for AcceptancePopulation {
    fn default() -> Self {
        Self { depths: [[(INITIAL_PRIOR, PRIOR_WEIGHT); 7]; 2] }
    }
}

impl AcceptancePopulation {
    pub fn probability(&self, source: ProposalSource, depth: usize) -> f64 {
        self.depths[source as usize][depth].0
    }

    /// The same samples as [`AcceptanceProfile::observe`].
    pub(super) fn observe(&mut self, offered: usize, accepted: usize, rejected: bool, copy_mask: u8) {
        let mut sample = |depth: usize, accepted: bool| {
            let (probability, weight) = &mut self.depths[usize::from(copy_mask & (1 << depth) != 0)][depth];
            *weight = (*weight + 1.0).min(POPULATION_MEMORY);
            *probability += (if accepted { 1.0 } else { 0.0 } - *probability) / *weight;
        };
        for depth in 0..accepted.min(offered) {
            sample(depth, true);
        }
        if rejected && accepted < offered {
            sample(accepted, false);
        }
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
