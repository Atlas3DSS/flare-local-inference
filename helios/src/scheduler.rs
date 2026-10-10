//! Bounded live-request resources and scheduling with CUDA-fenced retirement.

pub mod speculation;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::cache::{
    CheckpointId, ConditioningEvent, EvictionTarget, Fence, PageId, PrefixCheckpointCache,
    PrefixHistory, Publication, PublishResult, SnapshotId, StateSnapshot,
};
use crate::execution::{CompletionRecord, GraphKind, REQUEST_LANES};
use crate::{Result, Shutdown};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RequestHandle {
    pub slot: u32,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateSlotId(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct SchedulerLimits {
    pub max_active: usize,
    pub max_sequence_tokens: usize,
    pub prefill_chunk_size: usize,
    pub pages: u32,
    pub snapshots: u32,
}

impl SchedulerLimits {
    pub(crate) fn validate_prefill(&self) -> Result<()> {
        if self.prefill_chunk_size < 64
            || !self.prefill_chunk_size.is_multiple_of(64)
            || self.prefill_chunk_size > self.max_sequence_tokens
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Manifest,
                "prefill_chunk_size must be divisible by 64 and within 64..=context",
            ));
        }
        Ok(())
    }
}

/// Required token capacities for a validated, 64-aligned prefill chunk limit.
pub fn prefill_capacities(chunk_size: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(Some(128usize), |capacity| capacity.checked_mul(2))
        .take_while(move |capacity| *capacity < chunk_size)
        .chain(std::iter::once(chunk_size))
}

#[derive(Clone, Debug)]
pub struct SchedulerCapabilities {
    pub vocabulary: usize,
    pub page_tokens: usize,
    pub vision: crate::model::VisionCapabilities,
    pub live_state_base: u32,
    pub draft_pages_per_request: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub cache_lookups: u64,
    pub cache_hits: u64,
    pub reused_tokens: u64,
    pub snapshot_publications: u64,
    pub snapshot_evictions: u64,
    pub system_checkpoints: u64,
    pub prompt_checkpoints: u64,
    pub active_requests: usize,
    pub free_slots: usize,
    pub retiring_slots: usize,
    pub free_pages: usize,
    pub free_snapshots: usize,
    pub cached_checkpoints: usize,
    pub cached_pages: usize,
}

#[derive(Clone)]
pub struct RequestSpec {
    pub prompt: Vec<u32>,
    pub system_boundary: Option<usize>,
    pub conditioning: Vec<ConditioningEvent>,
    pub max_new_tokens: u32,
    pub grammar_constrained: bool,
    pub host_stops: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestPhase {
    Initialize,
    Restore,
    Vision,
    Prefill,
    Resume,
    Decode,
    Snapshot,
    Finishing,
}

#[derive(Clone, Copy, Debug)]
enum BoundaryKind {
    System,
    Prompt,
}
#[derive(Clone, Copy, Debug)]
struct BoundaryCapture {
    kind: BoundaryKind,
    snapshot: SnapshotId,
    consumed_tokens: u64,
    resume: RequestPhase,
}

pub struct LiveRequest {
    pub handle: RequestHandle,
    pub state_slot: StateSlotId,
    pub terminal_snapshot: SnapshotId,
    pub phase: RequestPhase,
    pub consumed_tokens: u64,
    pub generated_tokens: u32,
    pub prompt_tokens: usize,
    pub grammar_constrained: bool,
    pub error_code: Option<i64>,
    host_stops: bool,
    execution_limit: u64,
    history: PrefixHistory,
    tokens: Vec<u32>,
    prefix: Option<CheckpointId>,
    prefix_tokens: u64,
    admission_reused_tokens: u64,
    system_boundary: Option<usize>,
    system_snapshot: Option<SnapshotId>,
    prompt_snapshot: Option<SnapshotId>,
    boundary_capture: Option<BoundaryCapture>,
    prefix_snapshot: Option<SnapshotId>,
    shared_pages: Vec<PageId>,
    private_pages: Vec<PageId>,
    reserved_pages: Vec<PageId>,
    max_new_tokens: u32,
    scheduled_consumed: u64,
    last_use: Fence,
    pending: usize,
    terminal_snapshot_ready: bool,
    vision_patches: VecDeque<usize>,
    copy_proposal: Option<speculation::CopyProposal>,
}

impl LiveRequest {
    pub fn prefix(&self) -> Option<CheckpointId> {
        self.prefix
    }
    pub fn prefix_snapshot(&self) -> Option<SnapshotId> {
        self.prefix_snapshot
    }
    pub fn reused_prompt_tokens(&self) -> u64 {
        self.admission_reused_tokens
    }
    pub fn snapshot_destination(&self) -> SnapshotId {
        self.boundary_capture
            .map_or(self.terminal_snapshot, |c| c.snapshot)
    }
    pub fn snapshot_consumed(&self) -> u64 {
        self.boundary_capture
            .map_or(self.consumed_tokens, |c| c.consumed_tokens)
    }
    pub fn last_use(&self) -> Fence {
        self.last_use
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn copy_allowed(&self) -> bool {
        !self.grammar_constrained && !self.host_stops
    }
    pub fn host_stops(&self) -> bool {
        self.host_stops
    }
    pub fn copy_proposal(&self) -> Option<&speculation::CopyProposal> {
        self.copy_proposal
            .as_ref()
            .filter(|p| p.consumed == self.consumed_tokens)
    }
    pub fn pages(&self) -> impl Iterator<Item = PageId> + '_ {
        self.shared_pages.iter().chain(&self.private_pages).copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledLane {
    pub request: RequestHandle,
    pub start_position: u64,
    pub tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledStep {
    pub kind: GraphKind,
    pub batch_capacity: usize,
    pub token_capacity: usize,
    pub lanes: Vec<ScheduledLane>,
    pub grammar_constrained: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedToken {
    pub request: RequestHandle,
    pub token: u32,
    pub finished: bool,
}

struct RetiredRequest {
    pages: Vec<PageId>,
    snapshots: [Option<SnapshotId>; 3],
    slot: usize,
    through: Fence,
}
struct Slot {
    generation: u64,
    request: Option<LiveRequest>,
    retiring: bool,
}

pub struct RequestScheduler {
    limits: SchedulerLimits,
    capabilities: SchedulerCapabilities,
    identity: [u8; 32],
    cache: PrefixCheckpointCache,
    slots: Vec<Slot>,
    free_pages: BTreeSet<PageId>,
    free_snapshots: BTreeSet<SnapshotId>,
    retired: Vec<RetiredRequest>,
    submissions: BTreeMap<Fence, ScheduledStep>,
    completed: Fence,
    submitted: Fence,
    prefer_prefill: bool,
    stats: SchedulerStats,
    speculative_widths: Vec<usize>,
    speculative_fixed_width: Option<usize>,
    speculative_graph_budget: usize,
    speculative_catalog: [Vec<usize>; 3],
    proposal_source: speculation::ProposalSource,
    controller: speculation::WidthController,
}

impl RequestScheduler {
    pub fn new(
        identity: [u8; 32],
        limits: SchedulerLimits,
        capabilities: SchedulerCapabilities,
    ) -> Result<Self> {
        limits.validate_prefill()?;
        capabilities.vision.validate()?;
        if capabilities.draft_pages_per_request > limits.pages as usize
            || capabilities.vocabulary == 0
            || capabilities.vocabulary > i32::MAX as usize
            || capabilities
                .live_state_base
                .checked_add(REQUEST_LANES as u32)
                .is_none()
        {
            return Err("invalid model scheduling capabilities".into());
        }

        if !(1..=REQUEST_LANES).contains(&limits.max_active)
            || limits.max_sequence_tokens == 0
            || limits.pages == 0
            || limits.snapshots == 0
            || limits.max_sequence_tokens > i64::MAX as usize
        {
            return Err("invalid live request capacities".into());
        }
        let cache = PrefixCheckpointCache::new(
            identity,
            limits.pages,
            limits.snapshots,
            capabilities.page_tokens,
        )?;
        Ok(Self {
            limits,
            capabilities,
            identity,
            cache,
            slots: (0..limits.max_active)
                .map(|_| Slot {
                    generation: 0,
                    request: None,
                    retiring: false,
                })
                .collect(),
            free_pages: (0..limits.pages).map(PageId).collect(),
            free_snapshots: (0..limits.snapshots).map(SnapshotId).collect(),
            retired: Vec::new(),
            submissions: BTreeMap::new(),
            completed: Fence(0),
            submitted: Fence(0),
            prefer_prefill: true,
            stats: SchedulerStats::default(),
            speculative_widths: Vec::new(),
            speculative_fixed_width: None,
            speculative_graph_budget: 24,
            speculative_catalog: Default::default(),
            proposal_source: speculation::ProposalSource::Mtp,
            controller: speculation::WidthController::default(),
        })
    }

    pub fn enable_speculation(&mut self, widths: &[usize]) -> Result<()> {
        speculation::validate_widths(widths)?;
        if self.slots.iter().any(|s| s.request.is_some()) || self.submitted != Fence(0) {
            return Err("speculation must be configured before admission".into());
        }
        self.speculative_widths = widths.to_vec();
        self.refresh_speculative_catalog();
        Ok(())
    }
    pub fn fixed_speculative_width(&mut self, width: Option<usize>) -> Result<()> {
        if self.submitted != Fence(0)
            || width.is_some_and(|w| !self.speculative_widths.contains(&w))
        {
            return Err(
                "fixed speculative width must be prepared and configured before submissions".into(),
            );
        }
        self.speculative_fixed_width = width;
        self.refresh_speculative_catalog();
        Ok(())
    }
    fn refresh_speculative_catalog(&mut self) {
        self.speculative_catalog = speculation::graph_catalog(
            &self.speculative_widths,
            self.speculative_fixed_width,
            self.speculative_graph_budget,
        );
    }
    pub fn speculative_graph_budget(&mut self, budget: usize) -> Result<()> {
        if !(3..=24).contains(&budget)
            || self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
        {
            return Err(
                "speculative graph budget must be in 3..=24 and set before admission".into(),
            );
        }
        self.speculative_graph_budget = budget;
        self.refresh_speculative_catalog();
        Ok(())
    }
    pub fn proposal_source(&mut self, source: speculation::ProposalSource) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("proposal source must be configured before admission".into());
        }
        self.proposal_source = source;
        Ok(())
    }
    pub fn set_copy_proposal(
        &mut self,
        handle: RequestHandle,
        proposal: speculation::CopyProposal,
    ) -> Result<()> {
        let vocabulary = self.capabilities.vocabulary;
        let words = vocabulary.div_ceil(32);
        let request = self.request_mut(handle)?;
        if request.phase != RequestPhase::Decode
            || request.pending != 0
            || proposal.consumed != request.consumed_tokens
            || proposal.count > 7
            || proposal.tokens[..proposal.count]
                .iter()
                .any(|id| *id < 0 || *id >= vocabulary as i64)
            || proposal.tokens[proposal.count..].iter().any(|id| *id != -1)
            || proposal.grammar_masks.len()
                != if request.grammar_constrained {
                    (proposal.count + 1) * words
                } else {
                    0
                }
            || proposal
                .grammar_masks
                .chunks(words)
                .any(|mask| mask.iter().all(|word| *word == 0))
            || (request.host_stops && proposal.count != 0)
        {
            return Err("invalid copy proposal for committed request state".into());
        }
        request.copy_proposal = Some(proposal);
        Ok(())
    }
    pub fn observe_round(
        &mut self,
        step: &ScheduledStep,
        elapsed_ns: u64,
        emitted: usize,
    ) -> Result<()> {
        if step.kind != GraphKind::Speculative {
            return Err("timing supplied for a non-speculative step".into());
        }
        self.controller.observe(
            step.batch_capacity,
            step.lanes
                .iter()
                .map(|l| l.start_position)
                .max()
                .unwrap_or(0),
            step.token_capacity,
            elapsed_ns,
            emitted,
        )
    }
    pub fn submitted_step(&self, fence: Fence) -> Result<&ScheduledStep> {
        self.submissions
            .get(&fence)
            .ok_or_else(|| "unknown scheduled submission".into())
    }
    pub fn cache(&self) -> &PrefixCheckpointCache {
        &self.cache
    }
    pub fn stats(&self) -> SchedulerStats {
        let cache = self.cache.stats();
        SchedulerStats {
            active_requests: self
                .slots
                .iter()
                .filter(|slot| slot.request.is_some())
                .count(),
            free_slots: self
                .slots
                .iter()
                .filter(|slot| {
                    slot.request.is_none() && !slot.retiring && slot.generation < i64::MAX as u64
                })
                .count(),
            retiring_slots: self.slots.iter().filter(|slot| slot.retiring).count(),
            free_pages: self.free_pages.len(),
            free_snapshots: self.free_snapshots.len(),
            cached_checkpoints: cache.published_checkpoints,
            cached_pages: cache.cached_pages,
            ..self.stats
        }
    }
    pub fn completed(&self) -> Fence {
        self.completed
    }
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }
    pub fn free_snapshot_count(&self) -> usize {
        self.free_snapshots.len()
    }
    pub fn has_available_slot(&self) -> bool {
        self.slots.iter().any(|slot| {
            slot.request.is_none() && !slot.retiring && slot.generation < i64::MAX as u64
        })
    }
    pub fn request(&self, handle: RequestHandle) -> Result<&LiveRequest> {
        self.slots
            .get(handle.slot as usize)
            .and_then(|slot| slot.request.as_ref())
            .filter(|request| request.handle == handle)
            .ok_or_else(|| "stale or retired request handle".into())
    }
    fn request_mut(&mut self, handle: RequestHandle) -> Result<&mut LiveRequest> {
        self.slots
            .get_mut(handle.slot as usize)
            .and_then(|slot| slot.request.as_mut())
            .filter(|request| request.handle == handle)
            .ok_or_else(|| "stale or retired request handle".into())
    }

    fn ensure_resources(&mut self, pages: usize, snapshots: usize) -> Result<()> {
        let need_pages = pages.saturating_sub(self.free_pages.len());
        let need_snapshots = snapshots.saturating_sub(self.free_snapshots.len());
        if need_pages > 0 || need_snapshots > 0 {
            let evicted = self.cache.evict(EvictionTarget {
                pages: need_pages,
                snapshots: need_snapshots,
            });
            self.stats.snapshot_evictions += evicted.retired.len() as u64;
            let update = self.cache.advance_completed(self.completed)?;
            self.stats.snapshot_publications += update.published.len() as u64;
            let reusable = update.reusable;
            self.free_pages.extend(reusable.pages);
            self.free_snapshots.extend(reusable.snapshots);
        }
        if self.free_pages.len() < pages || self.free_snapshots.len() < snapshots {
            return Err(crate::Error::admission(
                crate::AdmissionError::Capacity,
                "live request resources are exhausted or awaiting GPU retirement",
            ));
        }
        Ok(())
    }

    pub fn admit(&mut self, spec: RequestSpec) -> Result<RequestHandle> {
        self.admit_with_prefix(spec, true)
    }

    #[cfg(any(test, feature = "server"))]
    pub(crate) fn admit_cold(&mut self, spec: RequestSpec) -> Result<RequestHandle> {
        self.admit_with_prefix(spec, false)
    }

    fn admit_with_prefix(
        &mut self,
        spec: RequestSpec,
        reuse_prefix: bool,
    ) -> Result<RequestHandle> {
        if spec.prompt.is_empty()
            || spec
                .system_boundary
                .is_some_and(|n| n == 0 || n > spec.prompt.len())
            || spec.max_new_tokens == 0
            || spec
                .prompt
                .len()
                .checked_add(spec.max_new_tokens as usize)
                .is_none_or(|count| count > self.limits.max_sequence_tokens)
        {
            return Err(crate::Error::admission(
                crate::AdmissionError::Invalid,
                "request requires a nonempty prompt, positive output limit, and bounded total tokens",
            ));
        }
        let slot = self
            .slots
            .iter()
            .position(|slot| {
                slot.request.is_none() && !slot.retiring && slot.generation < i64::MAX as u64
            })
            .ok_or_else(|| {
                crate::Error::admission(
                    crate::AdmissionError::Capacity,
                    "all request slots are active, retired, or generation-exhausted",
                )
            })?;
        let mut history = PrefixHistory::new(self.identity);
        history.append(&spec.prompt, &spec.conditioning)?;
        let prefix = if reuse_prefix {
            let prefix = self.cache.lookup(&history)?;
            self.stats.cache_lookups += 1;
            self.stats.cache_hits += u64::from(prefix.checkpoint.is_some());
            self.stats.reused_tokens += prefix.consumed_tokens;
            prefix
        } else {
            crate::cache::Lookup {
                checkpoint: None,
                consumed_tokens: 0,
                pages: Vec::new(),
                snapshot: None,
            }
        };
        if let Some(id) = prefix.checkpoint {
            self.cache.pin(id)?;
        }
        let extra = self.speculative_widths.last().copied().unwrap_or(1) - 1;
        let execution_limit = (spec.prompt.len() + spec.max_new_tokens as usize)
            .saturating_add(extra)
            .min(self.limits.max_sequence_tokens);
        let draft_page = if !self.speculative_widths.is_empty()
            && self.proposal_source == speculation::ProposalSource::Mtp
        {
            self.capabilities.draft_pages_per_request
        } else {
            0
        };
        let reserved_count = execution_limit.div_ceil(self.capabilities.page_tokens)
            - prefix.consumed_tokens as usize / self.capabilities.page_tokens
            + draft_page;
        let need_system = spec
            .system_boundary
            .is_some_and(|n| n as u64 > prefix.consumed_tokens);
        let need_prompt = spec.prompt.len() as u64 > prefix.consumed_tokens
            && spec.system_boundary != Some(spec.prompt.len())
            && spec.max_new_tokens > 1;
        if let Err(error) = self.ensure_resources(
            reserved_count,
            1 + usize::from(need_system) + usize::from(need_prompt),
        ) {
            if let Some(id) = prefix.checkpoint {
                self.cache.unpin(id)?;
            }
            return Err(error);
        }
        let terminal_snapshot = self.free_snapshots.pop_first().unwrap();
        let system_snapshot = need_system.then(|| self.free_snapshots.pop_first().unwrap());
        let prompt_snapshot = need_prompt.then(|| self.free_snapshots.pop_first().unwrap());
        let reserved_pages = (0..reserved_count)
            .map(|_| self.free_pages.pop_first().unwrap())
            .rev()
            .collect();
        let generation = self.slots[slot].generation + 1;
        let handle = RequestHandle {
            slot: slot as u32,
            generation,
        };
        self.slots[slot].generation = generation;
        self.slots[slot].request = Some(LiveRequest {
            handle,
            state_slot: StateSlotId(slot as u32 + self.capabilities.live_state_base),
            terminal_snapshot,
            phase: if prefix.checkpoint.is_some() {
                RequestPhase::Restore
            } else {
                RequestPhase::Initialize
            },
            consumed_tokens: prefix.consumed_tokens,
            generated_tokens: 0,
            prompt_tokens: spec.prompt.len(),
            grammar_constrained: spec.grammar_constrained,
            host_stops: spec.host_stops,
            execution_limit: execution_limit as u64,
            error_code: None,
            copy_proposal: None,
            history,
            tokens: spec.prompt,
            prefix: prefix.checkpoint,
            prefix_tokens: prefix.consumed_tokens,
            admission_reused_tokens: prefix.consumed_tokens,
            system_boundary: spec.system_boundary,
            system_snapshot,
            prompt_snapshot,
            boundary_capture: None,
            prefix_snapshot: prefix.snapshot,
            shared_pages: prefix.pages,
            private_pages: Vec::new(),
            reserved_pages,
            max_new_tokens: spec.max_new_tokens,
            scheduled_consumed: prefix.consumed_tokens,
            last_use: Fence(0),
            pending: 0,
            terminal_snapshot_ready: false,
            vision_patches: VecDeque::new(),
        });
        Ok(handle)
    }

    pub fn set_vision_images(&mut self, handle: RequestHandle, patches: &[usize]) -> Result<()> {
        let vision = &self.capabilities.vision;
        let maximum = *vision
            .patch_capacities
            .last()
            .ok_or("vision is unsupported")?;
        let maximum_images = vision.max_images;
        let merge = vision.patches_per_feature;
        let request = self.request_mut(handle)?;
        if request.last_use != Fence(0)
            || !request.vision_patches.is_empty()
            || patches.len() > maximum_images
            || patches
                .iter()
                .any(|count| *count == 0 || *count > maximum || *count % merge != 0)
        {
            return Err("vision images must be assigned once before request initialization".into());
        }
        request.vision_patches.extend(patches.iter().copied());
        Ok(())
    }

    /// Plans one operation without submitting GPU work or reserving new pages.
    pub fn next_step(&self) -> Option<ScheduledStep> {
        if self.submissions.len() >= 2
            || self
                .submissions
                .values()
                .any(|step| step.grammar_constrained)
        {
            return None;
        }
        let requests: Vec<_> = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .collect();
        let select = |phase: RequestPhase, allow_pending: bool| {
            requests
                .iter()
                .copied()
                .filter(|request| {
                    request.phase == phase
                        && (request.pending == 0
                            || (allow_pending
                                && request.pending < 2
                                && !request.grammar_constrained
                                && !request.host_stops))
                        && (phase != RequestPhase::Decode
                            || self.speculative_widths.is_empty()
                            || request.scheduled_consumed < self.limits.max_sequence_tokens as u64)
                })
                .collect::<Vec<_>>()
        };
        let (kind, selected) = if !select(RequestPhase::Initialize, false).is_empty() {
            (GraphKind::Init, select(RequestPhase::Initialize, false))
        } else if !select(RequestPhase::Restore, false).is_empty() {
            (GraphKind::Restore, select(RequestPhase::Restore, false))
        } else if !select(RequestPhase::Snapshot, false).is_empty() {
            (GraphKind::Snapshot, select(RequestPhase::Snapshot, false))
        } else if !select(RequestPhase::Vision, false).is_empty() {
            (
                GraphKind::Vision,
                vec![select(RequestPhase::Vision, false)[0]],
            )
        } else if !select(RequestPhase::Resume, false).is_empty() {
            (GraphKind::Resume, select(RequestPhase::Resume, false))
        } else {
            let prefill = select(RequestPhase::Prefill, false);
            let decode = select(RequestPhase::Decode, true);
            if !prefill.is_empty() && (self.prefer_prefill || decode.is_empty()) {
                (GraphKind::Prefill, vec![prefill[0]])
            } else if !decode.is_empty() {
                (
                    if self.speculative_widths.is_empty() {
                        GraphKind::Decode
                    } else {
                        GraphKind::Speculative
                    },
                    decode,
                )
            } else {
                return None;
            }
        };
        let token_barrier = selected
            .iter()
            .any(|r| r.grammar_constrained || r.host_stops);
        let width = if kind == GraphKind::Speculative {
            let batch = selected.len().next_power_of_two();
            let widths = &self.speculative_catalog[batch.trailing_zeros() as usize];
            if self.speculative_fixed_width.is_some() {
                *widths.last()?
            } else {
                self.controller.choose(
                    widths,
                    batch,
                    selected
                        .iter()
                        .map(|r| r.scheduled_consumed)
                        .max()
                        .unwrap_or(0),
                    8,
                )?
            }
        } else {
            1
        };
        let lanes: Vec<_> = selected
            .iter()
            .map(|request| ScheduledLane {
                request: request.handle,
                start_position: request.scheduled_consumed,
                tokens: match kind {
                    GraphKind::Prefill => (request
                        .system_boundary
                        .filter(|n| *n as u64 > request.consumed_tokens)
                        .unwrap_or(request.prompt_tokens)
                        - request.consumed_tokens as usize)
                        .min(self.limits.prefill_chunk_size),
                    GraphKind::Decode => 1,
                    GraphKind::Speculative => {
                        let remaining =
                            self.limits.max_sequence_tokens - request.scheduled_consumed as usize;
                        let limit = if request.host_stops
                            || (request.grammar_constrained
                                && self.proposal_source == speculation::ProposalSource::Mtp)
                        {
                            1
                        } else if self.proposal_source == speculation::ProposalSource::Copy {
                            request.copy_proposal().map_or(1, |p| p.count + 1)
                        } else {
                            width
                        };
                        width.min(remaining).min(limit)
                    }
                    _ => 0,
                },
            })
            .collect();
        let token_capacity = if kind == GraphKind::Prefill {
            prefill_capacities(self.limits.prefill_chunk_size)
                .find(|capacity| *capacity >= lanes[0].tokens)
                .unwrap()
        } else if kind == GraphKind::Vision {
            let patches = selected[0].vision_patches[0];
            self.capabilities
                .vision
                .patch_capacities
                .iter()
                .copied()
                .find(|capacity| *capacity >= patches)
                .unwrap()
        } else {
            width
        };
        let barrier = token_barrier || kind == GraphKind::Speculative;
        if !self.submissions.is_empty() && barrier {
            return None;
        }
        Some(ScheduledStep {
            kind,
            batch_capacity: selected.len().next_power_of_two(),
            token_capacity,
            lanes,
            grammar_constrained: barrier,
        })
    }

    /// Records resource use before enqueueing the matching GPU operation.
    /// A CUDA submission failure requires stream drain before retirement can advance.
    pub fn record_submission(&mut self, step: &ScheduledStep, fence: Fence) -> Result<()> {
        if fence <= self.submitted || fence.0 > i64::MAX as u64 {
            return Err("submission fences must increase within int64 range".into());
        }
        if self.submissions.len() >= 2
            || (step.grammar_constrained && !self.submissions.is_empty())
            || self
                .submissions
                .values()
                .any(|step| step.grammar_constrained)
        {
            return Err("submission window or grammar barrier is full".into());
        }
        if self.next_step().as_ref() != Some(step) {
            return Err("scheduled step is stale or does not match request state".into());
        }
        let mut needed = Vec::new();
        let draft_page = if !self.speculative_widths.is_empty()
            && self.proposal_source == speculation::ProposalSource::Mtp
        {
            self.capabilities.draft_pages_per_request
        } else {
            0
        };
        for lane in &step.lanes {
            let request = self.request(lane.request)?;
            let through = lane
                .start_position
                .checked_add(lane.tokens as u64)
                .ok_or("consumed token overflow")?;
            if through > self.limits.max_sequence_tokens as u64 {
                return Err("scheduled request exceeds sequence capacity".into());
            }
            let pages = (through.min(request.execution_limit) as usize)
                .div_ceil(self.capabilities.page_tokens)
                .saturating_sub(request.prefix_tokens as usize / self.capabilities.page_tokens)
                .saturating_add(draft_page)
                .saturating_sub(request.private_pages.len());
            if pages > request.reserved_pages.len() {
                return Err("scheduled step exceeds the request's reserved page budget".into());
            }
            needed.push((lane.request, pages, through));
        }
        for (handle, pages, through) in needed {
            let request = self.request_mut(handle)?;
            let allocated: Vec<_> = (0..pages)
                .map(|_| request.reserved_pages.pop().unwrap())
                .collect();
            request.private_pages.extend(allocated);
            request.scheduled_consumed = through;
            request.last_use = fence;
            request.pending += 1;
            if let Some(prefix) = request.prefix {
                self.cache.mark_used(prefix, fence)?;
            }
        }
        self.submitted = fence;
        if matches!(
            step.kind,
            GraphKind::Prefill | GraphKind::Decode | GraphKind::Speculative
        ) {
            self.prefer_prefill = step.kind != GraphKind::Prefill;
        }
        self.submissions.insert(fence, step.clone());
        Ok(())
    }

    pub fn complete(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord; REQUEST_LANES],
    ) -> Result<Vec<GeneratedToken>> {
        if self.submissions.first_key_value().map(|(next, _)| *next) != Some(fence) {
            return Err("completion fence must match the oldest submitted step".into());
        }
        let step = self.submissions.get(&fence).unwrap();
        if records.iter().enumerate().any(|(slot, record)| {
            record.status != 0
                && !step
                    .lanes
                    .iter()
                    .any(|lane| lane.request.slot as usize == slot)
        }) {
            return Err(
                "completion contains an active physical slot outside the submitted lanes".into(),
            );
        }
        // Validate every live lane before committing any completion state.
        for lane in &step.lanes {
            records[lane.request.slot as usize].validate_vector()?;
            let Ok(request) = self.request(lane.request) else {
                continue;
            };
            if matches!(
                step.kind,
                GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot | GraphKind::Vision
            ) {
                let record = records[lane.request.slot as usize];
                if !(1..=3).contains(&record.status)
                    || record.generation != lane.request.generation as i64
                    || record.submission_sequence != fence.0 as i64
                    || (record.status != 3
                        && (record.consumed_tokens != request.consumed_tokens as i64
                            || record.generated_tokens != request.generated_tokens as i64
                            || record.token_id != -1
                            || record.vector.is_some_and(|v| v.emitted_count != 0)))
                {
                    return Err(
                        "state-transfer completion is inactive or has mismatched generation, sequence or counters".into(),
                    );
                }
                continue;
            }
            let record = records[lane.request.slot as usize];
            let emitted = record.generated_tokens - request.generated_tokens as i64;
            let width = if step.kind == GraphKind::Speculative {
                step.token_capacity as i64
            } else {
                1
            };
            if let Some(vector) = record.vector {
                if vector.emitted_count != emitted
                    || (step.kind == GraphKind::Speculative
                        && (vector.verification_width != width
                            || record.consumed_tokens - request.consumed_tokens as i64 != emitted))
                {
                    return Err("token vector does not match committed request progress".into());
                }
            } else if step.kind == GraphKind::Speculative {
                return Err("speculative completion requires ABI3 token vector".into());
            }
            if record.generation != lane.request.generation as i64
                || record.submission_sequence != fence.0 as i64
                || !(1..=3).contains(&record.status)
                || record.consumed_tokens < request.consumed_tokens as i64
                || record.consumed_tokens as u64 > lane.start_position + lane.tokens as u64
                || record.generated_tokens < request.generated_tokens as i64
                || record.generated_tokens > request.generated_tokens as i64 + width
                || record.generated_tokens > request.max_new_tokens as i64
                || (record.status == 1
                    && step.kind != GraphKind::Speculative
                    && record.consumed_tokens as u64 != lane.start_position + lane.tokens as u64)
                || (record.status == 1
                    && matches!(
                        step.kind,
                        GraphKind::Decode | GraphKind::Speculative | GraphKind::Resume
                    )
                    && record.generated_tokens == request.generated_tokens as i64)
                || (record.status == 1
                    && matches!(
                        request.phase,
                        RequestPhase::Finishing | RequestPhase::Snapshot
                    ))
                || (record.generated_tokens > request.generated_tokens as i64
                    && u32::try_from(record.token_id).is_err())
            {
                return Err(
                    "completion counters, generation or token do not match the live request".into(),
                );
            }
        }
        let step = self.submissions.remove(&fence).unwrap();
        let mut generated = Vec::new();
        let mut boundaries = Vec::new();
        for lane in step.lanes {
            let Ok(request) = self.request_mut(lane.request) else {
                continue;
            };
            request.pending -= 1;
            if matches!(
                step.kind,
                GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot | GraphKind::Vision
            ) {
                let record = records[lane.request.slot as usize];
                if record.status == 3 {
                    request.error_code = Some(record.error_code);
                    request.terminal_snapshot_ready = false;
                    request.phase = RequestPhase::Finishing;
                } else if step.kind == GraphKind::Snapshot {
                    if request.boundary_capture.is_some() {
                        boundaries.push(lane.request);
                    } else {
                        request.terminal_snapshot_ready = true;
                        request.phase = RequestPhase::Finishing;
                    }
                } else {
                    if step.kind == GraphKind::Vision {
                        request.vision_patches.pop_front();
                    }
                    request.phase = if !request.vision_patches.is_empty() {
                        RequestPhase::Vision
                    } else if request.consumed_tokens == request.prompt_tokens as u64 {
                        RequestPhase::Resume
                    } else {
                        RequestPhase::Prefill
                    };
                }
                continue;
            }
            let record = records[lane.request.slot as usize];
            let finished =
                record.status >= 2 || record.generated_tokens == request.max_new_tokens as i64;
            if record.status == 3 {
                request.error_code = Some(record.error_code);
            }
            request.consumed_tokens = record.consumed_tokens as u64;
            if record.generated_tokens > request.generated_tokens as i64 {
                let count = (record.generated_tokens - request.generated_tokens as i64) as usize;
                for index in 0..count {
                    let token = record.vector.map_or(record.token_id, |v| v.tokens[index]) as u32;
                    request.history.append(&[token], &[])?;
                    request.tokens.push(token);
                    request.generated_tokens += 1;
                    generated.push(GeneratedToken {
                        request: lane.request,
                        token,
                        finished: finished && index + 1 == count,
                    });
                }
            }
            request.phase = if finished {
                if request.consumed_tokens > request.prefix_tokens && request.error_code.is_none() {
                    RequestPhase::Snapshot
                } else {
                    RequestPhase::Finishing
                }
            } else if request.consumed_tokens < request.prompt_tokens as u64 {
                RequestPhase::Prefill
            } else if request.generated_tokens == 0 {
                RequestPhase::Resume
            } else {
                RequestPhase::Decode
            };
            if step.kind == GraphKind::Prefill && request.error_code.is_none() {
                let endpoint = request.consumed_tokens;
                let capture = if request.system_boundary == Some(endpoint as usize) {
                    request
                        .system_snapshot
                        .map(|snapshot| (BoundaryKind::System, snapshot))
                } else {
                    None
                }
                .or_else(|| {
                    (endpoint == request.prompt_tokens as u64)
                        .then_some(request.prompt_snapshot)
                        .flatten()
                        .map(|snapshot| (BoundaryKind::Prompt, snapshot))
                });
                if let Some((kind, snapshot)) = capture {
                    request.boundary_capture = Some(BoundaryCapture {
                        kind,
                        snapshot,
                        consumed_tokens: endpoint,
                        resume: request.phase,
                    });
                    request.phase = RequestPhase::Snapshot;
                }
            }
            if request.pending == 0 {
                request.scheduled_consumed = request.consumed_tokens;
            }
        }
        self.advance_retirement(fence)?;
        for handle in boundaries {
            self.publish_boundary(handle)?;
        }
        Ok(generated)
    }

    fn publish_boundary(&mut self, handle: RequestHandle) -> Result<()> {
        let request = self.request(handle)?;
        let capture = request.boundary_capture.ok_or("missing boundary capture")?;
        let pages = capture.consumed_tokens as usize / self.capabilities.page_tokens
            - request.prefix_tokens as usize / self.capabilities.page_tokens;
        let proposal = Publication {
            parent: request.prefix,
            consumed_tokens: capture.consumed_tokens,
            tail_pages: request.private_pages[..pages].to_vec(),
            snapshot: StateSnapshot {
                id: capture.snapshot,
                consumed_tokens: capture.consumed_tokens,
            },
            ready_after: request.last_use,
        };
        let history = request.history.clone();
        let result = self.cache.publish(&history, &proposal)?;
        if let PublishResult::Created { id, ready } = result {
            if !ready {
                return Err("completed boundary copy was not GPU-ready".into());
            }
            self.cache.pin(id)?;
            if let Some(parent) = self.request(handle)?.prefix {
                self.cache.unpin(parent)?;
            }
            let request = self.request_mut(handle)?;
            request
                .shared_pages
                .extend(request.private_pages.drain(..pages));
            request.prefix = Some(id);
            request.prefix_tokens = capture.consumed_tokens;
            request.prefix_snapshot = Some(capture.snapshot);
            self.stats.snapshot_publications += 1;
        } else {
            self.free_snapshots.insert(capture.snapshot);
        }
        match capture.kind {
            BoundaryKind::System => self.stats.system_checkpoints += 1,
            BoundaryKind::Prompt => self.stats.prompt_checkpoints += 1,
        }
        let request = self.request_mut(handle)?;
        match capture.kind {
            BoundaryKind::System => request.system_snapshot = None,
            BoundaryKind::Prompt => request.prompt_snapshot = None,
        }
        request.boundary_capture = None;
        request.phase = if capture.resume == RequestPhase::Snapshot
            && request.prefix_tokens == request.consumed_tokens
        {
            RequestPhase::Finishing
        } else {
            capture.resume
        };
        Ok(())
    }

    fn advance_retirement(&mut self, fence: Fence) -> Result<()> {
        let update = self.cache.advance_completed(fence)?;
        self.stats.snapshot_publications += update.published.len() as u64;
        let reusable = update.reusable;
        self.free_pages.extend(reusable.pages);
        self.free_snapshots.extend(reusable.snapshots);
        self.completed = fence;
        let mut waiting = Vec::new();
        for retired in self.retired.drain(..) {
            if retired.through <= fence {
                self.free_pages.extend(retired.pages);
                self.free_snapshots
                    .extend(retired.snapshots.into_iter().flatten());
                self.slots[retired.slot].retiring = false;
            } else {
                waiting.push(retired);
            }
        }
        self.retired = waiting;
        Ok(())
    }

    /// Cancels scheduling immediately; resources remain reserved through last GPU use.
    pub fn cancel(&mut self, handle: RequestHandle) -> Result<()> {
        self.retire(handle, false).map(|_| ())
    }

    /// Publishes an immutable terminal snapshot after all writes have completed.
    pub fn finish(&mut self, handle: RequestHandle) -> Result<Option<CheckpointId>> {
        let request = self.request(handle)?;
        if request.phase != RequestPhase::Finishing || request.last_use > self.completed {
            return Err("terminal checkpoint publication requires a finished request and completed GPU writes".into());
        }
        self.retire(handle, true)
    }

    /// Releases live ownership after the executor confirms both streams drained.
    /// Unconfirmed shutdown leaves every live resource and pending fence intact.
    pub fn release_after_shutdown(&mut self, shutdown: &Shutdown) -> Result<()> {
        if !shutdown.drained {
            return Err("cannot release request resources before confirmed CUDA drain".into());
        }
        let handles: Vec<_> = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref().map(|request| request.handle))
            .collect();
        for handle in handles {
            self.cancel(handle)?;
        }
        self.submissions.clear();
        self.advance_retirement(self.submitted)
    }

    fn retire(&mut self, handle: RequestHandle, publish: bool) -> Result<Option<CheckpointId>> {
        let request = self.request(handle)?;
        let publication = (publish && request.terminal_snapshot_ready).then(|| Publication {
            parent: request.prefix,
            consumed_tokens: request.consumed_tokens,
            tail_pages: request.private_pages[..(request.consumed_tokens as usize
                / self.capabilities.page_tokens
                - request.prefix_tokens as usize / self.capabilities.page_tokens)]
                .to_vec(),
            snapshot: StateSnapshot {
                id: request.terminal_snapshot,
                consumed_tokens: request.consumed_tokens,
            },
            ready_after: request.last_use,
        });
        let result = match &publication {
            Some(proposal) => {
                let history = request.history.clone();
                Some(self.cache.publish(&history, proposal)?)
            }
            None => None,
        };
        let mut request = self.slots[handle.slot as usize].request.take().unwrap();
        let mut snapshot = Some(request.terminal_snapshot);
        if matches!(result, Some(PublishResult::Created { .. })) {
            if matches!(result, Some(PublishResult::Created { ready: true, .. })) {
                self.stats.snapshot_publications += 1;
            }
            request
                .private_pages
                .drain(..publication.as_ref().unwrap().tail_pages.len());
            snapshot = None;
        }
        if let Some(prefix) = request.prefix {
            self.cache.unpin(prefix)?;
        }
        self.slots[handle.slot as usize].retiring = true;
        request.private_pages.extend(request.reserved_pages);
        self.retired.push(RetiredRequest {
            pages: request.private_pages,
            snapshots: [snapshot, request.system_snapshot, request.prompt_snapshot],
            slot: handle.slot as usize,
            through: request.last_use,
        });
        self.advance_retirement(self.completed)?;
        Ok(result.map(PublishResult::id).or(request.prefix))
    }
}

#[cfg(test)]
mod boundary_tests;
#[cfg(test)]
mod tests;
