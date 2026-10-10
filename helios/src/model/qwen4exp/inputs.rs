use crate::{
    Result,
    execution::{GraphKind, Upload},
    model::{
        InputStep,
        inputs::{UploadBatch, append_pages},
    },
};
#[cfg(feature = "images")]
use crate::{model::ModelConfig, scheduler::RequestHandle};

pub type Admission = crate::model::Admission<super::Qwen4Exp>;

/// Reusable host upload storage; GraphExecutor copies these bytes before submit returns.
pub struct Inputs {
    staging: UploadBatch,
    page_counts: [(u64, usize); 4],
    draft_page_counts: [(u64, usize); 4],
}

impl Default for Inputs {
    fn default() -> Self {
        Self::new()
    }
}

impl Inputs {
    pub fn new() -> Self {
        Self {
            staging: UploadBatch::default(),
            page_counts: [(0, 0); 4],
            draft_page_counts: [(0, 0); 4],
        }
    }

    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.staging.uploads()
    }

    pub fn pack(&mut self, frame: &InputStep<'_, super::Qwen4Exp>) -> Result<()> {
        let InputStep {
            config,
            scheduler,
            step,
            admissions,
            slots,
            generations,
            lengths,
            samples,
        } = *frame;
        self.staging.clear();
        if !matches!(step.kind, GraphKind::Init | GraphKind::Restore) {
            self.staging.i32("row_slot_ids", 0, slots);
            self.staging.i64("row_generations", 0, generations);
            self.staging.i32("live_rows", 0, [step.lanes.len() as i32]);
        }
        if matches!(
            step.kind,
            GraphKind::Prefill
                | GraphKind::Decode
                | GraphKind::Speculative
                | GraphKind::Resume
                | GraphKind::Restore
                | GraphKind::Snapshot
        ) {
            if !matches!(step.kind, GraphKind::Restore | GraphKind::Snapshot) {
                self.staging.i32("sample_enabled", 0, samples);
            }
            for lane in &step.lanes {
                if !matches!(step.kind, GraphKind::Restore | GraphKind::Snapshot)
                    && let Some(mask) = &admissions[lane.request.slot as usize]
                        .as_ref()
                        .unwrap()
                        .grammar_mask
                {
                    self.staging.i32(
                        "sampling_grammar_mask",
                        lane.request.slot as usize * config.contract.vocabulary.div_ceil(32) * 4,
                        mask.iter().copied(),
                    );
                }
                let capacity = config.limits.max_sequence_tokens / 64;
                let live = scheduler.request(lane.request)?;
                for (name, skip, counts) in [
                    ("page_tables", 0, &mut self.page_counts),
                    ("draft_page_tables", 1, &mut self.draft_page_counts),
                ] {
                    if skip == 1 && !config.has_draft_cache() {
                        continue;
                    }
                    let available = live.pages().count().saturating_sub(skip).min(capacity);
                    append_pages(
                        &mut self.staging,
                        counts,
                        name,
                        live,
                        capacity,
                        skip..skip + available,
                    )?;
                }
            }
        }
        if step.kind == GraphKind::Speculative {
            self.staging.i32("query_lengths", 0, lengths);
            let mut ids = [-1i64; 28];
            let mut counts = [0; 4];
            let mut expected = [-1i64; 4];
            let mut copy_generations = [0i64; 4];
            let mut mask_counts = [0i32; 4];
            for lane in &step.lanes {
                let slot = lane.request.slot as usize;
                let live = scheduler.request(lane.request)?;
                if config.speculative_proposal_source
                    == crate::scheduler::speculation::ProposalSource::Copy
                {
                    let proposal = live
                        .copy_proposal()
                        .ok_or("missing copy plan for committed prefix")?;
                    if proposal.count + 1 < lane.tokens {
                        return Err(
                            "copy plan is shorter than the active verification prefix".into()
                        );
                    }
                    ids[slot * 7..slot * 7 + 7].copy_from_slice(&proposal.tokens);
                    counts[slot] = proposal.count as i32;
                    expected[slot] = proposal.consumed as i64;
                    copy_generations[slot] = lane.request.generation as i64;
                    mask_counts[slot] = (proposal.grammar_masks.len()
                        / config.contract.vocabulary.div_ceil(32))
                        as i32;
                    if !proposal.grammar_masks.is_empty() {
                        self.staging.i32(
                            "copy_grammar_mask",
                            slot * 8 * config.contract.vocabulary.div_ceil(32) * 4,
                            proposal.grammar_masks.iter().copied(),
                        );
                    }
                    continue;
                }
                if config.history_drafts && live.copy_allowed() {
                    let (copy, count) = crate::scheduler::speculation::copy_proposal(
                        live.tokens(),
                        step.token_capacity - 1,
                    );
                    ids[slot * 7..slot * 7 + 7].copy_from_slice(&copy);
                    counts[slot] = count as i32;
                    expected[slot] = live.consumed_tokens as i64;
                    copy_generations[slot] = lane.request.generation as i64;
                }
            }
            self.staging.i64("copy_token_ids", 0, ids);
            self.staging.i32("copy_token_count", 0, counts);
            self.staging.i64("copy_expected_consumed", 0, expected);
            self.staging.i64("copy_generation", 0, copy_generations);
            if config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
            {
                self.staging.i32("copy_grammar_mask_count", 0, mask_counts);
            }
        }
        if step.kind == GraphKind::Prefill {
            let lane = &step.lanes[0];
            if step.lanes.len() != 1
                || lane.tokens == 0
                || lane.tokens > config.limits.prefill_chunk_size
                || lane.tokens > step.token_capacity
                || step.token_capacity > config.limits.prefill_chunk_size
            {
                return Err(
                    "prefill requires one nonempty prompt chunk within its graph and configured capacities"
                        .into(),
                );
            }
            let admission = admissions[lane.request.slot as usize].as_ref().unwrap();
            let range = lane.start_position as usize..lane.start_position as usize + lane.tokens;
            let tokens = admission
                .prompt
                .get(range.clone())
                .ok_or("prefill exceeds prompt")?;
            let positions = admission
                .data
                .positions
                .get(range)
                .ok_or("prefill exceeds positions")?;
            self.staging.i32("query_lengths", 0, lengths);
            self.staging
                .i64("token_ids", 0, tokens.iter().map(|token| *token as i64));
            self.staging
                .i64("prefill_positions", 0, positions.iter().flatten().copied());
            self.staging.i32(
                "vision_token_indices",
                0,
                admission.data.vision_token_indices
                    [lane.start_position as usize..lane.start_position as usize + lane.tokens]
                    .iter()
                    .copied(),
            );
        }
        if matches!(
            step.kind,
            GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot
        ) {
            let mut source = [0; 4];
            let mut destination = [0; 4];
            let mut operation = [0; 4];
            let mut qsa = [-1; 4];
            let mut expected = [-1; 4];
            for (row, lane) in step.lanes.iter().enumerate() {
                let request = scheduler.request(lane.request)?;
                if step.kind == GraphKind::Snapshot {
                    source[row] = request.state_slot.0 as i64;
                    destination[row] = 5 + request.snapshot_destination().0 as i64;
                    expected[row] = request.snapshot_consumed() as i64;
                } else {
                    source[row] = if let Some(snapshot) = request.prefix_snapshot() {
                        5 + snapshot.0 as i64
                    } else {
                        request.state_slot.0 as i64
                    };
                    destination[row] = request.state_slot.0 as i64;
                    operation[row] = i32::from(step.kind == GraphKind::Init);
                    qsa[row] = request.state_slot.0 as i64;
                    expected[row] = request.consumed_tokens as i64;
                }
            }
            self.staging.i64("state_expected_consumed", 0, expected);
            self.staging.i64("state_source_slots", 0, source);
            self.staging.i64("state_destination_slots", 0, destination);
            self.staging.i32("state_operation", 0, operation);
            self.staging.i64("state_qsa_slots", 0, qsa);
            self.staging.i32(
                "state_accepted_tokens",
                0,
                (0..4).map(|row| {
                    i32::from(step.kind != GraphKind::Snapshot && row < step.lanes.len())
                }),
            );
            self.staging
                .i32("state_num_ops", 0, [step.lanes.len() as i32]);
        }
        if matches!(step.kind, GraphKind::Init | GraphKind::Restore) {
            self.staging
                .i32("admission_count", 0, [step.lanes.len() as i32]);
            self.staging.i32("admission_slots", 0, slots);
            self.staging.i64("admission_generations", 0, generations);
            for (row, lane) in step.lanes.iter().enumerate() {
                let admission = admissions[lane.request.slot as usize].as_ref().unwrap();
                self.staging.admission(config, row, admission.sampling())?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "images")]
    pub fn pack_vision(
        &mut self,
        config: &ModelConfig,
        handle: RequestHandle,
        images: &crate::frontend::images::ProcessedImages,
        image: usize,
    ) -> Result<()> {
        self.staging.clear();
        let geometry = images
            .geometry
            .get(image)
            .ok_or("missing scheduled image")?;
        let start: usize = images.geometry[..image]
            .iter()
            .map(|geometry| geometry.patches)
            .sum();
        let end = start + geometry.patches;
        if geometry.patches == 0
            || geometry.patches
                > *config
                    .contract
                    .vision
                    .patch_capacities
                    .last()
                    .ok_or("missing vision capacities")?
            || end / config.contract.vision.patches_per_feature
                > config.contract.vision.features_per_request
            || handle.slot >= 4
            || handle.generation == 0
            || handle.generation > i64::MAX as u64
        {
            return Err("invalid vision graph input extent".into());
        }
        self.staging.put(
            "vision_pixel_values",
            0,
            images
                .pixel_values
                .get(start * 1536..end * 1536)
                .ok_or("truncated vision pixels")?
                .iter()
                .copied()
                .flat_map(f32::to_le_bytes),
        );
        self.staging.i64(
            "vision_interpolation_indices",
            0,
            images
                .interpolation_indices
                .get(start..end)
                .ok_or("truncated vision interpolation indices")?
                .iter()
                .flatten()
                .copied(),
        );
        self.staging.put(
            "vision_interpolation_weights",
            0,
            images
                .interpolation_weights
                .get(start..end)
                .ok_or("truncated vision interpolation weights")?
                .iter()
                .flatten()
                .copied()
                .flat_map(f32::to_le_bytes),
        );
        self.staging.i64(
            "vision_position_ids",
            0,
            images
                .position_ids
                .get(start..end)
                .ok_or("truncated vision positions")?
                .iter()
                .flatten()
                .copied(),
        );
        self.staging
            .i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
        self.staging
            .i32("vision_num_patches", 0, [geometry.patches as i32]);
        self.staging.i32("vision_slot", 0, [handle.slot as i32]);
        self.staging
            .i64("vision_generation", 0, [handle.generation as i64]);
        self.staging.i32(
            "vision_feature_offset",
            0,
            [(start / config.contract.vision.patches_per_feature) as i32],
        );
        Ok(())
    }
}
