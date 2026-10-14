use crate::{
    Result,
    execution::{GraphKind, REQUEST_LANES, Upload},
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
    page_counts: [(u64, usize); REQUEST_LANES],
    draft_page_counts: [(u64, usize); REQUEST_LANES],
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
            page_counts: [(0, 0); REQUEST_LANES],
            draft_page_counts: [(0, 0); REQUEST_LANES],
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
        let capacity = config.limits.request_capacity();
        self.staging.clear();
        for lane in &step.lanes {
            let request = scheduler.request(lane.request)?;
            self.staging.i64(
                "request_state_rows",
                lane.request.slot as usize * 8 * 8,
                (0..8).map(|column| {
                    if column < scheduler.execution_rows() {
                        request.state_slot.0 as i64 + column as i64
                    } else {
                        0
                    }
                }),
            );
        }
        if !matches!(step.kind, GraphKind::Init | GraphKind::Restore) {
            self.staging
                .i32("row_slot_ids", 0, slots.into_iter().take(capacity));
            self.staging
                .i64("row_generations", 0, generations.into_iter().take(capacity));
            self.staging.i32("live_rows", 0, [step.lanes.len() as i32]);
        }
        if matches!(
            step.kind,
            GraphKind::Prefill
                | GraphKind::Decode
                | GraphKind::Speculative
                | GraphKind::Mixed
                | GraphKind::Resume
                | GraphKind::Restore
                | GraphKind::Snapshot
        ) {
            if !matches!(step.kind, GraphKind::Restore | GraphKind::Snapshot) {
                self.staging
                    .i32("sample_enabled", 0, samples.into_iter().take(capacity));
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
                let capacity = config.prepared_sequence_tokens / 64;
                let live = scheduler.request(lane.request)?;
                for (name, skip, counts) in [
                    ("page_tables", 0, &mut self.page_counts),
                    ("draft_page_tables", 1, &mut self.draft_page_counts),
                ] {
                    if skip == 1 && !config.has_learned_drafts() {
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
        if let Some(mixed) = &step.mixed {
            self.staging.i32(
                "prefill_row",
                0,
                [mixed.prefill_row.map_or(-1, |row| row as i32)],
            );
            self.staging
                .i32("query_lengths", 0, lengths.into_iter().take(capacity));
        }
        if step.kind == GraphKind::Speculative || step.mixed.as_ref().is_some_and(|m| m.speculative)
        {
            if step.kind == GraphKind::Speculative {
                self.staging
                    .i32("query_lengths", 0, lengths.into_iter().take(capacity));
            }
            let mut ids = [-1i64; 7 * REQUEST_LANES];
            let mut counts = [0; REQUEST_LANES];
            let mut expected = [-1i64; REQUEST_LANES];
            let mut copy_generations = [0i64; REQUEST_LANES];
            let mut mask_counts = [0i32; REQUEST_LANES];
            for (row, lane) in step.lanes.iter().enumerate() {
                if step.lane_kind(row) != GraphKind::Speculative {
                    continue;
                }
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
            self.staging
                .i64("copy_token_ids", 0, ids.into_iter().take(capacity * 7));
            self.staging
                .i32("copy_token_count", 0, counts.into_iter().take(capacity));
            self.staging.i64(
                "copy_expected_consumed",
                0,
                expected.into_iter().take(capacity),
            );
            self.staging.i64(
                "copy_generation",
                0,
                copy_generations.into_iter().take(capacity),
            );
            if config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
            {
                self.staging.i32(
                    "copy_grammar_mask_count",
                    0,
                    mask_counts.into_iter().take(capacity),
                );
            }
        }
        if let Some(row) = step
            .lanes
            .iter()
            .enumerate()
            .position(|(row, _)| step.lane_kind(row) == GraphKind::Prefill)
        {
            let lane = &step.lanes[row];
            let mixed = step.kind == GraphKind::Mixed;
            let token_capacity = if mixed {
                step.prefill_capacity()
            } else {
                step.token_capacity
            };
            let offset = step.lanes[..row]
                .iter()
                .map(|lane| lane.tokens)
                .sum::<usize>();
            if (!mixed && step.lanes.len() != 1)
                || lane.tokens == 0
                || lane.tokens > config.limits.prefill_chunk_size
                || lane.tokens > token_capacity
                || token_capacity > config.limits.prefill_chunk_size
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
            if !mixed {
                self.staging
                    .i32("query_lengths", 0, lengths.into_iter().take(capacity));
            }
            self.staging.i64(
                "token_ids",
                offset * 8,
                tokens.iter().map(|token| *token as i64),
            );
            self.staging.i64(
                "prefill_positions",
                offset * 3 * 8,
                positions.iter().flatten().copied(),
            );
            self.staging.i32(
                "vision_token_indices",
                offset * 4,
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
            self.staging.i64(
                "state_request_slots",
                0,
                step.lanes
                    .iter()
                    .map(|lane| lane.request.slot as i64)
                    .chain(std::iter::repeat(-1))
                    .take(capacity),
            );
            let mut source = [0; REQUEST_LANES];
            let mut destination = [0; REQUEST_LANES];
            let mut operation = [0; REQUEST_LANES];
            let mut qsa = [-1; REQUEST_LANES];
            let mut expected = [-1; REQUEST_LANES];
            for (row, lane) in step.lanes.iter().enumerate() {
                let request = scheduler.request(lane.request)?;
                if step.kind == GraphKind::Snapshot {
                    source[row] = request.state_slot.0 as i64;
                    destination[row] =
                        scheduler.snapshot_slot(request.snapshot_destination()).0 as i64;
                    expected[row] = request.snapshot_consumed() as i64;
                } else {
                    source[row] = if let Some(snapshot) = request.prefix_snapshot() {
                        scheduler.snapshot_slot(snapshot).0 as i64
                    } else {
                        request.state_slot.0 as i64
                    };
                    destination[row] = request.state_slot.0 as i64;
                    operation[row] = i32::from(step.kind == GraphKind::Init);
                    qsa[row] = lane.request.slot as i64 + 1;
                    expected[row] = request.consumed_tokens as i64;
                }
            }
            self.staging.i64(
                "state_expected_consumed",
                0,
                expected.into_iter().take(capacity),
            );
            self.staging
                .i64("state_source_slots", 0, source.into_iter().take(capacity));
            self.staging.i64(
                "state_destination_slots",
                0,
                destination.into_iter().take(capacity),
            );
            self.staging
                .i32("state_operation", 0, operation.into_iter().take(capacity));
            self.staging
                .i64("state_qsa_slots", 0, qsa.into_iter().take(capacity));
            self.staging.i32(
                "state_accepted_tokens",
                0,
                (0..capacity).map(|row| {
                    i32::from(step.kind != GraphKind::Snapshot && row < step.lanes.len())
                }),
            );
            self.staging
                .i32("state_num_ops", 0, [step.lanes.len() as i32]);
        }
        if matches!(step.kind, GraphKind::Init | GraphKind::Restore) {
            self.staging
                .i32("admission_count", 0, [step.lanes.len() as i32]);
            self.staging
                .i32("admission_slots", 0, slots.into_iter().take(capacity));
            self.staging.i64(
                "admission_generations",
                0,
                generations.into_iter().take(capacity),
            );
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
            || handle.slot as usize >= config.limits.max_active
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
