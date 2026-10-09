use crate::{
    Result,
    execution::{GraphKind, OwnedUpload, Upload},
    model::{ModelConfig, SamplingParameters},
    scheduler::{RequestHandle, RequestScheduler, ScheduledStep},
};

pub struct Admission {
    pub handle: RequestHandle,
    pub sampling: SamplingParameters,
    pub max_new_tokens: u32,
    pub prompt: Vec<u32>,
    /// Interleaved temporal, height and width positions for every prompt token.
    pub positions: Vec<[i64; 3]>,
    pub vision_token_indices: Vec<i32>,
    pub grammar_mask: Option<Vec<i32>>,
    pub ignore_eos: bool,
}

impl Admission {
    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        self.sampling.validate()?;
        if self.sampling.top_k > config.contract.vocabulary as i32
            || self.handle.slot >= 4
            || self.handle.generation == 0
            || self.handle.generation > i64::MAX as u64
            || self.max_new_tokens == 0
            || self.max_new_tokens > i32::MAX as u32
            || self.prompt.is_empty()
            || self.prompt.len() != self.positions.len()
            || self.prompt.len() != self.vision_token_indices.len()
            || self.vision_token_indices.iter().any(|index| {
                *index < -1 || *index >= config.contract.vision.features_per_request as i32
            })
            || self
                .grammar_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != config.contract.vocabulary.div_ceil(32))
            || self
                .prompt
                .iter()
                .any(|id| *id as usize >= config.contract.vocabulary)
            || self
                .positions
                .iter()
                .flatten()
                .any(|position| *position < 0)
        {
            return Err(
                "invalid prompt/admission extent, token, position or request generation".into(),
            );
        }
        Ok(())
    }
}

/// Reusable host upload storage; GraphExecutor copies these bytes before submit returns.
pub struct Inputs {
    uploads: Vec<OwnedUpload>,
    used: usize,
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
            uploads: Vec::with_capacity(40),
            used: 0,
            page_counts: [(0, 0); 4],
            draft_page_counts: [(0, 0); 4],
        }
    }

    fn put(&mut self, name: &str, offset: usize, bytes: impl Iterator<Item = u8>) {
        if self.used == self.uploads.len() {
            self.uploads.push(OwnedUpload {
                buffer: String::new(),
                offset: 0,
                bytes: Vec::new(),
            });
        }
        let upload = &mut self.uploads[self.used];
        upload.buffer.clear();
        upload.buffer.push_str(name);
        upload.offset = offset;
        upload.bytes.clear();
        upload.bytes.extend(bytes);
        self.used += 1;
    }
    fn i32(&mut self, name: &str, offset: usize, values: impl IntoIterator<Item = i32>) {
        self.put(name, offset, values.into_iter().flat_map(i32::to_le_bytes));
    }
    fn i64(&mut self, name: &str, offset: usize, values: impl IntoIterator<Item = i64>) {
        self.put(name, offset, values.into_iter().flat_map(i64::to_le_bytes));
    }
    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.uploads[..self.used]
            .iter()
            .map(|upload| Upload {
                buffer: &upload.buffer,
                offset: upload.offset,
                bytes: &upload.bytes,
            })
            .collect()
    }

    /// Packs after record_submission has reserved every page referenced by the step.
    pub fn pack(
        &mut self,
        config: &ModelConfig,
        scheduler: &RequestScheduler,
        step: &ScheduledStep,
        admissions: &[Option<Admission>; 4],
    ) -> Result<()> {
        self.used = 0;
        if step.lanes.is_empty()
            || step.lanes.len() > step.batch_capacity
            || step.batch_capacity > 4
        {
            return Err("invalid model lane count".into());
        }
        let mut seen = [false; 4];
        for lane in &step.lanes {
            if lane.request.slot >= 4
                || std::mem::replace(&mut seen[lane.request.slot as usize], true)
            {
                return Err("model rows require unique physical slots".into());
            }
            let admission = admissions[lane.request.slot as usize]
                .as_ref()
                .filter(|a| a.handle == lane.request)
                .ok_or("missing native request admission")?;
            if admission.handle != lane.request {
                return Err("stale native request admission".into());
            }
            scheduler.request(lane.request)?;
        }
        let mut slots = [-1; 4];
        let mut generations = [0; 4];
        let mut lengths = [0; 4];
        let mut samples = [0; 4];
        for (row, lane) in step.lanes.iter().enumerate() {
            slots[row] = lane.request.slot as i32;
            generations[row] = lane.request.generation as i64;
            lengths[row] = lane.tokens as i32;
            samples[row] = i32::from(
                step.kind != GraphKind::Prefill
                    || lane.start_position + lane.tokens as u64
                        == scheduler.request(lane.request)?.prompt_tokens as u64,
            );
        }
        if !matches!(step.kind, GraphKind::Init | GraphKind::Restore) {
            self.i32("row_slot_ids", 0, slots);
            self.i64("row_generations", 0, generations);
            self.i32("live_rows", 0, [step.lanes.len() as i32]);
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
                self.i32("sample_enabled", 0, samples);
            }
            for lane in &step.lanes {
                if !matches!(step.kind, GraphKind::Restore | GraphKind::Snapshot)
                    && let Some(mask) = &admissions[lane.request.slot as usize]
                        .as_ref()
                        .unwrap()
                        .grammar_mask
                {
                    self.i32(
                        "sampling_grammar_mask",
                        lane.request.slot as usize * config.contract.vocabulary.div_ceil(32) * 4,
                        mask.iter().copied(),
                    );
                }
                let slot = lane.request.slot as usize;
                let capacity = config.limits.max_sequence_tokens / 64;
                let live = scheduler.request(lane.request)?;
                for (name, skip, previous) in [
                    ("page_tables", 0, self.page_counts[slot]),
                    ("draft_page_tables", 1, self.draft_page_counts[slot]),
                ] {
                    if skip == 1 && !config.has_draft_cache() {
                        continue;
                    }
                    let count = if previous.0 == lane.request.generation {
                        previous.1
                    } else {
                        0
                    };
                    let available = live.pages().count().saturating_sub(skip).min(capacity);
                    if available <= count {
                        continue;
                    }
                    self.i32(
                        name,
                        (slot * capacity + count) * 4,
                        live.pages()
                            .skip(skip + count)
                            .take(available - count)
                            .map(|page| page.0 as i32),
                    );
                    if skip == 0 {
                        self.page_counts[slot] = (lane.request.generation, available);
                    } else {
                        self.draft_page_counts[slot] = (lane.request.generation, available);
                    }
                }
            }
        }
        if step.kind == GraphKind::Speculative {
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
                    if proposal.count + 1 < step.token_capacity {
                        return Err("copy plan is shorter than the selected graph width".into());
                    }
                    ids[slot * 7..slot * 7 + 7].copy_from_slice(&proposal.tokens);
                    counts[slot] = proposal.count as i32;
                    expected[slot] = proposal.consumed as i64;
                    copy_generations[slot] = lane.request.generation as i64;
                    mask_counts[slot] = (proposal.grammar_masks.len()
                        / config.contract.vocabulary.div_ceil(32))
                        as i32;
                    if !proposal.grammar_masks.is_empty() {
                        self.i32(
                            "copy_grammar_mask",
                            slot * 8 * config.contract.vocabulary.div_ceil(32) * 4,
                            proposal.grammar_masks.iter().copied(),
                        );
                    }
                    continue;
                }
                if config.speculative_copy && live.copy_allowed() {
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
            self.i64("copy_token_ids", 0, ids);
            self.i32("copy_token_count", 0, counts);
            self.i64("copy_expected_consumed", 0, expected);
            self.i64("copy_generation", 0, copy_generations);
            if config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
            {
                self.i32("copy_grammar_mask_count", 0, mask_counts);
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
                .positions
                .get(range)
                .ok_or("prefill exceeds positions")?;
            self.i32("query_lengths", 0, lengths);
            self.i64("token_ids", 0, tokens.iter().map(|token| *token as i64));
            self.i64("prefill_positions", 0, positions.iter().flatten().copied());
            self.i32(
                "vision_token_indices",
                0,
                admission.vision_token_indices
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
            self.i64("state_expected_consumed", 0, expected);
            self.i64("state_source_slots", 0, source);
            self.i64("state_destination_slots", 0, destination);
            self.i32("state_operation", 0, operation);
            self.i64("state_qsa_slots", 0, qsa);
            self.i32(
                "state_accepted_tokens",
                0,
                (0..4).map(|row| {
                    i32::from(step.kind != GraphKind::Snapshot && row < step.lanes.len())
                }),
            );
            self.i32("state_num_ops", 0, [step.lanes.len() as i32]);
        }
        if matches!(step.kind, GraphKind::Init | GraphKind::Restore) {
            self.i32("admission_count", 0, [step.lanes.len() as i32]);
            self.i32("admission_slots", 0, slots);
            self.i64("admission_generations", 0, generations);
            for (row, lane) in step.lanes.iter().enumerate() {
                let admission = admissions[lane.request.slot as usize].as_ref().unwrap();
                self.i64(
                    "admission_seed_bits",
                    row * 8,
                    [admission.sampling.seed as i64],
                );
                self.put(
                    "admission_parameters",
                    row * 24,
                    admission
                        .sampling
                        .values()
                        .into_iter()
                        .flat_map(f32::to_le_bytes),
                );
                self.i32("admission_top_k", row * 4, [admission.sampling.top_k]);
                self.i64(
                    "admission_max_new_tokens",
                    row * 8,
                    [admission.max_new_tokens as i64],
                );
                self.i64(
                    "admission_eos_token_ids",
                    row * 64,
                    config
                        .eos_token_ids
                        .iter()
                        .map(|id| *id as i64)
                        .chain(std::iter::repeat(-1))
                        .take(8),
                );
                self.i32(
                    "admission_eos_count",
                    row * 4,
                    [if admission.ignore_eos {
                        0
                    } else {
                        config.eos_token_ids.len() as i32
                    }],
                );
                let mut histogram = vec![0i32; config.contract.vocabulary];
                for token in &admission.prompt {
                    histogram[*token as usize] = histogram[*token as usize]
                        .checked_add(1)
                        .ok_or("prompt histogram overflow")?;
                }
                self.i32(
                    "admission_histogram",
                    row * config.contract.vocabulary * 4,
                    histogram,
                );
                if let Some(mask) = &admission.grammar_mask {
                    self.i32(
                        "admission_grammar_mask",
                        row * config.contract.vocabulary.div_ceil(32) * 4,
                        mask.iter().copied(),
                    );
                } else {
                    self.i32(
                        "admission_grammar_mask",
                        row * config.contract.vocabulary.div_ceil(32) * 4,
                        std::iter::repeat_n(-1, config.contract.vocabulary.div_ceil(32)),
                    );
                }
                self.i32(
                    "admission_grammar_enabled",
                    row * 4,
                    [i32::from(admission.grammar_mask.is_some())],
                );
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
        self.used = 0;
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
        self.put(
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
        self.i64(
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
        self.put(
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
        self.i64(
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
        self.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
        self.i32("vision_num_patches", 0, [geometry.patches as i32]);
        self.i32("vision_slot", 0, [handle.slot as i32]);
        self.i64("vision_generation", 0, [handle.generation as i64]);
        self.i32(
            "vision_feature_offset",
            0,
            [(start / config.contract.vision.patches_per_feature) as i32],
        );
        Ok(())
    }
}
