use super::Glm5Next;
use crate::{
    Result,
    execution::{GraphKind, REQUEST_LANES, Upload},
    model::{
        InputStep,
        inputs::{UploadBatch, append_pages},
    },
};

pub type Admission = crate::model::Admission<super::Glm5Next>;

pub struct Inputs {
    staging: UploadBatch,
    page_counts: [(u64, usize); REQUEST_LANES],
    draft_page_counts: [(u64, usize); REQUEST_LANES],
}

impl Default for Inputs {
    fn default() -> Self {
        Self {
            staging: UploadBatch::default(),
            page_counts: [(0, 0); REQUEST_LANES],
            draft_page_counts: [(0, 0); REQUEST_LANES],
        }
    }
}

impl Inputs {
    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.staging.uploads()
    }

    pub fn pack(&mut self, frame: &InputStep<'_, super::Glm5Next>) -> Result<()> {
        let InputStep {
            config,
            scheduler,
            step,
            admissions,
            slots,
            generations: epochs,
            lengths,
            samples,
        } = *frame;
        let capacity = config.limits.request_capacity();
        self.staging.clear();
        if step.kind == GraphKind::Vision {
            return Err("unsupported GLM graph kind".into());
        }
        let admit = matches!(step.kind, GraphKind::Init | GraphKind::Restore);
        if !admit {
            self.staging
                .i32("row_slot_ids", 0, slots.into_iter().take(capacity));
            self.staging
                .i64("row_generations", 0, epochs.into_iter().take(capacity));
            self.staging.i32("live_rows", 0, [step.lanes.len() as i32]);
        }
        if step.kind != GraphKind::Resume {
            let capacity = config.limits.max_sequence_tokens / Glm5Next::page_tokens(config);
            for lane in &step.lanes {
                let live = scheduler.request(lane.request)?;
                append_pages(
                    &mut self.staging,
                    &mut self.page_counts,
                    "page_tables",
                    live,
                    capacity,
                    0..live.pages().count().min(capacity),
                )?;
                if config.has_draft_cache() {
                    let available = live.pages().count().saturating_sub(1).min(capacity);
                    append_pages(
                        &mut self.staging,
                        &mut self.draft_page_counts,
                        "draft_page_tables",
                        live,
                        capacity,
                        1..1 + available,
                    )?;
                }
            }
        }
        if step.kind.samples() {
            if step.kind != GraphKind::Resume {
                self.staging
                    .i32("sample_enabled", 0, samples.into_iter().take(capacity));
                self.staging
                    .i32("query_lengths", 0, lengths.into_iter().take(capacity));
            }
            for lane in &step.lanes {
                if let Some(mask) = &admissions[lane.request.slot as usize]
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
            }
        }
        if step.kind == GraphKind::Prefill {
            let lane = &step.lanes[0];
            if step.lanes.len() != 1
                || lane.tokens == 0
                || lane.tokens > step.token_capacity
                || step.token_capacity > config.limits.prefill_chunk_size
            {
                return Err(
                    "GLM prefill requires one prompt chunk within its prepared capacity".into(),
                );
            }
            let admission = admissions[lane.request.slot as usize].as_ref().unwrap();
            let start = usize::try_from(lane.start_position)
                .map_err(|_| "prefill offset exceeds host range")?;
            let end = start
                .checked_add(lane.tokens)
                .ok_or("prefill extent overflow")?;
            let tokens = admission
                .prompt
                .get(start..end)
                .ok_or("GLM prefill exceeds prompt")?;
            self.staging
                .i64("token_ids", 0, tokens.iter().map(|t| *t as i64));
        } else if matches!(step.kind, GraphKind::Decode | GraphKind::Speculative) {
            let mut tokens = [0; REQUEST_LANES];
            for (row, lane) in step.lanes.iter().enumerate() {
                let live = scheduler.request(lane.request)?;
                if (step.kind == GraphKind::Decode && lane.tokens != 1)
                    || live.generated_tokens == 0
                {
                    return Err("GLM decode requires one pending sampled token per request".into());
                }
                tokens[row] = *live
                    .tokens()
                    .get(live.consumed_tokens as usize)
                    .ok_or("GLM request has no pending decode token")?
                    as i64;
            }
            self.staging
                .i64("token_ids", 0, tokens.into_iter().take(step.batch_capacity));
        }
        if admit || step.kind == GraphKind::Snapshot {
            let mut source = [-1; REQUEST_LANES];
            let mut destination = [-1; REQUEST_LANES];
            let mut operation = [0; REQUEST_LANES];
            let mut expected = [-1; REQUEST_LANES];
            for (row, lane) in step.lanes.iter().enumerate() {
                let live = scheduler.request(lane.request)?;
                match step.kind {
                    GraphKind::Init => {
                        destination[row] = live.state_slot.0 as i64;
                        operation[row] = 1;
                        expected[row] = 0;
                    }
                    GraphKind::Restore => {
                        source[row] = (capacity + 1) as i64
                            + live
                                .prefix_snapshot()
                                .ok_or("restore requires a GLM checkpoint")?
                                .0 as i64;
                        destination[row] = live.state_slot.0 as i64;
                        expected[row] = live.consumed_tokens as i64;
                    }
                    GraphKind::Snapshot => {
                        source[row] = live.state_slot.0 as i64;
                        destination[row] =
                            (capacity + 1) as i64 + live.snapshot_destination().0 as i64;
                        expected[row] = live.snapshot_consumed() as i64;
                    }
                    _ => unreachable!(),
                }
            }
            self.staging
                .i64("state_source_slots", 0, source.into_iter().take(capacity));
            self.staging.i64(
                "state_destination_slots",
                0,
                destination.into_iter().take(capacity),
            );
            self.staging.i64(
                "state_expected_consumed",
                0,
                expected.into_iter().take(capacity),
            );
            self.staging
                .i32("state_operation", 0, operation.into_iter().take(capacity));
            self.staging
                .i32("state_num_ops", 0, [step.lanes.len() as i32]);
        }
        if admit {
            self.staging
                .i32("admission_count", 0, [step.lanes.len() as i32]);
            self.staging
                .i32("admission_slots", 0, slots.into_iter().take(capacity));
            self.staging.i64(
                "admission_generations",
                0,
                epochs.into_iter().take(capacity),
            );
            for (row, lane) in step.lanes.iter().enumerate() {
                let admission = admissions[lane.request.slot as usize].as_ref().unwrap();
                self.staging.admission(config, row, admission.sampling())?;
            }
        }
        Ok(())
    }
}
