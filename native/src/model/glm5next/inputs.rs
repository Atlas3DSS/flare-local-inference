use super::PAGE_TOKENS;
use crate::{
    Result,
    execution::{GraphKind, Upload},
    model::{ModelConfig, inputs::UploadBatch},
    scheduler::{RequestScheduler, ScheduledStep},
};

pub type Admission = crate::model::Admission<super::Glm5Next>;

#[derive(Default)]
pub struct Inputs {
    staging: UploadBatch,
    page_counts: [(u64, usize); 4],
}

impl Inputs {
    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.staging.uploads()
    }

    /// Pack after the scheduler reserves every page needed by this submission.
    pub fn pack(
        &mut self,
        config: &ModelConfig,
        scheduler: &RequestScheduler,
        step: &ScheduledStep,
        admissions: &[Option<Admission>; 4],
    ) -> Result<()> {
        self.staging.clear();
        if matches!(step.kind, GraphKind::Vision | GraphKind::Speculative)
            || step.lanes.is_empty()
            || step.lanes.len() > step.batch_capacity
            || step.batch_capacity > 4
        {
            return Err("unsupported GLM graph kind or lane count".into());
        }
        let mut seen = [false; 4];
        let mut slots = [-1; 4];
        let mut epochs = [0; 4];
        let mut lengths = [0; 4];
        let mut samples = [0; 4];
        for (row, lane) in step.lanes.iter().enumerate() {
            let slot = lane.request.slot as usize;
            if slot >= 4 || std::mem::replace(&mut seen[slot], true) {
                return Err("GLM rows require unique physical slots".into());
            }
            admissions[slot]
                .as_ref()
                .filter(|a| a.handle == lane.request)
                .ok_or("missing or stale native GLM admission")?;
            let live = scheduler.request(lane.request)?;
            slots[row] = slot as i32;
            epochs[row] = lane.request.generation as i64;
            lengths[row] = lane.tokens as i32;
            samples[row] = i32::from(
                step.kind != GraphKind::Prefill
                    || lane.start_position + lane.tokens as u64 == live.prompt_tokens as u64,
            );
        }
        let admit = matches!(step.kind, GraphKind::Init | GraphKind::Restore);
        if !admit {
            self.staging.i32("row_slot_ids", 0, slots);
            self.staging.i64("row_generations", 0, epochs);
            self.staging.i32("live_rows", 0, [step.lanes.len() as i32]);
        }
        if step.kind != GraphKind::Resume {
            let capacity = config.limits.max_sequence_tokens / PAGE_TOKENS;
            for lane in &step.lanes {
                let slot = lane.request.slot as usize;
                let live = scheduler.request(lane.request)?;
                let previous = self.page_counts[slot];
                let count = if previous.0 == lane.request.generation {
                    previous.1
                } else {
                    0
                };
                let available = live.pages().count();
                if available > capacity || available < count {
                    return Err("GLM page table does not match its request capacity".into());
                }
                if available > count {
                    self.staging.i32(
                        "page_tables",
                        (slot * capacity + count) * 4,
                        live.pages().skip(count).map(|page| page.0 as i32),
                    );
                    self.page_counts[slot] = (lane.request.generation, available);
                }
            }
        }
        if step.kind.samples() {
            if step.kind != GraphKind::Resume {
                self.staging.i32("sample_enabled", 0, samples);
                self.staging.i32("query_lengths", 0, lengths);
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
        } else if step.kind == GraphKind::Decode {
            let mut tokens = [0; 4];
            for (row, lane) in step.lanes.iter().enumerate() {
                let live = scheduler.request(lane.request)?;
                if lane.tokens != 1 || live.generated_tokens == 0 {
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
            let mut source = [-1; 4];
            let mut destination = [-1; 4];
            let mut operation = [0; 4];
            let mut expected = [-1; 4];
            for (row, lane) in step.lanes.iter().enumerate() {
                let live = scheduler.request(lane.request)?;
                match step.kind {
                    GraphKind::Init => {
                        destination[row] = live.state_slot.0 as i64;
                        operation[row] = 1;
                        expected[row] = 0;
                    }
                    GraphKind::Restore => {
                        source[row] = 5 + live
                            .prefix_snapshot()
                            .ok_or("restore requires a GLM checkpoint")?
                            .0 as i64;
                        destination[row] = live.state_slot.0 as i64;
                        expected[row] = live.consumed_tokens as i64;
                    }
                    GraphKind::Snapshot => {
                        source[row] = live.state_slot.0 as i64;
                        destination[row] = 5 + live.snapshot_destination().0 as i64;
                        expected[row] = live.snapshot_consumed() as i64;
                    }
                    _ => unreachable!(),
                }
            }
            self.staging.i64("state_source_slots", 0, source);
            self.staging.i64("state_destination_slots", 0, destination);
            self.staging.i64("state_expected_consumed", 0, expected);
            self.staging.i32("state_operation", 0, operation);
            self.staging
                .i32("state_num_ops", 0, [step.lanes.len() as i32]);
        }
        if admit {
            self.staging
                .i32("admission_count", 0, [step.lanes.len() as i32]);
            self.staging.i32("admission_slots", 0, slots);
            self.staging.i64("admission_generations", 0, epochs);
            for (row, lane) in step.lanes.iter().enumerate() {
                let admission = admissions[lane.request.slot as usize].as_ref().unwrap();
                self.staging.admission(config, row, admission.sampling())?;
            }
        }
        Ok(())
    }
}
