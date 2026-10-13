//! Shared native uploads for paged text models with continuation rows.

use crate::{
    Result,
    execution::{GraphKind, REQUEST_LANES, Upload},
    model::{
        InputStep, ModelContract,
        inputs::{UploadBatch, append_pages},
    },
};

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
    pub fn clear(&mut self) {
        self.staging.clear();
    }

    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.staging.uploads()
    }

    pub fn pack<M: ModelContract>(&mut self, frame: &InputStep<'_, M>) -> Result<()> {
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
        for lane in &step.lanes {
            let request = scheduler.request(lane.request)?;
            self.staging.i64(
                "request_state_rows",
                lane.request.slot as usize * 8 * 8,
                (0..8).map(|column| {
                    if column < scheduler.execution_width() {
                        request.state_slot.0 as i64 + column as i64
                    } else {
                        0
                    }
                }),
            );
        }
        if step.kind == GraphKind::Vision {
            return Err("unsupported text graph kind".into());
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
            let capacity =
                config.limits.max_sequence_tokens / M::scheduler_capabilities(config).page_tokens;
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
                if M::scheduler_capabilities(config).draft_pages_per_request > 0 {
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
                    "text prefill requires one prompt chunk within its prepared capacity".into(),
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
                .ok_or("text prefill exceeds prompt")?;
            self.staging
                .i64("token_ids", 0, tokens.iter().map(|t| *t as i64));
        } else if matches!(step.kind, GraphKind::Decode | GraphKind::Speculative) {
            let mut tokens = [0; REQUEST_LANES];
            for (row, lane) in step.lanes.iter().enumerate() {
                let live = scheduler.request(lane.request)?;
                if (step.kind == GraphKind::Decode && lane.tokens != 1)
                    || live.generated_tokens == 0
                {
                    return Err("text decode requires one pending sampled token per request".into());
                }
                tokens[row] = *live
                    .tokens()
                    .get(live.consumed_tokens as usize)
                    .ok_or("text request has no pending decode token")?
                    as i64;
            }
            self.staging
                .i64("token_ids", 0, tokens.into_iter().take(step.batch_capacity));
        }
        if admit || step.kind == GraphKind::Snapshot {
            self.staging.i64(
                "state_request_slots",
                0,
                step.lanes
                    .iter()
                    .map(|lane| lane.request.slot as i64)
                    .chain(std::iter::repeat(-1))
                    .take(capacity),
            );
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
                        source[row] = scheduler
                            .snapshot_slot(
                                live.prefix_snapshot()
                                    .ok_or("restore lacks a prefix snapshot")?,
                            )
                            .0 as i64;
                        destination[row] = live.state_slot.0 as i64;
                        expected[row] = live.consumed_tokens as i64;
                    }
                    GraphKind::Snapshot => {
                        source[row] = live.state_slot.0 as i64;
                        destination[row] =
                            scheduler.snapshot_slot(live.snapshot_destination()).0 as i64;
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

pub(crate) fn required_buffers<M: ModelContract>(
    config: &crate::model::ModelConfig,
    page_tokens: usize,
) -> Vec<crate::model::catalog::BufferSpec> {
    let slots = config.limits.request_capacity();
    use crate::execution::Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
    let vocabulary = config.contract.vocabulary;
    let mut buffers = vec![
        ("request_state_rows", I64, vec![slots, 8]),
        ("memory_capacity", I64, vec![2]),
        ("state_request_slots", I64, vec![slots]),
        ("row_slot_ids", I32, vec![slots]),
        ("row_generations", I64, vec![slots]),
        ("live_rows", I32, vec![1]),
        ("query_lengths", I32, vec![slots]),
        ("token_ids", I64, vec![config.limits.prefill_chunk_size]),
        (
            "page_tables",
            I32,
            vec![slots, config.limits.max_sequence_tokens / page_tokens],
        ),
        ("sample_enabled", I32, vec![slots]),
        (
            "sampling_grammar_mask",
            I32,
            vec![slots, vocabulary.div_ceil(32)],
        ),
        ("admission_count", I32, vec![1]),
        ("admission_slots", I32, vec![slots]),
        ("admission_generations", I64, vec![slots]),
        ("admission_seed_bits", I64, vec![slots]),
        ("admission_parameters", F32, vec![slots, 6]),
        ("admission_top_k", I32, vec![slots]),
        ("admission_max_new_tokens", I64, vec![slots]),
        ("admission_eos_token_ids", I64, vec![slots, 8]),
        ("admission_eos_count", I32, vec![slots]),
        ("admission_histogram", I32, vec![slots, vocabulary]),
        (
            "admission_grammar_mask",
            I32,
            vec![slots, vocabulary.div_ceil(32)],
        ),
        ("admission_grammar_enabled", I32, vec![slots]),
        ("state_expected_consumed", I64, vec![slots]),
        ("state_source_slots", I64, vec![slots]),
        ("state_destination_slots", I64, vec![slots]),
        ("state_operation", I32, vec![slots]),
        ("state_num_ops", I32, vec![1]),
    ];
    if M::scheduler_capabilities(config).draft_pages_per_request > 0 {
        buffers.push((
            "draft_page_tables",
            I32,
            vec![slots, config.limits.max_sequence_tokens / page_tokens],
        ));
    }
    buffers
}

pub(crate) fn required_inputs(kind: GraphKind) -> &'static [&'static str] {
    match kind {
        GraphKind::Init | GraphKind::Restore => &[
            "request_state_rows",
            "admission_count",
            "admission_slots",
            "admission_generations",
            "admission_seed_bits",
            "admission_parameters",
            "admission_top_k",
            "admission_max_new_tokens",
            "admission_eos_token_ids",
            "admission_eos_count",
            "admission_histogram",
            "admission_grammar_mask",
            "admission_grammar_enabled",
            "page_tables",
            "state_request_slots",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_num_ops",
        ],
        GraphKind::Snapshot => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "page_tables",
            "state_request_slots",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_num_ops",
        ],
        GraphKind::Prefill | GraphKind::Decode | GraphKind::Speculative => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "query_lengths",
            "token_ids",
            "page_tables",
            "sample_enabled",
            "sampling_grammar_mask",
        ],
        GraphKind::Resume => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "sampling_grammar_mask",
        ],
        GraphKind::Vision => &[],
    }
}
