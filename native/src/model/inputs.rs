//! Shared request-row validation and host staging for captured graph inputs.

use crate::{
    Result,
    execution::{GraphKind, OwnedUpload, Upload},
    model::{Admission, ModelConfig, ModelContract, SamplingParameters},
    scheduler::{LiveRequest, RequestScheduler, ScheduledStep},
};

pub struct InputStep<'a, M: ModelContract> {
    pub config: &'a ModelConfig,
    pub scheduler: &'a RequestScheduler,
    pub step: &'a ScheduledStep,
    pub admissions: &'a [Option<Admission<M>>; 4],
    pub slots: [i32; 4],
    pub generations: [i64; 4],
    pub lengths: [i32; 4],
    pub samples: [i32; 4],
}

/// Pack after the scheduler reserves every page referenced by the submission.
pub fn pack_inputs<M: ModelContract>(
    inputs: &mut M::Inputs,
    config: &ModelConfig,
    scheduler: &RequestScheduler,
    step: &ScheduledStep,
    admissions: &[Option<Admission<M>>; 4],
) -> Result<()> {
    if step.lanes.is_empty() || step.lanes.len() > step.batch_capacity || step.batch_capacity > 4 {
        return Err("invalid model lane count".into());
    }
    let mut frame = InputStep {
        config,
        scheduler,
        step,
        admissions,
        slots: [-1; 4],
        generations: [0; 4],
        lengths: [0; 4],
        samples: [0; 4],
    };
    let mut seen = [false; 4];
    for (row, lane) in step.lanes.iter().enumerate() {
        let slot = lane.request.slot as usize;
        if slot >= 4 || std::mem::replace(&mut seen[slot], true) {
            return Err("model rows require unique physical slots".into());
        }
        admissions[slot]
            .as_ref()
            .filter(|admission| admission.handle == lane.request)
            .ok_or("missing or stale native request admission")?;
        let live = scheduler.request(lane.request)?;
        frame.slots[row] = slot as i32;
        frame.generations[row] = lane.request.generation as i64;
        frame.lengths[row] =
            i32::try_from(lane.tokens).map_err(|_| "model row exceeds token capacity")?;
        frame.samples[row] = i32::from(
            step.kind != GraphKind::Prefill
                || lane.start_position.checked_add(lane.tokens as u64)
                    == Some(live.prompt_tokens as u64),
        );
    }
    M::pack(inputs, &frame)
}

pub fn append_pages(
    staging: &mut UploadBatch,
    counts: &mut [(u64, usize); 4],
    name: &str,
    request: &LiveRequest,
    capacity: usize,
    pages: std::ops::Range<usize>,
) -> Result<()> {
    let slot = request.handle.slot as usize;
    let previous = counts[slot];
    let count = if previous.0 == request.handle.generation {
        previous.1
    } else {
        0
    };
    let available = pages
        .end
        .checked_sub(pages.start)
        .ok_or("invalid page-table window")?;
    if available > capacity || available < count {
        return Err("page table does not match its request capacity".into());
    }
    if available > count {
        if pages.end > request.pages().count() {
            return Err("page-table window exceeds reserved pages".into());
        }
        staging.i32(
            name,
            (slot * capacity + count) * 4,
            request
                .pages()
                .skip(pages.start + count)
                .take(available - count)
                .map(|page| page.0 as i32),
        );
        counts[slot] = (request.handle.generation, available);
    }
    Ok(())
}

pub struct SamplingAdmission<'a> {
    pub sampling: &'a SamplingParameters,
    pub max_new_tokens: u32,
    pub prompt: &'a [u32],
    pub grammar_mask: Option<&'a [i32]>,
    pub ignore_eos: bool,
}

/// Graph execution copies active staging bytes before submit returns.
#[derive(Default)]
pub struct UploadBatch {
    uploads: Vec<OwnedUpload>,
    used: usize,
}

impl UploadBatch {
    pub fn clear(&mut self) {
        self.used = 0;
    }
    pub fn put(&mut self, name: &str, offset: usize, bytes: impl Iterator<Item = u8>) {
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
    pub fn i32(&mut self, name: &str, offset: usize, values: impl IntoIterator<Item = i32>) {
        self.put(name, offset, values.into_iter().flat_map(i32::to_le_bytes));
    }
    pub fn i64(&mut self, name: &str, offset: usize, values: impl IntoIterator<Item = i64>) {
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

    pub fn admission(
        &mut self,
        config: &ModelConfig,
        row: usize,
        admission: SamplingAdmission<'_>,
    ) -> Result<()> {
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
        for token in admission.prompt {
            histogram[*token as usize] = histogram[*token as usize]
                .checked_add(1)
                .ok_or("prompt histogram overflow")?;
        }
        self.i32(
            "admission_histogram",
            row * config.contract.vocabulary * 4,
            histogram,
        );
        if let Some(mask) = admission.grammar_mask {
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
        Ok(())
    }
}
