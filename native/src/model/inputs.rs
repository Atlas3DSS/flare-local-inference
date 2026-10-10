//! Reusable host staging for the shared request-admission ABI.

use crate::{
    Result,
    execution::{OwnedUpload, Upload},
    model::{ModelConfig, SamplingParameters},
};

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
