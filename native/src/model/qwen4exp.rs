//! Qwen4Exp graph inputs, request interpretation, and output protocol.

mod config;
pub mod inputs;
#[cfg(feature = "server")]
pub mod output;
#[cfg(feature = "server")]
mod prompt;
#[cfg(test)]
mod tests;

use crate::{
    Result,
    execution::{
        CompletionRecord, ExecutionManifest, GraphKind, REQUEST_LANES, SamplingMode, Upload,
    },
    model::{ContractMetadata, ModelConfig, ModelContract, VisionCapabilities},
    scheduler::{RequestHandle, RequestScheduler, ScheduledStep, SchedulerCapabilities},
};

pub struct Qwen4Exp;
pub const INPUT_SCHEMA: &str = "qwen4exp.graph-inputs.v1";
pub const VOCABULARY: usize = 248320;

#[cfg(feature = "server")]
pub struct PromptData {
    pub positions: Vec<[i64; 3]>,
    pub vision_token_indices: Vec<i32>,
}

impl Qwen4Exp {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            vision: VisionCapabilities {
                features_per_request: 16384,
                max_images: 4,
                patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16384],
            },
        }
    }
    fn validate_metadata(metadata: &ContractMetadata) -> Result<()> {
        if metadata != &Self::metadata() {
            return Err(
                "unsupported Qwen4Exp graph-input schema, vocabulary, or vision capabilities"
                    .into(),
            );
        }
        Ok(())
    }
}

impl ModelContract for Qwen4Exp {
    type Admission = inputs::Admission;
    type Inputs = inputs::Inputs;
    fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        Self::validate(config, manifest)
    }
    fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities {
        SchedulerCapabilities {
            vocabulary: config.contract.vocabulary,
            vision: config.contract.vision.clone(),
            live_state_base: 1,
            candidate_state_base: 5,
            draft_pages_per_request: 1,
        }
    }
    fn graph(
        config: &ModelConfig,
        manifest: &ExecutionManifest,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        mode: SamplingMode,
    ) -> Result<usize> {
        Self::graph(config, manifest, kind, batch, tokens, mode)
    }
    fn pack(
        inputs: &mut Self::Inputs,
        config: &ModelConfig,
        scheduler: &RequestScheduler,
        step: &ScheduledStep,
        admissions: &[Option<Self::Admission>; REQUEST_LANES],
    ) -> Result<()> {
        inputs.pack(config, scheduler, step, admissions)
    }
    fn uploads(inputs: &Self::Inputs) -> Vec<Upload<'_>> {
        inputs.uploads()
    }
    fn admission_handle(admission: &Self::Admission) -> RequestHandle {
        admission.handle
    }
    fn grammar_updated(admission: &mut Self::Admission, mask: &[i32]) {
        admission
            .grammar_mask
            .as_mut()
            .expect("grammar admission has a mask")
            .copy_from_slice(mask);
    }
    fn validate_completion(
        config: &ModelConfig,
        kind: GraphKind,
        record: &CompletionRecord,
    ) -> Result<()> {
        if record.status != 3
            && kind == GraphKind::Speculative
            && record.vector.is_some_and(|vector| {
                vector.draft_cache_consumed
                    != if config.has_draft_cache() {
                        (record.consumed_tokens - 1).max(0)
                    } else {
                        -1
                    }
            })
        {
            return Err("Qwen4Exp draft cache does not match committed request progress".into());
        }
        if (record.status == 1 || record.status == 2)
            && record.token_id >= config.contract.vocabulary as i64
        {
            return Err("completion token exceeds prepared vocabulary".into());
        }
        if let Some(vector) = record.vector {
            if vector
                .tokens
                .iter()
                .any(|id| *id >= config.contract.vocabulary as i64)
            {
                return Err("completion vector token exceeds prepared vocabulary".into());
            }
        }
        Ok(())
    }
    fn candidate_boundary(record: &CompletionRecord) -> u64 {
        match record.vector {
            Some(vector) => vector.candidate_consumed.max(0) as u64,
            None => {
                (record.consumed_tokens as u64 / crate::cache::PAGE_SIZE as u64)
                    * crate::cache::PAGE_SIZE as u64
            }
        }
    }
    #[cfg(feature = "images")]
    fn pack_vision(
        inputs: &mut Self::Inputs,
        config: &ModelConfig,
        handle: RequestHandle,
        images: &crate::frontend::images::ProcessedImages,
        image: usize,
    ) -> Result<()> {
        inputs.pack_vision(config, handle, images, image)
    }
    #[cfg(feature = "server")]
    type PromptData = PromptData;
    #[cfg(feature = "server")]
    type OutputConfig = output::OutputConfig;
    #[cfg(feature = "server")]
    type Output = output::QwenOutput;
    #[cfg(feature = "server")]
    fn prepare_input(
        frontend: &crate::frontend::Frontend,
        processor: &crate::frontend::images::ImageProcessor,
        rendered: &crate::frontend::RenderedPrompt,
        images: Vec<crate::frontend::images::RgbImage>,
    ) -> Result<crate::model::PreparedInput<Self::PromptData>> {
        prompt::prepare_input(frontend, processor, rendered, images)
    }
    #[cfg(feature = "server")]
    fn system_boundary(
        frontend: &crate::frontend::Frontend,
        prompt_tokens: &[u32],
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<usize> {
        prompt::system_boundary(frontend, prompt_tokens, messages, options)
    }
    #[cfg(feature = "server")]
    fn prepare_output(
        frontend: &crate::frontend::Frontend,
        rendered: &crate::frontend::RenderedPrompt,
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
        structured: bool,
    ) -> Self::OutputConfig {
        output::prepare_output(frontend, rendered, messages, options, structured)
    }
    #[cfg(feature = "server")]
    fn validate_prompt(config: &ModelConfig, prompt: &crate::server::Prompt<Self>) -> Result<()> {
        prompt.sampling.validate()?;
        if prompt.tokens.is_empty()
            || prompt
                .system_boundary
                .is_some_and(|boundary| boundary == 0 || boundary >= prompt.tokens.len())
            || prompt.tokens.len() != prompt.data.positions.len()
            || prompt.tokens.len() != prompt.data.vision_token_indices.len()
            || prompt
                .data
                .vision_token_indices
                .iter()
                .any(|i| *i < -1 || *i >= config.contract.vision.features_per_request as i32)
            || prompt
                .tokens
                .iter()
                .any(|t| *t as usize >= config.contract.vocabulary)
            || prompt.data.positions.iter().flatten().any(|p| *p < 0)
            || prompt.sampling.top_k > config.contract.vocabulary as i32
            || prompt.max_new_tokens == 0
            || prompt.max_new_tokens > i32::MAX as u32
            || prompt
                .tokens
                .len()
                .checked_add(prompt.max_new_tokens as usize)
                .is_none_or(|n| n > config.limits.max_sequence_tokens)
        {
            return Err("prompt and output limit exceed the prepared model contract".into());
        }
        Ok(())
    }
    #[cfg(feature = "server")]
    fn text_only(data: &Self::PromptData) -> bool {
        data.vision_token_indices.iter().all(|i| *i == -1)
    }
    #[cfg(feature = "server")]
    fn admission(
        config: &ModelConfig,
        handle: RequestHandle,
        prompt: &crate::server::Prompt<Self>,
        ignore_eos: bool,
    ) -> Result<Self::Admission> {
        let admission = inputs::Admission {
            handle,
            sampling: prompt.sampling.clone(),
            max_new_tokens: prompt.max_new_tokens,
            prompt: prompt.tokens.clone(),
            positions: prompt.data.positions.clone(),
            vision_token_indices: prompt.data.vision_token_indices.clone(),
            grammar_mask: prompt.grammar.as_ref().map(|g| g.mask().to_vec()),
            ignore_eos,
        };
        admission.validate(config)?;
        Ok(admission)
    }
}
