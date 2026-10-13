//! Qwen4Exp graph inputs, request interpretation, and output protocol.

mod config;
pub mod inputs;
#[cfg(feature = "server")]
use crate::output::chatml as output;
#[cfg(feature = "server")]
mod prompt;
#[cfg(test)]
mod tests;

#[cfg(feature = "images")]
use crate::scheduler::RequestHandle;
use crate::{
    Result,
    execution::{CompletionRecord, ExecutionManifest, GraphKind, Upload},
    model::{ContractMetadata, ModelConfig, ModelContract, VisionCapabilities},
    scheduler::SchedulerCapabilities,
};

pub struct Qwen4Exp;
pub const INPUT_SCHEMA: &str = "qwen4exp.graph-inputs.v3";
pub const VOCABULARY: usize = 248320;

#[derive(Clone)]
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
            page_tokens: None,
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
    type Inputs = inputs::Inputs;
    fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        config::validate_config(config, manifest)
    }
    fn required_buffers(config: &ModelConfig) -> Vec<crate::model::catalog::BufferSpec> {
        config::required_buffers(config)
    }
    fn required_inputs(
        config: &ModelConfig,
        kind: GraphKind,
    ) -> impl Iterator<Item = &'static str> {
        config::required_inputs(config, kind)
    }
    fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities {
        SchedulerCapabilities {
            vocabulary: config.contract.vocabulary,
            page_tokens: 64,
            vision: config.contract.vision.clone(),

            draft_pages_per_request: 1,
            state_rows_per_request: config.state_rows_per_request,
        }
    }
    fn pack(inputs: &mut Self::Inputs, step: &crate::model::InputStep<'_, Self>) -> Result<()> {
        inputs.pack(step)
    }
    fn uploads(inputs: &Self::Inputs) -> Vec<Upload<'_>> {
        inputs.uploads()
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
                    != if config.has_learned_drafts() {
                        (record.consumed_tokens - 1).max(0)
                    } else {
                        -1
                    }
            })
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Completion,
                "Qwen4Exp draft cache does not match committed request progress",
            ));
        }
        if (record.status == 1 || record.status == 2)
            && record.token_id >= config.contract.vocabulary as i64
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Completion,
                "completion token exceeds prepared vocabulary",
            ));
        }
        if let Some(vector) = record.vector {
            if vector.model_reserved != -1 {
                return Err(crate::Error::execution(
                    crate::ExecutionError::Completion,
                    "Qwen4Exp v2 completion field 12 must be -1",
                ));
            }
            if vector
                .tokens
                .iter()
                .any(|id| *id >= config.contract.vocabulary as i64)
            {
                return Err(crate::Error::execution(
                    crate::ExecutionError::Completion,
                    "completion vector token exceeds prepared vocabulary",
                ));
            }
        }
        Ok(())
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
    type PromptData = PromptData;
    #[cfg(feature = "server")]
    type OutputConfig = output::OutputConfig;
    #[cfg(feature = "server")]
    type Output = output::ChatMlOutput;
    #[cfg(feature = "server")]
    fn prepare_input(
        frontend: &crate::frontend::Frontend,
        processor: Option<&crate::frontend::images::ImageProcessor>,
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
    ) -> Result<Self::OutputConfig> {
        Ok(output::prepare_output(
            frontend, rendered, messages, options, structured,
        ))
    }
    fn validate_prompt_data(
        config: &ModelConfig,
        tokens: &[u32],
        data: &Self::PromptData,
    ) -> Result<()> {
        if tokens.len() != data.positions.len()
            || tokens.len() != data.vision_token_indices.len()
            || data.vision_token_indices.iter().any(|index| {
                *index < -1 || *index >= config.contract.vision.features_per_request as i32
            })
            || data
                .positions
                .iter()
                .flatten()
                .any(|position| *position < 0)
        {
            return Err("invalid prompt position or vision feature metadata".into());
        }
        Ok(())
    }
    #[cfg(feature = "server")]
    fn text_only(data: &Self::PromptData) -> bool {
        data.vision_token_indices.iter().all(|i| *i == -1)
    }
}
