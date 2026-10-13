//! GLM5Next request semantics and argument-tag tool serialization.

mod config;
pub mod inputs;
#[cfg(feature = "server")]
pub mod output;
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

pub struct Glm5Next;
pub const INPUT_SCHEMA: &str = "glm5next.graph-inputs.v2";
pub const VOCABULARY: usize = 154880;
pub const PAGE_TOKENS: usize = 256;

impl Glm5Next {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            vision: VisionCapabilities::disabled(),
            page_tokens: None,
        }
    }

    /// Global tokens per page: 256 local records on every rank, times the decode context parallel size.
    pub fn page_tokens(config: &ModelConfig) -> usize {
        config.contract.page_tokens.unwrap_or(PAGE_TOKENS)
    }
}

impl ModelContract for Glm5Next {
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
        config::required_inputs(kind).iter().copied().chain(
            (config.has_learned_drafts() && kind != GraphKind::Resume && kind != GraphKind::Vision)
                .then_some("draft_page_tables"),
        )
    }
    fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities {
        SchedulerCapabilities {
            vocabulary: config.contract.vocabulary,
            page_tokens: Self::page_tokens(config),
            vision: config.contract.vision.clone(),

            draft_pages_per_request: usize::from(config.has_learned_drafts()),
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
        if (record.status == 1 || record.status == 2)
            && record.token_id >= config.contract.vocabulary as i64
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Completion,
                "completion token exceeds the GLM vocabulary",
            ));
        }
        if let Some(vector) = record.vector {
            if vector.model_reserved != -1
                || vector
                    .tokens
                    .iter()
                    .any(|id| *id >= config.contract.vocabulary as i64)
                || (record.status != 3
                    && kind == GraphKind::Speculative
                    && vector.draft_cache_consumed != (record.consumed_tokens - 1).max(0))
            {
                return Err(crate::Error::execution(
                    crate::ExecutionError::Completion,
                    "GLM speculative completion disagrees with committed draft state",
                ));
            }
        }
        Ok(())
    }

    #[cfg(feature = "images")]
    fn pack_vision(
        _: &mut Self::Inputs,
        _: &ModelConfig,
        _: RequestHandle,
        _: &crate::frontend::images::ProcessedImages,
        _: usize,
    ) -> Result<()> {
        Err("prepared GLM catalog has no vision graphs".into())
    }

    type PromptData = ();
    #[cfg(feature = "server")]
    type OutputConfig = output::OutputConfig;
    #[cfg(feature = "server")]
    type Output = output::GlmOutput;

    #[cfg(feature = "server")]
    fn prepare_input(
        frontend: &crate::frontend::Frontend,
        _: Option<&crate::frontend::images::ImageProcessor>,
        rendered: &crate::frontend::RenderedPrompt,
        images: Vec<crate::frontend::images::RgbImage>,
    ) -> Result<crate::model::PreparedInput<()>> {
        prompt::prepare_input(frontend, rendered, images)
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
        let output = output::prepare_output(frontend, rendered, messages, options, structured);
        if structured && output.reasoning {
            return Err(crate::Error::admission(
                crate::AdmissionError::Feature,
                "GLM structured output is not implemented for reasoning prompts",
            ));
        }
        Ok(output)
    }

    fn validate_prompt_data(_: &ModelConfig, _: &[u32], _: &()) -> Result<()> {
        Ok(())
    }
    #[cfg(feature = "server")]
    fn text_only(_: &()) -> bool {
        true
    }
}
