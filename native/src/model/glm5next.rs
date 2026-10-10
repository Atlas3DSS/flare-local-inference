//! GLM5Next request semantics and argument-tag tool serialization.

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

pub struct Glm5Next;
pub const INPUT_SCHEMA: &str = "glm5next.graph-inputs.v1";
pub const VOCABULARY: usize = 154880;
pub const PAGE_TOKENS: usize = 256;

impl Glm5Next {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            vision: VisionCapabilities::disabled(),
        }
    }
}

impl ModelContract for Glm5Next {
    type Admission = inputs::Admission;
    type Inputs = inputs::Inputs;

    fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        Self::validate(config, manifest)
    }

    fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities {
        SchedulerCapabilities {
            vocabulary: config.contract.vocabulary,
            page_tokens: PAGE_TOKENS,
            vision: config.contract.vision.clone(),
            live_state_base: 1,
            draft_pages_per_request: 0,
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
        _: GraphKind,
        record: &CompletionRecord,
    ) -> Result<()> {
        if record.vector.is_some()
            || ((record.status == 1 || record.status == 2)
                && record.token_id >= config.contract.vocabulary as i64)
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Completion,
                "completion exceeds the GLM scalar-token contract",
            ));
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

    #[cfg(feature = "server")]
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

    #[cfg(feature = "server")]
    fn validate_prompt(config: &ModelConfig, prompt: &crate::server::Prompt<Self>) -> Result<()> {
        prompt.sampling.validate()?;
        if prompt.tokens.is_empty()
            || prompt
                .system_boundary
                .is_some_and(|n| n == 0 || n >= prompt.tokens.len())
            || prompt
                .tokens
                .iter()
                .any(|t| *t as usize >= config.contract.vocabulary)
            || prompt.sampling.top_k > config.contract.vocabulary as i32
            || prompt.max_new_tokens == 0
            || prompt.max_new_tokens > i32::MAX as u32
            || prompt
                .tokens
                .len()
                .checked_add(prompt.max_new_tokens as usize)
                .is_none_or(|n| n > config.limits.max_sequence_tokens)
            || prompt.images.is_some()
            || !prompt.conditioning.is_empty()
        {
            return Err("prompt and output limit exceed the prepared GLM text contract".into());
        }
        Ok(())
    }

    #[cfg(feature = "server")]
    fn text_only(_: &()) -> bool {
        true
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
            grammar_mask: prompt.grammar.as_ref().map(|g| g.mask().to_vec()),
            ignore_eos,
        };
        admission.validate(config)?;
        Ok(admission)
    }
}
