//! Dense Qwen text requests with paged KV and opaque recurrent continuation rows.

use crate::{
    Result,
    execution::{ABI_VERSION, CompletionRecord, ExecutionManifest, GraphKind, Upload},
    model::{ContractMetadata, ModelConfig, ModelContract, VisionCapabilities, paged_text},
    scheduler::SchedulerCapabilities,
};

pub struct Qwen35;
pub const INPUT_SCHEMA: &str = "qwen38-dense.graph-inputs.v1";
pub const VOCABULARY: usize = 248320;
pub const PAGE_TOKENS: usize = 128;
pub type Inputs = paged_text::Inputs;

impl Qwen35 {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            vision: VisionCapabilities::disabled(),
            page_tokens: Some(PAGE_TOKENS),
        }
    }
}

impl ModelContract for Qwen35 {
    type Inputs = Inputs;
    type PromptData = ();

    fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        if config.contract != Self::metadata()
            || manifest.abi_version != ABI_VERSION
            || config.prepared_sequence_tokens > 262144
            || manifest.engram.is_some()
            || !config.speculative_widths.is_empty()
            || config.history_drafts
        {
            return Err(
                "unsupported dense Qwen text schema, cache layout, or completion ABI".into(),
            );
        }
        Ok(())
    }

    fn required_buffers(config: &ModelConfig) -> Vec<crate::model::catalog::BufferSpec> {
        paged_text::required_buffers::<Self>(config, PAGE_TOKENS)
    }

    fn required_inputs(_: &ModelConfig, kind: GraphKind) -> impl Iterator<Item = &'static str> {
        paged_text::required_inputs(kind).iter().copied()
    }

    fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities {
        SchedulerCapabilities {
            vocabulary: VOCABULARY,
            page_tokens: PAGE_TOKENS,
            vision: config.contract.vision.clone(),
            draft_pages_per_request: 0,
            state_rows_per_request: config.state_rows_per_request,
        }
    }

    fn pack(inputs: &mut Inputs, step: &crate::model::InputStep<'_, Self>) -> Result<()> {
        inputs.pack(step)
    }

    fn uploads(inputs: &Inputs) -> Vec<Upload<'_>> {
        inputs.uploads()
    }

    fn validate_completion(_: &ModelConfig, _: GraphKind, record: &CompletionRecord) -> Result<()> {
        if ((record.status == 1 || record.status == 2) && record.token_id >= VOCABULARY as i64)
            || record.vector.is_some()
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Completion,
                "invalid dense Qwen completion",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "images")]
    fn pack_vision(
        _: &mut Inputs,
        _: &ModelConfig,
        _: crate::scheduler::RequestHandle,
        _: &crate::frontend::images::ProcessedImages,
        _: usize,
    ) -> Result<()> {
        Err("dense Qwen text packages have no vision graphs".into())
    }

    fn validate_prompt_data(_: &ModelConfig, _: &[u32], _: &()) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "server")]
    fn text_only(_: &()) -> bool {
        true
    }

    #[cfg(feature = "server")]
    type OutputConfig = crate::output::chatml::OutputConfig;
    #[cfg(feature = "server")]
    type Output = crate::output::chatml::ChatMlOutput;

    #[cfg(feature = "server")]
    fn prepare_input(
        _: &crate::frontend::Frontend,
        _: Option<&crate::frontend::images::ImageProcessor>,
        rendered: &crate::frontend::RenderedPrompt,
        images: Vec<crate::frontend::images::RgbImage>,
    ) -> Result<crate::model::PreparedInput<()>> {
        if !images.is_empty() {
            return Err("dense Qwen text packages do not accept images".into());
        }
        Ok(crate::model::PreparedInput {
            tokens: rendered.token_ids.clone(),
            data: (),
            conditioning: vec![],
            images: None,
        })
    }

    #[cfg(feature = "server")]
    fn system_boundary(
        frontend: &crate::frontend::Frontend,
        tokens: &[u32],
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<usize> {
        crate::model::qwen4exp::Qwen4Exp::system_boundary(frontend, tokens, messages, options)
    }

    #[cfg(feature = "server")]
    fn prepare_output(
        frontend: &crate::frontend::Frontend,
        rendered: &crate::frontend::RenderedPrompt,
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
        structured: bool,
    ) -> Result<Self::OutputConfig> {
        Ok(crate::output::chatml::prepare_output(
            frontend, rendered, messages, options, structured,
        ))
    }
}
