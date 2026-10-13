//! MiMo paged text, DFlash completions and causal image-feature injection.

mod inputs;
#[cfg(feature = "server")]
mod prompt;
#[cfg(test)]
mod tests;

use crate::{
    Result,
    execution::{
        ABI_VERSION, CompletionRecord, ExecutionManifest, GraphKind, SPECULATIVE_ABI_VERSION,
        Upload,
    },
    model::{ContractMetadata, ModelConfig, ModelContract, VisionCapabilities, paged_text},
    scheduler::SchedulerCapabilities,
};

pub struct MiMo;
pub const INPUT_SCHEMA: &str = "mimo_v2.graph-inputs.v1";
pub const VOCABULARY: usize = 152576;
pub const PAGE_TOKENS: usize = 128;
pub const IMAGE_TOKEN: u32 = 151655;

#[derive(Clone, Default)]
pub struct PromptData {
    pub vision_token_indices: Vec<i32>,
}

impl MiMo {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            page_tokens: Some(PAGE_TOKENS),
            vision: VisionCapabilities {
                features_per_request: 16384,
                max_images: 4,
                patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16384],
            },
        }
    }
}

impl ModelContract for MiMo {
    type Inputs = inputs::Inputs;
    type PromptData = PromptData;

    fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        if config.contract != Self::metadata()
            || !matches!(manifest.abi_version, ABI_VERSION | SPECULATIVE_ABI_VERSION)
            || config.state_rows_per_request != Some(1)
            || config.limits.max_sequence_tokens > 1048576
            || manifest.engram.is_some()
        {
            return Err(
                "unsupported MiMo schema, vocabulary, cache layout or completion ABI".into(),
            );
        }
        if config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
            || config.history_drafts
            || config.speculative_widths.iter().any(|width| *width > 8)
        {
            return Err("MiMo speculation requires DFlash with at most seven drafts".into());
        }
        Ok(())
    }

    fn required_buffers(config: &ModelConfig) -> Vec<crate::model::catalog::BufferSpec> {
        use crate::execution::Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
        let mut buffers = paged_text::required_buffers::<Self>(config, PAGE_TOKENS);
        buffers.extend([
            (
                "vision_token_indices",
                I32,
                vec![config.limits.prefill_chunk_size],
            ),
            ("vision_pixel_values", F32, vec![16384, 1536]),
            ("vision_position_ids", I64, vec![16384, 2]),
            ("vision_column_indices", I32, vec![4096]),
            ("vision_row_indices", I32, vec![4096]),
            ("vision_cu_seqlens", I32, vec![2]),
            ("vision_num_patches", I32, vec![1]),
            ("vision_num_features", I32, vec![1]),
            ("vision_slot", I32, vec![1]),
            ("vision_generation", I64, vec![1]),
            ("vision_feature_offset", I32, vec![1]),
        ]);
        buffers
    }
    fn required_inputs(_: &ModelConfig, kind: GraphKind) -> impl Iterator<Item = &'static str> {
        let extra: &[&str] = match kind {
            GraphKind::Prefill | GraphKind::Mixed => &["vision_token_indices"],
            GraphKind::Vision => &[
                "vision_pixel_values",
                "vision_position_ids",
                "vision_column_indices",
                "vision_row_indices",
                "vision_cu_seqlens",
                "vision_num_patches",
                "vision_num_features",
                "vision_slot",
                "vision_generation",
                "vision_feature_offset",
            ],
            _ => &[],
        };
        paged_text::required_inputs(kind)
            .iter()
            .chain(extra)
            .copied()
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
    fn pack(inputs: &mut Self::Inputs, step: &crate::model::InputStep<'_, Self>) -> Result<()> {
        inputs.pack(step)
    }
    fn uploads(inputs: &Self::Inputs) -> Vec<Upload<'_>> {
        inputs.uploads()
    }
    fn validate_completion(
        _: &ModelConfig,
        kind: GraphKind,
        record: &CompletionRecord,
    ) -> Result<()> {
        if ((record.status == 1 || record.status == 2) && record.token_id >= VOCABULARY as i64)
            || record.vector.is_some_and(|v| {
                v.model_reserved != -1
                    || v.tokens.iter().any(|id| *id >= VOCABULARY as i64)
                    || (kind == GraphKind::Speculative
                        && record.status != 3
                        && v.draft_cache_consumed != record.consumed_tokens)
            })
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Completion,
                "invalid MiMo completion",
            ));
        }
        Ok(())
    }
    #[cfg(feature = "images")]
    fn pack_vision(
        inputs: &mut Self::Inputs,
        config: &ModelConfig,
        handle: crate::scheduler::RequestHandle,
        images: &crate::frontend::images::ProcessedImages,
        image: usize,
    ) -> Result<()> {
        inputs.pack_vision(config, handle, images, image)
    }
    fn validate_prompt_data(config: &ModelConfig, tokens: &[u32], data: &PromptData) -> Result<()> {
        if data.vision_token_indices.len() != tokens.len()
            || tokens
                .iter()
                .zip(&data.vision_token_indices)
                .any(|(token, index)| {
                    (*token == IMAGE_TOKEN) != (*index >= 0)
                        || *index < -1
                        || *index >= config.contract.vision.features_per_request as i32
                })
        {
            return Err("invalid MiMo image feature indices".into());
        }
        Ok(())
    }
    #[cfg(feature = "server")]
    fn text_only(data: &PromptData) -> bool {
        data.vision_token_indices.iter().all(|index| *index < 0)
    }
    #[cfg(feature = "server")]
    type OutputConfig = crate::output::chatml::OutputConfig;
    #[cfg(feature = "server")]
    type Output = crate::output::chatml::ChatMlOutput;
    #[cfg(feature = "server")]
    fn prepare_input(
        frontend: &crate::frontend::Frontend,
        processor: Option<&crate::frontend::images::ImageProcessor>,
        rendered: &crate::frontend::RenderedPrompt,
        images: Vec<crate::frontend::images::RgbImage>,
    ) -> Result<crate::model::PreparedInput<PromptData>> {
        prompt::prepare_input(frontend, processor, rendered, images)
    }
    #[cfg(feature = "server")]
    fn system_boundary(
        frontend: &crate::frontend::Frontend,
        tokens: &[u32],
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<usize> {
        let count = messages
            .iter()
            .take_while(|m| m["role"] == "system")
            .count();
        if count == 0 {
            return None;
        }
        let mut options = options.clone();
        options.insert("add_generation_prompt".into(), false.into());
        options.insert("continue_final_message".into(), false.into());
        let prefix = frontend.render_chat(&messages[..count], &options).ok()?;
        let length = prefix.token_ids.len();
        (length > 0 && length < tokens.len() && tokens.starts_with(&prefix.token_ids))
            .then_some(length)
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
