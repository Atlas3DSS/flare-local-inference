//! DeepSeek V4 Flash text/image requests, integrated drafts, and C4/C128 cache pages.

#[cfg(feature = "server")]
pub mod output;

use crate::{
    Result,
    execution::{
        ABI_VERSION, CompletionRecord, ExecutionManifest, GraphKind, SPECULATIVE_ABI_VERSION,
        Upload,
    },
    model::{ContractMetadata, ModelConfig, ModelContract, VisionCapabilities, paged_text},
    scheduler::SchedulerCapabilities,
};

pub struct DeepSeek4;
pub const INPUT_SCHEMA: &str = "deepseek4.graph-inputs.v1";
pub const VOCABULARY: usize = 129280;
pub const PAGE_TOKENS: usize = 128;

mod inputs;
#[cfg(feature = "server")]
mod prompt;
pub use inputs::Inputs;

#[derive(Clone, Default)]
pub struct PromptData {
    pub vision_token_indices: Vec<i32>,
    pub spans: Vec<std::ops::Range<usize>>,
}

impl DeepSeek4 {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            vision: VisionCapabilities {
                features_per_request: 1536,
                max_images: 4,
                patches_per_feature: 9,
                patch_capacities: vec![576, 1152, 2304, 3456],
            },
            page_tokens: Some(PAGE_TOKENS),
        }
    }
}

impl ModelContract for DeepSeek4 {
    type Inputs = Inputs;
    type PromptData = PromptData;

    fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        if config.contract != Self::metadata()
            || !matches!(manifest.abi_version, ABI_VERSION | SPECULATIVE_ABI_VERSION)
        {
            return Err(
                "unsupported DeepSeek V4 Flash schema, vocabulary, cache page size, or completion ABI"
                    .into(),
            );
        }
        if config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
            || config.history_drafts
        {
            return Err("DeepSeek V4 Flash speculation requires DSpark proposals".into());
        }
        if config.limits.max_sequence_tokens > 1048576 {
            return Err("DeepSeek V4 Flash context exceeds 1048576 tokens".into());
        }
        if manifest.engram.is_some() || config.limits.prefill_chunk_size < 512 {
            return Err(
                "DeepSeek V4 Flash requires image-capable prefill and no embedding tables".into(),
            );
        }
        Ok(())
    }

    fn required_buffers(config: &ModelConfig) -> Vec<crate::model::catalog::BufferSpec> {
        let mut buffers = paged_text::required_buffers::<Self>(config, PAGE_TOKENS);
        use crate::execution::Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
        for name in ["vision_token_indices", "vision_left", "vision_right"] {
            buffers.push((name, I32, vec![config.limits.prefill_chunk_size]));
        }
        buffers.extend([
            ("vision_pixel_values", F32, vec![3456, 588]),
            ("vision_position_ids", I64, vec![3456, 2]),
            ("vision_merge_indices", I32, vec![384, 9]),
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
            GraphKind::Prefill => &["vision_token_indices", "vision_left", "vision_right"],
            GraphKind::Vision => &[
                "vision_pixel_values",
                "vision_position_ids",
                "vision_merge_indices",
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
                "invalid DeepSeek completion",
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
    fn input_vocabulary(_: &ModelConfig) -> usize {
        VOCABULARY + 5
    }
    fn prefill_spans(data: &PromptData) -> &[std::ops::Range<usize>] {
        &data.spans
    }
    fn validate_prompt_data(config: &ModelConfig, tokens: &[u32], data: &PromptData) -> Result<()> {
        if data.vision_token_indices.len() != tokens.len()
            || data.spans.windows(2).any(|p| p[0].end > p[1].start)
            || data.spans.iter().any(|s| {
                s.start >= s.end
                    || s.end > tokens.len()
                    || s.end - s.start > 384
                    || s.end - s.start > config.limits.prefill_chunk_size
                    || tokens[s.start] != VOCABULARY as u32
                    || tokens[s.end - 1] != VOCABULARY as u32 + 4
            })
        {
            return Err("invalid DeepSeek image span metadata".into());
        }
        for (position, (token, index)) in tokens.iter().zip(&data.vision_token_indices).enumerate()
        {
            if (*token == VOCABULARY as u32 + 2) != (*index >= 0)
                || *index < -1
                || *index >= config.contract.vision.features_per_request as i32
                || (*token >= VOCABULARY as u32
                    && *token != VOCABULARY as u32 + 1
                    && !data.spans.iter().any(|s| s.contains(&position)))
            {
                return Err("invalid DeepSeek image token or feature index".into());
            }
        }
        Ok(())
    }
    #[cfg(feature = "server")]
    fn text_only(data: &PromptData) -> bool {
        data.spans.is_empty()
    }
    #[cfg(feature = "server")]
    type OutputConfig = output::OutputConfig;
    #[cfg(feature = "server")]
    type Output = output::DeepSeekOutput;
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
        _: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
        structured: bool,
    ) -> Result<Self::OutputConfig> {
        let prefix = rendered
            .text
            .rsplit_once("<｜Assistant｜>")
            .map(|(_, text)| text)
            .unwrap_or("");
        let reasoning = prefix
            .rfind("<think>")
            .is_some_and(|open| prefix.rfind("</think>").is_none_or(|close| open > close));
        if structured && reasoning {
            return Err("DeepSeek structured output requires non-thinking mode".into());
        }
        Ok(output::OutputConfig {
            verbatim: structured,
            reasoning,
            tools: options
                .get("tools")
                .or_else(|| frontend.template_default("tools"))
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{
            SamplingParameters,
            inputs::{SamplingAdmission, UploadBatch},
        },
        scheduler::SchedulerLimits,
    };

    fn config() -> ModelConfig {
        ModelConfig {
            contract: DeepSeek4::metadata(),
            identity: [0; 32],
            model_id: "deepseek-v4-flash".into(),
            limits: SchedulerLimits {
                max_active: 4,
                request_slots: 4,
                max_sequence_tokens: 4096,
                prefill_chunk_size: 512,
                pages: 128,
                snapshots: 8,
            },
            eos_token_ids: vec![1],
            sampling_defaults: SamplingParameters::default(),
            max_new_tokens: 128,
            speculative_widths: vec![1, 2, 3, 4, 5, 6],
            speculative_fixed_width: None,
            history_drafts: false,
            state_rows_per_request: None,
            speculative_proposal_source: crate::scheduler::speculation::ProposalSource::Mtp,
        }
    }

    #[test]
    fn image_sentinels_are_inputs_and_never_sampling_histogram_entries() {
        let config = config();
        let tokens = vec![20, 129280, 129282, 129283, 129284, 20];
        let data = PromptData {
            vision_token_indices: vec![-1, -1, 0, -1, -1, -1],
            spans: vec![1..5],
        };
        DeepSeek4::validate_prompt_data(&config, &tokens, &data).unwrap();
        assert_eq!(DeepSeek4::input_vocabulary(&config), VOCABULARY + 5);
        let mut uploads = UploadBatch::default();
        uploads
            .admission(
                &config,
                0,
                SamplingAdmission {
                    sampling: &config.sampling_defaults,
                    max_new_tokens: 1,
                    prompt: &tokens,
                    grammar_mask: None,
                    ignore_eos: false,
                },
            )
            .unwrap();
        let all = uploads.uploads();
        let histogram = all
            .iter()
            .find(|u| u.buffer == "admission_histogram")
            .unwrap();
        let values: Vec<_> = histogram
            .bytes
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), VOCABULARY);
        assert_eq!(values[20], 2);
        assert_eq!(values.iter().sum::<i32>(), 2);
        for bad in [
            PromptData {
                vision_token_indices: vec![-1; 6],
                spans: vec![1..5],
            },
            PromptData {
                vision_token_indices: vec![-1, -1, 1536, -1, -1, -1],
                spans: vec![1..5],
            },
            PromptData {
                vision_token_indices: data.vision_token_indices.clone(),
                spans: vec![1..4],
            },
            PromptData {
                vision_token_indices: data.vision_token_indices.clone(),
                spans: vec![],
            },
        ] {
            assert!(DeepSeek4::validate_prompt_data(&config, &tokens, &bad).is_err());
        }
    }

    #[test]
    fn vision_buffers_belong_only_to_image_and_prefill_graphs() {
        let config = config();
        assert!(
            DeepSeek4::required_inputs(&config, GraphKind::Vision)
                .any(|n| n == "vision_merge_indices")
        );
        assert!(
            DeepSeek4::required_inputs(&config, GraphKind::Prefill).any(|n| n == "vision_right")
        );
        for kind in [
            GraphKind::Decode,
            GraphKind::Speculative,
            GraphKind::Init,
            GraphKind::Restore,
        ] {
            assert!(!DeepSeek4::required_inputs(&config, kind).any(|n| n.starts_with("vision_")));
        }
    }
}
