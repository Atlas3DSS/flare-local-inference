//! DeepSeek V4.1 text requests and heterogeneous compressed-history pages.

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

pub struct DeepSeek41;
pub const INPUT_SCHEMA: &str = "deepseek41.graph-inputs.v1";
pub const VOCABULARY: usize = 129280;
pub const PAGE_TOKENS: usize = 128;

pub type Inputs = paged_text::Inputs;

impl DeepSeek41 {
    pub fn metadata() -> ContractMetadata {
        ContractMetadata {
            schema: INPUT_SCHEMA.into(),
            vocabulary: VOCABULARY,
            vision: VisionCapabilities::disabled(),
            page_tokens: Some(PAGE_TOKENS),
        }
    }
}

impl ModelContract for DeepSeek41 {
    type Inputs = Inputs;
    type PromptData = ();

    fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        if config.contract != Self::metadata()
            || !matches!(manifest.abi_version, ABI_VERSION | SPECULATIVE_ABI_VERSION)
        {
            return Err(
                "unsupported DeepSeek V4.1 schema, vocabulary, cache page size, or completion ABI"
                    .into(),
            );
        }
        if config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
            || config.history_drafts
        {
            return Err("DeepSeek V4.1 speculation requires DSpark proposals".into());
        }
        if config.limits.max_sequence_tokens > 1048576 {
            return Err("DeepSeek V4.1 context exceeds 1048576 tokens".into());
        }
        let index = manifest
            .engram
            .as_ref()
            .ok_or("DeepSeek requires Engram tables")?
            .index();
        if index.vocabulary != VOCABULARY
            || index.tables.len() != 2
            || index.tables.iter().any(|t| t.request.is_none())
            || index
                .tables
                .iter()
                .map(|t| t.hashes.as_str())
                .collect::<Vec<_>>()
                != ["engram_1_hashes", "engram_14_hashes"]
        {
            return Err("DeepSeek Engram catalog does not match its model contract".into());
        }
        Ok(())
    }

    fn required_buffers(config: &ModelConfig) -> Vec<crate::model::catalog::BufferSpec> {
        let mut buffers = paged_text::required_buffers::<Self>(config, PAGE_TOKENS);
        for name in ["engram_1_hashes", "engram_14_hashes"] {
            buffers.push((
                name,
                crate::execution::Dtype::Int64,
                vec![
                    config.limits.prefill_chunk_size.max(
                        config.limits.request_capacity()
                            * config.speculative_widths.iter().copied().max().unwrap_or(1),
                    ),
                    24,
                ],
            ));
        }
        buffers
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
        _: &mut Self::Inputs,
        _: &ModelConfig,
        _: crate::scheduler::RequestHandle,
        _: &crate::frontend::images::ProcessedImages,
        _: usize,
    ) -> Result<()> {
        Err("DeepSeek V4.1 text catalog has no vision graphs".into())
    }
    fn validate_prompt_data(_: &ModelConfig, _: &[u32], _: &()) -> Result<()> {
        Ok(())
    }
    #[cfg(feature = "server")]
    fn text_only(_: &()) -> bool {
        true
    }
    #[cfg(feature = "server")]
    type OutputConfig = output::OutputConfig;
    #[cfg(feature = "server")]
    type Output = output::DeepSeekOutput;
    #[cfg(feature = "server")]
    fn prepare_input(
        _: &crate::frontend::Frontend,
        _: Option<&crate::frontend::images::ImageProcessor>,
        rendered: &crate::frontend::RenderedPrompt,
        images: Vec<crate::frontend::images::RgbImage>,
    ) -> Result<crate::model::PreparedInput<()>> {
        if !images.is_empty() || rendered.token_ids.contains(&129264) {
            return Err("DeepSeek V4.1 text catalog does not accept images".into());
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
mod contract_tests {
    use super::*;
    use crate::{execution::TokenVector, model::SamplingParameters, scheduler::SchedulerLimits};

    fn config() -> ModelConfig {
        ModelConfig {
            contract: DeepSeek41::metadata(),
            identity: [0; 32],
            model_id: "deepseek".into(),
            limits: SchedulerLimits {
                max_active: 4,
                request_slots: 4,
                max_sequence_tokens: 4096,
                prefill_chunk_size: 4096,
                pages: 128,
                snapshots: 8,
            },
            eos_token_ids: vec![1],
            sampling_defaults: SamplingParameters::default(),
            max_new_tokens: 128,
            speculative_widths: vec![1, 2, 3, 4, 5, 6, 7, 8],
            speculative_fixed_width: None,
            history_drafts: false,
            speculative_proposal_source: crate::scheduler::speculation::ProposalSource::Mtp,
            state_rows_per_request: None,
        }
    }

    #[test]
    fn learned_drafts_do_not_require_separate_paged_storage() {
        let config = config();
        assert!(config.has_learned_drafts());
        assert_eq!(
            DeepSeek41::scheduler_capabilities(&config).draft_pages_per_request,
            0
        );
        assert!(
            !DeepSeek41::required_buffers(&config)
                .iter()
                .any(|(name, _, _)| *name == "draft_page_tables")
        );
        assert!(
            !DeepSeek41::required_inputs(&config, GraphKind::Speculative)
                .any(|name| name.ends_with("_hashes"))
        );
    }

    #[test]
    fn speculative_completion_requires_draft_context_through_the_accepted_target_prefix() {
        let config = config();
        let mut record = CompletionRecord {
            consumed_tokens: 133,
            status: 1,
            token_id: 7,
            vector: Some(TokenVector {
                draft_cache_consumed: 133,
                ..Default::default()
            }),
            ..Default::default()
        };
        DeepSeek41::validate_completion(&config, GraphKind::Speculative, &record).unwrap();
        record.vector.as_mut().unwrap().draft_cache_consumed = 132;
        assert!(DeepSeek41::validate_completion(&config, GraphKind::Speculative, &record).is_err());
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use crate::frontend::{Asset, Frontend, FrontendAssets};
    use serde_json::{Map, json};
    use sha2::{Digest, Sha256};

    #[test]
    #[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to the DeepSeek checkpoint"]
    fn canonical_text_template_renders_in_the_native_frontend() {
        let root =
            std::path::PathBuf::from(std::env::var_os("FLARE_DEEPSEEK_TEST_SNAPSHOT").unwrap());
        let asset = |path: std::path::PathBuf| Asset {
            sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
            path,
            range: None,
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset(root.join("tokenizer.json")),
                tokens: crate::frontend::test_support::tokens(&asset(
                    root.join("tokenizer_config.json"),
                ))
                .unwrap(),
                chat_template: asset(
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../ignition/recipes/deepseek41/assets/chat_template.jinja"),
                ),
            },
            Map::new(),
        )
        .unwrap();
        let messages = [json!({"role":"user", "content":"Hello"})];
        let mut options = Map::new();
        options.insert("add_generation_prompt".into(), true.into());
        let prompt = frontend.render_chat(&messages, &options).unwrap();
        assert_eq!(
            prompt.text,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hello<｜Assistant｜><think>"
        );
        assert!(!prompt.token_ids.is_empty());
        let output =
            DeepSeek41::prepare_output(&frontend, &prompt, &messages, &options, false).unwrap();
        assert!(output.reasoning);
        options.insert("enable_thinking".into(), false.into());
        let prompt = frontend.render_chat(&messages, &options).unwrap();
        assert_eq!(
            prompt.text,
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );
        let output =
            DeepSeek41::prepare_output(&frontend, &prompt, &messages, &options, false).unwrap();
        assert!(!output.reasoning);
    }
}
