//! Statically dispatched prepared-model semantics; CUDA transport remains model-independent.

pub mod qwen4exp;
use crate::{
    Result,
    execution::{
        CompletionRecord, ExecutionManifest, GraphKind, REQUEST_LANES, SamplingMode, Upload,
    },
    scheduler::{
        RequestHandle, RequestScheduler, ScheduledStep, SchedulerCapabilities, SchedulerLimits,
    },
};

#[cfg_attr(feature = "server", derive(serde::Deserialize))]
#[cfg_attr(feature = "server", serde(deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionCapabilities {
    pub features_per_request: usize,
    pub max_images: usize,
    pub patches_per_feature: usize,
    pub patch_capacities: Vec<usize>,
}

#[cfg_attr(feature = "server", derive(serde::Deserialize))]
#[cfg_attr(feature = "server", serde(deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractMetadata {
    pub schema: String,
    pub vocabulary: usize,
    pub vision: VisionCapabilities,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub contract: ContractMetadata,
    pub identity: [u8; 32],
    pub model_id: String,
    pub limits: SchedulerLimits,
    pub eos_token_ids: Vec<u32>,
    pub sampling_defaults: SamplingParameters,
    pub max_new_tokens: u32,
    pub speculative_widths: Vec<usize>,
    pub speculative_fixed_width: Option<usize>,
    pub speculative_copy: bool,
    pub speculative_proposal_source: crate::scheduler::speculation::ProposalSource,
}

#[derive(Clone, Debug)]
pub struct SamplingParameters {
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub top_k: i32,
}

impl Default for SamplingParameters {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 1.,
            top_p: 0.95,
            min_p: 0.,
            repetition_penalty: 1.,
            presence_penalty: 0.,
            frequency_penalty: 0.,
            top_k: 20,
        }
    }
}

impl SamplingParameters {
    pub(crate) fn values(&self) -> [f32; 6] {
        [
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        ]
    }
    pub fn validate(&self) -> Result<()> {
        if self.values().iter().any(|value| !value.is_finite())
            || self.temperature < 0.
            || !(0. < self.top_p && self.top_p <= 1.)
            || !(0. ..=1.).contains(&self.min_p)
            || self.repetition_penalty <= 0.
            || [self.temperature, self.top_p, self.repetition_penalty]
                .iter()
                .any(|value| *value > 0. && *value < f32::MIN_POSITIVE)
            || self.top_k < -1
        {
            return Err("invalid sampling parameters".into());
        }
        Ok(())
    }
}

impl ModelConfig {
    pub fn has_draft_cache(&self) -> bool {
        !self.speculative_widths.is_empty()
            && self.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Mtp
    }
}

pub trait ModelContract: Send + Sync + Sized + 'static {
    type Admission;
    type Inputs: Default;
    fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()>;
    fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities;
    fn graph(
        config: &ModelConfig,
        manifest: &ExecutionManifest,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        mode: SamplingMode,
    ) -> Result<usize>;
    fn pack(
        inputs: &mut Self::Inputs,
        config: &ModelConfig,
        scheduler: &RequestScheduler,
        step: &ScheduledStep,
        admissions: &[Option<Self::Admission>; REQUEST_LANES],
    ) -> Result<()>;
    fn uploads(inputs: &Self::Inputs) -> Vec<Upload<'_>>;
    fn admission_handle(admission: &Self::Admission) -> RequestHandle;
    fn grammar_updated(admission: &mut Self::Admission, mask: &[i32]);
    fn validate_completion(
        config: &ModelConfig,
        kind: GraphKind,
        record: &CompletionRecord,
    ) -> Result<()>;
    fn candidate_boundary(record: &CompletionRecord) -> u64;
    #[cfg(feature = "images")]
    fn pack_vision(
        inputs: &mut Self::Inputs,
        config: &ModelConfig,
        handle: RequestHandle,
        images: &crate::frontend::images::ProcessedImages,
        image: usize,
    ) -> Result<()>;
    #[cfg(feature = "server")]
    type PromptData: Send;
    #[cfg(feature = "server")]
    type OutputConfig: Send;
    #[cfg(feature = "server")]
    type Output: crate::output::ModelOutput<Config = Self::OutputConfig> + Send;
    #[cfg(feature = "server")]
    fn prepare_input(
        frontend: &crate::frontend::Frontend,
        processor: &crate::frontend::images::ImageProcessor,
        rendered: &crate::frontend::RenderedPrompt,
        images: Vec<crate::frontend::images::RgbImage>,
    ) -> Result<PreparedInput<Self::PromptData>>;
    #[cfg(feature = "server")]
    fn system_boundary(
        frontend: &crate::frontend::Frontend,
        prompt_tokens: &[u32],
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<usize>;
    #[cfg(feature = "server")]
    fn prepare_output(
        frontend: &crate::frontend::Frontend,
        rendered: &crate::frontend::RenderedPrompt,
        messages: &[serde_json::Value],
        options: &serde_json::Map<String, serde_json::Value>,
        structured: bool,
    ) -> Self::OutputConfig;
    #[cfg(feature = "server")]
    fn validate_prompt(config: &ModelConfig, prompt: &crate::server::Prompt<Self>) -> Result<()>;
    #[cfg(feature = "server")]
    fn text_only(data: &Self::PromptData) -> bool;
    #[cfg(feature = "server")]
    fn admission(
        config: &ModelConfig,
        handle: RequestHandle,
        prompt: &crate::server::Prompt<Self>,
        ignore_eos: bool,
    ) -> Result<Self::Admission>;
}

#[cfg(feature = "server")]
pub struct PreparedInput<D> {
    pub tokens: Vec<u32>,
    pub data: D,
    pub conditioning: Vec<crate::cache::ConditioningEvent>,
    pub images: Option<crate::frontend::images::ProcessedImages>,
}
