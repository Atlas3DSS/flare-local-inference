//! Model-independent request admission with model-specific prompt metadata.

use crate::{
    Result,
    model::{ModelConfig, ModelContract, SamplingParameters, inputs::SamplingAdmission},
    scheduler::RequestHandle,
};

pub struct Admission<M: ModelContract> {
    pub handle: RequestHandle,
    pub sampling: SamplingParameters,
    pub max_new_tokens: u32,
    pub prompt: Vec<u32>,
    pub data: M::PromptData,
    pub grammar_mask: Option<Vec<i32>>,
    pub ignore_eos: bool,
}

impl<M: ModelContract> Admission<M> {
    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        self.sampling.validate()?;
        if self.handle.slot as usize >= config.limits.max_active
            || self.handle.generation == 0
            || self.handle.generation > i64::MAX as u64
            || self.sampling.top_k > config.contract.vocabulary as i32
            || self.max_new_tokens == 0
            || self.max_new_tokens > i32::MAX as u32
            || self.prompt.is_empty()
            || self
                .prompt
                .iter()
                .any(|token| *token as usize >= config.contract.vocabulary)
            || self
                .grammar_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != config.contract.vocabulary.div_ceil(32))
        {
            return Err(
                "invalid prompt, sampling parameters, grammar mask, or request generation".into(),
            );
        }
        M::validate_prompt_data(config, &self.prompt, &self.data)
    }

    pub(crate) fn sampling(&self) -> SamplingAdmission<'_> {
        SamplingAdmission {
            sampling: &self.sampling,
            max_new_tokens: self.max_new_tokens,
            prompt: &self.prompt,
            grammar_mask: self.grammar_mask.as_deref(),
            ignore_eos: self.ignore_eos,
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn grammar_updated(&mut self, mask: &[i32]) {
        self.grammar_mask
            .as_mut()
            .expect("grammar admission has a mask")
            .copy_from_slice(mask);
    }

    #[cfg(feature = "server")]
    pub fn from_prompt(
        config: &ModelConfig,
        handle: RequestHandle,
        prompt: &crate::server::Prompt<M>,
        ignore_eos: bool,
    ) -> Result<Self> {
        let admission = Self {
            handle,
            sampling: prompt.sampling.clone(),
            max_new_tokens: prompt.max_new_tokens,
            prompt: prompt.tokens.clone(),
            data: prompt.data.clone(),
            grammar_mask: prompt
                .grammar
                .as_ref()
                .map(|grammar| grammar.mask().to_vec()),
            ignore_eos,
        };
        admission.validate(config)?;
        Ok(admission)
    }
}

#[cfg(feature = "server")]
pub fn validate_prompt<M: ModelContract>(
    config: &ModelConfig,
    prompt: &crate::server::Prompt<M>,
) -> Result<()> {
    prompt.sampling.validate()?;
    if prompt.tokens.is_empty()
        || prompt
            .system_boundary
            .is_some_and(|boundary| boundary == 0 || boundary >= prompt.tokens.len())
        || prompt
            .tokens
            .iter()
            .any(|token| *token as usize >= config.contract.vocabulary)
        || prompt.sampling.top_k > config.contract.vocabulary as i32
        || prompt.max_new_tokens == 0
        || prompt.max_new_tokens > i32::MAX as u32
        || prompt
            .tokens
            .len()
            .checked_add(prompt.max_new_tokens as usize)
            .is_none_or(|end| end > config.limits.max_sequence_tokens)
        || (!config.contract.vision.enabled()
            && (prompt.images.is_some() || !prompt.conditioning.is_empty()))
    {
        return Err("prompt and output limit exceed the prepared model contract".into());
    }
    M::validate_prompt_data(config, &prompt.tokens, &prompt.data)
}
