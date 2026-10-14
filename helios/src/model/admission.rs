//! Model-independent request admission with model-specific prompt metadata.

use crate::{
    Result,
    model::{ModelConfig, SamplingParameters},
    scheduler::RequestHandle,
};

/// Per-token frontend arrays extended by shared text continuation handling.
#[cfg(feature = "server")]
#[derive(Default)]
pub struct PromptMetadataMut<'a> {
    pub positions: Option<&'a mut Vec<[i64; 3]>>,
    pub vision_token_indices: Option<&'a mut Vec<i32>>,
}

#[cfg(feature = "server")]
impl PromptMetadataMut<'_> {
    pub(crate) fn extend_text(self, before: usize, after: usize) -> Result<()> {
        if after < before
            || self.positions.as_ref().is_some_and(|p| p.len() != before)
            || self
                .vision_token_indices
                .as_ref()
                .is_some_and(|v| v.len() != before || v.iter().any(|index| *index != -1))
        {
            return Err("text continuation metadata does not match its token history".into());
        }
        let next = self
            .positions
            .as_ref()
            .and_then(|positions| positions.last())
            .and_then(|position| position.iter().max().copied())
            .unwrap_or(-1)
            .checked_add(1)
            .ok_or("text continuation position overflow")?;
        let end = next
            .checked_add(
                i64::try_from(after - before)
                    .map_err(|_| "text continuation length exceeds position range")?,
            )
            .ok_or("text continuation position overflow")?;
        if let Some(positions) = self.positions {
            positions.extend((next..end).map(|position| [position; 3]));
        }
        if let Some(indices) = self.vision_token_indices {
            indices.resize(after, -1);
        }
        Ok(())
    }
}

pub struct Admission {
    pub handle: RequestHandle,
    pub sampling: SamplingParameters,
    pub max_new_tokens: u32,
    pub prompt: Vec<u32>,
    pub data: crate::model::PromptData,
    pub grammar_mask: Option<Vec<i32>>,
    pub ignore_eos: bool,
}

impl Admission {
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
                .any(|token| *token as usize >= config.contract.input_vocabulary)
            || self
                .grammar_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != config.contract.vocabulary.div_ceil(32))
        {
            return Err(
                "invalid prompt, sampling parameters, grammar mask, or request generation".into(),
            );
        }
        crate::model::validate_prompt_data(config, &self.prompt, &self.data)
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
        prompt: &crate::server::Prompt,
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
pub fn validate_prompt(config: &ModelConfig, prompt: &crate::server::Prompt) -> Result<()> {
    prompt.sampling.validate()?;
    if prompt
        .tokens
        .len()
        .checked_add(prompt.max_new_tokens as usize)
        .is_none_or(|end| end > config.limits.max_sequence_tokens)
    {
        return Err(crate::Error::admission(
            crate::AdmissionError::Context,
            format!(
                "prompt of {} tokens plus an output limit of {} tokens exceeds this server's context of {} tokens",
                prompt.tokens.len(),
                prompt.max_new_tokens,
                config.limits.max_sequence_tokens
            ),
        ));
    }
    if prompt.tokens.is_empty()
        || prompt
            .system_boundary
            .is_some_and(|boundary| boundary == 0 || boundary >= prompt.tokens.len())
        || prompt
            .tokens
            .iter()
            .any(|token| *token as usize >= config.contract.input_vocabulary)
        || prompt.sampling.top_k > config.contract.vocabulary as i32
        || prompt.max_new_tokens == 0
        || prompt.max_new_tokens > i32::MAX as u32
        || (!config.contract.vision.enabled()
            && (prompt.images.is_some() || !prompt.conditioning.is_empty()))
    {
        return Err("prompt, output limit, or inputs violate the prepared model contract".into());
    }
    crate::model::validate_prompt_data(config, &prompt.tokens, &prompt.data)
}
