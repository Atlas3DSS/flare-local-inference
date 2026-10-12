use super::Glm5Next;
use crate::{
    Result,
    execution::{ABI_VERSION, ExecutionManifest, GraphKind, SPECULATIVE_ABI_VERSION},
    model::ModelConfig,
};

pub(super) fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    let contract = crate::model::ContractMetadata {
        page_tokens: None,
        ..config.contract.clone()
    };
    if contract != Glm5Next::metadata()
        || !matches!(config.contract.page_tokens, None | Some(256 | 512 | 1024))
        || !matches!(manifest.abi_version, ABI_VERSION | SPECULATIVE_ABI_VERSION)
    {
        return Err(
            "unsupported GLM graph-input schema, vocabulary, capabilities, or completion ABI"
                .into(),
        );
    }
    if config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
        || config.history_drafts
    {
        return Err("GLM speculation requires learned MTP proposals".into());
    }
    if config.limits.max_sequence_tokens > 1048576 {
        return Err("invalid prepared GLM context capacity".into());
    }
    Ok(())
}

pub(super) fn required_buffers(config: &ModelConfig) -> Vec<crate::model::catalog::BufferSpec> {
    crate::model::paged_text::required_buffers(config, Glm5Next::page_tokens(config))
}

pub(super) fn required_inputs(kind: GraphKind) -> &'static [&'static str] {
    crate::model::paged_text::required_inputs(kind)
}
