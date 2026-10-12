use super::Glm5Next;
use super::PAGE_TOKENS;
use crate::{
    Result,
    execution::{ABI_VERSION, Dtype, ExecutionManifest, GraphKind, SPECULATIVE_ABI_VERSION},
    model::ModelConfig,
};

pub(super) fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    if config.contract != Glm5Next::metadata()
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

pub(super) fn required_buffers(config: &ModelConfig) -> Vec<(&'static str, Dtype, Vec<usize>)> {
    let slots = config.limits.request_capacity();
    use Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
    let vocabulary = config.contract.vocabulary;
    let mut buffers = vec![
        ("request_state_rows", I64, vec![slots, 8]),
        ("memory_capacity", I64, vec![2]),
        ("state_request_slots", I64, vec![slots]),
        ("row_slot_ids", I32, vec![slots]),
        ("row_generations", I64, vec![slots]),
        ("live_rows", I32, vec![1]),
        ("query_lengths", I32, vec![slots]),
        ("token_ids", I64, vec![config.limits.prefill_chunk_size]),
        (
            "page_tables",
            I32,
            vec![slots, config.limits.max_sequence_tokens / PAGE_TOKENS],
        ),
        ("sample_enabled", I32, vec![slots]),
        (
            "sampling_grammar_mask",
            I32,
            vec![slots, vocabulary.div_ceil(32)],
        ),
        ("admission_count", I32, vec![1]),
        ("admission_slots", I32, vec![slots]),
        ("admission_generations", I64, vec![slots]),
        ("admission_seed_bits", I64, vec![slots]),
        ("admission_parameters", F32, vec![slots, 6]),
        ("admission_top_k", I32, vec![slots]),
        ("admission_max_new_tokens", I64, vec![slots]),
        ("admission_eos_token_ids", I64, vec![slots, 8]),
        ("admission_eos_count", I32, vec![slots]),
        ("admission_histogram", I32, vec![slots, vocabulary]),
        (
            "admission_grammar_mask",
            I32,
            vec![slots, vocabulary.div_ceil(32)],
        ),
        ("admission_grammar_enabled", I32, vec![slots]),
        ("state_expected_consumed", I64, vec![slots]),
        ("state_source_slots", I64, vec![slots]),
        ("state_destination_slots", I64, vec![slots]),
        ("state_operation", I32, vec![slots]),
        ("state_num_ops", I32, vec![1]),
    ];
    if config.has_draft_cache() {
        buffers.push((
            "draft_page_tables",
            I32,
            vec![slots, config.limits.max_sequence_tokens / PAGE_TOKENS],
        ));
    }
    buffers
}

pub(super) fn required_inputs(kind: GraphKind) -> &'static [&'static str] {
    match kind {
        GraphKind::Init | GraphKind::Restore => &[
            "request_state_rows",
            "admission_count",
            "admission_slots",
            "admission_generations",
            "admission_seed_bits",
            "admission_parameters",
            "admission_top_k",
            "admission_max_new_tokens",
            "admission_eos_token_ids",
            "admission_eos_count",
            "admission_histogram",
            "admission_grammar_mask",
            "admission_grammar_enabled",
            "page_tables",
            "state_request_slots",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_num_ops",
        ],
        GraphKind::Snapshot => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "page_tables",
            "state_request_slots",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_num_ops",
        ],
        GraphKind::Prefill | GraphKind::Decode | GraphKind::Speculative => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "query_lengths",
            "token_ids",
            "page_tables",
            "sample_enabled",
            "sampling_grammar_mask",
        ],
        GraphKind::Resume => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "sampling_grammar_mask",
        ],
        GraphKind::Vision => &[],
    }
}
