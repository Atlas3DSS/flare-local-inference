use super::Glm5Next;
use super::PAGE_TOKENS;
use crate::{
    Result,
    execution::{ABI_VERSION, Dtype, ExecutionManifest, GraphKind},
    model::ModelConfig,
};

pub(super) fn validate_config(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    if config.contract != Glm5Next::metadata() || manifest.abi_version != ABI_VERSION {
        return Err(
            "unsupported GLM graph-input schema, vocabulary, capabilities, or completion ABI"
                .into(),
        );
    }
    if !config.speculative_widths.is_empty()
        || config.speculative_fixed_width.is_some()
        || config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
    {
        return Err("prepared GLM catalog does not support speculative execution".into());
    }
    if config.limits.max_sequence_tokens > 1048576 {
        return Err("invalid prepared GLM context capacity".into());
    }
    Ok(())
}

pub(super) fn required_buffers(config: &ModelConfig) -> Vec<(&'static str, Dtype, Vec<usize>)> {
    use Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
    let vocabulary = config.contract.vocabulary;
    vec![
        ("row_slot_ids", I32, vec![4]),
        ("row_generations", I64, vec![4]),
        ("live_rows", I32, vec![1]),
        ("query_lengths", I32, vec![4]),
        ("token_ids", I64, vec![config.limits.prefill_chunk_size]),
        (
            "page_tables",
            I32,
            vec![4, config.limits.max_sequence_tokens / PAGE_TOKENS],
        ),
        ("sample_enabled", I32, vec![4]),
        (
            "sampling_grammar_mask",
            I32,
            vec![4, vocabulary.div_ceil(32)],
        ),
        ("admission_count", I32, vec![1]),
        ("admission_slots", I32, vec![4]),
        ("admission_generations", I64, vec![4]),
        ("admission_seed_bits", I64, vec![4]),
        ("admission_parameters", F32, vec![4, 6]),
        ("admission_top_k", I32, vec![4]),
        ("admission_max_new_tokens", I64, vec![4]),
        ("admission_eos_token_ids", I64, vec![4, 8]),
        ("admission_eos_count", I32, vec![4]),
        ("admission_histogram", I32, vec![4, vocabulary]),
        (
            "admission_grammar_mask",
            I32,
            vec![4, vocabulary.div_ceil(32)],
        ),
        ("admission_grammar_enabled", I32, vec![4]),
        ("state_expected_consumed", I64, vec![4]),
        ("state_source_slots", I64, vec![4]),
        ("state_destination_slots", I64, vec![4]),
        ("state_operation", I32, vec![4]),
        ("state_num_ops", I32, vec![1]),
    ]
}

pub(super) fn required_inputs(kind: GraphKind) -> &'static [&'static str] {
    match kind {
        GraphKind::Init | GraphKind::Restore => &[
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
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_num_ops",
        ],
        GraphKind::Snapshot => &[
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "page_tables",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_num_ops",
        ],
        GraphKind::Prefill | GraphKind::Decode => &[
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
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "sampling_grammar_mask",
        ],
        GraphKind::Vision | GraphKind::Speculative => &[],
    }
}
