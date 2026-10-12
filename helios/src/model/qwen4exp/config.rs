use super::Qwen4Exp;
use crate::{
    Result,
    execution::{Dtype, ExecutionManifest, GraphKind},
    model::ModelConfig,
};

pub(super) fn validate_config(config: &ModelConfig, _: &ExecutionManifest) -> Result<()> {
    Qwen4Exp::validate_metadata(&config.contract)?;
    if config.limits.max_sequence_tokens > 262144 || config.limits.prefill_chunk_size > 262144 {
        return Err("invalid prepared Qwen4Exp context capacity".into());
    }
    Ok(())
}

pub(super) fn required_buffers(config: &ModelConfig) -> Vec<(&'static str, Dtype, Vec<usize>)> {
    let slots = config.limits.request_capacity();
    use Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
    let context = config.limits.max_sequence_tokens;
    let prefill_chunk_size = config.limits.prefill_chunk_size;
    let patch_capacity = *config.contract.vision.patch_capacities.last().unwrap();
    let mut buffers = vec![
        ("request_state_rows", I64, vec![slots, 8]),
        ("memory_capacity", I64, vec![2]),
        ("state_request_slots", I64, vec![slots]),
        ("row_slot_ids", I32, vec![slots]),
        ("row_generations", I64, vec![slots]),
        ("live_rows", I32, vec![1]),
        ("query_lengths", I32, vec![slots]),
        ("token_ids", I64, vec![prefill_chunk_size]),
        ("prefill_positions", I64, vec![prefill_chunk_size, 3]),
        ("page_tables", I32, vec![slots, context / 64]),
        ("sample_enabled", I32, vec![slots]),
        (
            "sampling_grammar_mask",
            I32,
            vec![slots, config.contract.vocabulary.div_ceil(32)],
        ),
        ("vision_token_indices", I32, vec![prefill_chunk_size]),
        ("vision_pixel_values", F32, vec![patch_capacity, 1536]),
        ("vision_interpolation_indices", I64, vec![patch_capacity, 4]),
        ("vision_interpolation_weights", F32, vec![patch_capacity, 4]),
        ("vision_position_ids", I64, vec![patch_capacity, 2]),
        ("vision_cu_seqlens", I32, vec![2]),
        ("vision_num_patches", I32, vec![1]),
        ("vision_slot", I32, vec![1]),
        ("vision_generation", I64, vec![1]),
        ("vision_feature_offset", I32, vec![1]),
        ("admission_count", I32, vec![1]),
        ("admission_slots", I32, vec![slots]),
        ("admission_generations", I64, vec![slots]),
        ("admission_seed_bits", I64, vec![slots]),
        ("admission_parameters", F32, vec![slots, 6]),
        ("admission_top_k", I32, vec![slots]),
        ("admission_max_new_tokens", I64, vec![slots]),
        ("admission_eos_token_ids", I64, vec![slots, 8]),
        ("admission_eos_count", I32, vec![slots]),
        (
            "admission_histogram",
            I32,
            vec![slots, config.contract.vocabulary],
        ),
        (
            "admission_grammar_mask",
            I32,
            vec![slots, config.contract.vocabulary.div_ceil(32)],
        ),
        ("admission_grammar_enabled", I32, vec![slots]),
        ("state_expected_consumed", I64, vec![slots]),
        ("state_source_slots", I64, vec![slots]),
        ("state_destination_slots", I64, vec![slots]),
        ("state_operation", I32, vec![slots]),
        ("state_qsa_slots", I64, vec![slots]),
        ("state_accepted_tokens", I32, vec![slots]),
        ("state_num_ops", I32, vec![1]),
    ];
    if !config.speculative_widths.is_empty() {
        buffers.extend([
            ("copy_token_ids", I64, vec![slots, 7]),
            ("copy_token_count", I32, vec![slots]),
            ("copy_expected_consumed", I64, vec![slots]),
            ("copy_generation", I64, vec![slots]),
        ]);
        if config.has_draft_cache() {
            buffers.push(("draft_page_tables", I32, vec![slots, context / 64]));
        } else {
            buffers.extend([
                (
                    "copy_grammar_mask",
                    I32,
                    vec![slots, 8, config.contract.vocabulary.div_ceil(32)],
                ),
                ("copy_grammar_mask_count", I32, vec![slots]),
            ]);
        }
    }
    buffers
}

pub(super) fn required_inputs(
    config: &ModelConfig,
    kind: GraphKind,
) -> impl Iterator<Item = &'static str> {
    let transfer = matches!(kind, GraphKind::Restore | GraphKind::Snapshot);
    let draft = config.has_draft_cache() && (transfer || kind.samples());
    let copy = kind == GraphKind::Speculative && !config.has_draft_cache();
    base_inputs(kind).iter().copied().chain(
        [
            transfer.then_some("page_tables"),
            draft.then_some("draft_page_tables"),
            copy.then_some("copy_grammar_mask"),
            copy.then_some("copy_grammar_mask_count"),
        ]
        .into_iter()
        .flatten(),
    )
}

fn base_inputs(kind: GraphKind) -> &'static [&'static str] {
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
            "state_request_slots",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_qsa_slots",
            "state_accepted_tokens",
            "state_num_ops",
        ],
        GraphKind::Snapshot => &[
            "request_state_rows",
            "state_request_slots",
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_qsa_slots",
            "state_accepted_tokens",
            "state_num_ops",
            "row_slot_ids",
            "row_generations",
            "live_rows",
        ],
        GraphKind::Prefill => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "query_lengths",
            "token_ids",
            "prefill_positions",
            "vision_token_indices",
            "page_tables",
            "sample_enabled",
            "sampling_grammar_mask",
        ],
        GraphKind::Speculative => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "page_tables",
            "sample_enabled",
            "sampling_grammar_mask",
            "copy_token_ids",
            "copy_token_count",
            "copy_expected_consumed",
            "copy_generation",
        ],
        GraphKind::Decode | GraphKind::Resume => &[
            "request_state_rows",
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "page_tables",
            "sample_enabled",
            "sampling_grammar_mask",
        ],
        GraphKind::Vision => &[
            "vision_pixel_values",
            "vision_interpolation_indices",
            "vision_interpolation_weights",
            "vision_position_ids",
            "vision_cu_seqlens",
            "vision_num_patches",
            "vision_slot",
            "vision_generation",
            "vision_feature_offset",
        ],
    }
}
