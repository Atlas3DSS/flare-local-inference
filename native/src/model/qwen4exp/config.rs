use super::Qwen4Exp;
use crate::{
    Result,
    execution::{Dtype, ExecutionManifest, GraphKind, SamplingMode},
    model::ModelConfig,
    scheduler::prefill_capacities,
};

impl Qwen4Exp {
    pub fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        manifest.validate()?;
        Self::validate_metadata(&config.contract)?;
        crate::scheduler::speculation::validate_widths(&config.speculative_widths)?;
        if config
            .speculative_fixed_width
            .is_some_and(|w| !config.speculative_widths.contains(&w))
        {
            return Err(
                "speculative_fixed_width must belong to the prepared speculative_widths".into(),
            );
        }
        if !config.speculative_widths.is_empty() {
            if manifest.abi_version != crate::execution::SPECULATIVE_ABI_VERSION {
                return Err("speculative catalog requires ABI3".into());
            }
            let mut buffers = vec![
                ("copy_token_ids", Dtype::Int64, vec![4, 7]),
                ("copy_token_count", Dtype::Int32, vec![4]),
                ("copy_expected_consumed", Dtype::Int64, vec![4]),
                ("copy_generation", Dtype::Int64, vec![4]),
            ];
            if config.has_draft_cache() {
                buffers.push((
                    "draft_page_tables",
                    Dtype::Int32,
                    vec![4, config.limits.max_sequence_tokens / 64],
                ));
            } else {
                buffers.extend([
                    (
                        "copy_grammar_mask",
                        Dtype::Int32,
                        vec![4, 8, config.contract.vocabulary.div_ceil(32)],
                    ),
                    ("copy_grammar_mask_count", Dtype::Int32, vec![4]),
                ]);
            }
            for (name, dtype, shape) in buffers {
                let buffer = manifest.buffer(name)?;
                if buffer.dtype != dtype || buffer.shape != shape {
                    return Err(format!("invalid speculative buffer {name}"));
                }
            }
            for width in &config.speculative_widths {
                for batch in [1, 2, 4]
                    .into_iter()
                    .filter(|b| *b <= config.limits.max_active.next_power_of_two())
                {
                    if batch * width > 8 {
                        continue;
                    }
                    Self::graph(
                        config,
                        manifest,
                        GraphKind::Speculative,
                        batch,
                        *width,
                        SamplingMode::General,
                    )?;
                }
            }
        }
        if config.speculative_widths.is_empty()
            && config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
        {
            return Err("copy proposal source requires speculative graphs".into());
        }
        config.sampling_defaults.validate()?;
        if config.sampling_defaults.top_k > config.contract.vocabulary as i32 {
            return Err("top_k exceeds the prepared vocabulary".into());
        }
        config.limits.validate_prefill()?;
        if config.limits.max_sequence_tokens > 262144 || config.limits.prefill_chunk_size > 262144 {
            return Err("invalid prepared Qwen4Exp model configuration".into());
        }
        for graph in &manifest.graphs {
            if (graph.kind == GraphKind::Vision
                && !config
                    .contract
                    .vision
                    .patch_capacities
                    .contains(&graph.token_capacity))
                || (graph.kind == GraphKind::Prefill
                    && !prefill_capacities(config.limits.prefill_chunk_size)
                        .any(|n| n == graph.token_capacity))
            {
                return Err(format!(
                    "graph {} has an undeclared Qwen4Exp capacity",
                    graph.name
                ));
            }
        }
        if config.model_id.is_empty()
            || config.limits.max_active != manifest.max_active
            || config.limits.max_sequence_tokens == 0
            || !config.limits.max_sequence_tokens.is_multiple_of(64)
            || config.limits.pages == 0
            || config.limits.pages > i32::MAX as u32
            || config.limits.snapshots == 0
            || config.limits.snapshots > i32::MAX as u32 - 5
            || config.contract.vision.features_per_request == 0
            || config.max_new_tokens == 0
            || config.max_new_tokens > i32::MAX as u32
            || config.eos_token_ids.len() > 8
            || config
                .eos_token_ids
                .iter()
                .any(|id| *id as usize >= config.contract.vocabulary)
        {
            return Err("invalid prepared Qwen4Exp model configuration".into());
        }
        for (name, dtype, shape) in required_buffers(config) {
            let buffer = manifest.buffer(name)?;
            if buffer.dtype != dtype || buffer.shape != shape {
                return Err(format!(
                    "invalid Qwen4Exp semantic buffer {name}: expected {dtype:?}{shape:?}"
                ));
            }
        }
        for kind in [
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Resume,
            GraphKind::Decode,
        ] {
            if kind == GraphKind::Decode && !config.speculative_widths.is_empty() {
                continue;
            }
            for batch in [1, 2, 4]
                .into_iter()
                .filter(|batch| *batch <= config.limits.max_active.next_power_of_two())
            {
                Self::graph(config, manifest, kind, batch, 1, SamplingMode::General)?;
            }
        }
        for tokens in prefill_capacities(config.limits.prefill_chunk_size) {
            Self::graph(
                config,
                manifest,
                GraphKind::Prefill,
                1,
                tokens,
                SamplingMode::General,
            )?;
        }
        for &patches in &config.contract.vision.patch_capacities {
            Self::graph(
                config,
                manifest,
                GraphKind::Vision,
                1,
                patches,
                SamplingMode::General,
            )?;
        }
        Ok(())
    }

    pub fn graph(
        config: &ModelConfig,
        manifest: &ExecutionManifest,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        sampling_mode: SamplingMode,
    ) -> Result<usize> {
        let mut candidates = manifest.graphs.iter().enumerate().filter(|(_, graph)| {
            graph.kind == kind
                && graph.batch_capacity == batch
                && graph.token_capacity == tokens
                && graph.sampling_mode == sampling_mode
        });
        let (index, graph) = candidates.next().ok_or_else(|| {
            format!("prepared model lacks {kind:?}/{sampling_mode:?} graph with batch {batch}, tokens {tokens}")
        })?;
        if candidates.next().is_some() {
            return Err("ambiguous prepared model graph capacity".into());
        }
        if config.has_draft_cache()
            && matches!(
                kind,
                GraphKind::Prefill | GraphKind::Decode | GraphKind::Resume | GraphKind::Speculative
            )
            && !graph
                .input_bindings
                .iter()
                .any(|binding| binding.buffer_name == "draft_page_tables")
        {
            return Err(format!(
                "prepared graph {} lacks required input draft_page_tables",
                graph.name
            ));
        }
        if kind == GraphKind::Speculative && !config.has_draft_cache() {
            for name in ["copy_grammar_mask", "copy_grammar_mask_count"] {
                if !graph.input_bindings.iter().any(|b| b.buffer_name == name) {
                    return Err(format!(
                        "prepared graph {} lacks required input {name}",
                        graph.name
                    ));
                }
            }
        }
        if matches!(kind, GraphKind::Restore | GraphKind::Snapshot) {
            for name in [
                Some("page_tables"),
                config.has_draft_cache().then_some("draft_page_tables"),
            ]
            .into_iter()
            .flatten()
            {
                if !graph
                    .input_bindings
                    .iter()
                    .any(|input| input.buffer_name == name)
                {
                    return Err(format!(
                        "prepared graph {} lacks required input {name}",
                        graph.name
                    ));
                }
            }
        }
        for name in required_inputs(kind) {
            if !graph
                .input_bindings
                .iter()
                .any(|binding| binding.buffer_name == *name)
            {
                return Err(format!(
                    "prepared graph {} lacks required input {name}",
                    graph.name
                ));
            }
        }
        Ok(index)
    }
}
pub(super) fn required_buffers(config: &ModelConfig) -> Vec<(&'static str, Dtype, Vec<usize>)> {
    use Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
    let context = config.limits.max_sequence_tokens;
    let prefill_chunk_size = config.limits.prefill_chunk_size;
    let patch_capacity = *config.contract.vision.patch_capacities.last().unwrap();
    vec![
        ("row_slot_ids", I32, vec![4]),
        ("row_generations", I64, vec![4]),
        ("live_rows", I32, vec![1]),
        ("query_lengths", I32, vec![4]),
        ("token_ids", I64, vec![prefill_chunk_size]),
        ("prefill_positions", I64, vec![prefill_chunk_size, 3]),
        ("page_tables", I32, vec![4, context / 64]),
        ("sample_enabled", I32, vec![4]),
        (
            "sampling_grammar_mask",
            I32,
            vec![4, config.contract.vocabulary.div_ceil(32)],
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
        ("admission_slots", I32, vec![4]),
        ("admission_generations", I64, vec![4]),
        ("admission_seed_bits", I64, vec![4]),
        ("admission_parameters", F32, vec![4, 6]),
        ("admission_top_k", I32, vec![4]),
        ("admission_max_new_tokens", I64, vec![4]),
        ("admission_eos_token_ids", I64, vec![4, 8]),
        ("admission_eos_count", I32, vec![4]),
        (
            "admission_histogram",
            I32,
            vec![4, config.contract.vocabulary],
        ),
        (
            "admission_grammar_mask",
            I32,
            vec![4, config.contract.vocabulary.div_ceil(32)],
        ),
        ("admission_grammar_enabled", I32, vec![4]),
        ("state_expected_consumed", I64, vec![4]),
        ("state_source_slots", I64, vec![4]),
        ("state_destination_slots", I64, vec![4]),
        ("state_operation", I32, vec![4]),
        ("state_qsa_slots", I64, vec![4]),
        ("state_accepted_tokens", I32, vec![4]),
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
            "state_expected_consumed",
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_qsa_slots",
            "state_accepted_tokens",
            "state_num_ops",
        ],
        GraphKind::Snapshot => &[
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
