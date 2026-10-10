use super::{Glm5Next, PAGE_TOKENS};
use crate::{
    Result,
    execution::{ABI_VERSION, Dtype, ExecutionManifest, GraphKind, SamplingMode},
    model::ModelConfig,
    scheduler::prefill_capacities,
};

impl Glm5Next {
    pub fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        Self::validate_inner(config, manifest).map_err(|error| match error {
            crate::Error::Execution(crate::ExecutionError::Manifest(_)) => error,
            _ => {
                crate::ExecutionError::Manifest(crate::Failure::caused_by(error.to_string(), error))
                    .into()
            }
        })
    }

    fn validate_inner(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
        manifest.validate()?;
        if config.contract != Self::metadata() || manifest.abi_version != ABI_VERSION {
            return Err(
                "unsupported GLM graph-input schema, vocabulary, capabilities, or completion ABI"
                    .into(),
            );
        }
        if !config.speculative_widths.is_empty()
            || config.speculative_fixed_width.is_some()
            || config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
        {
            return Err("prepared GLM catalog does not support speculative execution".into());
        }
        config.sampling_defaults.validate()?;
        config.limits.validate_prefill()?;
        if config.model_id.is_empty()
            || config.limits.max_active != manifest.max_active
            || config.limits.max_sequence_tokens == 0
            || config.limits.max_sequence_tokens > 1048576
            || !config
                .limits
                .max_sequence_tokens
                .is_multiple_of(PAGE_TOKENS)
            || config.limits.pages == 0
            || config.limits.pages > i32::MAX as u32
            || config.limits.snapshots == 0
            || config.limits.snapshots > i32::MAX as u32 - 5
            || config.max_new_tokens == 0
            || config.max_new_tokens > i32::MAX as u32
            || config.sampling_defaults.top_k > config.contract.vocabulary as i32
            || config.eos_token_ids.len() > 8
            || config
                .eos_token_ids
                .iter()
                .any(|id| *id as usize >= config.contract.vocabulary)
        {
            return Err("invalid prepared GLM model configuration".into());
        }
        for (name, dtype, shape) in required_buffers(config) {
            let buffer = manifest.buffer(name)?;
            if buffer.dtype != dtype || buffer.shape != shape {
                return Err(format!(
                    "invalid GLM semantic buffer {name}: expected {dtype:?}{shape:?}"
                )
                .into());
            }
        }
        for graph in &manifest.graphs {
            Self::graph(
                config,
                manifest,
                graph.kind,
                graph.batch_capacity,
                graph.token_capacity,
                graph.sampling_mode,
            )?;
        }
        for kind in [
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Resume,
            GraphKind::Decode,
        ] {
            for batch in [1, 2, 4]
                .into_iter()
                .filter(|b| *b <= config.limits.max_active.next_power_of_two())
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
        Ok(())
    }

    pub fn graph(
        config: &ModelConfig,
        manifest: &ExecutionManifest,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        mode: SamplingMode,
    ) -> Result<usize> {
        if matches!(kind, GraphKind::Vision | GraphKind::Speculative)
            || ![1, 2, 4].contains(&batch)
            || batch > config.limits.max_active.next_power_of_two()
            || if kind == GraphKind::Prefill {
                batch != 1
                    || !prefill_capacities(config.limits.prefill_chunk_size).any(|n| n == tokens)
            } else {
                tokens != 1
            }
        {
            return Err("undeclared GLM graph kind or capacity".into());
        }
        let mut matches = manifest.graphs.iter().enumerate().filter(|(_, g)| {
            g.kind == kind
                && g.batch_capacity == batch
                && g.token_capacity == tokens
                && g.sampling_mode == mode
        });
        let (index, graph) = matches.next().ok_or_else(|| {
            format!(
                "prepared model lacks {kind:?}/{mode:?} graph with batch {batch}, tokens {tokens}"
            )
        })?;
        if matches.next().is_some() {
            return Err("ambiguous prepared GLM graph capacity".into());
        }
        for name in required_inputs(kind) {
            if !graph.input_bindings.iter().any(|b| b.buffer_name == *name) {
                return Err(
                    format!("prepared graph {} lacks required input {name}", graph.name).into(),
                );
            }
        }
        Ok(index)
    }
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
