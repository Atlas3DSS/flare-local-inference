//! Qwen4Exp prepared graph semantics and native input packing.

mod inputs;
#[cfg(test)]
mod tests;
pub use inputs::{Admission, ModelInputs, SamplingParameters};

use crate::{
    Result,
    execution::{Dtype, ExecutionManifest, GraphKind, SamplingMode},
    scheduler::{SchedulerLimits, prefill_capacities},
};

pub const VOCABULARY: usize = 248_320;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub identity: [u8; 32],
    pub model_id: String,
    pub limits: SchedulerLimits,
    pub eos_token_ids: Vec<u32>,
    pub vision_features_per_request: usize,
    pub sampling_defaults: SamplingParameters,
    pub max_new_tokens: u32,
    pub speculative_widths: Vec<usize>,
    pub speculative_fixed_width: Option<usize>,
    pub speculative_copy: bool,
    pub speculative_proposal_source: crate::scheduler::speculation::ProposalSource,
}

impl ModelConfig {
    pub fn has_draft_cache(&self) -> bool {
        !self.speculative_widths.is_empty()
            && self.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Mtp
    }

    pub fn validate(&self, manifest: &ExecutionManifest) -> Result<()> {
        manifest.validate()?;
        crate::scheduler::speculation::validate_widths(&self.speculative_widths)?;
        if self
            .speculative_fixed_width
            .is_some_and(|w| !self.speculative_widths.contains(&w))
        {
            return Err(
                "speculative_fixed_width must belong to the prepared speculative_widths".into(),
            );
        }
        if !self.speculative_widths.is_empty() {
            if manifest.abi_version != crate::execution::SPECULATIVE_ABI_VERSION {
                return Err("speculative catalog requires ABI3".into());
            }
            let mut buffers = vec![
                ("copy_token_ids", Dtype::Int64, vec![4, 7]),
                ("copy_token_count", Dtype::Int32, vec![4]),
                ("copy_expected_consumed", Dtype::Int64, vec![4]),
                ("copy_generation", Dtype::Int64, vec![4]),
            ];
            if self.has_draft_cache() {
                buffers.push((
                    "draft_page_tables",
                    Dtype::Int32,
                    vec![4, self.limits.max_sequence_tokens / 64],
                ));
            } else {
                buffers.extend([
                    (
                        "copy_grammar_mask",
                        Dtype::Int32,
                        vec![4, 8, VOCABULARY.div_ceil(32)],
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
            for width in &self.speculative_widths {
                for batch in [1, 2, 4]
                    .into_iter()
                    .filter(|b| *b <= self.limits.max_active.next_power_of_two())
                {
                    if batch * width > 8 {
                        continue;
                    }
                    self.graph(
                        manifest,
                        GraphKind::Speculative,
                        batch,
                        *width,
                        SamplingMode::General,
                    )?;
                }
            }
        }
        if self.speculative_widths.is_empty()
            && self.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
        {
            return Err("copy proposal source requires speculative graphs".into());
        }
        self.sampling_defaults.validate()?;
        self.limits.validate_prefill()?;
        if self.model_id.is_empty()
            || self.limits.max_active != manifest.max_active
            || self.limits.max_sequence_tokens == 0
            || !self.limits.max_sequence_tokens.is_multiple_of(64)
            || self.limits.pages == 0
            || self.limits.pages > i32::MAX as u32
            || self.limits.snapshots == 0
            || self.limits.snapshots > i32::MAX as u32 - 9
            || self.vision_features_per_request != 16384
            || self.max_new_tokens == 0
            || self.max_new_tokens > i32::MAX as u32
            || self.eos_token_ids.len() > 8
            || self
                .eos_token_ids
                .iter()
                .any(|id| *id as usize >= VOCABULARY)
        {
            return Err("invalid prepared Qwen4Exp model configuration".into());
        }
        for (name, dtype, shape) in required_buffers(
            self.limits.max_sequence_tokens,
            self.limits.prefill_chunk_size,
        ) {
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
            if kind == GraphKind::Decode && !self.speculative_widths.is_empty() {
                continue;
            }
            for batch in [1, 2, 4]
                .into_iter()
                .filter(|batch| *batch <= self.limits.max_active.next_power_of_two())
            {
                self.graph(manifest, kind, batch, 1, SamplingMode::General)?;
            }
        }
        for tokens in prefill_capacities(self.limits.prefill_chunk_size) {
            self.graph(
                manifest,
                GraphKind::Prefill,
                1,
                tokens,
                SamplingMode::General,
            )?;
        }
        for patches in [256, 1024, 4096, 16384] {
            self.graph(
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
        &self,
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
        if self.has_draft_cache()
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
        if kind == GraphKind::Speculative && !self.has_draft_cache() {
            for name in ["copy_grammar_mask", "copy_grammar_mask_count"] {
                if !graph.input_bindings.iter().any(|b| b.buffer_name == name) {
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

fn required_buffers(
    context: usize,
    prefill_chunk_size: usize,
) -> Vec<(&'static str, Dtype, Vec<usize>)> {
    use Dtype::{Float32 as F32, Int32 as I32, Int64 as I64};
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
            vec![4, VOCABULARY.div_ceil(32)],
        ),
        ("vision_token_indices", I32, vec![prefill_chunk_size]),
        ("vision_pixel_values", F32, vec![16384, 1536]),
        ("vision_interpolation_indices", I64, vec![16384, 4]),
        ("vision_interpolation_weights", F32, vec![16384, 4]),
        ("vision_position_ids", I64, vec![16384, 2]),
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
        ("admission_histogram", I32, vec![4, VOCABULARY]),
        (
            "admission_grammar_mask",
            I32,
            vec![4, VOCABULARY.div_ceil(32)],
        ),
        ("admission_grammar_enabled", I32, vec![4]),
        ("state_source_slots", I64, vec![4]),
        ("state_destination_slots", I64, vec![4]),
        ("state_operation", I32, vec![4]),
        ("state_qsa_slots", I64, vec![4]),
        ("state_accepted_tokens", I32, vec![4]),
        ("state_num_ops", I32, vec![1]),
    ]
}

fn required_inputs(kind: GraphKind) -> &'static [&'static str] {
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
            "state_source_slots",
            "state_destination_slots",
            "state_operation",
            "state_qsa_slots",
            "state_accepted_tokens",
            "state_num_ops",
        ],
        GraphKind::Snapshot => &[
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

#[cfg(feature = "server")]
pub mod qwen4exp;
#[cfg(feature = "server")]
pub use qwen4exp::output::{OutputConfig, QwenOutput as OutputProtocol, prepare_output};
