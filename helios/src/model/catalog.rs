//! Shared validation and selection for every prepared model's graph catalog.

use crate::{
    Result,
    execution::{
        BATCH_CAPACITIES, Dtype, ExecutionManifest, GraphKind, SPECULATIVE_ABI_VERSION,
        SamplingMode,
    },
    model::{ModelConfig, ModelContract},
    scheduler::speculation,
};

pub type BufferSpec = (&'static str, Dtype, Vec<usize>);

pub fn validate<M: ModelContract>(
    config: &ModelConfig,
    manifest: &ExecutionManifest,
) -> Result<()> {
    validate_inner::<M>(config, manifest).map_err(|error| match error {
        crate::Error::Execution(crate::ExecutionError::Manifest(_)) => error,
        _ => crate::ExecutionError::Manifest(crate::Failure::caused_by(error.to_string(), error))
            .into(),
    })
}

fn validate_inner<M: ModelContract>(
    config: &ModelConfig,
    manifest: &ExecutionManifest,
) -> Result<()> {
    manifest.validate()?;
    M::validate_config(config, manifest)?;
    config.sampling_defaults.validate()?;
    config.limits.validate_prefill()?;
    let page_tokens = M::scheduler_capabilities(config).page_tokens;
    if config.model_id.is_empty()
        || config.limits.max_active == 0
        || config.limits.max_active > manifest.max_active
        || config.limits.request_slots != manifest.request_capacity()
        || config.limits.max_sequence_tokens == 0
        || config.limits.max_sequence_tokens > config.prepared_sequence_tokens
        || !config
            .prepared_sequence_tokens
            .is_multiple_of(page_tokens.max(1))
        || page_tokens == 0
        || !config
            .limits
            .max_sequence_tokens
            .is_multiple_of(page_tokens)
        || config.limits.pages == 0
        || config.limits.pages > i32::MAX as u32
        || config.limits.snapshots > i32::MAX as u32 - (config.limits.request_capacity() + 1) as u32
        || config.max_new_tokens == 0
        || config.max_new_tokens > i32::MAX as u32
        || config.sampling_defaults.top_k > config.contract.vocabulary as i32
        || config.eos_token_ids.len() > 8
        || config
            .eos_token_ids
            .iter()
            .any(|id| *id as usize >= config.contract.vocabulary)
    {
        return Err("invalid prepared model configuration".into());
    }
    speculation::validate_widths(&config.speculative_widths)?;
    if config
        .speculative_fixed_width
        .is_some_and(|w| !config.speculative_widths.contains(&w))
    {
        return Err(
            "speculative_fixed_width must belong to the prepared speculative_widths".into(),
        );
    }
    if config.speculative_widths.is_empty() {
        if config.speculative_proposal_source == speculation::ProposalSource::Copy {
            return Err("copy proposal source requires speculative graphs".into());
        }
    } else if manifest.abi_version != SPECULATIVE_ABI_VERSION {
        return Err("speculative catalog requires ABI3".into());
    }
    let packed = manifest
        .graphs
        .iter()
        .filter(|g| g.kind == GraphKind::Mixed)
        .map(|g| g.packed_tokens())
        .max()
        .unwrap_or(0);
    for (name, dtype, mut shape) in M::required_buffers(config) {
        let packed_input = matches!(
            name,
            "token_ids"
                | "prefill_positions"
                | "vision_token_indices"
                | "vision_left"
                | "vision_right"
        );
        if packed_input {
            shape[0] = shape[0].max(packed);
        }
        let buffer = manifest.buffer(name)?;
        let valid_shape = if packed_input {
            buffer.shape.len() == shape.len()
                && buffer.shape[0] >= shape[0]
                && buffer.shape[1..] == shape[1..]
        } else {
            buffer.shape == shape
        };
        if buffer.dtype != dtype || !valid_shape {
            return Err(
                format!("invalid semantic buffer {name}: expected {dtype:?}{shape:?}").into(),
            );
        }
    }
    for graph in &manifest.graphs {
        select::<M>(
            config,
            manifest,
            graph.kind,
            graph.batch_capacity,
            graph.token_capacity,
            graph.prefill_capacity,
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
        if kind == GraphKind::Decode && !config.speculative_widths.is_empty() {
            continue;
        }
        for batch in batches(config) {
            select::<M>(config, manifest, kind, batch, 1, 0, SamplingMode::General)?;
        }
    }
    select::<M>(
        config,
        manifest,
        GraphKind::Prefill,
        1,
        config.limits.prefill_chunk_size,
        0,
        SamplingMode::General,
    )?;
    for &tokens in &config.contract.vision.patch_capacities {
        select::<M>(
            config,
            manifest,
            GraphKind::Vision,
            1,
            tokens,
            0,
            SamplingMode::General,
        )?;
    }
    if !config.speculative_widths.is_empty() {
        for batch in batches(config) {
            if !manifest
                .graphs
                .iter()
                .any(|graph| graph.kind == GraphKind::Speculative && graph.batch_capacity == batch)
            {
                return Err(
                    format!("prepared model lacks speculative graphs for batch {batch}").into(),
                );
            }
        }
    }
    Ok(())
}

fn speculative_shape(config: &ModelConfig, batch: usize, tokens: usize) -> bool {
    batches(config).any(|capacity| capacity == batch)
        && config.speculative_widths.contains(&tokens)
        && config
            .speculative_fixed_width
            .is_none_or(|width| width == tokens)
}

fn batches(config: &ModelConfig) -> impl Iterator<Item = usize> {
    BATCH_CAPACITIES
        .into_iter()
        .filter(|batch| *batch <= config.limits.max_active.next_power_of_two())
}

pub fn select<M: ModelContract>(
    config: &ModelConfig,
    manifest: &ExecutionManifest,
    kind: GraphKind,
    batch: usize,
    tokens: usize,
    prefill: usize,
    mode: SamplingMode,
) -> Result<usize> {
    let declared = batches(config).any(|capacity| capacity == batch)
        && match kind {
            GraphKind::Mixed => {
                batch >= 2
                    && prefill > 0
                    && prefill.is_multiple_of(64)
                    && prefill <= config.limits.prefill_chunk_size
                    && if config.speculative_widths.is_empty() {
                        tokens == 1
                    } else {
                        speculative_shape(config, batch, tokens)
                    }
            }
            GraphKind::Prefill => {
                batch == 1
                    && tokens > 0
                    && tokens <= config.limits.prefill_chunk_size
                    && tokens.is_multiple_of(64)
            }
            GraphKind::Vision => {
                batch == 1 && config.contract.vision.patch_capacities.contains(&tokens)
            }
            GraphKind::Speculative => speculative_shape(config, batch, tokens),
            _ => tokens == 1,
        };
    if !declared || (kind != GraphKind::Mixed && prefill != 0) {
        return Err(format!(
            "undeclared prepared graph kind or capacity: {kind:?}, batch {batch}, tokens {tokens}"
        )
        .into());
    }
    let mut matches = manifest.graphs.iter().enumerate().filter(|(_, graph)| {
        graph.kind == kind
            && graph.batch_capacity == batch
            && graph.token_capacity == tokens
            && graph.prefill_capacity == prefill
            && graph.sampling_mode == mode
    });
    let (index, graph) = matches.next().ok_or_else(|| {
        format!("prepared model lacks {kind:?}/{mode:?} graph with batch {batch}, tokens {tokens}")
    })?;
    if matches.next().is_some() {
        return Err("ambiguous prepared model graph capacity".into());
    }
    for name in M::required_inputs(config, kind) {
        if !graph
            .input_bindings
            .iter()
            .any(|binding| binding.buffer_name == name)
        {
            return Err(
                format!("prepared graph {} lacks required input {name}", graph.name).into(),
            );
        }
    }
    Ok(index)
}
