use super::*;
use super::{
    config::required_buffers,
    inputs::{Admission, Inputs},
};
use crate::execution::SamplingMode;
use crate::model::catalog::{select, validate};
use crate::{
    cache::Fence,
    execution::{ABI_VERSION, BufferDescriptor, CompletionRecord, GraphDescriptor, InputBinding},
    scheduler::{RequestScheduler, RequestSpec},
};
use crate::{
    execution::Dtype,
    model::SamplingParameters,
    scheduler::{SchedulerLimits, prefill_capacities},
};

fn config() -> ModelConfig {
    ModelConfig {
        contract: Qwen4Exp::metadata(),
        identity: [1; 32],
        model_id: "qwen38".into(),
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 2048,
            prefill_chunk_size: 512,
            pages: 64,
            snapshots: 40,
        },
        eos_token_ids: vec![248046, 248044],
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 256,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        history_drafts: true,
        speculative_proposal_source: Default::default(),
    }
}
fn manifest(config: &ModelConfig) -> ExecutionManifest {
    let mut descriptions = required_buffers(config);
    let speculative = !config.speculative_widths.is_empty();
    descriptions.extend([
        ("frame_index", Dtype::Int32, vec![1]),
        ("submission_sequence", Dtype::Int64, vec![1]),
        (
            "completion_records",
            Dtype::Int64,
            vec![2, 4, if speculative { 24 } else { 8 }],
        ),
    ]);
    let mut address = 0x10000;
    let mut host = 0;
    let mut bindings = Vec::new();
    let buffers = descriptions
        .into_iter()
        .map(|(name, dtype, shape)| {
            let nbytes = shape.iter().product::<usize>() * dtype.size();
            let mut stride = 1;
            let mut strides = Vec::new();
            for extent in shape.iter().rev() {
                strides.push(stride);
                stride *= extent;
            }
            strides.reverse();
            let buffer = BufferDescriptor {
                name: name.into(),
                address,
                nbytes,
                dtype,
                shape,
                strides,
            };
            address += (nbytes + 7) & !7;
            if name != "completion_records" {
                bindings.push(InputBinding {
                    buffer_name: name.into(),
                    host_offset: host,
                    nbytes,
                });
                host += (nbytes + 7) & !7;
            }
            buffer
        })
        .collect();
    let mut graphs = Vec::new();
    for kind in [
        GraphKind::Init,
        GraphKind::Restore,
        GraphKind::Snapshot,
        GraphKind::Decode,
        GraphKind::Resume,
        GraphKind::Prefill,
        GraphKind::Vision,
        GraphKind::Speculative,
    ] {
        if kind == GraphKind::Speculative && !speculative {
            continue;
        }
        if kind == GraphKind::Decode && speculative {
            continue;
        }
        let capacities = match kind {
            GraphKind::Speculative => config
                .speculative_widths
                .iter()
                .flat_map(|&width| {
                    [1usize, 2, 4]
                        .into_iter()
                        .filter(move |batch| {
                            crate::scheduler::speculation::graph_width_masks(
                                &config.speculative_widths,
                                config.speculative_fixed_width,
                                config.limits.request_capacity(),
                            )[batch.trailing_zeros() as usize]
                                & (1 << (width - 1))
                                != 0
                        })
                        .map(move |batch| (batch, width))
                })
                .collect(),
            GraphKind::Prefill => prefill_capacities(config.limits.prefill_chunk_size)
                .map(|tokens| (1, tokens))
                .collect(),
            GraphKind::Vision => vec![(1, 256), (1, 1024), (1, 4096), (1, 16384)],
            _ => vec![(1, 1), (2, 1), (4, 1)],
        };
        for (batch, tokens) in capacities {
            let modes = &[SamplingMode::General];
            for &sampling_mode in modes {
                graphs.push(GraphDescriptor {
                    name: format!("{kind:?}-{batch}-{tokens}-{sampling_mode:?}"),
                    kind,
                    sampling_mode,
                    handle: graphs.len() + 1,
                    batch_capacity: batch,
                    token_capacity: tokens,
                    input_bindings: bindings.clone(),
                });
            }
        }
    }
    ExecutionManifest {
        engram: None,
        cache: None,
        offload: None,
        abi_version: if speculative {
            crate::execution::SPECULATIVE_ABI_VERSION
        } else {
            ABI_VERSION
        },
        context: 1,
        device: 0,
        max_active: 4,
        input_frame_bytes: host,
        buffers,
        graphs,
    }
}

#[test]
fn general_sampler_graphs_are_required_and_duplicate_variants_reject() {
    let config = config();
    let original = manifest(&config);
    assert_eq!(original.graphs.len(), 22);
    for kind in [GraphKind::Prefill, GraphKind::Decode, GraphKind::Resume] {
        {
            let mode = SamplingMode::General;
            let mut missing = original.clone();
            let index = missing
                .graphs
                .iter()
                .position(|graph| {
                    graph.kind == kind
                        && graph.sampling_mode == mode
                        && (kind != GraphKind::Prefill
                            || graph.token_capacity == config.limits.prefill_chunk_size)
                })
                .unwrap();
            let removed = missing.graphs.remove(index);
            assert!(matches!(
                validate::<Qwen4Exp>(&config, &missing).unwrap_err(),
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ));
            let selected = select::<Qwen4Exp>(
                &config,
                &original,
                kind,
                removed.batch_capacity,
                removed.token_capacity,
                mode,
            )
            .unwrap();
            assert_eq!(original.graphs[selected].name, removed.name);
            let mut duplicate = original.clone();
            let mut extra = removed;
            extra.name.push_str("-duplicate");
            extra.handle = duplicate.graphs.len() + 1;
            duplicate.graphs.push(extra);
            assert!(matches!(
                validate::<Qwen4Exp>(&config, &duplicate).unwrap_err(),
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ));
        }
    }
}

#[test]
fn shared_admission_enforces_slot_capacity_and_model_metadata() {
    let mut config = config();
    config.limits.max_active = 1;
    let mut admission = Admission {
        handle: crate::scheduler::RequestHandle {
            slot: 1,
            generation: 1,
        },
        sampling: SamplingParameters::default(),
        max_new_tokens: 16,
        prompt: vec![7],
        data: PromptData {
            positions: vec![[0; 3]],
            vision_token_indices: vec![-1],
        },
        grammar_mask: None,
        ignore_eos: false,
    };
    assert!(admission.validate(&config).is_err());
    admission.handle.slot = 0;
    admission.validate(&config).unwrap();
    admission.data.positions.clear();
    assert!(admission.validate(&config).is_err());
    admission.data.positions.push([0; 3]);
    admission.grammar_mask = Some(vec![-1]);
    assert!(admission.validate(&config).is_err());
    admission.grammar_mask = Some(vec![-1; VOCABULARY.div_ceil(32)]);
    admission.validate(&config).unwrap();
}

#[test]
fn optional_decode_graphs_in_speculative_catalogs_obey_the_shared_contract() {
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    let mut prepared = manifest(&config);
    let mut decode = prepared
        .graphs
        .iter()
        .find(|g| g.kind == GraphKind::Resume)
        .unwrap()
        .clone();
    decode.kind = GraphKind::Decode;
    decode.name = "optional-decode".into();
    decode.handle = prepared.graphs.len() + 1;
    prepared.graphs.push(decode);
    validate::<Qwen4Exp>(&config, &prepared).unwrap();
    prepared
        .graphs
        .last_mut()
        .unwrap()
        .input_bindings
        .retain(|b| b.buffer_name != "sampling_grammar_mask");
    assert!(matches!(
        validate::<Qwen4Exp>(&config, &prepared),
        Err(crate::Error::Execution(crate::ExecutionError::Manifest(_)))
    ));
}

#[test]
fn speculative_catalog_requires_each_batch_and_shifted_page_binding() {
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    let original = manifest(&config);
    validate::<Qwen4Exp>(&config, &original).unwrap();
    assert_eq!(original.graphs.len(), 43);
    assert!(
        original
            .graphs
            .iter()
            .filter(|g| g.kind == GraphKind::Speculative)
            .all(|g| g.token_capacity <= 8)
    );
    assert!(!original.graphs.iter().any(|g| g.kind == GraphKind::Decode));
    for batch in [1, 2, 4] {
        let mut missing = original.clone();
        missing
            .graphs
            .retain(|g| g.kind != GraphKind::Speculative || g.batch_capacity != batch);
        assert!(matches!(
            validate::<Qwen4Exp>(&config, &missing).unwrap_err(),
            crate::Error::Execution(crate::ExecutionError::Manifest(_))
        ));
    }
    let mut subset = original.clone();
    subset
        .graphs
        .retain(|g| g.kind != GraphKind::Speculative || g.token_capacity <= 3);
    validate::<Qwen4Exp>(&config, &subset).unwrap();
    assert!(
        select::<Qwen4Exp>(
            &config,
            &subset,
            GraphKind::Speculative,
            1,
            4,
            SamplingMode::General
        )
        .is_err()
    );
    for kind in [
        GraphKind::Prefill,
        GraphKind::Resume,
        GraphKind::Speculative,
    ] {
        let mut missing = original.clone();
        missing
            .graphs
            .iter_mut()
            .find(|g| g.kind == kind)
            .unwrap()
            .input_bindings
            .retain(|i| i.buffer_name != "draft_page_tables");
        assert!(matches!(
            validate::<Qwen4Exp>(&config, &missing).unwrap_err(),
            crate::Error::Execution(crate::ExecutionError::Manifest(_))
        ));
    }
    for name in [
        "copy_token_ids",
        "copy_token_count",
        "copy_expected_consumed",
        "copy_generation",
        "draft_page_tables",
    ] {
        let mut wrong = original.clone();
        wrong
            .buffers
            .iter_mut()
            .find(|b| b.name == name)
            .unwrap()
            .dtype = Dtype::Float32;
        assert!(validate::<Qwen4Exp>(&config, &wrong).is_err());
    }
    config.speculative_fixed_width = Some(9);
    assert!(matches!(
        validate::<Qwen4Exp>(&config, &original).unwrap_err(),
        crate::Error::Execution(crate::ExecutionError::Manifest(_))
    ));
    config.speculative_fixed_width = Some(4);
    let fixed = manifest(&config);
    validate::<Qwen4Exp>(&config, &fixed).unwrap();
    let shapes: Vec<_> = fixed
        .graphs
        .iter()
        .filter(|g| g.kind == GraphKind::Speculative)
        .map(|g| (g.batch_capacity, g.token_capacity))
        .collect();
    assert_eq!(shapes, [(1, 4), (2, 4), (4, 4)]);
    assert!(validate::<Qwen4Exp>(&config, &original).is_err());
}

#[test]
#[cfg(feature = "server")]
fn speculative_bootstrap_options_validate_before_assets_or_cuda() {
    use crate::server::Bootstrap;
    use serde_json::json;
    let mut config = config();
    config.speculative_widths = vec![1, 2, 4, 8];
    let adaptive = manifest(&config);
    let body = json!({"model_contract":metadata_json(),"model_id":"qwen38","identity":"01".repeat(32),
        "max_sequence_tokens":2048,"prefill_chunk_size":512,"pages":64,"snapshots":8,
        "eos_token_ids":[],"speculative_widths":[1,2,4,8]});
    for width in [json!(null), json!(1), json!(2), json!(4), json!(8)] {
        for copy in [true, false] {
            config.speculative_fixed_width = width.as_u64().map(|w| w as usize);
            let prepared = manifest(&config);
            let mut body = body.clone();
            body["speculative_fixed_width"] = width.clone();
            body["history_drafts"] = json!(copy);
            assert_eq!(
                Bootstrap::load::<Qwen4Exp>(&body.to_string(), &prepared)
                    .err()
                    .unwrap()
                    .to_string(),
                "assets must contain verified frontend descriptors"
            );
        }
    }
    for (key, value) in [
        ("speculative_fixed_width", json!(3)),
        ("speculative_fixed_width", json!("8")),
        ("history_drafts", json!(null)),
        ("history_drafts", json!(1)),
        ("speculative_widths", json!([1, 8, 4])),
        ("speculative_widths", json!([2, 4, 8])),
        ("speculative_proposal_source", json!(null)),
        ("speculative_proposal_source", json!("auto")),
    ] {
        let mut body = body.clone();
        body[key] = value;
        let error = Bootstrap::load::<Qwen4Exp>(&body.to_string(), &adaptive)
            .err()
            .unwrap();
        assert!(
            matches!(
                error,
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ),
            "{error}"
        );
    }
    config.speculative_fixed_width = None;
    config.speculative_proposal_source = crate::scheduler::speculation::ProposalSource::Copy;
    let copy_manifest = self::manifest(&config);
    let mut body = body;
    body["speculative_proposal_source"] = json!("copy");
    assert_eq!(
        Bootstrap::load::<Qwen4Exp>(&body.to_string(), &copy_manifest)
            .err()
            .unwrap()
            .to_string(),
        "assets must contain verified frontend descriptors"
    );
}

#[test]
fn mtp_copy_uploads_cover_zero_and_default_temperature_with_epoch_and_consumed_guards() {
    let mut config = config();
    config.speculative_widths = vec![1, 2, 4, 8];
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        Qwen4Exp::scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .enable_speculation(&config.speculative_widths)
        .unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let prompt = vec![7; 65];
    let handle = scheduler
        .admit(RequestSpec {
            system_boundary: None,
            prompt: prompt.clone(),
            conditioning: vec![],
            max_new_tokens: 16,
            grammar_constrained: false,
            host_stops: false,
        })
        .unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[0] = Some(Admission {
        handle,
        sampling: SamplingParameters {
            temperature: 0.,
            ..Default::default()
        },
        max_new_tokens: 16,
        prompt,
        data: PromptData {
            positions: (0..65).map(|p| [p; 3]).collect(),
            vision_token_indices: vec![-1; 65],
        },
        grammar_mask: None,
        ignore_eos: false,
    });
    let mut inputs = Inputs::new();
    let mut records = [CompletionRecord::default(); 4];
    for (fence, consumed, generated, token) in [(1, 0, 0, -1), (2, 65, 1, 7)] {
        let step = scheduler.next_step().unwrap();
        scheduler.record_submission(&step, Fence(fence)).unwrap();
        crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
            .unwrap();
        if fence == 2 {
            let uploads = inputs.uploads();
            let target = uploads.iter().find(|u| u.buffer == "page_tables").unwrap();
            let draft = uploads
                .iter()
                .find(|u| u.buffer == "draft_page_tables")
                .unwrap();
            assert_eq!(target.bytes.len(), 12);
            assert_eq!(draft.bytes, &target.bytes[4..]);
        }
        records[0] = CompletionRecord {
            generation: handle.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: consumed,
            generated_tokens: generated,
            token_id: token,
            status: 1,
            ..Default::default()
        };
        scheduler.complete(Fence(fence), &records).unwrap();
    }
    retain_prompt(&mut scheduler, 3);
    for fence in [4, 5] {
        admissions[0].as_mut().unwrap().sampling.temperature = if fence == 4 { 0. } else { 1. };
        let step = scheduler.next_step().unwrap();
        assert_eq!(step.token_capacity, 8);
        scheduler.record_submission(&step, Fence(fence)).unwrap();
        crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
            .unwrap();
        let uploads = inputs.uploads();
        assert!(!uploads.iter().any(|u| u.buffer.ends_with("page_tables")));
        let expected = uploads
            .iter()
            .find(|u| u.buffer == "copy_expected_consumed")
            .unwrap();
        assert_eq!(
            i64::from_le_bytes(expected.bytes[..8].try_into().unwrap()),
            65 + fence as i64 - 4
        );
        let count = uploads
            .iter()
            .find(|u| u.buffer == "copy_token_count")
            .unwrap();
        assert!(i32::from_le_bytes(count.bytes[..4].try_into().unwrap()) > 0);
        let generation = uploads
            .iter()
            .find(|u| u.buffer == "copy_generation")
            .unwrap();
        assert_eq!(
            i64::from_le_bytes(generation.bytes[..8].try_into().unwrap()),
            handle.generation as i64
        );
        assert!(generation.bytes[8..].iter().all(|byte| *byte == 0));
        config.history_drafts = false;
        crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
            .unwrap();
        let uploads = inputs.uploads();
        assert!(
            uploads
                .iter()
                .find(|u| u.buffer == "copy_token_count")
                .unwrap()
                .bytes
                .iter()
                .all(|b| *b == 0)
        );
        assert!(
            uploads
                .iter()
                .find(|u| u.buffer == "copy_generation")
                .unwrap()
                .bytes
                .iter()
                .all(|b| *b == 0)
        );
        config.history_drafts = true;
        assert!(scheduler.next_step().is_none());
        let mut vector = crate::execution::TokenVector {
            emitted_count: 1,
            offered_drafts: 7,
            accepted_drafts: 0,
            draft_cache_consumed: 65 + fence as i64 - 4,
            verification_width: 8,
            ..Default::default()
        };
        vector.tokens[0] = 7;
        records[0] = CompletionRecord {
            generation: handle.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: 66 + fence as i64 - 4,
            generated_tokens: fence as i64 - 2,
            token_id: 7,
            status: 1,
            vector: Some(vector),
            ..Default::default()
        };
        scheduler.complete(Fence(fence), &records).unwrap();
    }
}

#[test]
fn missing_semantic_inputs_and_graph_capacities_reject_before_cuda() {
    let config = config();
    let manifest = manifest(&config);
    validate::<Qwen4Exp>(&config, &manifest).unwrap();
    let mut missing = manifest.clone();
    missing
        .graphs
        .retain(|graph| graph.kind != GraphKind::Resume);
    assert!(matches!(
        validate::<Qwen4Exp>(&config, &missing).unwrap_err(),
        crate::Error::Execution(crate::ExecutionError::Manifest(_))
    ));
    let mut missing = manifest.clone();
    let graph = missing
        .graphs
        .iter_mut()
        .find(|graph| graph.kind == GraphKind::Prefill)
        .unwrap();
    graph
        .input_bindings
        .retain(|input| input.buffer_name != "vision_token_indices");
    assert!(matches!(
        validate::<Qwen4Exp>(&config, &missing).unwrap_err(),
        crate::Error::Execution(crate::ExecutionError::Manifest(_))
    ));
    let mut wrong = manifest;
    wrong
        .buffers
        .iter_mut()
        .find(|buffer| buffer.name == "admission_seed_bits")
        .unwrap()
        .dtype = Dtype::Float32;
    assert!(validate::<Qwen4Exp>(&config, &wrong).is_err());
}

#[test]
fn configured_prefill_catalog_and_backing_extents_are_required_before_cuda() {
    for chunk_size in [64, 3072, 4096, 8192] {
        let mut config = config();
        config.limits.max_sequence_tokens = 32768;
        config.limits.prefill_chunk_size = chunk_size;
        let original = manifest(&config);
        validate::<Qwen4Exp>(&config, &original).unwrap();
        let mut maximum_only = original.clone();
        maximum_only
            .graphs
            .retain(|graph| graph.kind != GraphKind::Prefill || graph.token_capacity == chunk_size);
        validate::<Qwen4Exp>(&config, &maximum_only).unwrap();
        {
            let mode = SamplingMode::General;
            let mut missing = original.clone();
            missing.graphs.retain(|graph| {
                !(graph.kind == GraphKind::Prefill
                    && graph.sampling_mode == mode
                    && graph.token_capacity == chunk_size)
            });
            let error = validate::<Qwen4Exp>(&config, &missing).unwrap_err();
            assert!(matches!(
                error,
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ));
            assert_eq!(
                error.to_string(),
                format!(
                    "prepared model lacks Prefill/General graph with batch 1, tokens {chunk_size}"
                )
            );
        }
        for name in ["token_ids", "prefill_positions", "vision_token_indices"] {
            let mut short = original.clone();
            let buffer = short
                .buffers
                .iter_mut()
                .find(|buffer| buffer.name == name)
                .unwrap();
            buffer.shape[0] /= 2;
            buffer.nbytes /= 2;
            for graph in &mut short.graphs {
                graph
                    .input_bindings
                    .iter_mut()
                    .find(|input| input.buffer_name == name)
                    .unwrap()
                    .nbytes /= 2;
            }
            assert!(matches!(
                validate::<Qwen4Exp>(&config, &short).unwrap_err(),
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ));
        }
    }
}

#[test]
#[cfg(feature = "server")]
fn bootstrap_defaults_to_4096_and_explicit_chunk_size_wins_before_asset_loading() {
    use crate::server::Bootstrap;
    use serde_json::json;

    let mut config = config();
    config.limits.max_sequence_tokens = 8192;
    config.limits.prefill_chunk_size = 4096;
    let mut body = json!({"model_contract": metadata_json(),"model_id": "qwen38", "identity": "01".repeat(32),
        "max_sequence_tokens": 8192, "pages": 64, "snapshots": 8, "eos_token_ids": []});
    let failure = |body: &serde_json::Value, config: &ModelConfig| {
        Bootstrap::load::<Qwen4Exp>(&body.to_string(), &manifest(config))
            .err()
            .unwrap()
    };
    assert_eq!(
        failure(&body, &config).to_string(),
        "assets must contain verified frontend descriptors"
    );
    config.limits.prefill_chunk_size = 512;
    assert!(matches!(
        failure(&body, &config),
        crate::Error::Execution(crate::ExecutionError::Manifest(_))
    ));
    body["prefill_chunk_size"] = json!(512);
    assert_eq!(
        failure(&body, &config).to_string(),
        "assets must contain verified frontend descriptors"
    );
    for invalid in [
        json!(0),
        json!(63),
        json!(65),
        json!(8193),
        json!("4096"),
        json!(null),
    ] {
        body["prefill_chunk_size"] = invalid;
        assert!(matches!(
            failure(&body, &config),
            crate::Error::Execution(crate::ExecutionError::Manifest(_))
        ));
    }
}

#[test]
fn admission_histogram_seed_and_partial_page_uploads_preserve_native_semantics() {
    let config = config();
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        Qwen4Exp::scheduler_capabilities(&config),
    )
    .unwrap();
    let handle = scheduler
        .admit(RequestSpec {
            system_boundary: None,
            prompt: vec![7; 65],
            conditioning: vec![],
            max_new_tokens: 4,
            grammar_constrained: false,
            host_stops: false,
        })
        .unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[handle.slot as usize] = Some(Admission {
        handle,
        sampling: SamplingParameters {
            seed: u64::MAX,
            temperature: 0.,
            ..Default::default()
        },
        max_new_tokens: 4,
        prompt: vec![7; 65],
        data: PromptData {
            positions: (0..65).map(|p| [p; 3]).collect(),
            vision_token_indices: vec![-1; 65],
        },
        grammar_mask: None,
        ignore_eos: false,
    });
    admissions[0].as_ref().unwrap().validate(&config).unwrap();
    let mut inputs = Inputs::new();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(1)).unwrap();
    crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    assert_state_bindings(&inputs, &scheduler, &step);
    let uploads = inputs.uploads();
    let seed = uploads
        .iter()
        .find(|u| u.buffer == "admission_seed_bits")
        .unwrap();
    assert_eq!(seed.bytes, &[255; 8]);
    let histogram = uploads
        .iter()
        .find(|u| u.buffer == "admission_histogram")
        .unwrap();
    assert_eq!(
        i32::from_le_bytes(histogram.bytes[28..32].try_into().unwrap()),
        65
    );
    for ignore_eos in [false, true, false] {
        admissions[0].as_mut().unwrap().ignore_eos = ignore_eos;
        crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
            .unwrap();
        assert_state_bindings(&inputs, &scheduler, &step);
        let uploads = inputs.uploads();
        let count = uploads
            .iter()
            .find(|upload| upload.buffer == "admission_eos_count")
            .unwrap();
        assert_eq!(
            i32::from_le_bytes(count.bytes.try_into().unwrap()),
            if ignore_eos { 0 } else { 2 }
        );
        let ids = uploads
            .iter()
            .find(|upload| upload.buffer == "admission_eos_token_ids")
            .unwrap();
        assert_eq!(
            i64::from_le_bytes(ids.bytes[..8].try_into().unwrap()),
            248046
        );
        assert_eq!(config.eos_token_ids, [248046, 248044]);
    }
    let mut initialized = [CompletionRecord::default(); 4];
    initialized[0] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 1,
        status: 1,
        token_id: -1,
        ..Default::default()
    };
    scheduler.complete(Fence(1), &initialized).unwrap();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    assert_state_bindings(&inputs, &scheduler, &step);
    assert_eq!(
        inputs
            .uploads()
            .iter()
            .find(|u| u.buffer == "page_tables")
            .unwrap()
            .bytes
            .len(),
        8
    );
    let mut records = [CompletionRecord::default(); 4];
    records[0] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 2,
        consumed_tokens: 65,
        generated_tokens: 1,
        token_id: 9,
        status: 1,
        ..Default::default()
    };
    scheduler.complete(Fence(2), &records).unwrap();
    retain_prompt(&mut scheduler, 3);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(4)).unwrap();
    crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    assert_state_bindings(&inputs, &scheduler, &step);
    assert!(!inputs.uploads().iter().any(|u| u.buffer == "page_tables"));
    assert!(
        !inputs
            .uploads()
            .iter()
            .any(|u| u.buffer == "token_ids" || u.buffer == "admission_histogram")
    );
}

#[test]
fn large_prefill_uploads_preserve_token_position_and_image_indices_without_tail_padding() {
    let mut config = config();
    config.limits.max_sequence_tokens = 8192;
    config.limits.prefill_chunk_size = 4096;
    config.limits.pages = 128;
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        Qwen4Exp::scheduler_capabilities(&config),
    )
    .unwrap();
    let prompt: Vec<u32> = (0..4113).collect();
    let handle = scheduler
        .admit(RequestSpec {
            system_boundary: None,
            prompt: prompt.clone(),
            conditioning: vec![],
            max_new_tokens: 4,
            grammar_constrained: false,
            host_stops: false,
        })
        .unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[handle.slot as usize] = Some(Admission {
        handle,
        sampling: SamplingParameters::default(),
        max_new_tokens: 4,
        data: PromptData {
            positions: prompt.iter().map(|&token| [token as i64; 3]).collect(),
            vision_token_indices: prompt
                .iter()
                .map(|&token| if token % 2 == 0 { token as i32 } else { -1 })
                .collect(),
        },
        prompt,
        grammar_mask: None,
        ignore_eos: false,
    });
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(1)).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 1,
        status: 1,
        token_id: -1,
        ..Default::default()
    };
    scheduler.complete(Fence(1), &records).unwrap();
    let mut inputs = Inputs::new();
    for (sequence, start, count, capacity) in [(2, 0, 4096, 4096), (3, 4096, 17, 128)] {
        let step = scheduler.next_step().unwrap();
        assert_eq!(
            (step.lanes[0].tokens, step.token_capacity),
            (count, capacity)
        );
        scheduler.record_submission(&step, Fence(sequence)).unwrap();
        crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
            .unwrap();
        assert_state_bindings(&inputs, &scheduler, &step);
        let uploads = inputs.uploads();
        let bytes = |name| {
            uploads
                .iter()
                .find(|upload| upload.buffer == name)
                .unwrap()
                .bytes
        };
        let tokens: Vec<u8> = (start..start + count)
            .flat_map(|id| (id as i64).to_le_bytes())
            .collect();
        let positions: Vec<u8> = (start..start + count)
            .flat_map(|id| [id as i64; 3])
            .flat_map(i64::to_le_bytes)
            .collect();
        let image_indices: Vec<u8> = (start..start + count)
            .map(|id| if id % 2 == 0 { id as i32 } else { -1 })
            .flat_map(i32::to_le_bytes)
            .collect();
        assert_eq!(bytes("token_ids"), tokens);
        assert_eq!(bytes("prefill_positions"), positions);
        assert_eq!(bytes("vision_token_indices"), image_indices);
        records[handle.slot as usize] = CompletionRecord {
            generation: handle.generation as i64,
            submission_sequence: sequence as i64,
            consumed_tokens: (start + count) as i64,
            generated_tokens: i64::from(sequence == 3),
            token_id: if sequence == 3 { 77 } else { -1 },
            status: 1,
            ..Default::default()
        };
        scheduler.complete(Fence(sequence), &records).unwrap();
    }
}

#[test]
fn admission_rejects_nonfinite_parameters_and_inconsistent_positions() {
    assert!(
        SamplingParameters {
            temperature: f32::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        SamplingParameters {
            top_p: f32::from_bits(1),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        SamplingParameters {
            temperature: 0.,
            top_k: -1,
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn copy_catalog_requires_masks_and_epoch_but_no_draft_cache_bindings() {
    use crate::scheduler::speculation::ProposalSource;
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    config.speculative_proposal_source = ProposalSource::Copy;
    let original = manifest(&config);
    validate::<Qwen4Exp>(&config, &original).unwrap();
    assert!(
        !original
            .buffers
            .iter()
            .any(|b| b.name == "draft_page_tables")
    );
    for name in [
        "copy_generation",
        "copy_grammar_mask",
        "copy_grammar_mask_count",
    ] {
        let mut missing = original.clone();
        missing
            .graphs
            .iter_mut()
            .find(|g| g.kind == GraphKind::Speculative)
            .unwrap()
            .input_bindings
            .retain(|b| b.buffer_name != name);
        assert!(matches!(
            validate::<Qwen4Exp>(&config, &missing).unwrap_err(),
            crate::Error::Execution(crate::ExecutionError::Manifest(_))
        ));
        let mut wrong = original.clone();
        wrong
            .buffers
            .iter_mut()
            .find(|b| b.name == name)
            .unwrap()
            .dtype = Dtype::Float32;
        assert!(validate::<Qwen4Exp>(&config, &wrong).is_err());
    }
    config.speculative_widths.clear();
    assert!(matches!(
        validate::<Qwen4Exp>(&config, &original).unwrap_err(),
        crate::Error::Execution(crate::ExecutionError::Manifest(_))
    ));
    assert!(ProposalSource::parse("auto").is_err());
}

#[test]
fn copy_staging_uses_physical_slots_stochastic_defaults_and_bonus_mask() {
    use crate::scheduler::speculation::{CopyProposal, ProposalSource};
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    config.speculative_fixed_width = Some(8);
    config.speculative_proposal_source = ProposalSource::Copy;
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        Qwen4Exp::scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .enable_speculation(&config.speculative_widths)
        .unwrap();
    scheduler.proposal_source(ProposalSource::Copy).unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let handles = (0..3)
        .map(|slot| {
            scheduler
                .admit(RequestSpec {
                    system_boundary: None,
                    prompt: vec![7; 65],
                    conditioning: vec![],
                    max_new_tokens: 16,
                    grammar_constrained: slot == 2,
                    host_stops: false,
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(1)).unwrap();
    let mut completed = [CompletionRecord::default(); 4];
    for h in &handles {
        completed[h.slot as usize] = CompletionRecord {
            generation: h.generation as i64,
            submission_sequence: 1,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
    }
    scheduler.complete(Fence(1), &completed).unwrap();
    for h in &handles[..2] {
        scheduler.cancel(*h).unwrap();
    }
    let handle = handles[2];
    assert_eq!(handle.slot, 2);
    let words = VOCABULARY.div_ceil(32);
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[2] = Some(Admission {
        handle,
        sampling: SamplingParameters::default(),
        max_new_tokens: 16,
        prompt: vec![7; 65],
        data: PromptData {
            positions: (0..65).map(|p| [p; 3]).collect(),
            vision_token_indices: vec![-1; 65],
        },
        grammar_mask: Some(vec![1; words]),
        ignore_eos: false,
    });
    let mut inputs = Inputs::new();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    assert!(
        !inputs
            .uploads()
            .iter()
            .any(|u| u.buffer == "draft_page_tables")
    );
    completed.fill(CompletionRecord::default());
    completed[2] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 2,
        consumed_tokens: 65,
        generated_tokens: 1,
        token_id: 7,
        status: 1,
        ..Default::default()
    };
    scheduler.complete(Fence(2), &completed).unwrap();
    retain_prompt(&mut scheduler, 3);
    let masks = [vec![1; words], vec![2; words], vec![4; words]].concat();
    scheduler
        .set_copy_proposal(
            handle,
            CopyProposal {
                consumed: 65,
                tokens: [8, 9, -1, -1, -1, -1, -1],
                count: 2,
                grammar_masks: masks.clone(),
            },
        )
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.token_capacity, 8);
    assert_eq!(step.lanes[0].tokens, 3);
    scheduler.record_submission(&step, Fence(4)).unwrap();
    crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    let uploads = inputs.uploads();
    let bytes = |name| uploads.iter().find(|u| u.buffer == name).unwrap().bytes;
    let i64s = |name| {
        bytes(name)
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let i32s = |name| {
        bytes(name)
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    assert_eq!(i64s("copy_generation"), [0, 0, handle.generation as i64, 0]);
    assert_eq!(i64s("copy_expected_consumed"), [-1, -1, 65, -1]);
    assert_eq!(i32s("query_lengths"), [3, 0, 0, 0]);
    assert_eq!(i32s("copy_token_count"), [0, 0, 2, 0]);
    assert_eq!(i32s("copy_grammar_mask_count"), [0, 0, 3, 0]);
    assert_eq!(&i64s("copy_token_ids")[14..21], &[8, 9, -1, -1, -1, -1, -1]);
    assert_eq!(i32s("copy_grammar_mask"), masks);
    assert_eq!(
        uploads
            .iter()
            .find(|u| u.buffer == "copy_grammar_mask")
            .unwrap()
            .offset,
        2 * 8 * words * 4
    );
    assert!(!uploads.iter().any(|u| u.buffer == "draft_page_tables"));
    completed[2] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 4,
        consumed_tokens: 66,
        generated_tokens: 2,
        token_id: 10,
        status: 1,
        vector: Some(crate::execution::TokenVector {
            emitted_count: 1,
            offered_drafts: 2,
            accepted_drafts: 0,
            copy_offered_mask: 3,
            draft_cache_consumed: -1,
            verification_width: 8,
            tokens: [10, -1, -1, -1, -1, -1, -1, -1],
            ..Default::default()
        }),
        ..Default::default()
    };
    scheduler.complete(Fence(4), &completed).unwrap();
    scheduler
        .set_copy_proposal(
            handle,
            CopyProposal {
                consumed: 66,
                tokens: [-1; 7],
                count: 0,
                grammar_masks: vec![16; words],
            },
        )
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.token_capacity, 8);
    assert_eq!(step.lanes[0].tokens, 1);
    scheduler.record_submission(&step, Fence(5)).unwrap();
    crate::model::pack_inputs::<Qwen4Exp>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    let uploads = inputs.uploads();
    let counts = uploads
        .iter()
        .find(|u| u.buffer == "copy_grammar_mask_count")
        .unwrap()
        .bytes;
    assert_eq!(i32::from_le_bytes(counts[8..12].try_into().unwrap()), 1);
    let mask = uploads
        .iter()
        .find(|u| u.buffer == "copy_grammar_mask")
        .unwrap();
    assert_eq!(mask.bytes.len(), words * 4);
    assert!(
        mask.bytes
            .chunks_exact(4)
            .all(|b| i32::from_le_bytes(b.try_into().unwrap()) == 16)
    );
}

#[cfg(feature = "server")]
fn metadata_json() -> serde_json::Value {
    serde_json::json!({"schema":super::INPUT_SCHEMA,"vocabulary":248320,"vision":{"features_per_request":16384,"max_images":4,"patches_per_feature":4,"patch_capacities":[256,1024,4096,16384]}})
}

#[test]
#[cfg(feature = "server")]
fn closed_metadata_rejects_unknown_or_mismatched_contract_before_assets() {
    use crate::server::Bootstrap;
    use serde_json::json;
    let config = config();
    let manifest = manifest(&config);
    let body = json!({"model_id":"qwen38","identity":"01".repeat(32),
        "max_sequence_tokens":2048,"prefill_chunk_size":512,"pages":64,"snapshots":8,
        "eos_token_ids":[],"model_contract":metadata_json()});
    assert_eq!(
        Bootstrap::load::<Qwen4Exp>(&body.to_string(), &manifest)
            .err()
            .unwrap()
            .to_string(),
        "assets must contain verified frontend descriptors"
    );
    let mut cases = Vec::new();
    let mut value = body.clone();
    value.as_object_mut().unwrap().remove("model_contract");
    cases.push(value);
    for invalid in [json!(null), json!({}), json!("qwen4exp.graph-inputs.v3")] {
        let mut value = body.clone();
        value["model_contract"] = invalid;
        cases.push(value);
    }
    for (key, invalid) in [
        ("schema", json!("qwen4exp.graph-inputs.v1")),
        ("vocabulary", json!(248319)),
        ("vocabulary", json!(true)),
        ("extra", json!(0)),
    ] {
        let mut value = body.clone();
        value["model_contract"][key] = invalid;
        cases.push(value);
    }
    for (key, invalid) in [
        ("max_images", json!(0)),
        ("features_per_request", json!(16383)),
        ("patches_per_feature", json!(2)),
        ("patch_capacities", json!([256, 1024, 16384, 4096])),
        ("extra", json!(1)),
    ] {
        let mut value = body.clone();
        value["model_contract"]["vision"][key] = invalid;
        cases.push(value);
    }
    for value in cases {
        let error = Bootstrap::load::<Qwen4Exp>(&value.to_string(), &manifest)
            .err()
            .unwrap();
        assert!(
            matches!(
                error,
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ),
            "{error}: {value}"
        );
    }
    let mut manifest = manifest;
    manifest
        .graphs
        .iter_mut()
        .find(|g| g.kind == GraphKind::Vision)
        .unwrap()
        .token_capacity = 512;
    manifest.validate().unwrap();
    assert!(matches!(
        validate::<Qwen4Exp>(&config, &manifest).unwrap_err(),
        crate::Error::Execution(crate::ExecutionError::Manifest(_))
    ));
}

#[test]
fn model_validates_draft_cache_and_completion_vocabulary() {
    use crate::execution::TokenVector;
    let mut config = config();
    config.speculative_widths = vec![1, 2, 4, 8];
    let mut record = CompletionRecord {
        consumed_tokens: 65,
        token_id: 7,
        status: 1,
        ..Default::default()
    };
    record.vector = Some(TokenVector {
        emitted_count: 1,
        tokens: [7, -1, -1, -1, -1, -1, -1, -1],
        draft_cache_consumed: 64,
        ..Default::default()
    });
    Qwen4Exp::validate_completion(&config, GraphKind::Speculative, &record).unwrap();
    for value in [0, 64, 65, -2] {
        record.vector.as_mut().unwrap().model_reserved = value;
        for kind in [
            GraphKind::Prefill,
            GraphKind::Decode,
            GraphKind::Speculative,
            GraphKind::Resume,
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Vision,
        ] {
            assert!(matches!(
                Qwen4Exp::validate_completion(&config, kind, &record).unwrap_err(),
                crate::Error::Execution(crate::ExecutionError::Completion(_))
            ));
        }
    }
    record.vector.as_mut().unwrap().model_reserved = -1;
    config.speculative_proposal_source = crate::scheduler::speculation::ProposalSource::Copy;
    assert!(Qwen4Exp::validate_completion(&config, GraphKind::Speculative, &record).is_err());
    record.vector.as_mut().unwrap().draft_cache_consumed = -1;
    Qwen4Exp::validate_completion(&config, GraphKind::Speculative, &record).unwrap();
    // Ordinary completions do not claim speculative draft-cache conservation.
    record.vector.as_mut().unwrap().draft_cache_consumed = 64;
    Qwen4Exp::validate_completion(&config, GraphKind::Resume, &record).unwrap();
    record.vector.as_mut().unwrap().tokens[0] = config.contract.vocabulary as i64;
    assert!(Qwen4Exp::validate_completion(&config, GraphKind::Speculative, &record).is_err());
}

#[test]
fn state_transfer_and_resume_upload_fixture() {
    let config = config();
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        Qwen4Exp::scheduler_capabilities(&config),
    )
    .unwrap();
    let mut inputs = Inputs::new();
    for cycle in 0..2 {
        let prompt = vec![7; 64];
        let handle = scheduler
            .admit(RequestSpec {
                system_boundary: None,
                prompt: prompt.clone(),
                conditioning: vec![],
                max_new_tokens: 2,
                grammar_constrained: false,
                host_stops: false,
            })
            .unwrap();
        let mut admissions: [_; 4] = std::array::from_fn(|_| None);
        admissions[handle.slot as usize] = Some(Admission {
            handle,
            sampling: SamplingParameters::default(),
            max_new_tokens: 2,
            prompt,
            data: PromptData {
                positions: (0..64).map(|i| [i, i + 1, i + 2]).collect(),
                vision_token_indices: vec![-1; 64],
            },
            grammar_mask: None,
            ignore_eos: false,
        });
        for (offset, kind, consumed, generated, token, status) in if cycle == 0 {
            vec![
                (1, GraphKind::Init, 0, 0, -1, 1),
                (2, GraphKind::Prefill, 64, 1, 7, 1),
                (3, GraphKind::Snapshot, 64, 1, -1, 1),
                (4, GraphKind::Decode, 65, 2, 8, 2),
                (5, GraphKind::Snapshot, 65, 2, -1, 1),
            ]
        } else {
            vec![
                (1, GraphKind::Restore, 64, 0, -1, 1),
                (2, GraphKind::Resume, 64, 1, 7, 1),
                (3, GraphKind::Decode, 65, 2, 8, 2),
                (4, GraphKind::Snapshot, 65, 2, -1, 1),
            ]
        } {
            let fence = Fence(cycle * 5 + offset);
            let step = scheduler.next_step().unwrap();
            assert_eq!(step.kind, kind);
            scheduler.record_submission(&step, fence).unwrap();
            crate::model::pack_inputs::<Qwen4Exp>(
                &mut inputs,
                &config,
                &scheduler,
                &step,
                &admissions,
            )
            .unwrap();
            assert_state_bindings(&inputs, &scheduler, &step);
            assert!(!inputs.uploads().is_empty());
            if matches!(
                kind,
                GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot
            ) {
                let uploads = inputs.uploads();
                let values = |name| {
                    uploads
                        .iter()
                        .find(|u| u.buffer == name)
                        .unwrap()
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(values("state_expected_consumed"), [consumed, -1, -1, -1]);
                if kind == GraphKind::Snapshot {
                    assert_eq!(values("state_source_slots"), [1, 0, 0, 0]);
                    assert_eq!(
                        values("state_destination_slots")[0],
                        scheduler
                            .snapshot_slot(
                                scheduler.request(handle).unwrap().snapshot_destination()
                            )
                            .0 as i64
                    );
                }
            }
            let mut records = [CompletionRecord::default(); 4];
            records[handle.slot as usize] = CompletionRecord {
                generation: handle.generation as i64,
                submission_sequence: fence.0 as i64,
                consumed_tokens: consumed,
                generated_tokens: generated,
                token_id: token,
                status,
                ..Default::default()
            };
            scheduler.complete(fence, &records).unwrap();
        }
        scheduler.finish(handle).unwrap();
    }
}

#[test]
#[cfg(feature = "images")]
fn vision_upload_fixture_retains_second_image_feature_offset() {
    use crate::frontend::images::{ImageGeometry, ProcessedImages};
    let config = config();
    let images = ProcessedImages {
        pixel_values: (0..8 * 1536).map(|i| (i as f32 - 71.) / 32.).collect(),
        geometry: (0..2)
            .map(|_| ImageGeometry {
                height: 32,
                width: 32,
                grid_thw: [1, 2, 2],
                patches: 4,
                tokens: 1,
            })
            .collect(),
        position_ids: (0..8).map(|i| [i, i + 1]).collect(),
        interpolation_indices: (0..8).map(|i| [i, i + 1, i + 2, i + 3]).collect(),
        interpolation_weights: vec![[0.25; 4]; 8],
        cu_seqlens: vec![0, 4, 8],
    };
    let mut inputs = Inputs::new();
    for image in 0..2 {
        inputs
            .pack_vision(
                &config,
                RequestHandle {
                    slot: 3,
                    generation: 19,
                },
                &images,
                image,
            )
            .unwrap();
        let uploads = inputs.uploads();
        assert_eq!(
            uploads
                .iter()
                .find(|u| u.buffer == "vision_feature_offset")
                .unwrap()
                .bytes,
            (image as i32).to_le_bytes()
        );
    }
}

fn assert_state_bindings(
    inputs: &Inputs,
    scheduler: &RequestScheduler,
    step: &crate::scheduler::ScheduledStep,
) {
    let uploads = inputs.uploads();
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        let binding = uploads
            .iter()
            .find(|upload| {
                upload.buffer == "request_state_rows"
                    && upload.offset == lane.request.slot as usize * 64
            })
            .unwrap();
        let rows: Vec<_> = binding
            .bytes
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[0], request.state_slot.0 as i64);
        for column in 0..scheduler.execution_width() {
            assert_eq!(rows[column], request.state_slot.0 as i64 + column as i64);
        }
        assert!(
            rows[scheduler.execution_width()..]
                .iter()
                .all(|row| *row == 0)
        );
    }
}

fn retain_prompt(s: &mut RequestScheduler, fence: u64) {
    let step = s.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Snapshot);
    s.record_submission(&step, Fence(fence)).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let request = s.request(lane.request).unwrap();
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: request.consumed_tokens as i64,
            generated_tokens: request.generated_tokens as i64,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
    }
    s.complete(Fence(fence), &records).unwrap();
}

#[test]
fn exact_restore_uploads_use_packed_endpoints_and_physical_page_rows() {
    for end in [1usize, 2, 3, 15, 63, 64, 65, 66, 67] {
        for mtp in [false, true] {
            let mut config = config();
            if mtp {
                config.speculative_widths = vec![1, 2, 4, 8];
            }
            let mut scheduler = RequestScheduler::new(
                config.identity,
                config.limits,
                Qwen4Exp::scheduler_capabilities(&config),
            )
            .unwrap();
            if mtp {
                scheduler
                    .enable_speculation(&config.speculative_widths)
                    .unwrap();
            }
            let spec = |token| RequestSpec {
                prompt: vec![token; end],
                system_boundary: None,
                conditioning: vec![],
                max_new_tokens: 1,
                grammar_constrained: false,
                host_stops: false,
            };
            let seed = scheduler.admit(spec(7)).unwrap();
            let mut fence = 0;
            for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
                fence += 1;
                let step = scheduler.next_step().unwrap();
                assert_eq!(step.kind, kind);
                scheduler.record_submission(&step, Fence(fence)).unwrap();
                let generated = i64::from(kind != GraphKind::Init);
                let mut records = [CompletionRecord::default(); 4];
                records[0] = CompletionRecord {
                    generation: seed.generation as i64,
                    submission_sequence: fence as i64,
                    consumed_tokens: if kind == GraphKind::Init {
                        0
                    } else {
                        end as i64
                    },
                    generated_tokens: generated,
                    token_id: if kind == GraphKind::Prefill { 8 } else { -1 },
                    status: if kind == GraphKind::Prefill { 2 } else { 1 },
                    ..Default::default()
                };
                scheduler.complete(Fence(fence), &records).unwrap();
            }
            let checkpoint = scheduler.finish(seed).unwrap().unwrap();
            let blocked = scheduler.admit(spec(9)).unwrap();
            let restored = scheduler.admit(spec(7)).unwrap();
            assert_eq!(restored.slot, 1);
            fence += 1;
            let init = scheduler.next_step().unwrap();
            assert_eq!(init.kind, GraphKind::Init);
            scheduler.record_submission(&init, Fence(fence)).unwrap();
            let mut records = [CompletionRecord::default(); 4];
            records[blocked.slot as usize] = CompletionRecord {
                generation: blocked.generation as i64,
                submission_sequence: fence as i64,
                token_id: -1,
                status: 1,
                ..Default::default()
            };
            scheduler.complete(Fence(fence), &records).unwrap();
            fence += 1;
            let step = scheduler.next_step().unwrap();
            assert_eq!(step.kind, GraphKind::Restore);
            scheduler.record_submission(&step, Fence(fence)).unwrap();
            let mut admissions: [_; 4] = std::array::from_fn(|_| None);
            admissions[1] = Some(Admission {
                handle: restored,
                sampling: SamplingParameters::default(),
                max_new_tokens: 1,
                prompt: vec![7; end],
                data: PromptData {
                    positions: (0..end).map(|i| [i as i64; 3]).collect(),
                    vision_token_indices: vec![-1; end],
                },
                grammar_mask: None,
                ignore_eos: false,
            });
            let mut inputs = Inputs::new();
            crate::model::pack_inputs::<Qwen4Exp>(
                &mut inputs,
                &config,
                &scheduler,
                &step,
                &admissions,
            )
            .unwrap();
            let uploads = inputs.uploads();
            let i64s = |name| {
                uploads
                    .iter()
                    .find(|u| u.buffer == name)
                    .unwrap()
                    .bytes
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(i64s("state_expected_consumed"), [end as i64, -1, -1, -1]);
            assert_eq!(
                i64s("state_source_slots"),
                [
                    scheduler
                        .snapshot_slot(scheduler.cache().checkpoint(checkpoint).unwrap().snapshot())
                        .0 as i64,
                    0,
                    0,
                    0
                ]
            );
            assert_eq!(
                i64s("state_destination_slots"),
                [
                    scheduler.request(restored).unwrap().state_slot.0 as i64,
                    0,
                    0,
                    0
                ]
            );
            let pages = scheduler
                .request(restored)
                .unwrap()
                .pages()
                .collect::<Vec<_>>();
            assert_eq!(pages.len(), end.div_ceil(64) + usize::from(mtp));
            let table = uploads.iter().find(|u| u.buffer == "page_tables").unwrap();
            assert_eq!(table.offset, config.limits.max_sequence_tokens / 64 * 4);
            assert_eq!(
                table.bytes,
                pages
                    .iter()
                    .flat_map(|p| (p.0 as i32).to_le_bytes())
                    .collect::<Vec<_>>()
            );
            if mtp {
                let table = uploads
                    .iter()
                    .find(|u| u.buffer == "draft_page_tables")
                    .unwrap();
                assert_eq!(table.offset, config.limits.max_sequence_tokens / 64 * 4);
                assert_eq!(
                    table.bytes,
                    pages[1..]
                        .iter()
                        .flat_map(|p| (p.0 as i32).to_le_bytes())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn maintenance_contract_requires_exact_endpoint_and_private_page_bindings() {
    for mtp in [false, true] {
        let mut config = config();
        if mtp {
            config.speculative_widths = vec![1, 2, 4, 8];
        }
        let valid = manifest(&config);
        validate::<Qwen4Exp>(&config, &valid).unwrap();
        for kind in [GraphKind::Init, GraphKind::Restore, GraphKind::Snapshot] {
            let mut broken = valid.clone();
            for graph in broken.graphs.iter_mut().filter(|g| g.kind == kind) {
                graph
                    .input_bindings
                    .retain(|b| b.buffer_name != "state_expected_consumed");
            }
            assert!(matches!(
                validate::<Qwen4Exp>(&config, &broken).unwrap_err(),
                crate::Error::Execution(crate::ExecutionError::Manifest(_))
            ));
        }
        for kind in [GraphKind::Restore, GraphKind::Snapshot] {
            for name in ["page_tables", "draft_page_tables"]
                .into_iter()
                .take(if mtp { 2 } else { 1 })
            {
                let mut broken = valid.clone();
                for graph in broken.graphs.iter_mut().filter(|g| g.kind == kind) {
                    graph.input_bindings.retain(|b| b.buffer_name != name);
                }
                assert!(matches!(
                    validate::<Qwen4Exp>(&config, &broken).unwrap_err(),
                    crate::Error::Execution(crate::ExecutionError::Manifest(_))
                ));
            }
        }
        let mut broken = valid;
        broken
            .buffers
            .iter_mut()
            .find(|b| b.name == "state_expected_consumed")
            .unwrap()
            .shape = vec![2, 2];
        assert!(validate::<Qwen4Exp>(&config, &broken).is_err());
    }
}

#[test]
fn reduced_concurrency_preserves_captured_buffer_strides() {
    let mut config = config();
    let mut captured = manifest(&config);
    config.limits.max_active = 1;
    captured.graphs.retain(|graph| graph.batch_capacity == 1);
    validate::<Qwen4Exp>(&config, &captured).unwrap();
    assert_eq!(config.limits.request_capacity(), 4);
    assert_eq!(captured.request_capacity(), 4);
    assert_eq!(captured.buffer("page_tables").unwrap().shape[0], 4);
    config.limits.request_slots = 1;
    assert!(validate::<Qwen4Exp>(&config, &captured).is_err());
}
