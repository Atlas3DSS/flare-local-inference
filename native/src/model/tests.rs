use super::*;
use crate::{
    cache::Fence,
    execution::{ABI_VERSION, BufferDescriptor, CompletionRecord, GraphDescriptor, InputBinding},
    scheduler::{RequestScheduler, RequestSpec},
};

fn config() -> ModelConfig {
    ModelConfig {
        identity: [1; 32],
        model_id: "qwen38".into(),
        limits: SchedulerLimits {
            max_active: 4,
            max_sequence_tokens: 2048,
            prefill_chunk_size: 512,
            pages: 64,
            snapshots: 8,
        },
        eos_token_ids: vec![248046, 248044],
        vision_features_per_request: 16384,
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 256,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        speculative_copy: true,
        speculative_proposal_source: Default::default(),
    }
}
fn manifest(config: &ModelConfig) -> ExecutionManifest {
    let mut descriptions = required_buffers(
        config.limits.max_sequence_tokens,
        config.limits.prefill_chunk_size,
    );
    let speculative = !config.speculative_widths.is_empty();
    if speculative {
        descriptions.extend([
            ("copy_token_ids", Dtype::Int64, vec![4, 7]),
            ("copy_token_count", Dtype::Int32, vec![4]),
            ("copy_expected_consumed", Dtype::Int64, vec![4]),
            ("copy_generation", Dtype::Int64, vec![4]),
            (
                "draft_page_tables",
                Dtype::Int32,
                vec![4, config.limits.max_sequence_tokens / 64],
            ),
        ]);
        if !config.has_draft_cache() {
            descriptions.retain(|(name, _, _)| *name != "draft_page_tables");
            descriptions.extend([
                (
                    "copy_grammar_mask",
                    Dtype::Int32,
                    vec![4, 8, VOCABULARY.div_ceil(32)],
                ),
                ("copy_grammar_mask_count", Dtype::Int32, vec![4]),
            ]);
        }
    }
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
                    [1, 2, 4]
                        .into_iter()
                        .filter(move |batch| batch * width <= 8)
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
    assert_eq!(original.graphs.len(), 25);
    for kind in [GraphKind::Prefill, GraphKind::Decode, GraphKind::Resume] {
        {
            let mode = SamplingMode::General;
            let mut missing = original.clone();
            let index = missing
                .graphs
                .iter()
                .position(|graph| graph.kind == kind && graph.sampling_mode == mode)
                .unwrap();
            let removed = missing.graphs.remove(index);
            assert!(config.validate(&missing).unwrap_err().contains("lacks"));
            let selected = config
                .graph(
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
            assert!(
                config
                    .validate(&duplicate)
                    .unwrap_err()
                    .contains("ambiguous")
            );
        }
    }
}

#[test]
fn speculative_catalog_requires_every_width_mode_and_shifted_page_binding() {
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    let original = manifest(&config);
    config.validate(&original).unwrap();
    assert_eq!(original.graphs.len(), 36);
    assert!(
        original
            .graphs
            .iter()
            .filter(|g| g.kind == GraphKind::Speculative)
            .all(|g| g.batch_capacity * g.token_capacity <= 8)
    );
    assert!(!original.graphs.iter().any(|g| g.kind == GraphKind::Decode));
    for index in original
        .graphs
        .iter()
        .enumerate()
        .filter_map(|(i, g)| (g.kind == GraphKind::Speculative).then_some(i))
    {
        let mut missing = original.clone();
        missing.graphs.remove(index);
        assert!(
            config
                .validate(&missing)
                .unwrap_err()
                .contains("lacks Speculative")
        );
    }
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
        assert!(
            config
                .validate(&missing)
                .unwrap_err()
                .contains("draft_page_tables")
        );
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
        assert!(config.validate(&wrong).is_err());
    }
    config.speculative_fixed_width = Some(9);
    assert!(
        config
            .validate(&original)
            .unwrap_err()
            .contains("speculative_fixed_width")
    );
    config.speculative_fixed_width = Some(8);
    config.validate(&original).unwrap();
}

#[test]
#[cfg(feature = "server")]
fn speculative_bootstrap_options_validate_before_assets_or_cuda() {
    use crate::server::Bootstrap;
    use serde_json::json;
    let mut config = config();
    config.speculative_widths = vec![1, 2, 4, 8];
    let manifest = manifest(&config);
    let body = json!({"model_id":"qwen38","identity":"01".repeat(32),
        "max_sequence_tokens":2048,"prefill_chunk_size":512,"pages":64,"snapshots":8,
        "eos_token_ids":[],"speculative_widths":[1,2,4,8]});
    for width in [json!(null), json!(1), json!(2), json!(4), json!(8)] {
        for copy in [true, false] {
            let mut body = body.clone();
            body["speculative_fixed_width"] = width.clone();
            body["speculative_copy"] = json!(copy);
            assert_eq!(
                Bootstrap::load(&body.to_string(), &manifest).err().unwrap(),
                "assets must contain verified frontend descriptors"
            );
        }
    }
    for (key, value) in [
        ("speculative_fixed_width", json!(3)),
        ("speculative_fixed_width", json!("8")),
        ("speculative_copy", json!(null)),
        ("speculative_copy", json!(1)),
        ("speculative_widths", json!([1, 8, 4])),
        ("speculative_widths", json!([2, 4, 8])),
        ("speculative_proposal_source", json!(null)),
        ("speculative_proposal_source", json!("auto")),
    ] {
        let mut body = body.clone();
        body[key] = value;
        let error = Bootstrap::load(&body.to_string(), &manifest).err().unwrap();
        assert!(error.contains("speculative"), "{error}");
    }
    config.speculative_proposal_source = crate::scheduler::speculation::ProposalSource::Copy;
    let copy_manifest = self::manifest(&config);
    let mut body = body;
    body["speculative_proposal_source"] = json!("copy");
    assert_eq!(
        Bootstrap::load(&body.to_string(), &copy_manifest)
            .err()
            .unwrap(),
        "assets must contain verified frontend descriptors"
    );
}

#[test]
fn mtp_copy_uploads_cover_zero_and_default_temperature_with_epoch_and_consumed_guards() {
    let mut config = config();
    config.speculative_widths = vec![1, 2, 4, 8];
    let mut scheduler = RequestScheduler::new(config.identity, config.limits).unwrap();
    scheduler
        .enable_speculation(&config.speculative_widths)
        .unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let prompt = vec![7; 65];
    let handle = scheduler
        .admit(RequestSpec {
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
        positions: (0..65).map(|p| [p; 3]).collect(),
        vision_token_indices: vec![-1; 65],
        grammar_mask: None,
        ignore_eos: false,
    });
    let mut inputs = ModelInputs::new();
    let mut records = [CompletionRecord::default(); 4];
    for (fence, consumed, generated, token) in [(1, 0, 0, -1), (2, 65, 1, 7)] {
        let step = scheduler.next_step().unwrap();
        scheduler.record_submission(&step, Fence(fence)).unwrap();
        inputs
            .pack(&config, &scheduler, &step, &admissions)
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
    for fence in [3, 4] {
        admissions[0].as_mut().unwrap().sampling.temperature = if fence == 3 { 0. } else { 1. };
        let step = scheduler.next_step().unwrap();
        assert_eq!(step.token_capacity, 8);
        scheduler.record_submission(&step, Fence(fence)).unwrap();
        inputs
            .pack(&config, &scheduler, &step, &admissions)
            .unwrap();
        let uploads = inputs.uploads();
        assert!(!uploads.iter().any(|u| u.buffer.ends_with("page_tables")));
        let expected = uploads
            .iter()
            .find(|u| u.buffer == "copy_expected_consumed")
            .unwrap();
        assert_eq!(
            i64::from_le_bytes(expected.bytes[..8].try_into().unwrap()),
            65 + fence as i64 - 3
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
        config.speculative_copy = false;
        inputs
            .pack(&config, &scheduler, &step, &admissions)
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
        config.speculative_copy = true;
        assert!(scheduler.next_step().is_none());
        let mut vector = crate::execution::TokenVector {
            emitted_count: 1,
            offered_drafts: 7,
            accepted_drafts: 0,
            draft_cache_consumed: 65 + fence as i64 - 3,
            verification_width: 8,
            ..Default::default()
        };
        vector.tokens[0] = 7;
        records[0] = CompletionRecord {
            generation: handle.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: 66 + fence as i64 - 3,
            generated_tokens: fence as i64 - 1,
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
    config.validate(&manifest).unwrap();
    let mut missing = manifest.clone();
    missing
        .graphs
        .retain(|graph| graph.kind != GraphKind::Resume);
    assert!(config.validate(&missing).unwrap_err().contains("Resume"));
    let mut missing = manifest.clone();
    let graph = missing
        .graphs
        .iter_mut()
        .find(|graph| graph.kind == GraphKind::Prefill)
        .unwrap();
    graph
        .input_bindings
        .retain(|input| input.buffer_name != "vision_token_indices");
    assert!(
        config
            .validate(&missing)
            .unwrap_err()
            .contains("vision_token_indices")
    );
    let mut wrong = manifest;
    wrong
        .buffers
        .iter_mut()
        .find(|buffer| buffer.name == "admission_seed_bits")
        .unwrap()
        .dtype = Dtype::Float32;
    assert!(config.validate(&wrong).is_err());
}

#[test]
fn configured_prefill_catalog_and_backing_extents_are_required_before_cuda() {
    for chunk_size in [64, 3072, 4096, 8192] {
        let mut config = config();
        config.limits.max_sequence_tokens = 32768;
        config.limits.prefill_chunk_size = chunk_size;
        let original = manifest(&config);
        config.validate(&original).unwrap();
        {
            let mode = SamplingMode::General;
            let mut missing = original.clone();
            missing.graphs.retain(|graph| {
                !(graph.kind == GraphKind::Prefill
                    && graph.sampling_mode == mode
                    && graph.token_capacity == chunk_size)
            });
            let error = config.validate(&missing).unwrap_err();
            assert!(error.contains("lacks Prefill") && error.contains(&chunk_size.to_string()));
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
            assert!(
                config
                    .validate(&short)
                    .unwrap_err()
                    .contains(&format!("semantic buffer {name}"))
            );
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
    let mut body = json!({"model_id": "qwen38", "identity": "01".repeat(32),
        "max_sequence_tokens": 8192, "pages": 64, "snapshots": 8, "eos_token_ids": []});
    let failure = |body: &serde_json::Value, config: &ModelConfig| {
        Bootstrap::load(&body.to_string(), &manifest(config))
            .err()
            .unwrap()
    };
    assert_eq!(
        failure(&body, &config),
        "assets must contain verified frontend descriptors"
    );
    config.limits.prefill_chunk_size = 512;
    assert!(failure(&body, &config).contains("semantic buffer token_ids"));
    body["prefill_chunk_size"] = json!(512);
    assert_eq!(
        failure(&body, &config),
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
        assert!(failure(&body, &config).contains("prefill_chunk_size"));
    }
}

#[test]
fn admission_histogram_seed_and_partial_page_uploads_preserve_native_semantics() {
    let config = config();
    let mut scheduler = RequestScheduler::new(config.identity, config.limits).unwrap();
    let handle = scheduler
        .admit(RequestSpec {
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
        positions: (0..65).map(|p| [p; 3]).collect(),
        vision_token_indices: vec![-1; 65],
        grammar_mask: None,
        ignore_eos: false,
    });
    admissions[0].as_ref().unwrap().validate().unwrap();
    let mut inputs = ModelInputs::new();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(1)).unwrap();
    inputs
        .pack(&config, &scheduler, &step, &admissions)
        .unwrap();
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
        inputs
            .pack(&config, &scheduler, &step, &admissions)
            .unwrap();
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
    inputs
        .pack(&config, &scheduler, &step, &admissions)
        .unwrap();
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
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(3)).unwrap();
    inputs
        .pack(&config, &scheduler, &step, &admissions)
        .unwrap();
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
    let mut scheduler = RequestScheduler::new(config.identity, config.limits).unwrap();
    let prompt: Vec<u32> = (0..4113).collect();
    let handle = scheduler
        .admit(RequestSpec {
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
        positions: prompt.iter().map(|&token| [token as i64; 3]).collect(),
        vision_token_indices: prompt
            .iter()
            .map(|&token| if token % 2 == 0 { token as i32 } else { -1 })
            .collect(),
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
    let mut inputs = ModelInputs::new();
    for (sequence, start, count, capacity) in [(2, 0, 4096, 4096), (3, 4096, 17, 64)] {
        let step = scheduler.next_step().unwrap();
        assert_eq!(
            (step.lanes[0].tokens, step.token_capacity),
            (count, capacity)
        );
        scheduler.record_submission(&step, Fence(sequence)).unwrap();
        inputs
            .pack(&config, &scheduler, &step, &admissions)
            .unwrap();
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
    config.validate(&original).unwrap();
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
        assert!(config.validate(&missing).unwrap_err().contains(name));
        let mut wrong = original.clone();
        wrong
            .buffers
            .iter_mut()
            .find(|b| b.name == name)
            .unwrap()
            .dtype = Dtype::Float32;
        assert!(config.validate(&wrong).is_err());
    }
    config.speculative_widths.clear();
    assert!(
        config
            .validate(&original)
            .unwrap_err()
            .contains("requires speculative")
    );
    assert!(ProposalSource::parse("auto").is_err());
}

#[test]
fn copy_staging_uses_physical_slots_stochastic_defaults_and_bonus_mask() {
    use crate::scheduler::speculation::{CopyProposal, ProposalSource};
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    config.speculative_proposal_source = ProposalSource::Copy;
    let mut scheduler = RequestScheduler::new(config.identity, config.limits).unwrap();
    scheduler
        .enable_speculation(&config.speculative_widths)
        .unwrap();
    scheduler.proposal_source(ProposalSource::Copy).unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let handles = (0..3)
        .map(|slot| {
            scheduler
                .admit(RequestSpec {
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
        positions: (0..65).map(|p| [p; 3]).collect(),
        vision_token_indices: vec![-1; 65],
        grammar_mask: Some(vec![1; words]),
        ignore_eos: false,
    });
    let mut inputs = ModelInputs::new();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    inputs
        .pack(&config, &scheduler, &step, &admissions)
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
    assert_eq!(step.token_capacity, 3);
    scheduler.record_submission(&step, Fence(3)).unwrap();
    inputs
        .pack(&config, &scheduler, &step, &admissions)
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
        submission_sequence: 3,
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
            verification_width: 3,
            tokens: [10, -1, -1, -1, -1, -1, -1, -1],
            ..Default::default()
        }),
        ..Default::default()
    };
    scheduler.complete(Fence(3), &completed).unwrap();
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
    assert_eq!(step.token_capacity, 1);
    scheduler.record_submission(&step, Fence(4)).unwrap();
    inputs
        .pack(&config, &scheduler, &step, &admissions)
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
