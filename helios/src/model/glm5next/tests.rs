use super::inputs::{Admission, Inputs};
use super::*;
use crate::execution::SamplingMode;
use crate::model::catalog::{select, validate};
use crate::{
    cache::Fence,
    execution::{ABI_VERSION, BufferDescriptor, Dtype, GraphDescriptor, InputBinding},
    model::SamplingParameters,
    scheduler::{
        RequestHandle, RequestScheduler, RequestSpec, SchedulerLimits, prefill_capacities,
    },
};

pub(super) fn config() -> ModelConfig {
    ModelConfig {
        contract: Glm5Next::metadata(),
        identity: [9; 32],
        model_id: "glm53".into(),
        limits: SchedulerLimits {
            max_active: 4,
            max_sequence_tokens: 16384,
            prefill_chunk_size: 4096,
            pages: 64,
            snapshots: 16,
        },
        eos_token_ids: vec![154820, 154827, 154829],
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 128,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        speculative_copy: false,
        speculative_proposal_source: Default::default(),
    }
}

fn manifest(config: &ModelConfig) -> ExecutionManifest {
    let mut descriptions = super::config::required_buffers(config);
    descriptions.extend([
        ("frame_index", Dtype::Int32, vec![1]),
        ("submission_sequence", Dtype::Int64, vec![1]),
        ("completion_records", Dtype::Int64, vec![2, 4, 8]),
    ]);
    let (mut address, mut host) = (0x10000, 0);
    let mut bindings = Vec::new();
    let buffers = descriptions
        .into_iter()
        .map(|(name, dtype, shape)| {
            let nbytes = shape.iter().product::<usize>() * dtype.size();
            let mut stride = 1;
            let mut strides = shape
                .iter()
                .rev()
                .map(|n| {
                    let result = stride;
                    stride *= n;
                    result
                })
                .collect::<Vec<_>>();
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
        GraphKind::Resume,
        GraphKind::Decode,
        GraphKind::Prefill,
    ] {
        let capacities: Vec<_> = if kind == GraphKind::Prefill {
            prefill_capacities(config.limits.prefill_chunk_size)
                .map(|n| (1, n))
                .collect()
        } else {
            [1, 2, 4]
                .into_iter()
                .filter(|b| *b <= config.limits.max_active.next_power_of_two())
                .map(|b| (b, 1))
                .collect()
        };
        for (batch, tokens) in capacities {
            graphs.push(GraphDescriptor {
                name: format!("{kind:?}-{batch}-{tokens}"),
                kind,
                sampling_mode: SamplingMode::General,
                handle: graphs.len() + 1,
                batch_capacity: batch,
                token_capacity: tokens,
                input_bindings: bindings
                    .iter()
                    .filter(|b| {
                        super::config::required_inputs(kind).contains(&b.buffer_name.as_str())
                            || ["frame_index", "submission_sequence"]
                                .contains(&b.buffer_name.as_str())
                    })
                    .cloned()
                    .collect(),
            });
        }
    }
    ExecutionManifest {
        abi_version: ABI_VERSION,
        context: 1,
        device: 0,
        max_active: config.limits.max_active,
        input_frame_bytes: host,
        buffers,
        graphs,
    }
}

#[test]
fn catalog_requires_each_capacity_and_semantic_binding() {
    for chunk in [64, 3072, 4096, 8192] {
        let mut config = config();
        config.limits.prefill_chunk_size = chunk;
        let original = manifest(&config);
        validate::<Glm5Next>(&config, &original).unwrap();
        for index in 0..original.graphs.len() {
            let mut missing = original.clone();
            let graph = missing.graphs.remove(index);
            assert!(
                validate::<Glm5Next>(&config, &missing).is_err(),
                "{}",
                graph.name
            );
            for input in super::config::required_inputs(graph.kind) {
                let mut missing = original.clone();
                missing.graphs[index]
                    .input_bindings
                    .retain(|b| b.buffer_name != *input);
                assert!(
                    validate::<Glm5Next>(&config, &missing).is_err(),
                    "{} {input}",
                    graph.name
                );
            }
        }
    }
    let mut config = config();
    let mut invalid = manifest(&config);
    invalid
        .buffers
        .iter_mut()
        .find(|b| b.name == "page_tables")
        .unwrap()
        .shape[1] *= 4;
    assert!(validate::<Glm5Next>(&config, &invalid).is_err());
    let mut duplicate = manifest(&config);
    let mut graph = duplicate.graphs[0].clone();
    graph.name.push_str("-duplicate");
    graph.handle = 999;
    duplicate.graphs.push(graph);
    assert!(validate::<Glm5Next>(&config, &duplicate).is_err());
    config.speculative_widths = vec![1, 2];
    assert!(validate::<Glm5Next>(&config, &manifest(&config)).is_err());
}

#[test]
fn disabled_vision_is_explicit_and_partial_geometry_rejects() {
    let config = config();
    let mut caps = Glm5Next::scheduler_capabilities(&config);
    assert_eq!(caps.page_tokens, 256);
    assert_eq!(caps.draft_pages_per_request, 0);
    RequestScheduler::new(config.identity, config.limits, caps.clone()).unwrap();
    caps.vision.max_images = 1;
    assert!(RequestScheduler::new(config.identity, config.limits, caps).is_err());
}

#[cfg(feature = "server")]
#[test]
#[ignore = "requires FLARE_GLM_TEST_SNAPSHOT pointing to the GLM checkpoint"]
fn bootstrap_loads_glm_assets_without_a_vision_processor() {
    use serde_json::{Map, json};
    use sha2::{Digest, Sha256};
    let config = config();
    let manifest = manifest(&config);
    let snapshot = std::path::PathBuf::from(std::env::var_os("FLARE_GLM_TEST_SNAPSHOT").unwrap());
    let assets: Map<_, _> = [
        ("tokenizer", "tokenizer.json"),
        ("tokenizer_config", "tokenizer_config.json"),
        ("chat_template", "chat_template.jinja"),
        ("generation_config", "generation_config.json"),
    ]
    .into_iter()
    .map(|(role, name)| {
        let path = snapshot.join(name);
        let hash = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
        (role.into(), json!({"path": path, "sha256": hash}))
    })
    .collect();
    let config = json!({
        "identity": "09".repeat(32), "model_id": config.model_id,
        "model_contract": {"schema": INPUT_SCHEMA, "vocabulary": VOCABULARY,
            "vision": {"features_per_request":0, "max_images":0, "patches_per_feature":0, "patch_capacities":[]}},
        "max_sequence_tokens": config.limits.max_sequence_tokens, "prefill_chunk_size": config.limits.prefill_chunk_size,
        "pages": config.limits.pages, "snapshots": config.limits.snapshots,
        "eos_token_ids": config.eos_token_ids, "assets": assets,
    });
    let bootstrap =
        crate::server::Bootstrap::load::<Glm5Next>(&config.to_string(), &manifest).unwrap();
    assert!(bootstrap.images.is_none());
    assert_eq!(bootstrap.model.sampling_defaults.temperature, 1.);
    assert_eq!(bootstrap.model.sampling_defaults.top_p, 0.95);
    let prompt = bootstrap
        .frontend
        .render_chat(&[json!({"role":"user", "content":"Hello"})], &Map::new())
        .unwrap();
    assert!(prompt.text.ends_with("<|assistant|><think>"));
}

fn add(
    scheduler: &mut RequestScheduler,
    admissions: &mut [Option<Admission>; 4],
    prompt: Vec<u32>,
) -> RequestHandle {
    let handle = scheduler
        .admit(RequestSpec {
            prompt: prompt.clone(),
            system_boundary: Some(267),
            conditioning: vec![],
            max_new_tokens: 8,
            grammar_constrained: true,
            host_stops: false,
        })
        .unwrap();
    admissions[handle.slot as usize] = Some(Admission {
        data: (),
        handle,
        sampling: SamplingParameters {
            seed: u64::MAX - 5,
            ..Default::default()
        },
        max_new_tokens: 8,
        prompt,
        grammar_mask: Some(vec![-1; VOCABULARY.div_ceil(32)]),
        ignore_eos: false,
    });
    handle
}

fn values(inputs: &Inputs, name: &str) -> Vec<i64> {
    inputs
        .uploads()
        .iter()
        .filter(|u| u.buffer == name)
        .flat_map(|u| u.bytes.chunks_exact(8))
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn native_uploads_preserve_exact_checkpoints_and_broadcast_pending_tokens() {
    let config = config();
    let manifest = manifest(&config);
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        Glm5Next::scheduler_capabilities(&config),
    )
    .unwrap();
    let mut admissions = std::array::from_fn(|_| None);
    let original = add(&mut scheduler, &mut admissions, vec![7; 535]);
    let mut inputs = Inputs::default();
    let mut fence = 0;
    let mut observed = Vec::new();
    loop {
        let step = scheduler.next_step().unwrap();
        fence += 1;
        scheduler.record_submission(&step, Fence(fence)).unwrap();
        crate::model::pack_inputs::<Glm5Next>(&mut inputs, &config, &scheduler, &step, &admissions)
            .unwrap();
        let index = select::<Glm5Next>(
            &config,
            &manifest,
            step.kind,
            step.batch_capacity,
            step.token_capacity,
            SamplingMode::General,
        )
        .unwrap();
        for upload in inputs.uploads() {
            assert!(
                manifest.graphs[index]
                    .input_bindings
                    .iter()
                    .any(|b| b.buffer_name == upload.buffer)
            );
            assert!(
                upload.offset + upload.bytes.len()
                    <= manifest.buffer(upload.buffer).unwrap().nbytes
            );
        }
        let live = scheduler.request(original).unwrap();
        let (mut consumed, mut generated, mut token) =
            (live.consumed_tokens, live.generated_tokens, -1);
        match step.kind {
            GraphKind::Init => {
                assert_eq!(values(&inputs, "state_source_slots"), [-1; 4]);
                assert_eq!(values(&inputs, "state_destination_slots")[0], 1);
                assert_eq!(values(&inputs, "admission_seed_bits")[0], -6);
            }
            GraphKind::Prefill => {
                assert_eq!(values(&inputs, "token_ids"), vec![7; step.lanes[0].tokens]);
                consumed += step.lanes[0].tokens as u64;
                if consumed == 535 {
                    generated += 1;
                    token = 42;
                }
            }
            GraphKind::Snapshot => {
                assert_eq!(values(&inputs, "state_source_slots")[0], 1);
                assert_eq!(
                    values(&inputs, "state_expected_consumed")[0],
                    consumed as i64
                );
                observed.push(consumed);
            }
            GraphKind::Decode => {
                assert_eq!(values(&inputs, "token_ids"), vec![42]);
                consumed += 1;
                generated += 1;
                token = 43;
            }
            kind => panic!("unexpected {kind:?}"),
        }
        let mut records = [CompletionRecord::default(); 4];
        records[original.slot as usize] = CompletionRecord {
            generation: original.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: token,
            status: 1,
            ..Default::default()
        };
        scheduler.complete(Fence(fence), &records).unwrap();
        if step.kind == GraphKind::Decode {
            break;
        }
    }
    assert_eq!(observed, [267, 535]);
    let branch = add(
        &mut scheduler,
        &mut admissions,
        [vec![7; 267], vec![9; 100]].concat(),
    );
    let live = scheduler.request(branch).unwrap();
    assert_eq!(live.reused_prompt_tokens(), 267);
    let source = 5 + live.prefix_snapshot().unwrap().0 as i64;
    let exact = add(&mut scheduler, &mut admissions, vec![7; 535]);
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Restore);
    fence += 1;
    scheduler.record_submission(&step, Fence(fence)).unwrap();
    crate::model::pack_inputs::<Glm5Next>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    assert_eq!(values(&inputs, "state_source_slots")[0], source);
    assert_eq!(
        values(&inputs, "state_destination_slots")[0],
        branch.slot as i64 + 1
    );
    assert_eq!(values(&inputs, "state_expected_consumed")[0], 267);
    let pages = inputs
        .uploads()
        .into_iter()
        .find(|u| u.buffer == "page_tables")
        .unwrap();
    assert_eq!(pages.offset, branch.slot as usize * (16384 / 256) * 4);
    assert_eq!(pages.bytes.len(), 8);
    assert_eq!(step.batch_capacity, 2);
    assert_eq!(values(&inputs, "state_expected_consumed")[..2], [267, 535]);
    let mut records = [CompletionRecord::default(); 4];
    records[branch.slot as usize] = CompletionRecord {
        generation: branch.generation as i64,
        submission_sequence: fence as i64,
        consumed_tokens: 267,
        token_id: -1,
        status: 1,
        ..Default::default()
    };
    records[exact.slot as usize] = CompletionRecord {
        generation: exact.generation as i64,
        submission_sequence: fence as i64,
        consumed_tokens: 535,
        token_id: -1,
        status: 1,
        ..Default::default()
    };
    scheduler.complete(Fence(fence), &records).unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Resume);
    fence += 1;
    scheduler.record_submission(&step, Fence(fence)).unwrap();
    crate::model::pack_inputs::<Glm5Next>(&mut inputs, &config, &scheduler, &step, &admissions)
        .unwrap();
    assert!(inputs.uploads().iter().all(|u| {
        [
            "row_slot_ids",
            "row_generations",
            "live_rows",
            "sampling_grammar_mask",
        ]
        .contains(&u.buffer)
    }));
    assert_eq!(
        values(&inputs, "row_generations")[0],
        exact.generation as i64
    );
}
