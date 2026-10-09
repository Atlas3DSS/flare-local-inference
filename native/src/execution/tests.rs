use super::*;
use crate::cache::Fence;

fn manifest() -> ExecutionManifest {
    let buffers = vec![
        BufferDescriptor {
            name: "frame_index".into(),
            address: 0x1000,
            nbytes: 4,
            dtype: Dtype::Int32,
            shape: vec![1],
            strides: vec![1],
        },
        BufferDescriptor {
            name: "submission_sequence".into(),
            address: 0x2000,
            nbytes: 8,
            dtype: Dtype::Int64,
            shape: vec![1],
            strides: vec![1],
        },
        BufferDescriptor {
            name: "completion_records".into(),
            address: 0x3000,
            nbytes: 512,
            dtype: Dtype::Int64,
            shape: vec![2, 4, 8],
            strides: vec![32, 8, 1],
        },
        BufferDescriptor {
            name: "tokens".into(),
            address: 0x4000,
            nbytes: 32,
            dtype: Dtype::Int64,
            shape: vec![4],
            strides: vec![1],
        },
    ];
    let graph = GraphDescriptor {
        name: "decode4".into(),
        kind: GraphKind::Decode,
        sampling_mode: SamplingMode::General,
        handle: 0x5000,
        batch_capacity: 4,
        token_capacity: 1,
        input_bindings: vec![
            InputBinding {
                buffer_name: "frame_index".into(),
                host_offset: 0,
                nbytes: 4,
            },
            InputBinding {
                buffer_name: "submission_sequence".into(),
                host_offset: 8,
                nbytes: 8,
            },
            InputBinding {
                buffer_name: "tokens".into(),
                host_offset: 16,
                nbytes: 32,
            },
        ],
    };
    ExecutionManifest {
        abi_version: ABI_VERSION,
        context: 1,
        device: 0,
        max_active: 4,
        input_frame_bytes: 48,
        buffers,
        graphs: vec![graph],
    }
}

#[test]
fn execution_abi_and_unsupported_sampling_modes_reject_before_cuda() {
    let mut value = manifest();
    value.abi_version = 1;
    assert!(value.validate().unwrap_err().contains("ABI version"));
    value.abi_version = ABI_VERSION;
    value.validate().unwrap();
    assert!(SamplingMode::try_from("greedy").is_err());
    assert!(SamplingMode::try_from("automatic").is_err());
}

#[test]
fn required_completion_and_input_layouts_are_checked_before_cuda() {
    let original = manifest();
    original.validate().unwrap();
    let mut changed = original.clone();
    changed.buffers[2].shape = vec![4, 2, 8];
    changed.buffers[2].strides = vec![16, 8, 1];
    assert!(
        changed
            .validate()
            .unwrap_err()
            .contains("required buffer layout")
    );
    changed = original.clone();
    changed.buffers[3].address = 0x3008;
    assert!(changed.validate().unwrap_err().contains("overlap"));
    changed = original.clone();
    changed.buffers[3].strides = vec![2];
    assert!(changed.validate().unwrap_err().contains("contiguous"));
    changed = original.clone();
    changed.buffers[3].address = usize::MAX - 8;
    assert!(changed.validate().is_err());
    changed = original;
    changed.graphs[0]
        .input_bindings
        .retain(|binding| binding.buffer_name != "frame_index");
    assert!(changed.validate().unwrap_err().contains("reserved"));
}

#[test]
fn graph_variants_share_one_nonoverlapping_input_layout() {
    let mut manifest = manifest();
    let mut graph = manifest.graphs[0].clone();
    graph.name = "decode2".into();
    graph.handle += 1;
    graph.batch_capacity = 2;
    manifest.graphs.push(graph);
    manifest.validate().unwrap();
    manifest.graphs[1].input_bindings[2].host_offset = 8;
    assert!(
        manifest
            .validate()
            .unwrap_err()
            .contains("different host offsets")
    );
    manifest.graphs.pop();
    manifest.graphs[0].input_bindings[2].host_offset = 8;
    assert!(manifest.validate().unwrap_err().contains("overlap"));
}

#[test]
fn all_prefill_buckets_are_valid_but_other_capacities_are_not() {
    let mut manifest = manifest();
    manifest.graphs[0].kind = GraphKind::Prefill;
    manifest.graphs[0].batch_capacity = 1;
    for capacity in [1, 16, 64, 128, 256, 512, 3072, 4096, 8192, 262144] {
        manifest.graphs[0].token_capacity = capacity;
        manifest.validate().unwrap();
    }
    for capacity in [0, 2, 15, 32, 63, 65, 513, 4095, 262208, usize::MAX] {
        manifest.graphs[0].token_capacity = capacity;
        assert!(manifest.validate().is_err());
    }
}

#[test]
fn partial_uploads_reject_overlaps_reserved_inputs_and_misalignment() {
    let manifest = manifest();
    transport::validate_uploads(
        &manifest,
        0,
        &[Upload {
            buffer: "tokens",
            offset: 8,
            bytes: &[1; 8],
        }],
    )
    .unwrap();
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[Upload {
                buffer: "tokens",
                offset: 1,
                bytes: &[1; 8]
            }]
        )
        .is_err()
    );
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[Upload {
                buffer: "tokens",
                offset: 32,
                bytes: &[1; 8]
            }]
        )
        .is_err()
    );
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[Upload {
                buffer: "frame_index",
                offset: 0,
                bytes: &[0; 4]
            }]
        )
        .is_err()
    );
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[
                Upload {
                    buffer: "tokens",
                    offset: 0,
                    bytes: &[1; 16]
                },
                Upload {
                    buffer: "tokens",
                    offset: 8,
                    bytes: &[2; 8]
                }
            ]
        )
        .is_err()
    );
}

#[test]
fn physical_lane_records_reject_stale_sequences_and_impossible_status() {
    let mut records = [[0i64; 8]; 4];
    records[3] = [7, 9, 65, 2, 88, 1, 0, 0];
    let bytes = |records: [[i64; 8]; 4]| {
        records
            .into_iter()
            .flatten()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let decoded = transport::decode_records(&bytes(records), Fence(9)).unwrap();
    assert_eq!(decoded[3].generation, 7);
    assert_eq!(decoded[3].token_id, 88);
    assert_eq!(decoded[0].status, 0);
    assert!(transport::decode_records(&bytes(records), Fence(10)).is_err());
    records[3][0] = 0;
    assert!(transport::decode_records(&bytes(records), Fence(9)).is_err());
    records[3] = [7, 9, 65, 2, -1, 2, 1, 0];
    assert!(transport::decode_records(&bytes(records), Fence(9)).is_err());
}

#[test]
fn abi3_requires_vector_storage_and_speculative_widths() {
    let mut m = manifest();
    m.abi_version = SPECULATIVE_ABI_VERSION;
    assert!(m.validate().is_err());
    m.buffers[2].shape = vec![2, 4, 24];
    m.buffers[2].strides = vec![96, 24, 1];
    m.buffers[2].nbytes = 1536;
    m.graphs[0].kind = GraphKind::Speculative;
    for batch in [1, 2, 4] {
        m.graphs[0].batch_capacity = batch;
        for width in 1..=8 {
            m.graphs[0].token_capacity = width;
            assert_eq!(m.validate().is_ok(), batch * width <= 8);
        }
    }
    m.graphs[0].batch_capacity = 1;
    m.graphs[0].token_capacity = 9;
    assert!(m.validate().is_err());
    m.graphs[0].token_capacity = 8;
    m.abi_version = ABI_VERSION;
    assert!(m.validate().is_err());
}

#[test]
fn vector_parser_rejects_counts_tails_and_stale_epochs_without_cuda() {
    let mut vector = TokenVector {
        emitted_count: 3,
        offered_drafts: 7,
        accepted_drafts: 3,
        copy_offered_mask: 3,
        candidate_consumed: 64,
        draft_cache_consumed: 66,
        verification_width: 8,
        ..Default::default()
    };
    vector.tokens[..3].copy_from_slice(&[40, 41, 42]);
    let good = CompletionRecord {
        generation: 3,
        submission_sequence: 9,
        consumed_tokens: 67,
        generated_tokens: 4,
        token_id: 42,
        status: 2,
        vector: Some(vector),
        ..Default::default()
    };
    let encoded = |row: CompletionRecord| {
        let mut bytes = vec![0; REQUEST_LANES * VECTOR_RECORD_FIELDS * 8];
        for (dst, field) in bytes[..VECTOR_RECORD_FIELDS * 8]
            .chunks_exact_mut(8)
            .zip(row.fields())
        {
            dst.copy_from_slice(&field.to_le_bytes());
        }
        bytes
    };
    assert_eq!(
        transport::decode_records(&encoded(good), Fence(9)).unwrap()[0],
        good
    );
    for change in 0..8 {
        let mut row = good;
        let v = row.vector.as_mut().unwrap();
        match change {
            0 => v.emitted_count = 9,
            1 => v.tokens[3] = 1,
            2 => v.accepted_drafts = 1,
            3 => v.offered_drafts = 8,
            4 => v.copy_offered_mask = 128,
            5 => v.candidate_consumed = 65,
            6 => row.token_id = 40,
            _ => row.submission_sequence = 8,
        }
        assert!(
            transport::decode_records(&encoded(row), Fence(9)).is_err(),
            "change{change}"
        );
    }
}
