use super::*;

#[test]
fn tensor_parallel_capacity_rejects_more_than_sixteen_ranks() {
    let pairs: Vec<_> = (0..16)
        .map(|_| {
            let (local, peer) = crate::execution::rank_stream::test_pair().unwrap();
            (RankStream::from(local), peer)
        })
        .collect();
    let (sockets, _peers): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let result = unsafe { TensorParallelExecutor::new(manifest(), sockets) };
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("at most sixteen")
    );
    let result = unsafe { threaded::ThreadedExecutor::new(manifest(), vec![manifest(); 16]) };
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("at most sixteen")
    );
}

#[test]
fn completion_wire_decodes_every_configured_lane() {
    for capacity in [1, 2, 8, 64] {
        for fields in [8, 24] {
            let mut bytes = vec![0u8; capacity * fields * 8];
            let last = (capacity - 1) * fields * 8;
            for (field, value) in [(0, 7i64), (1, 9), (4, -1), (5, 1)] {
                bytes[last + field * 8..last + (field + 1) * 8]
                    .copy_from_slice(&value.to_le_bytes());
            }
            if fields == 24 {
                for (field, value) in [(12, -1i64), (15, 1)] {
                    bytes[last + field * 8..last + (field + 1) * 8]
                        .copy_from_slice(&value.to_le_bytes());
                }
                for field in 16..24 {
                    bytes[last + field * 8..last + (field + 1) * 8]
                        .copy_from_slice(&(-1i64).to_le_bytes());
                }
            }
            let records = transport::decode_records(&bytes, Fence(9), capacity, fields).unwrap();
            assert_eq!(records.len(), capacity);
            assert_eq!(records[capacity - 1].generation, 7);
            assert!(
                transport::decode_records(&bytes[..bytes.len() - 8], Fence(9), capacity, fields)
                    .is_err()
            );
        }
    }
}
use crate::cache::Fence;
use crate::{Error, ExecutionError};

pub(super) fn manifest() -> ExecutionManifest {
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
        prefill_capacity: 0,
        estimated_duration_ns: 1_000_000,
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
        engram: None,
        collectives: None,
        cache: None,
        offload: None,
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
    assert!(matches!(
        value.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
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
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    changed = original.clone();
    changed.buffers[3].address = 0x3008;
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    changed = original.clone();
    changed.buffers[3].strides = vec![2];
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    changed = original.clone();
    changed.buffers[3].address = usize::MAX - 8;
    assert!(changed.validate().is_err());
    changed = original;
    changed.graphs[0]
        .input_bindings
        .retain(|binding| binding.buffer_name != "frame_index");
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
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
    assert!(matches!(
        manifest.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    manifest.graphs.pop();
    manifest.graphs[0].input_bindings[2].host_offset = 8;
    assert!(matches!(
        manifest.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
}

#[test]
fn prefill_transport_alignment_is_independent_of_model_context_limit() {
    let mut manifest = manifest();
    manifest.graphs[0].kind = GraphKind::Prefill;
    manifest.graphs[0].batch_capacity = 1;
    for capacity in [1, 16, 64, 128, 256, 512, 3072, 4096, 8192, 262144, 262208] {
        manifest.graphs[0].token_capacity = capacity;
        manifest.validate().unwrap();
    }
    for capacity in [0, 2, 15, 32, 63, 65, 513, 4095, usize::MAX] {
        manifest.graphs[0].token_capacity = capacity;
        assert!(manifest.validate().is_err());
    }
}

#[test]
fn partial_uploads_reject_overlaps_reserved_inputs_and_misalignment() {
    let manifest = manifest();
    let mut storage = Vec::new();
    transport::validate_uploads(
        &manifest,
        0,
        &[Upload {
            buffer: "tokens",
            offset: 8,
            bytes: &[1; 8],
        }],
        &mut storage,
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
            }],
            &mut storage,
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
            }],
            &mut storage,
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
            }],
            &mut storage,
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
            ],
            &mut storage,
        )
        .is_err()
    );
}

#[test]
fn upload_and_completion_failures_keep_distinct_domains() {
    let manifest = manifest();
    let mut storage = Vec::new();
    let uploads = [Upload {
        buffer: "tokens",
        offset: 1,
        bytes: &[1; 8],
    }];
    let error = match transport::validate_uploads(&manifest, 0, &uploads, &mut storage) {
        Err(error) => error,
        Ok(_) => panic!("misaligned upload was accepted"),
    };
    assert!(matches!(error, Error::Execution(ExecutionError::Upload(_))));
    assert_eq!(
        error.to_string(),
        "unaligned or out-of-range upload: tokens"
    );
    assert!(storage.is_empty());

    let error = transport::decode_records(&[0; 8], Fence(1), 4, 8).unwrap_err();
    assert!(matches!(
        error,
        Error::Execution(ExecutionError::Completion(_))
    ));
    assert_eq!(error.to_string(), "completion extent mismatch");
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
    let decoded = transport::decode_records(&bytes(records), Fence(9), 4, 8).unwrap();
    assert_eq!(decoded[3].generation, 7);
    assert_eq!(decoded[3].token_id, 88);
    assert_eq!(decoded[0].status, 0);
    assert!(transport::decode_records(&bytes(records), Fence(10), 4, 8).is_err());
    records[3][0] = 0;
    assert!(transport::decode_records(&bytes(records), Fence(9), 4, 8).is_err());
    records[3] = [7, 9, 65, 2, -1, 2, 1, 0];
    assert!(transport::decode_records(&bytes(records), Fence(9), 4, 8).is_err());
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
            assert!(m.validate().is_ok());
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
fn split_graphs_require_single_lane_confidence_and_deliver_embeddings_only_to_verification() {
    let mut m = manifest();
    m.max_active = 1;
    m.abi_version = SPECULATIVE_ABI_VERSION;
    m.buffers[2].shape = vec![2, 1, 24];
    m.buffers[2].strides = vec![24, 24, 1];
    m.buffers[2].nbytes = 384;
    m.graphs[0].batch_capacity = 1;
    m.buffers.push(BufferDescriptor {
        name: "proposal_confidence".into(),
        address: 0x40000,
        nbytes: 28,
        dtype: Dtype::Float32,
        shape: vec![1, 7],
        strides: vec![7, 1],
    });
    for (name, kind, delivery) in [
        ("proposal", GraphKind::Proposal, false),
        ("verification", GraphKind::Verification, true),
    ] {
        assert_eq!(GraphKind::try_from(name).unwrap(), kind);
        assert_eq!(crate::engram::consumes(kind), delivery);
        assert!(crate::model::contract::GraphOperation::try_from(kind).is_ok());
        m.graphs[0].kind = kind;
        for width in 1..=8 {
            m.graphs[0].token_capacity = width;
            if kind == GraphKind::Proposal {
                if width == 1 {
                    assert!(m.validate().is_err());
                    continue;
                }
                let confidence = m.buffers.last_mut().unwrap();
                confidence.shape = vec![1, width - 1];
                confidence.strides = vec![width - 1, 1];
                confidence.nbytes = (width - 1) * 4;
            }
            assert!(m.validate().is_ok());
        }
        m.graphs[0].token_capacity = 9;
        assert!(m.validate().is_err());
    }
}

#[test]
fn vector_parser_rejects_counts_tails_and_stale_epochs_without_cuda() {
    let mut vector = TokenVector {
        emitted_count: 3,
        offered_drafts: 7,
        accepted_drafts: 3,
        copy_offered_mask: 3,
        model_reserved: 65,
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
        let mut bytes = vec![0; 4 * VECTOR_RECORD_FIELDS * 8];
        for (dst, field) in bytes[..VECTOR_RECORD_FIELDS * 8]
            .chunks_exact_mut(8)
            .zip(row.fields())
        {
            dst.copy_from_slice(&field.to_le_bytes());
        }
        bytes
    };
    assert_eq!(
        transport::decode_records(&encoded(good), Fence(9), 4, VECTOR_RECORD_FIELDS).unwrap()[0],
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
            5 => v.draft_cache_consumed = -2,
            6 => row.token_id = 40,
            _ => row.submission_sequence = 8,
        }
        assert!(
            transport::decode_records(&encoded(row), Fence(9), 4, VECTOR_RECORD_FIELDS).is_err(),
            "change{change}"
        );
    }
}

#[test]
fn resolved_uploads_reject_unknown_missing_duplicate_and_invalid_ranges() {
    let manifest = manifest();
    manifest.validate().unwrap();
    let mut storage = Vec::new();
    let valid = Upload {
        buffer: "tokens",
        offset: 8,
        bytes: &[1; 8],
    };
    for uploads in [
        vec![Upload {
            buffer: "missing",
            offset: 0,
            bytes: &[0; 8],
        }],
        vec![Upload {
            buffer: "completion_records",
            offset: 0,
            bytes: &[0; 8],
        }],
        vec![Upload {
            buffer: "submission_sequence",
            offset: 0,
            bytes: &[0; 8],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &[],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &[0; 7],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: usize::MAX - 7,
            bytes: &[0; 16],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: 24,
            bytes: &[0; 16],
        }],
        vec![
            Upload {
                buffer: "tokens",
                offset: 8,
                bytes: &[0; 8],
            },
            Upload {
                buffer: "tokens",
                offset: 8,
                bytes: &[0; 8],
            },
        ],
        vec![
            Upload {
                buffer: "tokens",
                offset: 0,
                bytes: &[0; 16],
            },
            Upload {
                buffer: "tokens",
                offset: 8,
                bytes: &[0; 16],
            },
        ],
    ] {
        assert!(transport::validate_uploads(&manifest, 0, &uploads, &mut storage).is_err());
    }
    assert!(transport::validate_uploads(&manifest, 1, &[], &mut storage).is_err());
    let mut missing = manifest.clone();
    missing.buffers.retain(|buffer| buffer.name != "tokens");
    assert!(transport::validate_uploads(&missing, 0, &[valid], &mut storage).is_err());
    missing = manifest.clone();
    missing.graphs[0]
        .input_bindings
        .retain(|binding| binding.buffer_name != "frame_index");
    assert!(transport::validate_uploads(&missing, 0, &[], &mut storage).is_err());

    // Failure can leave scratch descriptors, but the next validation clears them.
    let accepted = [Upload {
        buffer: "tokens",
        offset: 24,
        bytes: &[5; 8],
    }];
    let resolved = transport::validate_uploads(&manifest, 0, &accepted, &mut storage).unwrap();
    assert_eq!(
        resolved.copies().collect::<Vec<_>>(),
        vec![
            transport::ResolvedUpload {
                host_offset: 40,
                device_address: 0x4018,
                nbytes: 8
            },
            transport::ResolvedUpload {
                host_offset: 0,
                device_address: 0x1000,
                nbytes: 4
            },
            transport::ResolvedUpload {
                host_offset: 8,
                device_address: 0x2000,
                nbytes: 8
            },
        ]
    );
}

#[test]
fn failed_validation_never_changes_a_staged_frame() {
    let manifest = manifest();
    let mut storage = Vec::new();
    let mut frame = vec![0xA5; manifest.input_frame_bytes];
    let before = frame.clone();
    let uploads = [
        Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &[1; 8],
        },
        Upload {
            buffer: "tokens",
            offset: 4,
            bytes: &[2; 8],
        },
    ];
    if let Ok(resolved) = transport::validate_uploads(&manifest, 0, &uploads, &mut storage) {
        resolved.stage(&mut frame, 0, 1);
        panic!("invalid command reached frame staging");
    }
    assert_eq!(frame, before);
    // An omitted upload is valid; only executor-owned input words are staged.
    let resolved = transport::validate_uploads(&manifest, 0, &[], &mut storage).unwrap();
    resolved.stage(&mut frame, 1, 2);
    assert_eq!(&frame[..4], &1i32.to_le_bytes());
    assert_eq!(&frame[8..16], &2i64.to_le_bytes());
    assert_eq!(&frame[4..8], &before[4..8]);
    assert_eq!(&frame[16..], &before[16..]);
    assert_eq!(resolved.copies().count(), 2);
}

#[test]
fn reusable_resolution_preserves_two_frames_and_exact_partial_copy_ranges() {
    let manifest = manifest();
    manifest.validate().unwrap();
    let mut storage = Vec::with_capacity(12);
    let pointer = storage.as_ptr();
    let capacity = storage.capacity();
    let mut frames = [
        vec![0xA5; manifest.input_frame_bytes],
        vec![0x5A; manifest.input_frame_bytes],
    ];
    let mut device = vec![0xCC; 0x4020];
    let first = {
        let bytes = 0x0123_4567_89AB_CDEFi64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 8,
            bytes: &bytes,
        }];
        let resolved = transport::validate_uploads(&manifest, 0, &uploads, &mut storage).unwrap();
        resolved.stage(&mut frames[0], 0, 10);
        let copies = resolved.copies().collect::<Vec<_>>();
        assert_eq!(&frames[0][24..32], &bytes);
        copies
    };
    let staged = frames[0].clone();
    {
        let bytes = (-77i64).to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 24,
            bytes: &bytes,
        }];
        let resolved = transport::validate_uploads(&manifest, 0, &uploads, &mut storage).unwrap();
        resolved.stage(&mut frames[1], 1, 11);
        assert_eq!(&frames[1][40..48], &bytes);
    }
    assert_eq!(frames[0], staged);
    assert_eq!(storage.as_ptr(), pointer);
    assert_eq!(storage.capacity(), capacity);
    // The first pinned frame remains sufficient after source bytes and descriptors
    // were replaced. This CPU transfer model checks the actual resolved ranges.
    for binding in first {
        device[binding.device_address..binding.device_address + binding.nbytes]
            .copy_from_slice(&frames[0][binding.host_offset..binding.host_offset + binding.nbytes]);
    }
    assert_eq!(&device[0x1000..0x1004], &0i32.to_le_bytes());
    assert_eq!(&device[0x2000..0x2008], &10i64.to_le_bytes());
    assert_eq!(
        &device[0x4008..0x4010],
        &0x0123_4567_89AB_CDEFi64.to_le_bytes()
    );
    assert_eq!(&device[0x4000..0x4008], &[0xCC; 8]);
    assert_eq!(&device[0x4010..0x4020], &[0xCC; 16]);
    let uploads = [Upload {
        buffer: "tokens",
        offset: 16,
        bytes: &[0x71; 8],
    }];
    let resolved = transport::validate_uploads(&manifest, 0, &uploads, &mut storage).unwrap();
    resolved.stage(&mut frames[0], 0, 12);
    for binding in resolved.copies() {
        device[binding.device_address..binding.device_address + binding.nbytes]
            .copy_from_slice(&frames[0][binding.host_offset..binding.host_offset + binding.nbytes]);
    }
    assert_eq!(
        &device[0x4008..0x4010],
        &0x0123_4567_89AB_CDEFi64.to_le_bytes()
    );
    assert_eq!(&device[0x4010..0x4018], &[0x71; 8]);
    assert_eq!(&device[0x4018..0x4020], &[0xCC; 8]);
    assert_eq!(&device[0x2000..0x2008], &12i64.to_le_bytes());
}

#[test]
fn fragmented_upload_descriptors_reuse_their_highwater_capacity() {
    let manifest = manifest();
    let mut storage = Vec::with_capacity(1);
    let uploads = [24, 0, 16, 8].map(|offset| Upload {
        buffer: "tokens",
        offset,
        bytes: &[0x31; 8],
    });
    let mut frame = vec![0xA5; manifest.input_frame_bytes];
    transport::validate_uploads(&manifest, 0, &uploads, &mut storage)
        .unwrap()
        .stage(&mut frame, 0, 1);
    let pointer = storage.as_ptr();
    let capacity = storage.capacity();
    assert!(capacity >= 4);
    assert_eq!(&frame[16..48], &[0x31; 32]);
    for sequence in 2..258 {
        let count = sequence as usize % 5;
        let resolved =
            transport::validate_uploads(&manifest, 0, &uploads[..count], &mut storage).unwrap();
        assert_eq!(resolved.copies().count(), count + 2);
        resolved.stage(&mut frame, 0, sequence);
        assert_eq!(storage.as_ptr(), pointer);
        assert_eq!(storage.capacity(), capacity);
    }
}

#[test]
fn mixed_manifest_requires_independent_prompt_capacity_and_a_role_input() {
    let mut m = manifest();
    m.abi_version = SPECULATIVE_ABI_VERSION;
    m.buffers[2].shape = vec![2, 4, 24];
    m.buffers[2].strides = vec![96, 24, 1];
    m.buffers[2].nbytes = 1536;
    m.graphs[0].kind = GraphKind::Mixed;
    m.graphs[0].prefill_capacity = 512;
    m.graphs[0].token_capacity = 8;
    assert!(m.validate().is_err());
    m.buffers.push(BufferDescriptor {
        name: "prefill_row".into(),
        address: 0x6000,
        nbytes: 4,
        dtype: Dtype::Int32,
        shape: vec![1],
        strides: vec![1],
    });
    let offset = m.input_frame_bytes;
    m.input_frame_bytes += 8;
    m.graphs[0].input_bindings.push(InputBinding {
        buffer_name: "prefill_row".into(),
        host_offset: offset,
        nbytes: 4,
    });
    m.validate().unwrap();
    assert_eq!(m.graphs[0].packed_tokens(), 536);
    for (batch, width, prefill) in [
        (1, 8, 512),
        (4, 9, 512),
        (4, 8, 0),
        (4, 8, 65),
        (4, 8, 1usize << 31),
    ] {
        let mut bad = m.clone();
        bad.graphs[0].batch_capacity = batch;
        bad.graphs[0].token_capacity = width;
        bad.graphs[0].prefill_capacity = prefill;
        assert!(bad.validate().is_err());
    }
    m.graphs[0].kind = GraphKind::Speculative;
    assert!(m.validate().is_err());
}
