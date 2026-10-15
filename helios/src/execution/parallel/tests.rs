use std::io::Write;

use super::*;
use crate::execution::{CompletionRecord, Dtype};

#[test]
fn logical_layout_excludes_process_addresses_and_detects_contract_changes() {
    let manifest = super::super::tests::manifest();
    let expected = layout(&manifest);
    let mut peer = manifest.clone();
    peer.context += 10;
    peer.device += 1;
    for buffer in &mut peer.buffers {
        buffer.address += 0x10000;
    }
    peer.graphs[0].handle += 0x10000;
    assert_eq!(expected, layout(&peer));
    peer.buffers[3].dtype = Dtype::Float32;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.graphs[0].token_capacity += 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.graphs[0].prefill_capacity = 64;
    assert_ne!(expected, layout(&peer));
    peer = manifest;
    peer.graphs[0].input_bindings[2].host_offset += 8;
    assert_ne!(expected, layout(&peer));
}

#[test]
fn rank_layout_requires_equal_cache_capacities_and_snapshot_exclusions() {
    let mut manifest = super::super::tests::manifest();
    let mut cache = crate::offload::geometry::tests::geometry(64, &[("kv", 4096)], 65536);
    cache.snapshot_pages = vec![100..120, 115..135];
    manifest.cache = Some(cache);
    let expected = layout(&manifest);
    let mut peer = manifest.clone();
    let cache = peer.cache.as_mut().unwrap();
    cache.row.address += 0x10000;
    cache.planes[0].address += 0x20000;
    assert_eq!(expected, layout(&peer));
    peer.cache.as_mut().unwrap().snapshot_pages[0].start += 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.cache.as_mut().unwrap().page_capacity -= 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.cache.as_mut().unwrap().row_capacity -= 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest;
    peer.cache = None;
    assert_ne!(expected, layout(&peer));
}

#[test]
fn fragmented_completion_remains_pending_until_all_bytes_arrive() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut channel = Channel::new(receiver).unwrap();
    channel.nonblocking(true).unwrap();
    sender.write_all(&[COMPLETE, 3]).unwrap();
    assert!(channel.poll().unwrap().is_none());
    sender.write_all(&[0, 0, 0, 10]).unwrap();
    assert!(channel.poll().unwrap().is_none());
    sender.write_all(&[11, 12]).unwrap();
    assert_eq!(channel.poll().unwrap(), Some((COMPLETE, vec![10, 11, 12])));
    drop(sender);
    assert!(
        channel
            .poll()
            .unwrap_err()
            .to_string()
            .contains("disconnected")
    );
}

#[test]
fn packet_bounds_and_peer_errors_are_reported() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut channel = Channel::new(receiver).unwrap();
    sender.write_all(&[COMPLETE, 255, 255, 255, 255]).unwrap();
    assert!(
        channel
            .poll()
            .unwrap_err()
            .to_string()
            .contains("oversized")
    );
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut sender = Channel::new(sender).unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    sender.send(FAILED, b"rank fence differs").unwrap();
    assert!(
        receiver
            .receive(READY)
            .unwrap_err()
            .to_string()
            .contains("rank fence differs")
    );
}

#[test]
fn upload_decoder_rejects_unbounded_or_truncated_fragments() {
    let manifest = super::super::tests::manifest();
    let mut bytes = Vec::new();
    number(&mut bytes, usize::MAX);
    assert!(uploads(&manifest, &mut Reader(&bytes)).is_err());
    bytes.clear();
    for number_value in [1, 3, 8, 8] {
        number(&mut bytes, number_value);
    }
    bytes.extend_from_slice(&7_i64.to_le_bytes());
    let decoded = uploads(&manifest, &mut Reader(&bytes)).unwrap();
    assert_eq!(decoded[0].buffer, "tokens");
    assert_eq!(decoded[0].offset, 8);
    assert_eq!(decoded[0].bytes, 7_i64.to_le_bytes());
    assert!(uploads(&manifest, &mut Reader(&bytes[..bytes.len() - 1])).is_err());
}

fn completed(status: i64) -> FrameCompletion {
    let record = CompletionRecord {
        generation: 7,
        submission_sequence: 3,
        consumed_tokens: 65,
        generated_tokens: 0,
        token_id: -1,
        status,
        error_code: 0,
        reserved: 0,
        vector: None,
    };
    FrameCompletion {
        proposal_confidence: None,
        fence: Fence(3),
        graph_index: 2,
        timing: Default::default(),
        records: [record; 4].into(),
    }
}

#[test]
fn peers_agree_on_request_consumption_without_sampling() {
    let mut root = completed(2);
    root.records[0].generated_tokens = 2;
    root.records[0].token_id = 19;
    let mut peer = completed(1);
    agree(&root, &peer).unwrap();
    peer.records[0].consumed_tokens -= 1;
    assert!(agree(&root, &peer).is_err());
    peer = completed(1);
    peer.records[0].generation += 1;
    assert!(agree(&root, &peer).is_err());
    peer = completed(3);
    assert!(agree(&root, &peer).is_err());
    root = completed(3);
    agree(&root, &peer).unwrap();
}

#[test]
fn completion_wire_preserves_the_execution_abi() {
    let mut root = completed(1);
    root.timing.gpu_ns[0] = Some(1250);
    root.timing.wall_ns = 2345;
    root.timing.queued = true;
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    assert_eq!(reader.number().unwrap(), 3);
    assert_eq!(reader.number().unwrap(), 2);
    assert_eq!(timing(&mut reader).unwrap(), root.timing);
    let records = super::super::transport::decode_records(reader.0, Fence(3), 4, 8).unwrap();
    for (expected, actual) in root.records.iter().zip(records.iter()) {
        assert_eq!(expected.fields(), actual.fields());
    }
}

#[test]
fn rank_timing_rejects_truncation_and_preserves_zero_duration() {
    for gpu in [None, Some(0), Some(731)] {
        let mut root = completed(1);
        root.timing.gpu_ns[0] = gpu;
        root.timing.wall_ns = 1234;
        let bytes = completion(&root);
        for length in 0..32 {
            assert!(timing(&mut Reader(&bytes[16..16 + length])).is_err());
        }
        assert_eq!(timing(&mut Reader(&bytes[16..])).unwrap(), root.timing);
    }
    let mut invalid = vec![];
    for value in [2, 0, 1234, 0] {
        number(&mut invalid, value);
    }
    assert!(timing(&mut Reader(&invalid)).is_err());
}

#[test]
fn rank_completion_requires_every_record() {
    let root = completed(1);
    let mut peer = completed(1);
    peer.records = [peer.records[0]; 3].into();
    assert!(agree(&root, &peer).is_err());
}
