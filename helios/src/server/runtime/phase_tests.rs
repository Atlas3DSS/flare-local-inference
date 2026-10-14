use super::*;
use crate::{
    cache::Fence,
    execution::{CompletionRecord, FrameCompletion},
    frontend::{Asset, Frontend, FrontendAssets},
    model::qwen4exp::{PromptData, Qwen4Exp},
    scheduler::SchedulerLimits,
};
use sha2::{Digest, Sha256};
use std::ops::ControlFlow;
use std::sync::atomic::AtomicUsize;

struct Fixture {
    config: ModelConfig,
    frontend: Frontend,
    stop: AtomicBool,
    metrics: Arc<super::super::metrics::Metrics>,
}

impl Fixture {
    fn new() -> Self {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "flare-worker-phase-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let vocabulary: tokenizers::models::bpe::Vocab = ["a", "b", "c", "d", "x"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
        let mut tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocabulary, vec![])
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer
            .add_special_tokens([tokenizers::AddedToken::from("<eos>", true)])
            .unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = directory.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset(
                    "tokenizer.json",
                    tokenizer.to_string(false).unwrap().as_bytes(),
                ),
                tokens: Default::default(),
                chat_template: asset("chat.jinja", b"{{ messages[0].content }}"),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        Self {
            config: ModelConfig {
                prepared_sequence_tokens: 128,
                contract: Qwen4Exp::metadata(),
                identity: [23; 32],
                model_id: "qwen-worker-phases".into(),
                limits: SchedulerLimits {
                    request_slots: 4,
                    max_active: 4,
                    max_sequence_tokens: 128,
                    prefill_chunk_size: 64,
                    pages: 16,
                    snapshots: 16,
                },
                eos_token_ids: vec![5],
                sampling_defaults: SamplingParameters::default(),
                max_new_tokens: 16,
                speculative_widths: vec![],
                speculative_fixed_width: None,
                history_drafts: false,
                speculative_proposal_source: Default::default(),
                state_rows_per_request: None,
            },
            frontend,
            stop: AtomicBool::new(false),
            metrics: Arc::new(Default::default()),
        }
    }

    fn worker(&self, receiver: mpsc::Receiver<Vec<Incoming<Qwen4Exp>>>) -> Worker<'_, Qwen4Exp> {
        self.worker_for(receiver)
    }

    fn worker_for<M: ModelContract>(
        &self,
        receiver: mpsc::Receiver<Vec<Incoming<M>>>,
    ) -> Worker<'_, M> {
        let mut scheduler = RequestScheduler::new(
            self.config.identity,
            self.config.limits,
            M::scheduler_capabilities(&self.config),
        )
        .unwrap();
        let mut graphs = Vec::new();
        for batch in [1, 2, 4] {
            for kind in [
                GraphKind::Init,
                GraphKind::Restore,
                GraphKind::Snapshot,
                GraphKind::Resume,
                GraphKind::Decode,
                GraphKind::Speculative,
                GraphKind::Prefill,
            ] {
                for width in 1..=if kind == GraphKind::Speculative { 8 } else { 1 } {
                    graphs.push(crate::execution::GraphDescriptor {
                        name: format!("{kind:?}-{batch}-{width}"),
                        kind,
                        sampling_mode: crate::execution::SamplingMode::General,
                        handle: 1,
                        batch_capacity: batch,
                        token_capacity: if kind == GraphKind::Prefill {
                            64
                        } else {
                            width
                        },
                        prefill_capacity: 0,
                        estimated_duration_ns: 1000,
                        input_bindings: vec![],
                    });
                }
            }
        }
        scheduler.measured_graph_durations(&graphs).unwrap();
        let mut worker = Worker::new(
            &self.config,
            &self.frontend,
            &self.stop,
            &self.metrics,
            scheduler,
            receiver,
            4,
        );
        worker.inputs = Some(Default::default());
        worker
    }

    fn incoming(&self, tokens: Vec<u32>, limit: u32) -> (Incoming<Qwen4Exp>, Generation) {
        let length = tokens.len();
        self.incoming_for(
            tokens,
            limit,
            PromptData {
                positions: (0..length).map(|n| [n as i64; 3]).collect(),
                vision_token_indices: vec![-1; length],
            },
        )
    }

    fn incoming_for<M: ModelContract>(
        &self,
        tokens: Vec<u32>,
        limit: u32,
        data: M::PromptData,
    ) -> (Incoming<M>, Generation) {
        let (events, receiver) = async_mpsc::channel(64);
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        (
            Incoming {
                _capacity: None,
                arrival_ns: super::super::observation::monotonic_ns(),
                publications: Default::default(),
                prompt: Prompt {
                    tokens,
                    system_boundary: None,
                    data,
                    images: None,
                    conditioning: vec![],
                    sampling: SamplingParameters::default(),
                    max_new_tokens: limit,
                    stop_sequences: vec![],
                    stop_strings: vec![],
                    ignore_eos: false,
                    grammar: None,
                },
                events,
                cancelled: cancelled.clone(),
                failure: failure.clone(),
                metrics: self.metrics.clone(),
                observation: Some(false),
                host_lookup: true,
                readmitted: false,
            },
            Generation {
                events: receiver,
                cancelled,
                failure,
                enqueue_ns: None,
            },
        )
    }
}

fn planned(worker: &mut Worker<'_, Qwen4Exp>, kind: GraphKind, sequence: u64) -> ScheduledPlan {
    let (step, _) = worker.plan(0).unwrap();
    let step = step.unwrap();
    assert_eq!(step.kind, kind);
    worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
    step
}

fn completion(
    worker: &Worker<'_, impl ModelContract>,
    step: &ScheduledStep,
    sequence: u64,
    sampled: Option<u32>,
) -> FrameCompletion {
    let mut records = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let live = worker.scheduler.request(lane.request).unwrap();
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: sequence as i64,
            consumed_tokens: (lane.start_position + lane.tokens as u64) as i64,
            generated_tokens: live.generated_tokens as i64 + i64::from(sampled.is_some()),
            token_id: sampled.map_or(-1, i64::from),
            status: 1,
            ..Default::default()
        };
    }
    FrameCompletion {
        fence: Fence(sequence),
        graph_index: 0,
        timing: Default::default(),
        records: records.into(),
    }
}

fn advance(worker: &mut Worker<'_, Qwen4Exp>, kind: GraphKind, sequence: u64, token: Option<u32>) {
    let step = planned(worker, kind, sequence);
    let frame = completion(worker, &step, sequence, token);
    worker.complete(kind, frame).unwrap();
}

#[test]
fn all_lanes_validate_before_scheduler_or_observation_mutation() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (first, mut first_generation) = fixture.incoming(vec![0], 8);
    let (second, mut second_generation) = fixture.incoming(vec![1], 8);
    sender.send(vec![first]).unwrap();
    sender.send(vec![second]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let step = planned(&mut worker, GraphKind::Init, 1);
    assert_eq!(step.lanes.len(), 2);
    let mut frame = completion(&worker, &step, 1, None);
    frame.records[step.lanes[1].request.slot as usize].token_id =
        fixture.config.contract.vocabulary as i64;
    assert!(matches!(
        worker.complete(GraphKind::Init, frame),
        Err(Error::Execution(ExecutionError::Completion(_)))
    ));
    assert_eq!(worker.scheduler.completed(), Fence(0));
    assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(1000));
    assert_eq!(worker.scheduler.submitted_step(Fence(1)).unwrap(), &*step);
    for lane in &step.lanes {
        let live = worker.scheduler.request(lane.request).unwrap();
        assert_eq!(live.phase, RequestPhase::Initialize);
        assert_eq!((live.consumed_tokens, live.generated_tokens), (0, 0));
    }
    assert!(first_generation.events.try_recv().is_err());
    assert!(second_generation.events.try_recv().is_err());
    worker
        .complete(GraphKind::Init, completion(&worker, &step, 1, None))
        .unwrap();
    assert_eq!(worker.scheduler.completed(), Fence(1));
    assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(1000));
}

#[test]
fn worker_observes_prefill_decode_and_prerequisite_timings() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 13], 8);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    for (kind, fence, token) in [
        (GraphKind::Init, 1, None),
        (GraphKind::Prefill, 2, Some(1)),
        (GraphKind::Snapshot, 3, None),
        (GraphKind::Decode, 4, Some(2)),
    ] {
        let step = planned(&mut worker, kind, fence);
        let mut frame = completion(&worker, &step, fence, token);
        frame.timing.gpu_ns = [
            Some(100),
            Some(200),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        frame.timing.wall_ns = 350;
        worker.complete(kind, frame).unwrap();
        assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(900));
    }
}

#[test]
fn worker_discovers_indeterminate_costs_and_records_actual_progress() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 13], 8);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker
        .scheduler
        .graph_durations(&crate::scheduler::tests::duration_graphs(4))
        .unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let h = worker.active[0].as_ref().unwrap().request;
    for (kind, fence, token) in [
        (GraphKind::Init, 1, None),
        (GraphKind::Prefill, 2, Some(1)),
        (GraphKind::Snapshot, 3, None),
        (GraphKind::Decode, 4, Some(2)),
    ] {
        let step = planned(&mut worker, kind, fence);
        assert_eq!(step.estimated_duration_ns(), 0);
        assert!(step.requires_feedback);
        let mut frame = completion(&worker, &step, fence, token);
        frame.timing.gpu_ns = [
            Some(100),
            Some(200),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        worker.complete(kind, frame).unwrap();
        assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(200));
    }
    let accounting = worker.scheduler.request(h).unwrap().accounting();
    assert_eq!(accounting.committed_prefill_tokens, 13);
    assert_eq!(accounting.committed_output_tokens, 2);
}

#[test]
fn output_clock_advances_only_after_stop_buffering_releases_tokens() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming(vec![0], 8);
    let arrival = incoming.arrival_ns;
    incoming.prompt.stop_sequences = vec![vec![1, 2]];
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let h = worker.active[0].as_ref().unwrap().request;
    assert_eq!(
        worker.scheduler.request(h).unwrap().accounting().arrival_ns,
        arrival
    );
    advance(&mut worker, GraphKind::Init, 1, None);
    advance(&mut worker, GraphKind::Prefill, 2, Some(1));
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .published_deliveries,
        0
    );
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_ns,
        None
    );
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::ObservedToken { token: 1, .. })
    ));
    assert!(generation.events.try_recv().is_err());
    advance(&mut worker, GraphKind::Snapshot, 3, None);
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_ns,
        None
    );
    advance(&mut worker, GraphKind::Decode, 4, Some(3));
    let published = worker
        .scheduler
        .request(h)
        .unwrap()
        .accounting()
        .last_output_ns;
    assert!(published.is_some());
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .published_deliveries,
        1
    );
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_fence,
        Some(Fence(4))
    );
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::ObservedToken { token: 3, .. })
    ));
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::Token(1))
    ));
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::Token(3))
    ));
    advance(&mut worker, GraphKind::Decode, 5, Some(1));
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .published_deliveries,
        1
    );
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_ns,
        published
    );
}

#[test]
fn packing_sees_reserved_pages_and_pack_failure_retains_submission_until_drain() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 65], 8);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    advance(&mut worker, GraphKind::Init, 1, None);
    let handle = worker.active[0].as_ref().unwrap().request;
    assert_eq!(worker.scheduler.request(handle).unwrap().pages().count(), 0);
    let step = planned(&mut worker, GraphKind::Prefill, 2);
    let pages: Vec<_> = worker.scheduler.request(handle).unwrap().pages().collect();
    assert_eq!(pages.len(), 1);
    let uploads = Qwen4Exp::uploads(worker.inputs.as_ref().unwrap());
    let page_upload = uploads
        .iter()
        .find(|upload| upload.buffer == "page_tables")
        .unwrap();
    assert_eq!(page_upload.offset, 0);
    assert_eq!(page_upload.bytes, (pages[0].0 as i32).to_le_bytes());
    worker
        .complete(GraphKind::Prefill, completion(&worker, &step, 2, None))
        .unwrap();
    let step = worker.plan(0).unwrap().0.unwrap();
    assert_eq!(
        (step.kind, step.lanes[0].start_position),
        (GraphKind::Prefill, 64)
    );
    worker.admissions[handle.slot as usize] = None;
    assert!(worker.reserve_and_pack(&step, Fence(3)).is_err());
    assert_eq!(worker.scheduler.submitted_step(Fence(3)).unwrap(), &*step);
    assert_eq!(worker.scheduler.request(handle).unwrap().pages().count(), 2);
    assert_eq!(
        worker.scheduler.request(handle).unwrap().last_use(),
        Fence(3)
    );
    let free = worker.scheduler.free_page_count();
    worker.retire(&[handle]).unwrap();
    assert_eq!(worker.scheduler.free_page_count(), free);
    worker
        .scheduler
        .release_after_shutdown(&Shutdown {
            drained: true,
            error: None,
        })
        .unwrap();
    assert_eq!(
        worker.scheduler.free_page_count(),
        fixture.config.limits.pages as usize
    );
}

#[test]
fn exact_system_and_prompt_checkpoints_publish_only_after_snapshot_completion() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming(vec![0; 13], 2);
    incoming.prompt.system_boundary = Some(10);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    advance(&mut worker, GraphKind::Init, 1, None);
    advance(&mut worker, GraphKind::Prefill, 2, None);
    assert_eq!(
        worker.scheduler.request(handle).unwrap().consumed_tokens,
        10
    );
    assert_eq!(worker.scheduler.stats().system_checkpoints, 0);
    assert!(!worker.publish().unwrap());
    let snapshot = planned(&mut worker, GraphKind::Snapshot, 3);
    let uploads = Qwen4Exp::uploads(worker.inputs.as_ref().unwrap());
    let expected = uploads
        .iter()
        .find(|u| u.buffer == "state_expected_consumed")
        .unwrap();
    assert_eq!(&expected.bytes[..8], &10i64.to_le_bytes());
    assert_eq!(worker.scheduler.stats().system_checkpoints, 0);
    worker
        .complete(GraphKind::Snapshot, completion(&worker, &snapshot, 3, None))
        .unwrap();
    assert_eq!(worker.scheduler.stats().system_checkpoints, 1);
    assert_eq!(
        worker
            .scheduler
            .request(handle)
            .unwrap()
            .reused_prompt_tokens(),
        0
    );
    advance(&mut worker, GraphKind::Prefill, 4, Some(1));
    assert_eq!(
        worker.scheduler.request(handle).unwrap().consumed_tokens,
        13
    );
    assert_eq!(worker.scheduler.stats().prompt_checkpoints, 0);
    assert!(!worker.publish().unwrap());
    advance(&mut worker, GraphKind::Snapshot, 5, None);
    assert_eq!(worker.scheduler.stats().prompt_checkpoints, 1);
    advance(&mut worker, GraphKind::Decode, 6, Some(2));
    assert!(!worker.publish().unwrap());
    advance(&mut worker, GraphKind::Snapshot, 7, None);
    assert!(worker.publish().unwrap());
    assert!(!worker.publish().unwrap());
    assert_eq!(worker.scheduler.stats().snapshot_publications, 3);
    let metrics = fixture.metrics.snapshot(false);
    assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 2));
    let mut finished = 0;
    while let Ok(event) = generation.events.try_recv() {
        if let GenerationEvent::Finished {
            generated_tokens,
            reused_prompt_tokens,
            reason,
            ..
        } = event
        {
            assert_eq!(
                (generated_tokens, reused_prompt_tokens, reason),
                (2, 0, "length")
            );
            finished += 1;
        }
    }
    assert_eq!(finished, 1);
    let (repeat, _generation) = fixture.incoming(vec![0; 13], 2);
    sender.send(vec![repeat]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let live = worker
        .scheduler
        .request(worker.active[0].as_ref().unwrap().request)
        .unwrap();
    assert_eq!(
        (live.phase, live.reused_prompt_tokens()),
        (RequestPhase::Restore, 13)
    );
    advance(&mut worker, GraphKind::Restore, 8, None);
    advance(&mut worker, GraphKind::Resume, 9, Some(1));
    let metrics = fixture.metrics.snapshot(false);
    assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 3));
}

#[test]
fn disabled_host_store_preserves_decode_prefix_reuse_and_cancellation() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, mut generation) = fixture.incoming(vec![0; 13], 1);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert!(worker.store.is_none());
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    for (kind, fence, token) in [
        (GraphKind::Init, 1, None),
        (GraphKind::Prefill, 2, Some(1)),
        (GraphKind::Snapshot, 3, None),
    ] {
        advance(&mut worker, kind, fence, token);
        assert!(!worker.service_store().unwrap());
        assert!(worker.scheduler.take_publications().is_empty());
    }
    assert!(worker.publish().unwrap());
    assert!(worker.scheduler.take_publications().is_empty());
    assert!(matches!(
        generation.events.try_recv().unwrap(),
        GenerationEvent::ObservedToken { token: 1, .. }
    ));
    let mut finished = false;
    while let Ok(event) = generation.events.try_recv() {
        match event {
            GenerationEvent::Finished {
                generated_tokens: 1,
                reused_prompt_tokens: 0,
                ..
            } => finished = true,
            GenerationEvent::Error(error) => panic!("{error}"),
            _ => {}
        }
    }
    assert!(finished);
    let (incoming, mut generation) = fixture.incoming(vec![0; 13], 2);
    sender.send(vec![incoming]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    let request = worker.scheduler.request(handle).unwrap();
    assert_eq!(
        (request.phase, request.reused_prompt_tokens()),
        (RequestPhase::Restore, 13)
    );
    advance(&mut worker, GraphKind::Restore, 4, None);
    advance(&mut worker, GraphKind::Resume, 5, Some(2));
    assert!(matches!(
        generation.events.try_recv().unwrap(),
        GenerationEvent::ObservedToken { token: 2, .. }
    ));
    generation.cancelled.store(true, Ordering::Release);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.scheduler.request(handle).is_err());
    assert!(!worker.service_store().unwrap());
    assert_eq!(fixture.metrics.cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(worker.prefill_started.capacity(), 0);
    assert_eq!(worker.prefill_rate, crate::offload::INITIAL_PREFILL_RATE);
    let stats = worker.scheduler.stats();
    assert_eq!((stats.cache_hits, stats.reused_tokens), (1, 13));
    assert_eq!(
        (stats.host_lookups, stats.host_hits, stats.inbound_waiting),
        (0, 0, 0)
    );
    let metrics = fixture.metrics.snapshot(false);
    assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 2));
    assert_eq!(
        (metrics.host_memory_records, metrics.host_disk_records),
        (0, 0)
    );
    assert_eq!((metrics.host_bytes_out, metrics.host_bytes_in), (0, 0));
    let mut shutdown = Shutdown {
        drained: true,
        error: None,
    };
    worker.close_store(&mut shutdown);
    assert!(shutdown.drained);
    assert!(shutdown.error.is_none());
}

#[test]
fn speculative_dashboard_counts_live_rounds_and_batched_forward_passes() {
    for copy in [false, true] {
        let mut fixture = Fixture::new();
        fixture.config.speculative_widths = vec![1, 4];
        fixture.config.speculative_fixed_width = Some(4);
        fixture.config.history_drafts = copy;
        if copy {
            fixture.config.speculative_proposal_source =
                crate::scheduler::speculation::ProposalSource::Copy;
        }
        let (sender, receiver) = mpsc::channel();
        let (incoming, _generation) = fixture.incoming(vec![0; 13], 8);
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        worker.scheduler.enable_speculation(&[1, 4]).unwrap();
        worker.scheduler.fixed_speculative_width(Some(4)).unwrap();
        worker
            .scheduler
            .proposal_source(fixture.config.speculative_proposal_source)
            .unwrap();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(&mut worker, GraphKind::Prefill, 2, Some(0));
        advance(&mut worker, GraphKind::Snapshot, 3, None);
        if copy {
            let handle = worker.active[0].as_ref().unwrap().request;
            worker
                .scheduler
                .set_copy_proposal(
                    handle,
                    crate::scheduler::speculation::CopyProposal {
                        consumed: 13,
                        tokens: [0, 0, 0, -1, -1, -1, -1],
                        count: 3,
                        grammar_masks: vec![],
                    },
                )
                .unwrap();
        }
        let step = planned(&mut worker, GraphKind::Speculative, 4);
        assert_eq!(step.token_capacity, 4);
        let mut frame = completion(&worker, &step, 4, None);
        frame.timing.gpu_ns[0] = Some(1000);
        frame.timing.wall_ns = 1500;
        let record = &mut frame.records[step.lanes[0].request.slot as usize];
        record.consumed_tokens = 16;
        record.generated_tokens = 4;
        record.token_id = 2;
        record.vector = Some(crate::execution::TokenVector {
            emitted_count: 3,
            offered_drafts: 3,
            accepted_drafts: 2,
            copy_offered_mask: if copy { 7 } else { 0 },
            verification_width: 4,
            draft_cache_consumed: if copy { -1 } else { 15 },
            tokens: [1, 1, 2, -1, -1, -1, -1, -1],
            ..Default::default()
        });
        let mut invalid = frame.records;
        invalid[step.lanes[0].request.slot as usize].generated_tokens = 20;
        assert!(
            worker
                .complete(
                    GraphKind::Speculative,
                    FrameCompletion {
                        records: invalid,
                        ..frame
                    }
                )
                .is_err()
        );
        assert_eq!(fixture.metrics.snapshot(!copy).generated_tokens, 1);
        worker.complete(GraphKind::Speculative, frame).unwrap();
        let profile = worker
            .scheduler
            .request(step.lanes[0].request)
            .unwrap()
            .acceptance();
        let source = fixture.config.speculative_proposal_source;
        assert!(profile.probability(source, 0).unwrap() > 0.5);
        assert!(profile.probability(source, 2).unwrap() < 0.5);
        assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(1000));
        let metrics = fixture.metrics.snapshot(!copy);
        assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 4));
        assert_eq!((metrics.accepted_drafts, metrics.draft_rounds), (2, 1));
        assert_eq!(metrics.verify_steps, 1);
        assert_eq!(metrics.draft_steps, if copy { 0 } else { 3 });
    }
}

#[test]
fn observations_precede_grammar_failures_and_token_stop_completion() {
    for invalid in [false, true] {
        let fixture = Fixture::new();
        let compiler = crate::grammar::GrammarCompiler::new(
            fixture.frontend.tokenizer(),
            fixture.config.contract.vocabulary,
            &[5],
        )
        .unwrap();
        let (sender, receiver) = mpsc::channel();
        let (mut incoming, mut generation) = fixture.incoming(vec![0], 8);
        if invalid {
            incoming.prompt.grammar = Some(compiler.compile_lark("start: \"a\"").unwrap());
        } else {
            incoming.prompt.stop_sequences = vec![vec![0]];
        }
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(
            &mut worker,
            GraphKind::Prefill,
            2,
            Some(if invalid { 4 } else { 0 }),
        );
        assert!(
            matches!(generation.events.try_recv().unwrap(), GenerationEvent::ObservedToken { token, timestamp_ns: None } if token == if invalid { 4 } else { 0 })
        );
        match generation.events.try_recv().unwrap() {
            GenerationEvent::Error(Error::Execution(ExecutionError::Completion(_))) if invalid => {
                assert_eq!(fixture.metrics.failed.load(Ordering::Relaxed), 1);
            }
            GenerationEvent::Finished {
                reason: "stop",
                termination: crate::output::Termination::UserStop,
                ..
            } if !invalid => {
                assert_eq!(fixture.metrics.completed.load(Ordering::Relaxed), 1);
            }
            event => panic!("unexpected completion event: {event:?}"),
        }
        assert!(generation.events.try_recv().is_err());
        assert_eq!(worker.scheduler.stats().active_requests, 0);
        assert!(worker.active.iter().all(Option::is_none));
        assert!(worker.admissions.iter().all(Option::is_none));
    }
}

#[test]
fn fatal_broadcast_preserves_first_failure_and_reaches_pending_and_ingress() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (first, first_generation) = fixture.incoming(vec![0], 8);
    first.fail(Error::execution(
        ExecutionError::Completion,
        "first completion failure",
    ));
    sender.send(vec![first]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let (pending, pending_generation) = fixture.incoming(vec![1], 8);
    worker.pending.push_back(pending);
    let (queued, queued_generation) = fixture.incoming(vec![2], 8);
    sender.send(vec![queued]).unwrap();
    worker.fail_all(&Error::execution(ExecutionError::Worker, "worker failed"));
    assert_eq!(
        first_generation.failure().to_string(),
        "first completion failure"
    );
    assert!(matches!(
        first_generation.failure(),
        Error::Execution(ExecutionError::Completion(_))
    ));
    for generation in [pending_generation, queued_generation] {
        assert_eq!(generation.failure().to_string(), "worker failed");
        assert!(matches!(
            generation.failure(),
            Error::Execution(ExecutionError::Worker(_))
        ));
    }
    assert_eq!(fixture.metrics.failed.load(Ordering::Relaxed), 3);
    assert_eq!(worker.scheduler.stats().active_requests, 1);
}

/// A host store that answers lookups from a script and reports completions on demand.
struct FakeStore {
    hit: Option<crate::offload::HostHit>,
    reject_fetch: bool,
    stored: Vec<crate::offload::PublishedCheckpoint>,
    fetched: Vec<(crate::offload::HostHit, crate::offload::InboundTicket)>,
    events: VecDeque<crate::offload::StoreEvent>,
    close_error: Option<Error>,
    close_panics: bool,
    close_calls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl FakeStore {
    fn new(hit: Option<crate::offload::HostHit>) -> Self {
        Self {
            hit,
            reject_fetch: false,
            stored: Vec::new(),
            fetched: Vec::new(),
            events: VecDeque::new(),
            close_error: None,
            close_panics: false,
            close_calls: Arc::new(AtomicUsize::new(0)),
            drops: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl crate::offload::PrefixStore for FakeStore {
    fn lookup(
        &mut self,
        history: &crate::cache::PrefixHistory,
        resident: u64,
    ) -> Option<crate::offload::HostHit> {
        self.hit.clone().filter(|hit| {
            hit.consumed_tokens > resident && hit.consumed_tokens <= history.token_count()
        })
    }
    fn store(&mut self, checkpoint: &crate::offload::PublishedCheckpoint) -> Result<bool> {
        self.stored.push(checkpoint.clone());
        Ok(true)
    }
    fn fetch(
        &mut self,
        hit: crate::offload::HostHit,
        ticket: crate::offload::InboundTicket,
    ) -> Result<()> {
        if self.reject_fetch {
            return Err("fetch rejected".into());
        }
        self.fetched.push((hit, ticket));
        Ok(())
    }
    fn poll(&mut self, events: &mut Vec<crate::offload::StoreEvent>) {
        events.extend(self.events.drain(..));
    }
    fn stats(&self) -> crate::offload::StoreStats {
        crate::offload::StoreStats::default()
    }
    fn close(&mut self) -> Result<()> {
        self.close_calls.fetch_add(1, Ordering::Relaxed);
        assert!(!self.close_panics, "transfer drain panicked");
        self.close_error.clone().map_or(Ok(()), Err)
    }
}

impl Drop for FakeStore {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn shutdown_releases_the_store_after_both_drains_complete() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let store = FakeStore::new(None);
    let close_calls = store.close_calls.clone();
    let drops = store.drops.clone();
    worker.attach_store(Some(Box::new(store)));
    let mut shutdown = Shutdown {
        drained: true,
        error: None,
    };
    worker.close_store(&mut shutdown);
    assert!(shutdown.drained);
    assert!(shutdown.error.is_none());
    assert!(worker.store.is_none());
    assert_eq!(close_calls.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    worker.close_store(&mut shutdown);
    drop(worker);
    assert_eq!(close_calls.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn shutdown_retains_the_store_when_its_drain_fails() {
    for previous_error in [None, Some(Error::from("model failed"))] {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.close_error = Some("transfer drain failed".into());
        let close_calls = store.close_calls.clone();
        let drops = store.drops.clone();
        worker.attach_store(Some(Box::new(store)));
        let expected_error = previous_error
            .clone()
            .unwrap_or_else(|| "transfer drain failed".into());
        let mut shutdown = Shutdown {
            drained: true,
            error: previous_error,
        };
        worker.close_store(&mut shutdown);
        assert!(!shutdown.drained);
        assert_eq!(
            shutdown.error.unwrap().to_string(),
            expected_error.to_string()
        );
        assert!(worker.store.is_none());
        drop(worker);
        assert_eq!(close_calls.load(Ordering::Relaxed), 1);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn shutdown_retains_the_store_without_closing_when_the_executor_did_not_drain() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let store = FakeStore::new(None);
    let close_calls = store.close_calls.clone();
    let drops = store.drops.clone();
    worker.attach_store(Some(Box::new(store)));
    let mut shutdown = Shutdown {
        drained: false,
        error: Some("model drain failed".into()),
    };
    worker.close_store(&mut shutdown);
    assert!(!shutdown.drained);
    assert_eq!(shutdown.error.unwrap().to_string(), "model drain failed");
    assert!(worker.store.is_none());
    drop(worker);
    assert_eq!(close_calls.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
}

#[test]
fn shutdown_retains_the_store_when_its_drain_panics() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.close_panics = true;
    let close_calls = store.close_calls.clone();
    let drops = store.drops.clone();
    worker.attach_store(Some(Box::new(store)));
    let mut shutdown = Shutdown {
        drained: true,
        error: None,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker.close_store(&mut shutdown);
    }));
    assert!(result.is_err());
    assert!(!shutdown.drained);
    assert!(worker.store.is_none());
    drop(worker);
    assert_eq!(close_calls.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
}

fn host_hit(history: &crate::cache::PrefixHistory, consumed: u64) -> crate::offload::HostHit {
    let key = history.key_at(consumed).unwrap();
    crate::offload::HostHit {
        key,
        consumed_tokens: consumed,
        resident_tokens: 0,
        pages: (consumed / 64) as usize,
        bytes: 1 << 20,
        tier: crate::offload::Tier::Memory,
        estimated_seconds: 0.0,
        chain: crate::offload::index::Match {
            key,
            consumed_tokens: consumed,
            resident_tokens: 0,
            nodes: vec![key],
            skip_pages: 0,
            pages: (consumed / 64) as usize,
        },
    }
}

fn fake_store<'w>(worker: &'w mut Worker<'_, Qwen4Exp>) -> &'w mut FakeStore {
    let store = worker.store.as_mut().unwrap().as_mut() as *mut dyn crate::offload::PrefixStore;
    // SAFETY: The worker holds the only store and it was constructed as a FakeStore.
    unsafe { &mut *(store as *mut FakeStore) }
}

#[test]
fn host_hit_holds_the_request_inbound_until_the_restore_completes() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(&history, 70))))));
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    let live = worker.scheduler.request(handle).unwrap();
    assert_eq!(
        (
            live.phase,
            live.consumed_tokens,
            live.reused_prompt_tokens()
        ),
        (RequestPhase::Inbound, 70, 70)
    );
    assert!(worker.plan(0).unwrap().0.is_none());
    assert_eq!(worker.scheduler.stats().inbound_waiting, 1);
    assert_eq!(
        (
            worker.scheduler.stats().host_lookups,
            worker.scheduler.stats().host_hits
        ),
        (1, 1)
    );
    let (hit, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
    assert_eq!((ticket.key, ticket.pages.len()), (hit.key, 1));
    fake_store(&mut worker)
        .events
        .push_back(crate::offload::StoreEvent::Fetched {
            key: ticket.key,
            result: Ok(1 << 20),
        });
    assert!(worker.service_store().unwrap());
    let live = worker.scheduler.request(handle).unwrap();
    assert_eq!(
        (live.phase, live.pages().count()),
        (RequestPhase::Restore, 1)
    );
    let restored = live.prefix().unwrap();
    // The restored checkpoint became ready during this pass; the next pass
    // drains its publication and copies it out under its own pin.
    assert_eq!(fake_store(&mut worker).stored.len(), 0);
    assert!(worker.service_store().unwrap());
    assert_eq!(fake_store(&mut worker).stored.len(), 1);
    let publication = fake_store(&mut worker).stored[0].clone();
    assert_eq!(
        (
            publication.id,
            publication.chain_pages.len(),
            publication.tail_pages
        ),
        (restored, 1, 1)
    );
    assert_eq!(publication.key, history.key_at(70).unwrap());
    assert_eq!(
        worker
            .scheduler
            .cache()
            .checkpoint(restored)
            .unwrap()
            .pin_count(),
        2
    );
    fake_store(&mut worker)
        .events
        .push_back(crate::offload::StoreEvent::Stored {
            checkpoint: restored,
            key: publication.key,
            result: Ok(1 << 20),
        });
    assert!(worker.service_store().unwrap());
    assert_eq!(
        worker
            .scheduler
            .cache()
            .checkpoint(restored)
            .unwrap()
            .pin_count(),
        1
    );
    advance(&mut worker, GraphKind::Restore, 1, None);
    advance(&mut worker, GraphKind::Resume, 2, Some(1));
    assert_eq!(
        worker.scheduler.request(handle).unwrap().phase,
        RequestPhase::Decode
    );
    assert_eq!(fixture.metrics.snapshot(false).prefill_tokens, 0);
}

#[test]
fn host_hit_recomputes_when_snapshot_backing_would_prevent_admission() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker
        .scheduler
        .share_snapshot_pages(&vec![0..16; 16])
        .unwrap();
    worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(&history, 70))))));
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    assert_eq!(
        worker.scheduler.request(handle).unwrap().phase,
        RequestPhase::Initialize
    );
    assert_eq!(worker.scheduler.free_page_count(), 14);
    assert_eq!(worker.scheduler.free_snapshot_count(), 16);
    assert!(fake_store(&mut worker).fetched.is_empty());
}

#[test]
fn failed_restores_readmit_the_request_without_the_host_store() {
    for reject_at_fetch in [false, true] {
        let fixture = Fixture::new();
        let (sender, receiver) = mpsc::channel();
        let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
        let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
        history.append(&incoming.prompt.tokens, &[]).unwrap();
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(Some(host_hit(&history, 70)));
        store.reject_fetch = reject_at_fetch;
        worker.attach_store(Some(Box::new(store)));
        let free_pages = worker.scheduler.free_page_count();
        let free_snapshots = worker.scheduler.free_snapshot_count();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        if !reject_at_fetch {
            let handle = worker.active[0].as_ref().unwrap().request;
            assert_eq!(
                worker.scheduler.request(handle).unwrap().phase,
                RequestPhase::Inbound
            );
            let (_, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
            fake_store(&mut worker)
                .events
                .push_back(crate::offload::StoreEvent::Fetched {
                    key: ticket.key,
                    result: Err("transfer failed".into()),
                });
            assert!(worker.service_store().unwrap());
            assert!(worker.active[0].is_none());
            assert_eq!(worker.pending.len(), 1);
            assert!(!worker.pending[0].host_lookup);
            assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        }
        let handle = worker.active[0].as_ref().unwrap().request;
        let live = worker.scheduler.request(handle).unwrap();
        assert_eq!(
            (
                live.phase,
                live.consumed_tokens,
                live.reused_prompt_tokens()
            ),
            (RequestPhase::Initialize, 0, 0)
        );
        assert_eq!(fixture.metrics.admitted.load(Ordering::Relaxed), 1);
        assert_eq!(worker.scheduler.stats().inbound_waiting, 0);
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(&mut worker, GraphKind::Prefill, 2, None);
        advance(&mut worker, GraphKind::Prefill, 3, Some(1));
        assert_eq!(
            worker.scheduler.request(handle).unwrap().consumed_tokens,
            70
        );
        worker.retire(&[handle]).unwrap();
        worker
            .scheduler
            .release_after_shutdown(&Shutdown {
                drained: true,
                error: None,
            })
            .unwrap();
        assert_eq!(worker.scheduler.free_page_count(), free_pages);
        assert_eq!(worker.scheduler.free_snapshot_count(), free_snapshots);
    }
}

fn group_runtime(
    fixture: &Fixture,
    capacity: usize,
) -> (
    ModelRuntime<Qwen4Exp>,
    mpsc::Receiver<Vec<Incoming<Qwen4Exp>>>,
) {
    let (sender, receiver) = mpsc::sync_channel(2);
    (
        ModelRuntime {
            sender,
            stop: Arc::new(AtomicBool::new(false)),
            healthy: Arc::new(AtomicBool::new(true)),
            config: Arc::new(fixture.config.clone()),
            event_capacity: 16,
            capacity: Arc::new(tokio::sync::Semaphore::new(capacity)),
            thread: Mutex::new(None),
            shutdown: Mutex::new(None),
            finished: Arc::new((Mutex::new(false), Condvar::new())),
            metrics: fixture.metrics.clone(),
        },
        receiver,
    )
}
#[test]
fn grouped_submission_validates_all_children_before_enqueue_and_releases_capacity() {
    let fixture = Fixture::new();
    let (runtime, receiver) = group_runtime(&fixture, 4);
    let good = fixture.incoming(vec![1, 2], 4).0.prompt;
    let mut bad = fixture.incoming(vec![1, 2], 4).0.prompt;
    bad.tokens.clear();
    assert!(runtime.generate_many(vec![good, bad]).is_err());
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert_eq!(runtime.capacity.available_permits(), 4);
    let prompts = (0..3)
        .map(|_| fixture.incoming(vec![1, 2], 4).0.prompt)
        .collect();
    let generations = runtime.generate_many(prompts).unwrap();
    let group = receiver.try_recv().unwrap();
    assert_eq!(group.len(), 3);
    assert_eq!(runtime.capacity.available_permits(), 1);
    let prompts = (0..2)
        .map(|_| fixture.incoming(vec![1, 2], 4).0.prompt)
        .collect();
    assert!(runtime.generate_many(prompts).is_err());
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(generations);
    assert!(group.iter().all(|r| r.cancelled.load(Ordering::Acquire)));
    drop(group);
    assert_eq!(runtime.capacity.available_permits(), 4);
}

fn exercise_text_preemption<M: ModelContract>(
    metadata: crate::model::ContractMetadata,
    data: impl Fn(usize) -> M::PromptData,
    evict_checkpoint: bool,
) {
    let mut fixture = Fixture::new();
    fixture.config.contract = metadata;
    fixture.config.prepared_sequence_tokens = 4096;
    fixture.config.limits.max_sequence_tokens = 4096;
    fixture.config.limits.pages = 6;
    fixture.config.limits.max_active = 2;
    let page_tokens = M::scheduler_capabilities(&fixture.config).page_tokens;
    let budget = (page_tokens * 4) as u32;
    let (sender, receiver) = mpsc::channel();
    let (first, mut first_generation) = fixture.incoming_for::<M>(vec![0; 60], budget, data(60));
    let (second, mut second_generation) = fixture.incoming_for::<M>(vec![2; 60], budget, data(60));
    sender.send(vec![first, second]).unwrap();
    let mut worker = fixture.worker_for::<M>(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert_eq!(worker.scheduler.free_page_count(), 2);
    let older = worker.active[0].as_ref().unwrap().request;
    let mut sequence = 0;
    while worker.parked.is_empty() {
        sequence += 1;
        assert!(sequence < (page_tokens * 8) as u64, "no request was parked");
        let step = worker
            .plan(0)
            .unwrap()
            .0
            .expect("page pressure must make progress");
        worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
        let sample = match step.kind {
            GraphKind::Decode | GraphKind::Resume => Some(1),
            GraphKind::Prefill => {
                let lane = &step.lanes[0];
                let live = worker.scheduler.request(lane.request).unwrap();
                (lane.start_position as usize + lane.tokens == live.prompt_tokens).then_some(1)
            }
            _ => None,
        };
        let frame = completion(&worker, &step, sequence, sample);
        worker.complete(step.kind, frame).unwrap();
        worker.publish().unwrap();
        for generation in [&mut first_generation, &mut second_generation] {
            while let Ok(event) = generation.events.try_recv() {
                assert!(!matches!(
                    event,
                    GenerationEvent::Error(_) | GenerationEvent::AdmissionRejected(_)
                ));
            }
        }
    }
    assert!(worker.scheduler.stats().preemptions > 0);
    let (incoming, parked) = worker.parked.front_mut().unwrap();
    assert!(parked.carried.generated_tokens > 0);
    let history_len = incoming.prompt.tokens.len();
    assert_eq!(history_len, 60 + parked.carried.generated_tokens as usize);
    assert_eq!(
        incoming.prompt.max_new_tokens + parked.carried.generated_tokens,
        budget
    );
    M::validate_prompt_data(
        &fixture.config,
        &incoming.prompt.tokens,
        &incoming.prompt.data,
    )
    .unwrap();
    let metadata = M::prompt_metadata(&mut incoming.prompt.data);
    if let Some(positions) = metadata.positions {
        assert_eq!(positions.last(), Some(&[history_len as i64 - 1; 3]));
    }
    if let Some(indices) = metadata.vision_token_indices {
        assert_eq!(indices, &vec![-1; history_len]);
    }
    worker.retire(&[older]).unwrap();
    if evict_checkpoint {
        let pressure = worker
            .scheduler
            .admit(RequestSpec {
                prompt: vec![3; page_tokens * 5],
                max_new_tokens: 1,
                growable_output: false,
                system_boundary: None,
                conditioning: vec![],
                grammar_constrained: false,
                host_stops: false,
            })
            .unwrap();
        worker.scheduler.cancel(pressure).unwrap();
    }
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.parked.is_empty());
    let resumed = worker.active.iter().flatten().next().unwrap().request;
    let live = worker.scheduler.request(resumed).unwrap();
    if evict_checkpoint {
        assert!(live.consumed_tokens < history_len as u64 - 1);
    } else {
        assert_eq!(live.phase, RequestPhase::Restore);
        assert_eq!(live.consumed_tokens, history_len as u64 - 1);
    }
    let mut recomputed = false;
    for _ in 0..history_len / 64 + 8 {
        sequence += 1;
        let step = worker.plan(0).unwrap().0.unwrap();
        recomputed |= step.kind == GraphKind::Prefill
            && step.lanes[0].start_position < history_len as u64 - 1;
        worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
        let sample = match step.kind {
            GraphKind::Decode | GraphKind::Resume => Some(1),
            GraphKind::Prefill => {
                let lane = &step.lanes[0];
                (lane.start_position as usize + lane.tokens == history_len).then_some(1)
            }
            _ => None,
        };
        let frame = completion(&worker, &step, sequence, sample);
        worker.complete(step.kind, frame).unwrap();
        worker.publish().unwrap();
        if worker.scheduler.request(resumed).unwrap().generated_tokens > 0 {
            assert_eq!(recomputed, evict_checkpoint);
            assert!(
                worker.active[resumed.slot as usize]
                    .as_ref()
                    .unwrap()
                    .carried
                    .generated_tokens
                    > 0
            );
            return;
        }
    }
    panic!("readmitted request did not generate output");
}

#[test]
fn text_preemption_restores_or_recomputes_every_attention_contract() {
    use crate::model::{deepseek4, deepseek41::DeepSeek41, glm5next::Glm5Next, mimo};
    for evict in [false, true] {
        exercise_text_preemption::<Qwen4Exp>(
            Qwen4Exp::metadata(),
            |n| PromptData {
                positions: (0..n).map(|p| [p as i64; 3]).collect(),
                vision_token_indices: vec![-1; n],
            },
            evict,
        );
        exercise_text_preemption::<Glm5Next>(Glm5Next::metadata(), |_| (), evict);
        exercise_text_preemption::<DeepSeek41>(DeepSeek41::metadata(), |_| (), evict);
        exercise_text_preemption::<deepseek4::DeepSeek4>(
            deepseek4::DeepSeek4::metadata(),
            |n| deepseek4::PromptData {
                vision_token_indices: vec![-1; n],
                spans: vec![],
            },
            evict,
        );
        exercise_text_preemption::<mimo::MiMo>(
            mimo::MiMo::metadata(),
            |n| mimo::PromptData {
                vision_token_indices: vec![-1; n],
            },
            evict,
        );
    }
}

#[test]
fn text_admission_waits_for_pages_and_conditioned_requests_reserve_their_limit() {
    let mut fixture = Fixture::new();
    fixture.config.prepared_sequence_tokens = 2048;
    fixture.config.limits.max_sequence_tokens = 2048;
    fixture.config.limits.pages = 4;
    let (sender, receiver) = mpsc::channel();
    let (first, _first_generation) = fixture.incoming(vec![0; 60], 1024);
    let (second, _second_generation) = fixture.incoming(vec![1; 60], 1024);
    let (third, mut third_generation) = fixture.incoming(vec![2; 60], 1024);
    sender.send(vec![first, second, third]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert_eq!(worker.pending.len(), 1);
    assert!(third_generation.events.try_recv().is_err());
    let first = worker.active[0].as_ref().unwrap().request;
    worker.retire(&[first]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.pending.is_empty());
    assert_eq!(worker.active.iter().flatten().count(), 2);

    let (sender, receiver) = mpsc::channel();
    let (mut conditioned, mut generation) = fixture.incoming(vec![0; 60], 1024);
    conditioned.prompt.conditioning.push(ConditioningEvent {
        position: 0,
        identity: [3; 32],
    });
    sender.send(vec![conditioned]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.active.iter().all(Option::is_none));
    assert!(matches!(
        generation.events.try_recv().unwrap(),
        GenerationEvent::AdmissionRejected(_)
    ));
}
