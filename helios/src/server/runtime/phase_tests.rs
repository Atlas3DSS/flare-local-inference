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
                tokenizer_config: asset("tokenizer_config.json", b"{}"),
                chat_template: asset("chat.jinja", b"{{ messages[0].content }}"),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        Self {
            config: ModelConfig {
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
            },
            frontend,
            stop: AtomicBool::new(false),
            metrics: Arc::new(Default::default()),
        }
    }

    fn worker(&self, receiver: mpsc::Receiver<Incoming<Qwen4Exp>>) -> Worker<'_, Qwen4Exp> {
        let scheduler = RequestScheduler::new(
            self.config.identity,
            self.config.limits,
            Qwen4Exp::scheduler_capabilities(&self.config),
        )
        .unwrap();
        let mut worker = Worker::new(
            &self.config,
            &self.frontend,
            &self.stop,
            &self.metrics,
            scheduler,
            receiver,
            4,
            None,
        );
        worker.inputs = Some(Default::default());
        worker
    }

    fn incoming(&self, tokens: Vec<u32>, limit: u32) -> (Incoming<Qwen4Exp>, Generation) {
        let (events, receiver) = async_mpsc::channel(64);
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let length = tokens.len();
        (
            Incoming {
                prompt: Prompt {
                    tokens,
                    system_boundary: None,
                    data: PromptData {
                        positions: (0..length).map(|n| [n as i64; 3]).collect(),
                        vision_token_indices: vec![-1; length],
                    },
                    images: None,
                    conditioning: vec![],
                    sampling: SamplingParameters::default(),
                    max_new_tokens: limit,
                    stop_sequences: vec![],
                    stop_strings: vec![],
                    grammar: None,
                },
                events,
                cancelled: cancelled.clone(),
                failure: failure.clone(),
                metrics: self.metrics.clone(),
                qualification: false,
                observation: Some(false),
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

fn planned(worker: &mut Worker<'_, Qwen4Exp>, kind: GraphKind, sequence: u64) -> ScheduledStep {
    let (step, _) = worker.plan(0).unwrap();
    let step = step.unwrap();
    assert_eq!(step.kind, kind);
    worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
    step
}

fn completion(
    worker: &Worker<'_, Qwen4Exp>,
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
        elapsed_ns: None,
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
    sender.send(first).unwrap();
    sender.send(second).unwrap();
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
    assert_eq!(worker.scheduler.submitted_step(Fence(1)).unwrap(), &step);
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
}

#[test]
fn packing_sees_reserved_pages_and_pack_failure_retains_submission_until_drain() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 65], 8);
    sender.send(incoming).unwrap();
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
    assert_eq!(worker.scheduler.submitted_step(Fence(3)).unwrap(), &step);
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
    sender.send(incoming).unwrap();
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
    sender.send(repeat).unwrap();
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
        sender.send(incoming).unwrap();
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
        frame.elapsed_ns = Some(1000);
        let record = &mut frame.records[step.lanes[0].request.slot as usize];
        record.consumed_tokens = 16;
        record.generated_tokens = 4;
        record.token_id = 2;
        record.vector = Some(crate::execution::TokenVector {
            emitted_count: 3,
            offered_drafts: 3,
            accepted_drafts: 2,
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
        sender.send(incoming).unwrap();
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
    sender.send(first).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let (pending, pending_generation) = fixture.incoming(vec![1], 8);
    worker.pending.push_back(pending);
    let (queued, queued_generation) = fixture.incoming(vec![2], 8);
    sender.send(queued).unwrap();
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
