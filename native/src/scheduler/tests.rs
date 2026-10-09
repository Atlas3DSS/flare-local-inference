use super::*;

fn scheduler(active: usize, pages: u32, snapshots: u32) -> RequestScheduler {
    RequestScheduler::new(
        [9; 32],
        SchedulerLimits {
            max_active: active,
            max_sequence_tokens: 2048,
            prefill_chunk_size: 512,
            pages,
            snapshots,
        },
    )
    .unwrap()
}
fn spec(tokens: usize, max_new: u32) -> RequestSpec {
    RequestSpec {
        prompt: (0..tokens as u32).collect(),
        conditioning: vec![],
        max_new_tokens: max_new,
        grammar_constrained: false,
        host_stops: false,
    }
}
fn records(
    handle: RequestHandle,
    fence: u64,
    consumed: i64,
    generated: i64,
    token: i64,
    status: i64,
) -> [CompletionRecord; 4] {
    let mut records = [CompletionRecord::default(); 4];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: fence as i64,
        consumed_tokens: consumed,
        generated_tokens: generated,
        token_id: token,
        status,
        error_code: 0,
        reserved: 0,
        vector: None,
    };
    records
}
fn init(scheduler: &mut RequestScheduler, fence: u64) {
    let step = scheduler.next_step().unwrap();
    assert!(matches!(step.kind, GraphKind::Init | GraphKind::Restore));
    scheduler.record_submission(&step, Fence(fence)).unwrap();
    let completions = state_records(scheduler, &step, fence);
    scheduler.complete(Fence(fence), &completions).unwrap();
}

#[test]
fn vision_images_run_after_admission_before_prefill_and_cancel_waits_for_copy() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(65, 2)).unwrap();
    scheduler.set_vision_images(handle, &[256, 1152]).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    assert_eq!((step.kind, step.token_capacity), (GraphKind::Vision, 256));
    scheduler.record_submission(&step, Fence(2)).unwrap();
    assert!(scheduler.next_step().is_none());
    scheduler
        .complete(Fence(2), &records(handle, 2, 0, 0, -1, 1))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!((step.kind, step.token_capacity), (GraphKind::Vision, 4096));
    scheduler.record_submission(&step, Fence(3)).unwrap();
    scheduler
        .complete(Fence(3), &records(handle, 3, 0, 0, -1, 1))
        .unwrap();
    assert_eq!(scheduler.next_step().unwrap().kind, GraphKind::Prefill);
    scheduler.cancel(handle).unwrap();
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    scheduler.set_vision_images(handle, &[256]).unwrap();
    init(&mut scheduler, 4);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(5)).unwrap();
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.admit(spec(64, 1)).is_err());
    scheduler
        .complete(Fence(5), &records(handle, 5, 0, 0, -1, 1))
        .unwrap();
    assert!(scheduler.admit(spec(64, 1)).is_ok());
}

fn freeze(scheduler: &mut RequestScheduler, handle: RequestHandle, fence: u64) {
    assert!(scheduler.finish(handle).is_err());
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Snapshot);
    scheduler.record_submission(&step, Fence(fence)).unwrap();
    assert!(scheduler.finish(handle).is_err());
    let completions = state_records(scheduler, &step, fence);
    scheduler.complete(Fence(fence), &completions).unwrap();
}

fn state_records(
    scheduler: &RequestScheduler,
    step: &ScheduledStep,
    fence: u64,
) -> [CompletionRecord; 4] {
    let mut output = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        let slot = lane.request.slot as usize;
        output[slot] = records(
            lane.request,
            fence,
            request.consumed_tokens as i64,
            request.generated_tokens as i64,
            -1,
            1,
        )[slot];
    }
    output
}

#[test]
fn inactive_state_completion_never_initializes_or_publishes_a_checkpoint() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(1)).unwrap();
    assert!(
        scheduler
            .complete(Fence(1), &[CompletionRecord::default(); 4])
            .is_err()
    );
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Initialize
    );
    let mut corrupt = state_records(&scheduler, &step, 1);
    corrupt[handle.slot as usize].consumed_tokens = 64;
    assert!(scheduler.complete(Fence(1), &corrupt).is_err());
    let mut corrupt = state_records(&scheduler, &step, 1);
    corrupt[handle.slot as usize].token_id = 7;
    assert!(scheduler.complete(Fence(1), &corrupt).is_err());
    scheduler
        .complete(Fence(1), &state_records(&scheduler, &step, 1))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler.record_candidate(handle, 64, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(handle, 2, 64, 1, 88, 2))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(3)).unwrap();
    assert!(
        scheduler
            .complete(Fence(3), &[CompletionRecord::default(); 4])
            .is_err()
    );
    assert!(!scheduler.request(handle).unwrap().candidate_frozen);
    assert!(scheduler.finish(handle).is_err());
    scheduler
        .complete(Fence(3), &state_records(&scheduler, &step, 3))
        .unwrap();
    assert!(scheduler.finish(handle).unwrap().is_some());
}

#[test]
fn prefill_buckets_mask_capacity_and_allocate_only_live_pages() {
    let mut scheduler = scheduler(1, 32, 2);
    let handle = scheduler.admit(spec(529, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    assert_eq!(
        (step.kind, step.token_capacity, step.lanes[0].tokens),
        (GraphKind::Prefill, 512, 512)
    );
    scheduler.record_submission(&step, Fence(2)).unwrap();
    assert_eq!(scheduler.free_page_count(), 23);
    scheduler.record_candidate(handle, 512, Fence(2)).unwrap();
    assert!(scheduler.record_candidate(handle, 529, Fence(2)).is_err());
    scheduler
        .complete(Fence(2), &records(handle, 2, 512, 0, -1, 1))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!((step.token_capacity, step.lanes[0].tokens), (64, 17));
    scheduler.record_submission(&step, Fence(3)).unwrap();
    assert_eq!(scheduler.request(handle).unwrap().pages().count(), 9);
    scheduler
        .complete(Fence(3), &records(handle, 3, 529, 1, 77, 1))
        .unwrap();
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Decode
    );
}

#[test]
fn configured_prefill_chunks_and_tails_preserve_alignment_and_live_page_extents() {
    for (chunk_size, tail, tail_capacity) in [(4096, 17, 64), (3072, 2050, 3072), (8192, 513, 1024)]
    {
        let mut scheduler = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                max_active: 1,
                max_sequence_tokens: 32768,
                prefill_chunk_size: chunk_size,
                pages: 512,
                snapshots: 2,
            },
        )
        .unwrap();
        let handle = scheduler.admit(spec(chunk_size + tail, 3)).unwrap();
        init(&mut scheduler, 1);
        let step = scheduler.next_step().unwrap();
        assert_eq!(
            (step.token_capacity, step.lanes[0].tokens),
            (chunk_size, chunk_size)
        );
        assert_eq!(step.lanes[0].start_position, 0);
        scheduler.record_submission(&step, Fence(2)).unwrap();
        assert_eq!(
            scheduler.request(handle).unwrap().pages().count(),
            chunk_size / 64
        );
        scheduler
            .record_candidate(handle, chunk_size as u64, Fence(2))
            .unwrap();
        scheduler
            .complete(Fence(2), &records(handle, 2, chunk_size as i64, 0, -1, 1))
            .unwrap();
        let step = scheduler.next_step().unwrap();
        assert_eq!(
            (step.token_capacity, step.lanes[0].tokens),
            (tail_capacity, tail)
        );
        assert_eq!(step.lanes[0].start_position, chunk_size as u64);
        scheduler.record_submission(&step, Fence(3)).unwrap();
        assert_eq!(
            scheduler.request(handle).unwrap().pages().count(),
            (chunk_size + tail).div_ceil(64)
        );
        scheduler
            .complete(
                Fence(3),
                &records(handle, 3, (chunk_size + tail) as i64, 1, 77, 1),
            )
            .unwrap();
        assert_eq!(
            scheduler.request(handle).unwrap().phase,
            RequestPhase::Decode
        );
    }
}

#[test]
fn invalid_prefill_limits_reject_before_request_resources_are_created() {
    for chunk_size in [0, 1, 16, 63, 65, 4095, 4160, usize::MAX] {
        let result = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                max_active: 1,
                max_sequence_tokens: 4096,
                prefill_chunk_size: chunk_size,
                pages: 64,
                snapshots: 2,
            },
        );
        assert!(result.err().unwrap().contains("prefill_chunk_size"));
    }
}

#[test]
fn cancellation_delays_slot_page_and_snapshot_reuse_until_gpu_fence() {
    let mut scheduler = scheduler(1, 2, 1);
    let handle = scheduler.admit(spec(64, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_page_count(), 0);
    assert_eq!(scheduler.free_snapshot_count(), 0);
    assert!(scheduler.admit(spec(64, 3)).is_err());
    scheduler
        .complete(Fence(2), &records(handle, 2, 64, 1, 10, 1))
        .unwrap();
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (2, 1)
    );
    let reused = scheduler.admit(spec(64, 3)).unwrap();
    assert_eq!(reused.slot, handle.slot);
    assert!(reused.generation > handle.generation);
    assert!(scheduler.cancel(handle).is_err());
}

#[test]
fn cancellation_waits_for_both_submissions_and_reused_slot_rejects_old_epoch() {
    let mut scheduler = scheduler(1, 4, 1);
    let old = scheduler.admit(spec(4, 8)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(old, 2, 4, 1, 77, 1))
        .unwrap();
    for fence in [3, 4] {
        let step = scheduler.next_step().unwrap();
        scheduler.record_submission(&step, Fence(fence)).unwrap();
    }
    scheduler.cancel(old).unwrap();
    assert_eq!(
        (
            scheduler.stats().active_requests,
            scheduler.stats().retiring_slots
        ),
        (0, 1)
    );
    scheduler
        .complete(Fence(3), &records(old, 3, 5, 2, 78, 1))
        .unwrap();
    assert_eq!(scheduler.stats().free_slots, 0);
    assert_eq!(scheduler.free_snapshot_count(), 0);
    assert!(scheduler.admit(spec(4, 8)).is_err());
    scheduler
        .complete(Fence(4), &records(old, 4, 6, 3, 79, 1))
        .unwrap();
    assert_eq!(
        (
            scheduler.stats().free_slots,
            scheduler.free_page_count(),
            scheduler.free_snapshot_count()
        ),
        (1, 4, 1)
    );
    let reused = scheduler.admit(spec(4, 8)).unwrap();
    assert_eq!(reused.slot, old.slot);
    assert!(reused.generation > old.generation);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(5)).unwrap();
    assert!(
        scheduler
            .complete(Fence(5), &records(old, 5, 0, 0, -1, 1))
            .is_err()
    );
    assert_eq!(
        scheduler.request(reused).unwrap().phase,
        RequestPhase::Initialize
    );
    scheduler
        .complete(Fence(5), &records(reused, 5, 0, 0, -1, 1))
        .unwrap();
    scheduler.cancel(reused).unwrap();
}

#[test]
fn snapshot_pressure_waits_for_cancelled_restore_before_reusing_storage() {
    let mut scheduler = scheduler(2, 4, 2);
    let unrelated = || {
        let mut request = spec(65, 1);
        request.prompt[0] = 999;
        request
    };
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler.record_candidate(seed, 64, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(seed, 2, 64, 1, 88, 2))
        .unwrap();
    freeze(&mut scheduler, seed, 3);
    let checkpoint = scheduler.finish(seed).unwrap().unwrap();
    let restored = scheduler.admit(spec(64, 1)).unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Restore);
    scheduler.record_submission(&step, Fence(4)).unwrap();
    assert!(scheduler.admit(unrelated()).is_err());
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        1
    );
    scheduler.cancel(restored).unwrap();
    assert!(scheduler.admit(unrelated()).is_err());
    assert_eq!(scheduler.stats().snapshot_evictions, 1);
    assert!(scheduler.cache().checkpoint(checkpoint).is_err());
    assert_eq!(scheduler.free_snapshot_count(), 0);
    scheduler
        .complete(Fence(4), &records(restored, 4, 64, 0, -1, 1))
        .unwrap();
    assert_eq!(
        (
            scheduler.stats().free_slots,
            scheduler.free_page_count(),
            scheduler.free_snapshot_count()
        ),
        (2, 4, 2)
    );
    let fresh = scheduler.admit(unrelated()).unwrap();
    assert_eq!(
        scheduler.request(fresh).unwrap().phase,
        RequestPhase::Initialize
    );
    scheduler.cancel(fresh).unwrap();
}

#[test]
fn terminal_checkpoint_full_prompt_hit_restores_and_resumes_without_prefill() {
    let mut scheduler = scheduler(2, 8, 4);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let prefill = scheduler.next_step().unwrap();
    scheduler.record_submission(&prefill, Fence(2)).unwrap();
    scheduler.record_candidate(handle, 64, Fence(2)).unwrap();
    assert_eq!(scheduler.stats().snapshot_publications, 0);
    assert!(scheduler.finish(handle).is_err());
    scheduler
        .complete(Fence(2), &records(handle, 2, 64, 1, 99, 2))
        .unwrap();
    freeze(&mut scheduler, handle, 3);
    let checkpoint = scheduler.finish(handle).unwrap().unwrap();
    assert_eq!(scheduler.stats().snapshot_publications, 1);
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        0
    );
    let first = scheduler.admit(spec(64, 2)).unwrap();
    let second = scheduler.admit(spec(64, 2)).unwrap();
    let stats = scheduler.stats();
    assert_eq!(
        (stats.cache_lookups, stats.cache_hits, stats.reused_tokens),
        (3, 2, 128)
    );
    assert_eq!(
        (
            stats.active_requests,
            stats.free_slots,
            stats.cached_checkpoints
        ),
        (2, 0, 1)
    );
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        2
    );
    assert_eq!(
        scheduler
            .request(first)
            .unwrap()
            .pages()
            .collect::<Vec<_>>(),
        scheduler
            .request(second)
            .unwrap()
            .pages()
            .collect::<Vec<_>>()
    );
    init(&mut scheduler, 4);
    let resume = scheduler.next_step().unwrap();
    assert_eq!(resume.kind, GraphKind::Resume);
    assert_eq!(resume.batch_capacity, 2);
    scheduler.cancel(first).unwrap();
    scheduler.cancel(second).unwrap();
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        0
    );
}

#[test]
fn cold_admission_skips_matching_checkpoint_but_retains_capacity_eviction() {
    let mut scheduler = scheduler(1, 2, 2);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let prefill = scheduler.next_step().unwrap();
    scheduler.record_submission(&prefill, Fence(2)).unwrap();
    scheduler.record_candidate(handle, 64, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(handle, 2, 64, 1, 99, 2))
        .unwrap();
    freeze(&mut scheduler, handle, 3);
    scheduler.finish(handle).unwrap().unwrap();

    let warm = scheduler.admit(spec(64, 64)).unwrap();
    assert_eq!(
        scheduler.request(warm).unwrap().phase,
        RequestPhase::Restore
    );
    assert_eq!(scheduler.request(warm).unwrap().reused_prompt_tokens(), 64);
    scheduler.cancel(warm).unwrap();
    let lookups = scheduler.stats().cache_lookups;
    let cold = scheduler.admit_cold(spec(64, 64)).unwrap();
    let request = scheduler.request(cold).unwrap();
    assert_eq!(request.phase, RequestPhase::Initialize);
    assert_eq!(request.consumed_tokens, 0);
    assert_eq!(request.reused_prompt_tokens(), 0);
    assert!(request.prefix().is_none());
    assert_eq!(scheduler.stats().cache_lookups, lookups);
    assert_eq!(scheduler.stats().snapshot_evictions, 1);
    assert_eq!(scheduler.free_page_count(), 0);
    scheduler.cancel(cold).unwrap();
    assert_eq!(scheduler.free_page_count(), 2);
    assert_eq!(scheduler.free_snapshot_count(), 2);
}

#[test]
fn appended_conversation_outlives_snapshot_capacity_without_losing_prefix_pages() {
    let mut scheduler = scheduler(1, 32, 3);
    for turn in 1..=12 {
        let tokens = turn * PAGE_SIZE;
        let handle = scheduler.admit(spec(tokens, 1)).unwrap_or_else(|error| {
            panic!(
                "turn {turn} failed admission: {error}; {:?}",
                scheduler.stats()
            )
        });
        assert_eq!(
            scheduler.request(handle).unwrap().consumed_tokens,
            (tokens - PAGE_SIZE) as u64
        );
        let fence = (turn as u64 - 1) * 3 + 1;
        init(&mut scheduler, fence);
        let step = scheduler.next_step().unwrap();
        assert_eq!(step.kind, GraphKind::Prefill);
        assert_eq!(step.lanes[0].tokens, PAGE_SIZE);
        scheduler
            .record_submission(&step, Fence(fence + 1))
            .unwrap();
        scheduler
            .record_candidate(handle, tokens as u64, Fence(fence + 1))
            .unwrap();
        scheduler
            .complete(
                Fence(fence + 1),
                &records(handle, fence + 1, tokens as i64, 1, 77, 2),
            )
            .unwrap();
        freeze(&mut scheduler, handle, fence + 2);
        let checkpoint = scheduler.finish(handle).unwrap().unwrap();
        let pages = scheduler.cache().page_chain(checkpoint).unwrap();
        assert_eq!(pages.len(), turn);
        assert_eq!(pages.iter().copied().collect::<BTreeSet<_>>().len(), turn);
        assert_eq!(scheduler.stats().cached_pages, turn);
        assert_eq!(scheduler.free_page_count(), 32 - turn);
    }
    scheduler.ensure_resources(32, 3).unwrap();
    assert_eq!(scheduler.free_page_count(), 32);
    assert_eq!(scheduler.free_snapshot_count(), 3);
}

#[test]
fn branching_requests_own_distinct_tail_pages_and_candidate_snapshots() {
    let mut scheduler = scheduler(2, 8, 4);
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler.record_candidate(seed, 64, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(seed, 2, 64, 1, 77, 2))
        .unwrap();
    freeze(&mut scheduler, seed, 3);
    scheduler.finish(seed).unwrap();
    let first = scheduler.admit(spec(65, 4)).unwrap();
    let second = scheduler.admit(spec(65, 4)).unwrap();
    init(&mut scheduler, 4);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(5)).unwrap();
    scheduler
        .complete(Fence(5), &records(first, 5, 65, 1, 88, 1))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Decode);
    scheduler.record_submission(&step, Fence(6)).unwrap();
    scheduler
        .complete(Fence(6), &records(first, 6, 66, 2, 89, 1))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.lanes[0].request, second);
    scheduler.record_submission(&step, Fence(7)).unwrap();
    scheduler
        .complete(Fence(7), &records(second, 7, 65, 1, 90, 1))
        .unwrap();
    let first = scheduler.request(first).unwrap();
    let second = scheduler.request(second).unwrap();
    assert_eq!(first.shared_pages, second.shared_pages);
    assert_ne!(first.private_pages, second.private_pages);
    assert_ne!(first.candidate_snapshot, second.candidate_snapshot);
}

#[test]
fn decode_allows_two_steps_and_duplicate_finished_record_emits_once() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(63, 2)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(handle, 2, 63, 1, 88, 1))
        .unwrap();
    for fence in [3, 4] {
        let step = scheduler.next_step().unwrap();
        scheduler.record_submission(&step, Fence(fence)).unwrap();
    }
    assert!(scheduler.next_step().is_none());
    let generated = scheduler
        .complete(Fence(3), &records(handle, 3, 64, 2, 89, 2))
        .unwrap();
    assert_eq!(generated.len(), 1);
    assert!(scheduler.finish(handle).is_err());
    let generated = scheduler
        .complete(Fence(4), &records(handle, 4, 64, 2, 89, 2))
        .unwrap();
    assert!(generated.is_empty());
    assert_eq!(scheduler.request(handle).unwrap().tokens().len(), 65);
    scheduler.finish(handle).unwrap();
}

#[test]
fn grammar_barrier_and_bad_completion_do_not_advance_request_state() {
    let mut scheduler = scheduler(1, 8, 2);
    let mut request = spec(65, 3);
    request.grammar_constrained = true;
    let handle = scheduler.admit(request).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    let bad = records(handle, 2, 66, 1, 88, 1);
    assert!(scheduler.complete(Fence(2), &bad).is_err());
    assert_eq!(scheduler.request(handle).unwrap().consumed_tokens, 0);
    scheduler
        .complete(Fence(2), &records(handle, 2, 65, 1, 88, 1))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(3)).unwrap();
    assert!(scheduler.next_step().is_none());
    scheduler
        .complete(Fence(3), &records(handle, 3, 66, 2, 89, 1))
        .unwrap();
}

#[test]
fn resource_exhaustion_rolls_back_admission_and_page_reservation() {
    let mut scheduler = scheduler(2, 1, 1);
    assert!(scheduler.admit(spec(65, 3)).is_err());
    assert_eq!(scheduler.free_page_count(), 1);
    assert_eq!(scheduler.free_snapshot_count(), 1);
    let handle = scheduler.admit(spec(60, 3)).unwrap();
    assert!(scheduler.admit(spec(10, 3)).is_err());
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    assert_eq!(scheduler.free_page_count(), 0);
    assert_eq!(scheduler.request(handle).unwrap().last_use(), Fence(2));
    scheduler.cancel(handle).unwrap();
    scheduler
        .complete(Fence(2), &records(handle, 2, 60, 1, 8, 1))
        .unwrap();
    assert_eq!(scheduler.free_snapshot_count(), 1);
}

#[test]
fn uncertain_drain_keeps_ownership_and_confirmed_failure_reclaims_submissions() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(64, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    assert!(
        scheduler
            .release_after_shutdown(&Shutdown {
                drained: false,
                error: Some("device error".into())
            })
            .is_err()
    );
    assert!(scheduler.request(handle).is_ok());
    assert_eq!(scheduler.free_page_count(), 6);
    scheduler
        .release_after_shutdown(&Shutdown {
            drained: true,
            error: Some("graph failed".into()),
        })
        .unwrap();
    assert!(scheduler.request(handle).is_err());
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (8, 2)
    );
    assert!(scheduler.admit(spec(64, 3)).is_ok());
}

#[test]
fn concurrent_identical_terminal_publications_release_duplicate_resources() {
    let mut scheduler = scheduler(2, 8, 4);
    let first = scheduler.admit(spec(64, 1)).unwrap();
    let second = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let mut checkpoint = None;
    for (handle, fence) in [(first, 2), (second, 4)] {
        let step = scheduler.next_step().unwrap();
        assert_eq!(step.lanes[0].request, handle);
        scheduler.record_submission(&step, Fence(fence)).unwrap();
        scheduler
            .record_candidate(handle, 64, Fence(fence))
            .unwrap();
        scheduler
            .complete(Fence(fence), &records(handle, fence, 64, 1, 88, 2))
            .unwrap();
        freeze(&mut scheduler, handle, fence + 1);
        let result = scheduler.finish(handle).unwrap();
        if checkpoint.is_none() {
            checkpoint = result;
        } else {
            assert_eq!(checkpoint, result);
        }
    }
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (7, 3)
    );
    assert_eq!(scheduler.stats().snapshot_publications, 1);
    let mut unrelated = spec(64, 1);
    unrelated.prompt[0] = 999;
    scheduler.ensure_resources(8, 1).unwrap();
    assert_eq!(scheduler.stats().snapshot_evictions, 1);
    assert_eq!(scheduler.stats().cached_checkpoints, 0);
    scheduler.admit(unrelated).unwrap();
}

#[test]
fn failed_candidate_freeze_never_publishes_uninitialized_snapshot_storage() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    scheduler.record_candidate(handle, 64, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(handle, 2, 64, 1, 88, 2))
        .unwrap();
    let step = scheduler.next_step().unwrap();
    assert_eq!(step.kind, GraphKind::Snapshot);
    scheduler.record_submission(&step, Fence(3)).unwrap();
    let mut error = records(handle, 3, 64, 1, -1, 3);
    error[handle.slot as usize].error_code = 6;
    scheduler.complete(Fence(3), &error).unwrap();
    assert_eq!(scheduler.request(handle).unwrap().error_code, Some(6));
    assert_eq!(scheduler.finish(handle).unwrap(), None);
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (8, 2)
    );
}

fn speculative_scheduler(prompt: usize, budget: u32) -> (RequestScheduler, RequestHandle) {
    let mut s = scheduler(1, 64, 4);
    s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    let h = s.admit(spec(prompt, budget)).unwrap();
    init(&mut s, 1);
    let step = s.next_step().unwrap();
    s.record_submission(&step, Fence(2)).unwrap();
    s.complete(Fence(2), &records(h, 2, prompt as i64, 1, 900, 1))
        .unwrap();
    for (width, nanos) in [(1, 100), (2, 150), (4, 200)] {
        s.controller
            .observe(1, prompt as u64, width, nanos, width)
            .unwrap();
    }
    (s, h)
}
fn vector_records(
    h: RequestHandle,
    fence: u64,
    consumed: i64,
    generated: i64,
    tokens: &[i64],
    width: i64,
    status: i64,
) -> [CompletionRecord; 4] {
    let mut row = records(
        h,
        fence,
        consumed,
        generated,
        tokens.last().copied().unwrap_or(-1),
        status,
    );
    let mut v = crate::execution::TokenVector {
        emitted_count: tokens.len() as i64,
        offered_drafts: if tokens.is_empty() { 0 } else { width - 1 },
        accepted_drafts: (tokens.len() as i64).min(width - 1),
        draft_cache_consumed: consumed - 1,
        verification_width: width,
        ..Default::default()
    };
    v.tokens[..tokens.len()].copy_from_slice(tokens);
    row[h.slot as usize].vector = Some(v);
    row
}
#[test]
fn speculative_every_committed_prefix_updates_counts_and_tokens_once() {
    for count in 1..=8 {
        let (mut s, h) = speculative_scheduler(65, 32);
        let step = s.next_step().unwrap();
        assert_eq!(step.kind, GraphKind::Speculative);
        assert_eq!(step.token_capacity, 8);
        s.record_submission(&step, Fence(3)).unwrap();
        let ids = (10..10 + count as i64).collect::<Vec<_>>();
        let events = s
            .complete(
                Fence(3),
                &vector_records(h, 3, 65 + count as i64, 1 + count as i64, &ids, 8, 1),
            )
            .unwrap();
        assert_eq!(
            events.iter().map(|t| t.token).collect::<Vec<_>>(),
            ids.iter().map(|t| *t as u32).collect::<Vec<_>>()
        );
        assert_eq!(s.request(h).unwrap().consumed_tokens, 65 + count as u64);
        assert_eq!(s.request(h).unwrap().generated_tokens, 1 + count as u32);
        assert_eq!(s.request(h).unwrap().scheduled_consumed, 65 + count as u64);
    }
}
#[test]
fn speculative_rounds_wait_for_feedback_and_use_actual_counts() {
    let (mut s, h) = speculative_scheduler(65, 32);
    let first = s.next_step().unwrap();
    assert!(first.grammar_constrained);
    s.record_submission(&first, Fence(3)).unwrap();
    assert!(s.next_step().is_none());
    assert!(s.record_submission(&first, Fence(4)).is_err());
    s.complete(Fence(3), &vector_records(h, 3, 67, 3, &[10, 11], 8, 1))
        .unwrap();
    s.controller = Default::default();
    for (w, ns) in [(1, 1), (2, 1000), (4, 1000), (8, 1000)] {
        s.controller.observe(1, 67, w, ns, 1).unwrap();
    }
    let second = s.next_step().unwrap();
    assert_eq!(second.token_capacity, 1);
    assert_eq!(second.lanes[0].start_position, 67);
    s.record_submission(&second, Fence(4)).unwrap();
    s.complete(Fence(4), &vector_records(h, 4, 68, 4, &[12], 1, 1))
        .unwrap();
    assert_eq!(s.request(h).unwrap().scheduled_consumed, 68);
    assert_eq!(s.request(h).unwrap().tokens().last(), Some(&12));
}
#[test]
fn speculative_budget_clipping_reserves_verification_rows_without_queueing_terminal_work() {
    let (mut s, h) = speculative_scheduler(60, 2);
    let first = s.next_step().unwrap();
    assert_eq!(first.token_capacity, 8);
    s.record_submission(&first, Fence(3)).unwrap();
    assert_eq!(s.request(h).unwrap().pages().count(), 3);
    assert!(s.next_step().is_none());
    let out = s
        .complete(Fence(3), &vector_records(h, 3, 61, 2, &[7], 8, 2))
        .unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].finished);
    assert!(s.next_step().is_none());
    s.finish(h).unwrap();
    assert_eq!(
        (
            s.stats().free_pages,
            s.stats().free_snapshots,
            s.stats().free_slots
        ),
        (64, 4, 1)
    );
}
#[test]
fn speculative_bad_vector_is_transactional_and_cancel_waits_for_round_fence() {
    let (mut s, h) = speculative_scheduler(65, 32);
    let first = s.next_step().unwrap();
    s.record_submission(&first, Fence(3)).unwrap();
    let bad = vector_records(h, 3, 68, 3, &[10, 11], 8, 1);
    assert!(s.complete(Fence(3), &bad).is_err());
    assert_eq!(s.request(h).unwrap().generated_tokens, 1);
    s.cancel(h).unwrap();
    assert!(s.admit(spec(3, 2)).is_err());
    s.complete(Fence(3), &vector_records(h, 3, 67, 3, &[10, 11], 8, 1))
        .unwrap();
    let next = s.admit(spec(3, 2)).unwrap();
    assert_eq!(next.slot, h.slot);
    assert_eq!(next.generation, h.generation + 1);
}
#[test]
fn grammar_and_host_stops_force_one_token_and_one_outstanding_round() {
    for grammar in [false, true] {
        let (mut s, h) = speculative_scheduler(65, 32);
        s.request_mut(h).unwrap().grammar_constrained = grammar;
        s.request_mut(h).unwrap().host_stops = !grammar;
        let step = s.next_step().unwrap();
        assert_eq!(step.token_capacity, 1);
        assert!(step.grammar_constrained);
        s.record_submission(&step, Fence(3)).unwrap();
        assert!(s.next_step().is_none());
        s.complete(Fence(3), &vector_records(h, 3, 66, 2, &[10], 1, 1))
            .unwrap();
        assert!(s.next_step().is_some());
    }
}
#[test]
fn copied_continuations_are_observed_tokens_and_controller_never_exceeds_catalog() {
    use super::speculation::{WidthController, copy_proposal, validate_widths};
    let tokens = [1, 2, 3, 4, 90, 91, 92, 8, 1, 2, 3, 4];
    let (ids, count) = copy_proposal(&tokens, 3);
    assert_eq!(count, 3);
    assert_eq!(&ids[..3], &[90, 91, 92]);
    assert_eq!(copy_proposal(&[1, 2, 3, 4], 7).1, 0);
    assert_eq!(copy_proposal(&tokens, 0).1, 0);
    for bad in [vec![2], vec![1, 9], vec![1, 4, 2]] {
        assert!(validate_widths(&bad).is_err());
    }
    let mut c = WidthController::default();
    assert_eq!(c.choose(&[1, 2, 4, 8], 1, 100, 3), Some(1));
    c.observe(1, 100, 1, 100, 1).unwrap();
    assert_eq!(c.choose(&[1, 2, 4, 8], 1, 100, 3), Some(2));
    c.observe(1, 100, 2, 50, 2).unwrap();
    assert_eq!(c.choose(&[1, 2, 4, 8], 1, 100, 3), Some(2));
}

#[test]
fn draft_boundary_page_remains_private_across_published_prefix_branches() {
    let mut scheduler = scheduler(2, 16, 4);
    scheduler.enable_speculation(&[1, 2, 4, 8]).unwrap();
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.next_step().unwrap();
    scheduler.record_submission(&step, Fence(2)).unwrap();
    let initial = scheduler.request(seed).unwrap().pages().collect::<Vec<_>>();
    assert_eq!(initial.len(), 2);
    scheduler.record_candidate(seed, 64, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(seed, 2, 64, 1, 77, 2))
        .unwrap();
    freeze(&mut scheduler, seed, 3);
    scheduler.finish(seed).unwrap();
    assert_eq!(scheduler.stats().cached_pages, 1);
    assert_eq!(scheduler.free_page_count(), 15);
    let first = scheduler.admit(spec(64, 4)).unwrap();
    let second = scheduler.admit(spec(64, 4)).unwrap();
    init(&mut scheduler, 4);
    let a = scheduler
        .request(first)
        .unwrap()
        .pages()
        .collect::<Vec<_>>();
    let b = scheduler
        .request(second)
        .unwrap()
        .pages()
        .collect::<Vec<_>>();
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(a[0], initial[0]);
    assert_eq!(b[0], initial[0]);
    assert_ne!(a[1], b[1]);
    assert_ne!(a[1], a[0]);
    assert_ne!(b[1], b[0]);
    scheduler.cancel(first).unwrap();
    scheduler.cancel(second).unwrap();
    assert_eq!(scheduler.free_page_count(), 15);
}

#[test]
fn fixed_width_shrinks_at_context_end_and_never_at_remaining_output_budget() {
    let mut scheduler = scheduler(1, 16, 4);
    scheduler.limits.max_sequence_tokens = 128;
    scheduler.enable_speculation(&[1, 2, 4, 8]).unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let h = scheduler.admit(spec(125, 3)).unwrap();
    init(&mut scheduler, 1);
    let prefill = scheduler.next_step().unwrap();
    scheduler.record_submission(&prefill, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(h, 2, 125, 1, 9, 1))
        .unwrap();
    let first = scheduler.next_step().unwrap();
    assert_eq!(first.token_capacity, 2);
    scheduler.record_submission(&first, Fence(3)).unwrap();
    assert_eq!(scheduler.request(h).unwrap().pages().count(), 3);
    assert!(scheduler.next_step().is_none());
    scheduler
        .complete(Fence(3), &vector_records(h, 3, 127, 3, &[10, 11], 2, 2))
        .unwrap();
    assert!(scheduler.next_step().is_none());
    scheduler.finish(h).unwrap();
    assert_eq!(scheduler.free_page_count(), 16);
}

#[test]
fn bad_intermediate_vector_token_does_not_partially_commit() {
    let (mut s, h) = speculative_scheduler(65, 32);
    let step = s.next_step().unwrap();
    s.record_submission(&step, Fence(3)).unwrap();
    let mut bad = vector_records(h, 3, 67, 3, &[10, 11], 8, 1);
    bad[0].vector.as_mut().unwrap().tokens[0] = -2;
    assert!(s.complete(Fence(3), &bad).is_err());
    assert_eq!(s.request(h).unwrap().consumed_tokens, 65);
    assert_eq!(s.request(h).unwrap().generated_tokens, 1);
    assert_eq!(s.request(h).unwrap().tokens().len(), 66);
    s.complete(Fence(3), &vector_records(h, 3, 67, 3, &[10, 11], 8, 1))
        .unwrap();
}

#[test]
fn speculative_fixed_and_adaptive_catalogs_stay_within_eight_total_rows() {
    for active in [1, 2, 3, 4] {
        let mut s = scheduler(active, 64, 8);
        let widths = (1..=8).collect::<Vec<_>>();
        s.enable_speculation(&widths).unwrap();
        s.fixed_speculative_width(Some(7)).unwrap();
        let handles = (0..active)
            .map(|_| s.admit(spec(65, 32)).unwrap())
            .collect::<Vec<_>>();
        init(&mut s, 1);
        for (index, h) in handles.iter().enumerate() {
            s.prefer_prefill = true;
            let step = s.next_step().unwrap();
            assert_eq!(step.kind, GraphKind::Prefill);
            let fence = 2 + index as u64;
            s.record_submission(&step, Fence(fence)).unwrap();
            s.complete(Fence(fence), &records(*h, fence, 65, 1, 9, 1))
                .unwrap();
        }
        let step = s.next_step().unwrap();
        let batch = active.next_power_of_two();
        let width = 7.min(8 / batch);
        assert_eq!((step.batch_capacity, step.token_capacity), (batch, width));
        assert!(step.grammar_constrained);
        s.speculative_fixed_width = None;
        s.controller = Default::default();
        for w in 1..=8 / batch {
            assert_eq!(s.next_step().unwrap().token_capacity, w);
            s.controller
                .observe(batch, 65, w, 1000, w * active)
                .unwrap();
        }
        assert_eq!(s.next_step().unwrap().token_capacity, 8 / batch);
        assert!(
            s.controller
                .observe(batch, 65, 8 / batch + 1, 1000, 1)
                .is_err()
        );
    }
}

fn copied(consumed: u64, count: usize, grammar: bool) -> speculation::CopyProposal {
    let mut tokens = [-1; 7];
    for (index, token) in tokens[..count].iter_mut().enumerate() {
        *token = index as i64 + 10;
    }
    speculation::CopyProposal {
        consumed,
        tokens,
        count,
        grammar_masks: if grammar {
            vec![-1; (count + 1) * crate::model::VOCABULARY.div_ceil(32)]
        } else {
            vec![]
        },
    }
}

#[test]
fn copy_width_is_limited_by_every_legal_chain_and_recomputed_after_completion() {
    for (available, grammar, expected) in
        [(0, false, 1), (2, false, 3), (7, false, 4), (2, true, 3)]
    {
        let mut s = scheduler(2, 32, 4);
        s.enable_speculation(&speculation::WIDTHS).unwrap();
        s.proposal_source(speculation::ProposalSource::Copy)
            .unwrap();
        s.fixed_speculative_width(Some(8)).unwrap();
        let mut request = spec(65, 32);
        request.grammar_constrained = grammar;
        let a = s.admit(request).unwrap();
        let b = s.admit(spec(65, 32)).unwrap();
        init(&mut s, 1);
        for (i, h) in [a, b].into_iter().enumerate() {
            s.prefer_prefill = true;
            let step = s.next_step().unwrap();
            s.record_submission(&step, Fence(2 + i as u64)).unwrap();
            s.complete(Fence(2 + i as u64), &records(h, 2 + i as u64, 65, 1, 9, 1))
                .unwrap();
        }
        s.set_copy_proposal(a, copied(65, available, grammar))
            .unwrap();
        s.set_copy_proposal(b, copied(65, 7, false)).unwrap();
        let step = s.next_step().unwrap();
        assert_eq!((step.batch_capacity, step.token_capacity), (2, expected));
        assert!(step.grammar_constrained);
        s.record_submission(&step, Fence(4)).unwrap();
        assert!(s.next_step().is_none());
        let mut complete = vector_records(a, 4, 66, 2, &[40], expected as i64, 1);
        let other = vector_records(b, 4, 66, 2, &[41], expected as i64, 1);
        complete[b.slot as usize] = other[b.slot as usize];
        // Copy-only records cannot claim an MTP cache exists.
        assert!(s.complete(Fence(4), &complete).is_err());
        for h in [a, b] {
            complete[h.slot as usize]
                .vector
                .as_mut()
                .unwrap()
                .draft_cache_consumed = -1;
        }
        let events = s.complete(Fence(4), &complete).unwrap();
        assert_eq!(events.len(), 2);
        assert!(s.request(a).unwrap().copy_proposal().is_none());
        assert_eq!(s.next_step().unwrap().token_capacity, 1);
        assert!(s.set_copy_proposal(a, copied(65, 7, grammar)).is_err());
        s.cancel(a).unwrap();
        s.cancel(b).unwrap();
        assert_eq!((s.free_page_count(), s.free_snapshot_count()), (32, 4));
    }
}

#[test]
fn copy_source_uses_no_draft_page_and_host_stops_stay_width_one() {
    let mut s = scheduler(1, 2, 2);
    s.limits.max_sequence_tokens = 128;
    s.enable_speculation(&speculation::WIDTHS).unwrap();
    s.proposal_source(speculation::ProposalSource::Copy)
        .unwrap();
    s.fixed_speculative_width(Some(8)).unwrap();
    let mut request = spec(64, 64);
    request.host_stops = true;
    let h = s.admit(request).unwrap();
    init(&mut s, 1);
    let prefill = s.next_step().unwrap();
    s.record_submission(&prefill, Fence(2)).unwrap();
    s.complete(Fence(2), &records(h, 2, 64, 1, 9, 1)).unwrap();
    assert_eq!(s.request(h).unwrap().pages().count(), 1);
    assert!(s.set_copy_proposal(h, copied(64, 1, false)).is_err());
    s.set_copy_proposal(h, copied(64, 0, false)).unwrap();
    assert_eq!(s.next_step().unwrap().token_capacity, 1);
    s.cancel(h).unwrap();
    assert_eq!(s.free_page_count(), 2);
    // MTP requires a private draft page in addition to these two target pages.
    let mut mtp = scheduler(1, 2, 2);
    mtp.limits.max_sequence_tokens = 128;
    mtp.enable_speculation(&speculation::WIDTHS).unwrap();
    assert!(mtp.admit(spec(64, 64)).is_err());
}
