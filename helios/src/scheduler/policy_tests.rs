use super::*;
use crate::execution::{ExecutionTiming, TokenVector};

const MS: u64 = 1_000_000;

fn engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits {
            max_active: active,
            request_slots: active.next_power_of_two(),
            max_sequence_tokens: 8192,
            prefill_chunk_size: 512,
            pages: 1024,
            snapshots: 256,
        },
        SchedulerCapabilities {
            vocabulary: 1024,
            page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            state_rows_per_request: None,
        },
    )
    .unwrap();
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            GraphKind::Prefill => (g.token_capacity as u64 / 64 + 1) * MS,
            GraphKind::Decode => MS + (g.batch_capacity as u64 - 1) * MS / 4,
            GraphKind::Speculative => {
                MS + (g.batch_capacity as u64 - 1) * MS / 4 + (g.token_capacity as u64 - 1) * MS / 2
            }
            _ => MS / 10,
        };
    }
    s.measured_graph_durations(&graphs).unwrap();
    if speculative {
        s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    }
    s.configure_policy(PolicyConfig {
        max_slice_ns: 6 * MS,
        submission_window_ns: 6 * MS,
        service_gap_ns: 12 * MS,
        deficit_scale_ns: 4 * MS,
        ..PolicyConfig::default()
    })
    .unwrap();
    s
}

fn add(s: &mut RequestScheduler, id: u32, prompt: usize, output: u32) -> RequestHandle {
    s.admit(RequestSpec {
        prompt: vec![id; prompt],
        system_boundary: None,
        conditioning: vec![],
        max_new_tokens: output,
        grammar_constrained: false,
        host_stops: false,
    })
    .unwrap()
}

fn complete(
    s: &mut RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
    publish: bool,
) {
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    for lane in &plan.lanes {
        let Ok(r) = s.request(lane.request) else {
            continue;
        };
        let count = match plan.kind {
            GraphKind::Prefill => {
                usize::from(r.consumed_tokens + lane.tokens as u64 == r.prompt_tokens as u64)
            }
            GraphKind::Resume | GraphKind::Decode => 1,
            GraphKind::Speculative => lane.tokens.min(accepted + 1),
            _ => 0,
        };
        let consumed = if plan.kind == GraphKind::Speculative {
            r.consumed_tokens + count as u64
        } else {
            lane.start_position + lane.tokens as u64
        };
        let generated = r.generated_tokens + count as u32;
        let mut record = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: if count == 0 { -1 } else { 99 },
            status: if count > 0 && generated == r.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
        if plan.kind == GraphKind::Speculative {
            let mut tokens = [-1; 8];
            tokens[..count].fill(99);
            record.vector = Some(TokenVector {
                emitted_count: count as i64,
                offered_drafts: (plan.token_capacity - 1) as i64,
                accepted_drafts: (count - 1) as i64,
                verification_width: plan.token_capacity as i64,
                draft_cache_consumed: -1,
                tokens,
                ..Default::default()
            });
        }
        records[lane.request.slot as usize] = record;
    }
    let output = s.complete(fence, &records).unwrap();
    if publish {
        for token in output {
            s.published_output(token.request).unwrap();
        }
    }
    let finished: Vec<_> = s
        .slots
        .iter()
        .filter_map(|slot| slot.request.as_ref())
        .filter(|r| r.phase == RequestPhase::Finishing && r.pending == 0)
        .map(|r| r.handle)
        .collect();
    for h in finished {
        s.finish(h).unwrap();
    }
}

fn step(s: &mut RequestScheduler, accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("runnable trace must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + plan.estimated_duration_ns())
        .unwrap();
    complete(s, &plan, fence, accepted, true);
    plan
}

fn ready(s: &mut RequestScheduler, handles: &[RequestHandle]) {
    for _ in 0..200 {
        if handles
            .iter()
            .all(|h| s.request(*h).unwrap().phase == RequestPhase::Decode)
        {
            return;
        }
        step(s, 0);
    }
    panic!("requests did not reach decode");
}

#[test]
fn entitlement_integrates_arrival_cancellation_and_overlapping_frames_once() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 32);
    let a = s.plan().unwrap();
    s.commit(&a, Fence(1)).unwrap();
    s.advance_to(40_000).unwrap();
    let second = add(&mut s, 2, 65, 32);
    s.advance_to(80_000).unwrap();
    let b = s.plan().unwrap();
    assert_eq!(b.lanes[0].request, second);
    s.commit(&b, Fence(2)).unwrap();
    s.advance_to(140_000).unwrap();
    assert_eq!(
        s.request(first).unwrap().accounting.entitlement_ns,
        90_000.0
    );
    assert_eq!(
        s.request(second).unwrap().accounting.entitlement_ns,
        50_000.0
    );
    complete(&mut s, &a, Fence(1), 0, false);
    s.advance_to(160_000).unwrap();
    assert_eq!(
        s.request(second).unwrap().accounting.entitlement_ns,
        60_000.0
    );
    s.cancel(second).unwrap();
    s.advance_to(200_000).unwrap();
    assert_eq!(
        s.request(first).unwrap().accounting.entitlement_ns,
        140_000.0
    );
    assert_eq!(
        s.request(first).unwrap().accounting.committed_credit_ns,
        0.0
    );
    assert!(s.advance_to(199_999).is_err());
    complete(&mut s, &b, Fence(2), 0, false);
    s.advance_to(MS).unwrap();
    assert_eq!(
        s.request(first).unwrap().accounting.entitlement_ns,
        140_000.0
    );
}

#[test]
fn external_dependencies_pause_entitlement_and_preserve_deficit() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 32);
    let second = add(&mut s, 2, 65, 32);
    s.request_mut(second).unwrap().phase = RequestPhase::Inbound;
    let plan = s.plan().unwrap();
    s.commit(&plan, Fence(1)).unwrap();
    s.advance_to(20_000).unwrap();
    assert_eq!(
        s.request(first).unwrap().accounting.entitlement_ns,
        20_000.0
    );
    assert_eq!(s.request(second).unwrap().accounting.entitlement_ns, 0.0);
    s.request_mut(second).unwrap().phase = RequestPhase::Restore;
    s.advance_to(40_000).unwrap();
    assert_eq!(
        s.request(first).unwrap().accounting.entitlement_ns,
        30_000.0
    );
    assert_eq!(s.request(second).unwrap().accounting.deficit_ns, 10_000.0);
}

#[test]
fn committed_output_earns_credit_and_publication_has_its_own_clock() {
    let mut s = engine(1, true);
    s.fixed_speculative_width(Some(4)).unwrap();
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    let before = *s.request(h).unwrap().accounting();
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + plan.estimated_duration_ns())
        .unwrap();
    complete(&mut s, &plan, fence, 1, false);
    let after = *s.request(h).unwrap().accounting();
    assert_eq!(
        after.committed_credit_ns - before.committed_credit_ns,
        2.0 * before.decode_credit_ns
    );
    assert_eq!(after.last_output_ns, before.last_output_ns);
    assert_eq!(after.last_progress_ns, s.now_ns);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(100 * MS),
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
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(
        s.request(h).unwrap().accounting.decode_credit_ns,
        before.decode_credit_ns
    );
    s.advance_to(s.now_ns + MS).unwrap();
    s.published_output(h).unwrap();
    assert_eq!(
        s.request(h).unwrap().accounting.last_output_ns,
        Some(s.now_ns)
    );
}

#[test]
fn short_arrival_gets_output_while_long_prefill_keeps_progressing() {
    let mut s = engine(2, false);
    let long = add(&mut s, 1, 4096, 32);
    step(&mut s, 0);
    step(&mut s, 0);
    let previous = s.request(long).unwrap().consumed_tokens;
    let arrival = s.now_ns;
    let short = add(&mut s, 2, 65, 32);
    let mut first_output = None;
    for _ in 0..30 {
        let plan = step(&mut s, 0);
        assert!(plan.estimated_duration_ns() <= s.policy.max_slice_ns);
        if first_output.is_none() {
            first_output = s
                .request(short)
                .ok()
                .and_then(|r| r.accounting.last_output_ns);
        }
    }
    assert!(first_output.unwrap() - arrival <= 2 * s.policy.service_gap_ns);
    assert!(s.request(long).unwrap().consumed_tokens > previous);
}

#[test]
fn speculative_cohorts_follow_acceptance_until_a_deficit_requires_other_requests() {
    let mut s = engine(4, true);
    let handles: Vec<_> = (1..=4).map(|id| add(&mut s, id, 1, 64)).collect();
    ready(&mut s, &handles);
    let mut graphs = super::tests::duration_graphs(4);
    for g in &mut graphs {
        g.estimated_duration_ns = match (g.kind, g.batch_capacity, g.token_capacity) {
            (GraphKind::Speculative, 2, 4) => 2 * MS,
            (GraphKind::Speculative, _, 1) => 2 * MS,
            (GraphKind::Speculative, _, _) => 10 * MS,
            _ => MS,
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let now = s.now_ns;
    for (index, h) in handles.iter().enumerate() {
        let r = s.request_mut(*h).unwrap();
        for _ in 0..128 {
            r.acceptance
                .observe(7, if index < 2 { 7 } else { 0 }, index >= 2, 0);
        }
        r.accounting.deficit_ns = 0.0;
        r.accounting.last_output_ns = Some(now);
    }
    let plan = s.plan().unwrap();
    assert_eq!((plan.batch_capacity, plan.token_capacity), (2, 4));
    assert_eq!(
        plan.lanes.iter().map(|l| l.request).collect::<Vec<_>>(),
        handles[..2]
    );
    for graph in &mut graphs {
        if graph.kind == GraphKind::Speculative && graph.batch_capacity == 1 {
            graph.estimated_duration_ns = 10 * MS;
        }
    }
    s.costs = super::tests::measured_costs(&graphs);
    let protected = s.plan().unwrap();
    assert_eq!((protected.batch_capacity, protected.token_capacity), (4, 1));
    for graph in &mut graphs {
        if graph.kind == GraphKind::Speculative && graph.token_capacity == 1 {
            graph.estimated_duration_ns = 2 * MS;
        }
    }
    s.costs = super::tests::measured_costs(&graphs);
    s.request_mut(handles[2]).unwrap().accounting.deficit_ns = 5.0 * MS as f64;
    assert!(
        s.plan()
            .unwrap()
            .lanes
            .iter()
            .any(|l| l.request == handles[2])
    );
}

#[test]
fn an_overdue_feedback_request_drains_the_queue_before_another_pipeline_step() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 1, 64);
    ready(&mut s, &[first]);
    s.advance_to(2 * s.policy.service_gap_ns).unwrap();
    s.published_output(first).unwrap();
    let plan = s.plan().unwrap();
    assert!(!plan.requires_feedback);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let overdue = add(&mut s, 2, 1, 64);
    s.request_mut(overdue).unwrap().grammar_constrained = true;
    s.set_arrival(overdue, 0).unwrap();
    assert!(s.plan().is_none());
    s.advance_to(s.now_ns + plan.estimated_duration_ns())
        .unwrap();
    complete(&mut s, &plan, fence, 0, true);
    let rescue = s.plan().unwrap();
    assert_eq!(rescue.kind, GraphKind::Init);
    assert_eq!(rescue.lanes[0].request, overdue);
}

#[test]
fn overdue_initialization_keeps_urgency_until_useful_progress() {
    let mut s = engine(2, false);
    let h = add(&mut s, 1, 1, 64);
    ready(&mut s, &[h]);
    let delayed = add(&mut s, 2, 1, 64);
    s.set_arrival(delayed, 0).unwrap();
    s.advance_to(2 * s.policy.service_gap_ns).unwrap();
    s.published_output(h).unwrap();
    let plan = step(&mut s, 0);
    assert_eq!(plan.kind, GraphKind::Init);
    assert_eq!(plan.lanes[0].request, delayed);
    assert!(plan.recovery_lateness_ns() > 0.0);
    assert_eq!(s.request(delayed).unwrap().accounting.last_progress_ns, 0);
    assert_eq!(s.request(delayed).unwrap().accounting.last_output_ns, None);
    let plan = step(&mut s, 0);
    assert_eq!(plan.kind, GraphKind::Prefill);
    assert_eq!(plan.lanes[0].request, delayed);
}

#[test]
fn large_cohort_search_covers_high_slots_without_growing_scratch() {
    let mut s = engine(64, true);
    let handles: Vec<_> = (1..=64).map(|id| add(&mut s, id, 1, 64)).collect();
    for h in &handles {
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 1;
        r.scheduled_consumed = 1;
        r.generated_tokens = 1;
    }
    s.advance_to(2 * s.service_gap(64)).unwrap();
    for h in &handles[..63] {
        s.published_output(*h).unwrap();
    }
    let capacity = s.candidates.borrow().capacity();
    let plan = s.plan().unwrap();
    assert!(plan.lanes.iter().any(|l| l.request == handles[63]));
    assert!(plan.recovery_lateness_ns() > 0.0);
    assert_eq!(s.candidates.borrow().capacity(), capacity);
    assert!(plan.lanes.iter().all(|l| l.tokens <= plan.token_capacity));
}

#[test]
fn unmeasured_subsets_do_not_split_a_measured_cohort() {
    for active in [4, 8] {
        let mut s = engine(active, true);
        let handles: Vec<_> = (1..=active as u32)
            .map(|id| add(&mut s, id, 1, 64))
            .collect();
        ready(&mut s, &handles);
        // The complete cohort's shapes are measured; smaller batches never ran or expired.
        let mut graphs = super::tests::duration_graphs(active);
        for g in &mut graphs {
            g.estimated_duration_ns = match (g.kind, g.batch_capacity) {
                (GraphKind::Speculative, b) if b < active => 0,
                (GraphKind::Speculative, _) => MS + g.token_capacity as u64 * MS / 4,
                _ => MS,
            };
        }
        s.costs = super::tests::measured_costs(&graphs);
        for _ in 0..16 {
            let plan = step(&mut s, 1);
            assert_eq!(plan.kind, GraphKind::Speculative);
            assert_eq!(plan.lanes.len(), active);
            let positions: std::collections::HashSet<_> =
                plan.lanes.iter().map(|l| l.start_position).collect();
            assert_eq!(positions.len(), 1, "cohort members advance together");
        }
    }
}

#[test]
fn overdue_requests_are_not_rescued_one_at_a_time_through_unmeasured_graphs() {
    let mut s = engine(16, true);
    let decoding: Vec<_> = (1..=8).map(|id| add(&mut s, id, 1, 256)).collect();
    ready(&mut s, &decoding);
    // Another request is still prefilling, so no speculative cohort serves every runnable request.
    add(&mut s, 9, 4096, 64);
    let mut graphs = super::tests::duration_graphs(16);
    for g in &mut graphs {
        g.estimated_duration_ns = match (g.kind, g.batch_capacity, g.token_capacity) {
            (GraphKind::Speculative, 8, w) => 2 * MS + w as u64 * MS,
            (GraphKind::Speculative, 1, 1) => 3 * MS / 2,
            (GraphKind::Speculative, _, _) => 0,
            (GraphKind::Prefill, _, n) => (n as u64 / 64 + 1) * MS,
            _ => MS / 10,
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let mut lanes = 0;
    let mut rounds = 0;
    for _ in 0..64 {
        let plan = step(&mut s, 1);
        if plan.kind == GraphKind::Speculative {
            lanes += plan.lanes.len();
            rounds += 1;
        }
    }
    assert!(rounds > 0);
    assert!(
        lanes >= 6 * rounds,
        "{lanes} lanes over {rounds} speculative rounds"
    );
}

#[test]
fn prefix_restore_earns_no_credit_for_reused_prompt_tokens() {
    let mut s = engine(1, false);
    let seed = add(&mut s, 1, 65, 1);
    while s.request(seed).is_ok() {
        step(&mut s, 0);
    }
    let h = add(&mut s, 1, 65, 32);
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 65);
    assert_eq!(s.request(h).unwrap().accounting.prefill_credit_ns, 0.0);
    let restore = step(&mut s, 0);
    assert_eq!(restore.kind, GraphKind::Restore);
    assert_eq!(s.request(h).unwrap().accounting.committed_credit_ns, 0.0);
    let resume = step(&mut s, 0);
    assert_eq!(resume.kind, GraphKind::Resume);
    let r = s.request(h).unwrap();
    assert_eq!(
        r.accounting.committed_credit_ns,
        r.accounting.decode_credit_ns
    );
}

#[test]
fn a_plan_keeps_its_duration_when_the_graph_estimate_changes() {
    let mut s = engine(1, false);
    add(&mut s, 1, 1, 32);
    let plan = s.plan().unwrap();
    let estimate = plan.estimated_duration_ns();
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(80 * MS),
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
                None,
            ],
            ..Default::default()
        },
    );
    assert_ne!(s.estimated_duration_ns(&plan), Some(estimate));
    assert_eq!(plan.estimated_duration_ns(), estimate);
    s.commit(&plan, Fence(1)).unwrap();
    assert_eq!(s.window_remaining(), Some(estimate));
}

#[test]
fn a_finished_prompt_waiting_for_snapshot_earns_no_further_entitlement() {
    let mut s = engine(2, false);
    let finished = add(&mut s, 1, 1, 1);
    step(&mut s, 0);
    step(&mut s, 0);
    assert_eq!(s.request(finished).unwrap().phase, RequestPhase::Snapshot);
    let before = s.request(finished).unwrap().accounting.entitlement_ns;
    let live = add(&mut s, 2, 1, 32);
    let plan = s.plan().unwrap();
    s.commit(&plan, Fence(s.submitted.0 + 1)).unwrap();
    s.advance_to(s.now_ns + MS).unwrap();
    assert_eq!(
        s.request(finished).unwrap().accounting.entitlement_ns,
        before
    );
    assert_eq!(
        s.request(live).unwrap().accounting.entitlement_ns,
        MS as f64
    );
}

#[test]
fn submission_window_uses_remaining_time_and_waits_on_an_overrun() {
    let mut s = engine(2, false);
    let handles: Vec<_> = (1..=2).map(|id| add(&mut s, id, 1, 64)).collect();
    ready(&mut s, &handles);
    let mut graphs = super::tests::duration_graphs(2);
    for g in &mut graphs {
        g.estimated_duration_ns = if g.batch_capacity == 1 {
            4 * MS
        } else {
            20 * MS
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let start = s.now_ns;
    let a = s.plan().unwrap();
    let fa = Fence(s.submitted.0 + 1);
    s.commit(&a, fa).unwrap();
    assert!(s.plan().is_none());
    s.advance_to(start + 2 * MS).unwrap();
    let b = s.plan().unwrap();
    assert_ne!(a.lanes[0].request, b.lanes[0].request);
    s.commit(&b, Fence(fa.0 + 1)).unwrap();
    assert!(s.plan().is_none());
    s.advance_to(start + 5 * MS).unwrap();
    assert_eq!(s.window_remaining(), None);
    assert!(!s.submission_window_available(MS));
    complete(&mut s, &a, fa, 0, true);
    assert_eq!(s.window_remaining(), Some(3 * MS));
}

#[test]
fn atomic_prefill_can_make_progress_when_its_minimum_graph_exceeds_the_slice() {
    let mut s = engine(1, false);
    s.configure_policy(PolicyConfig {
        max_slice_ns: MS,
        submission_window_ns: MS,
        ..s.policy
    })
    .unwrap();
    let h = add(&mut s, 1, 256, 32);
    s.set_prefill_spans(h, &[0..256]).unwrap();
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(
        (plan.kind, plan.token_capacity, plan.lanes[0].tokens),
        (GraphKind::Prefill, 256, 256)
    );
    assert!(plan.estimated_duration_ns() > s.policy.max_slice_ns);
    step(&mut s, 0);
    assert_eq!(s.request(h).unwrap().consumed_tokens, 256);
}

#[test]
fn oversized_atomic_work_requires_an_empty_submission_queue() {
    let mut s = engine(2, false);
    s.configure_policy(PolicyConfig {
        max_slice_ns: MS / 20,
        ..s.policy
    })
    .unwrap();
    add(&mut s, 1, 1, 32);
    add(&mut s, 2, 1, 32);
    // A step that serves every runnable request keeps nobody waiting, so it may run over the slice alone.
    let plan = s.plan().unwrap();
    assert_eq!(plan.lanes.len(), 2);
    assert!(plan.estimated_duration_ns() > s.policy.max_slice_ns);
    s.commit(&plan, Fence(1)).unwrap();
    assert!(s.plan().is_none());
    assert!(!s.submission_window_available(plan.estimated_duration_ns()));
    assert!(
        s.commit(&plan, Fence(2))
            .unwrap_err()
            .to_string()
            .contains("execution-time window")
    );
}

fn cold_engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = engine(active, speculative);
    s.graph_durations(&super::tests::duration_graphs(active))
        .unwrap();
    s
}

fn measured_step(s: &mut RequestScheduler, duration_ns: u64, accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("runnable request must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + duration_ns).unwrap();
    complete(s, &plan, fence, accepted, true);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(duration_ns),
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
                None,
            ],
            ..Default::default()
        },
    );
    plan
}

#[test]
fn indeterminate_and_equal_costs_naturally_choose_full_useful_prefill() {
    for known in [false, true] {
        let mut s = cold_engine(1, false);
        if known {
            let mut graphs = super::tests::duration_graphs(1);
            for graph in &mut graphs {
                graph.estimated_duration_ns = MS;
            }
            s.measured_graph_durations(&graphs).unwrap();
        }
        let h = add(&mut s, 1, 4096, 64);
        measured_step(&mut s, MS / 10, 0);
        let plan = measured_step(&mut s, 5 * MS, 0);
        assert_eq!(plan.kind, GraphKind::Prefill);
        assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (512, 512));
        let credit = s.request(h).unwrap().accounting();
        assert!(credit.prefill_credit_ns > 0.0);
        if !known {
            assert_eq!(credit.committed_credit_ns, 5.0 * MS as f64);
            assert_eq!(credit.deficit_ns, (MS / 10) as f64);
        }
    }
}

#[test]
fn indeterminate_and_equal_costs_naturally_choose_full_useful_speculation() {
    for known in [false, true] {
        for (limit, width) in [(64, 8), (3, 2)] {
            let mut s = cold_engine(1, true);
            if known {
                let mut graphs = super::tests::duration_graphs(1);
                for graph in &mut graphs {
                    graph.estimated_duration_ns = MS;
                }
                s.measured_graph_durations(&graphs).unwrap();
            }
            add(&mut s, 1, 1, limit);
            for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
                assert_eq!(measured_step(&mut s, MS / 10, 0).kind, kind);
            }
            let plan = s.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Speculative);
            assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (width, width));
        }
    }
}

#[test]
fn indeterminate_execution_occupies_the_submission_window_until_completion() {
    let mut s = cold_engine(2, false);
    add(&mut s, 1, 1, 64);
    let plan = s.plan().unwrap();
    assert_eq!(plan.estimated_duration_ns(), 0);
    assert!(plan.requires_feedback);
    s.commit(&plan, Fence(1)).unwrap();
    add(&mut s, 2, 1, 64);
    assert!(!s.submission_window_available(0));
    assert!(!s.submission_window_available(MS));
    assert!(s.plan().is_none());
    s.advance_to(MS).unwrap();
    complete(&mut s, &plan, Fence(1), 0, true);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(MS),
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
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(MS));
    assert!(s.plan().is_some());
}

#[test]
fn graph_expiry_uses_its_last_valid_observation_and_restarts_the_ema() {
    let mut s = cold_engine(1, false);
    add(&mut s, 1, 1, 64);
    let plan = s.plan().unwrap();
    let ttl = s.policy.estimate_ttl_ns;
    s.advance_to(10 * MS).unwrap();
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(MS),
                Some(2 * MS),
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
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.advance_to(10 * MS + ttl - 1).unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(0),
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
                None,
            ],
            ..Default::default()
        },
    );
    s.advance_to(10 * MS + ttl).unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(0));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            wall_ns: MS,
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(0));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(8 * MS),
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
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(8 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(16 * MS),
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
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(9 * MS));
}

#[test]
fn poisoned_graph_expires_wins_again_and_recovers_from_actual_runtime() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 1, 64);
    ready(&mut s, &[h]);
    let mut graphs = super::tests::duration_graphs(1);
    for graph in &mut graphs {
        graph.estimated_duration_ns = if graph.token_capacity == 8 {
            2 * MS
        } else {
            4 * MS
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let plan = s.plan().unwrap();
    assert_eq!(plan.token_capacity, 8);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(1_000 * MS),
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
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(126_750_000));
    assert_ne!(s.plan().unwrap().token_capacity, 8);
    let ttl = s.policy.estimate_ttl_ns;
    s.advance_to(ttl / 2).unwrap();
    for width in [1, 2, 4] {
        let mut other = (*plan).clone();
        other.token_capacity = width;
        s.observe_execution(
            &other,
            ExecutionTiming {
                gpu_ns: [
                    Some(4 * MS),
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
                    None,
                ],
                ..Default::default()
            },
        );
    }
    s.advance_to(ttl).unwrap();
    let retry = measured_step(&mut s, 2 * MS, 7);
    assert_eq!(
        (retry.token_capacity, retry.estimated_duration_ns()),
        (8, 0)
    );
    assert_eq!(s.estimated_duration_ns(&retry), Some(2 * MS));
}

#[test]
fn repeated_observations_keep_the_winner_fresh_while_neglected_shapes_expire() {
    let mut s = cold_engine(1, true);
    s.configure_policy(PolicyConfig {
        estimate_ttl_ns: 20 * MS,
        ..s.policy
    })
    .unwrap();
    let h = add(&mut s, 1, 1, 1024);
    for _ in 0..3 {
        measured_step(&mut s, MS / 10, 0);
    }
    let mut widths = Vec::new();
    for _ in 0..4 {
        widths.push(measured_step(&mut s, MS, 7).token_capacity);
    }
    assert_eq!(widths, [8, 4, 2, 1]);
    let rate = s.request(h).unwrap().accounting.decode_credit_ns;
    for _ in 0..15 {
        assert_eq!(measured_step(&mut s, MS, 7).token_capacity, 8);
    }
    let mut retried = Vec::new();
    for _ in 0..10 {
        let plan = measured_step(&mut s, MS, 7);
        if plan.estimated_duration_ns() == 0 {
            retried.push(plan.token_capacity);
        }
    }
    assert!(retried.contains(&4) && retried.contains(&2) && retried.contains(&1));
    assert!(!retried.contains(&8));
    assert_eq!(s.request(h).unwrap().accounting.decode_credit_ns, rate);
    assert!(s.request(h).unwrap().accounting.committed_credit_ns > 0.0);
}

#[test]
fn an_uncontended_prefill_takes_its_whole_chunk_over_the_slice() {
    let mut s = engine(1, false);
    add(&mut s, 1, 2048, 32);
    let plan = loop {
        let plan = s.plan().unwrap();
        if plan.kind == GraphKind::Prefill {
            break plan;
        }
        step(&mut s, 0);
    };
    assert_eq!(plan.token_capacity, 512);
    assert!(plan.estimated_duration_ns() > s.policy.max_slice_ns);
}

#[test]
fn slow_rounds_keep_speculating_when_every_round_exceeds_the_service_gap() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 1, 64);
    ready(&mut s, &[h]);
    let mut graphs = super::tests::duration_graphs(1);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            // Two-Spark GLM rounds: 45 ms for one token, 5 ms more per draft.
            GraphKind::Speculative | GraphKind::Decode => {
                45 * MS + (g.token_capacity as u64 - 1) * 5 * MS
            }
            _ => MS,
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let now = s.now_ns;
    let r = s.request_mut(h).unwrap();
    for _ in 0..128 {
        r.acceptance.observe(7, 7, false, 0);
    }
    r.accounting.last_output_ns = Some(now);
    assert!(s.service_gap(1) >= 3 * 45 * MS);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Speculative);
    assert!(plan.token_capacity > 1, "width {}", plan.token_capacity);
}

#[test]
fn batched_slow_rounds_keep_speculating_when_their_rounds_exceed_one_requests_gap() {
    let mut s = engine(8, true);
    let handles: Vec<_> = (0..8).map(|i| add(&mut s, i + 1, 1, 256)).collect();
    ready(&mut s, &handles);
    let mut graphs = super::tests::duration_graphs(8);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            // Four-Spark GLM rounds: 30 ms for one request's token, 53 ms for eight, and each
            // draft adds 5 ms per request; a width-3 round for eight costs about three of one's gaps.
            GraphKind::Speculative | GraphKind::Decode => {
                let base = if g.batch_capacity >= 8 { 53 } else { 30 };
                (base + (g.token_capacity as u64 - 1) * 2 * g.batch_capacity as u64) * MS
            }
            _ => MS,
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let now = s.now_ns;
    for h in &handles {
        let r = s.request_mut(*h).unwrap();
        for _ in 0..128 {
            r.acceptance.observe(7, 7, false, 0);
        }
        r.accounting.last_output_ns = Some(now);
    }
    assert!(s.service_gap(8) >= 3 * 53 * MS);
    assert!(s.service_gap(1) < s.service_gap(8));
    let mut widths = Vec::new();
    for _ in 0..24 {
        let plan = step(&mut s, 3);
        if plan.kind == GraphKind::Speculative {
            widths.push((plan.lanes.len(), plan.token_capacity));
        }
    }
    let wide = widths
        .iter()
        .filter(|(lanes, width)| *lanes == 8 && *width > 1)
        .count();
    assert!(wide * 4 >= widths.len() * 3, "batched widths {widths:?}");
}
