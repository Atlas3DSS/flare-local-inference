//! Opt-in SRPT prefill ordering (`PolicyConfig::prefill_srpt`): order, ties, cached prefixes, the in-flight barrier,
//! no stall on an absent leader, bounded aging with non-preemption, and decode scheduling unchanged.
use super::*;

const MS: u64 = 1_000_000;

/// Four request slots, 128-token pages, prefill chunk `chunk` (catalog 128 or 128/256). `prefill_ns` overrides
/// prefill graph costs by capacity.
fn engine(chunk: usize, srpt: bool, prefill_ns: &[(usize, u64)], aging: Option<(u64, u64)>) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits { request_slots: 4, max_active: 4, max_sequence_tokens: 8192, prefill_chunk_size: chunk,
                          pages: 512, snapshots: 16 },
        SchedulerCapabilities {
            vocabulary: 248320,
            page_tokens: 128,
            vision: crate::model::VisionCapabilities {
                feature_slots: 4, features_per_request: 16384, max_images: 4, patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16384],
            },
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    let mut graphs = super::super::tests::duration_graphs(4);
    for graph in &mut graphs {
        graph.estimated_duration_ns *= 1000;
        if graph.kind == GraphKind::Prefill {
            if let Some(&(_, ns)) = prefill_ns.iter().find(|(c, _)| *c == graph.token_capacity) {
                graph.estimated_duration_ns = ns;
            }
        }
    }
    s.measured_graph_durations(&graphs).unwrap();
    let (max_wait, credit) = aging.unwrap_or((256, 32));
    s.configure_policy(PolicyConfig {
        prefill_srpt: srpt,
        prefill_srpt_max_wait_chunks: max_wait,
        prefill_srpt_aging_tokens_per_chunk: credit,
        ..PolicyConfig::agent()
    })
    .unwrap();
    s
}

fn request(tag: u32, prompt: usize, output: u32) -> RequestSpec {
    let mut prompt: Vec<u32> = (0..prompt as u32).collect();
    prompt[0] = 100_000 + tag;
    RequestSpec {
        message_boundaries: vec![],
        prompt,
        conditioning: vec![],
        max_new_tokens: output,
        grammar_constrained: false,
        host_stops: false,
        growable_output: false,
    }
}

fn records(s: &RequestScheduler, plan: &ScheduledPlan, fence: Fence) -> [CompletionRecord; 4] {
    let step = &plan.step;
    let mut records = [CompletionRecord::default(); 4];
    for (row, lane) in step.lanes.iter().enumerate() {
        let r = s.request(lane.request).unwrap();
        let end = lane.start_position + lane.tokens as u64;
        let emitted = match step.lane_kind(row) {
            GraphKind::Prefill => usize::from(end == r.prompt_tokens as u64),
            GraphKind::Decode | GraphKind::Resume => 1,
            _ => 0,
        };
        let generated = r.generated_tokens + emitted as u32;
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: end as i64,
            generated_tokens: generated as i64,
            token_id: if emitted > 0 { 900 } else { -1 },
            status: if emitted > 0 && generated == r.max_new_tokens { 2 } else { 1 },
            ..Default::default()
        };
    }
    records
}

/// Plans, commits and completes one step; returns the plan.
fn run_step(s: &mut RequestScheduler) -> ScheduledPlan {
    let plan = s.plan().expect("a schedulable step");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = records(s, &plan, fence);
    s.advance_to(s.now_ns + plan.estimated_duration_ns.max(MS)).unwrap();
    s.complete(fence, &records, Default::default()).unwrap();
    plan
}

/// Request handles of the prefill lanes of a plan.
fn prefill_lanes(plan: &ScheduledPlan) -> Vec<RequestHandle> {
    plan.step.lanes.iter().enumerate()
        .filter(|(row, _)| plan.step.lane_kind(*row) == GraphKind::Prefill)
        .map(|(_, lane)| lane.request)
        .collect()
}

/// Runs until every prompt in `handles` is consumed; returns the prefill lane sequence (by handle).
fn prefill_sequence(s: &mut RequestScheduler, handles: &[RequestHandle]) -> Vec<RequestHandle> {
    let mut sequence = Vec::new();
    for _ in 0..2000 {
        if handles.iter().all(|h| s.request(*h).map_or(true, |r| r.consumed_tokens >= r.prompt_tokens as u64)) {
            return sequence;
        }
        sequence.extend(prefill_lanes(&run_step(s)));
    }
    panic!("prompts did not finish");
}

/// Order in which prompts finished their prefill (first index after which a handle no longer appears).
fn finish_order(sequence: &[RequestHandle], handles: &[RequestHandle]) -> Vec<usize> {
    let mut last: Vec<(usize, usize)> = handles.iter().enumerate()
        .map(|(i, h)| (sequence.iter().rposition(|x| x == h).unwrap(), i)).collect();
    last.sort();
    last.into_iter().map(|(_, i)| i).collect()
}

/// Number of switches between prompts in a prefill sequence (prompts contiguous => handles.len() - 1).
fn switches(sequence: &[RequestHandle]) -> usize {
    sequence.windows(2).filter(|w| w[0] != w[1]).count()
}

#[test]
fn srpt_serves_the_shortest_remaining_prompt_first_and_finishes_each_prompt() {
    let mut s = engine(128, true, &[], None);
    let lengths = [4096, 1024, 2048, 512];
    let handles: Vec<_> = lengths.iter().enumerate().map(|(i, &n)| s.admit(request(i as u32, n, 1)).unwrap()).collect();
    let sequence = prefill_sequence(&mut s, &handles);
    assert_eq!(finish_order(&sequence, &handles), [3, 1, 2, 0]);   // 512 -> 1024 -> 2048 -> 4096
    assert_eq!(switches(&sequence), 3, "each prompt runs to completion before the next starts");
    assert_eq!(sequence.len(), (4096 + 1024 + 2048 + 512) / 128);
}

#[test]
fn srpt_is_off_in_every_built_in_policy() {
    for policy in [PolicyConfig::default(), PolicyConfig::throughput(), PolicyConfig::agent()] {
        assert!(!policy.prefill_srpt);
        assert_eq!((policy.prefill_srpt_max_wait_chunks, policy.prefill_srpt_aging_tokens_per_chunk), (256, 32));
    }
}

#[test]
fn equal_prompts_break_ties_by_arrival_then_slot() {
    let mut s = engine(128, true, &[], None);
    let a = s.admit(request(0, 1024, 1)).unwrap();
    s.advance_to(s.now_ns + 1).unwrap();
    let b = s.admit(request(1, 1024, 1)).unwrap();
    let c = s.admit(request(2, 1024, 1)).unwrap();             // same arrival as b: lower slot first
    s.advance_to(s.now_ns + 1).unwrap();
    let d = s.admit(request(3, 1024, 1)).unwrap();
    assert!(b.slot < c.slot);
    let handles = [d, c, b, a];
    let sequence = prefill_sequence(&mut s, &handles);
    assert_eq!(finish_order(&sequence, &handles), [3, 2, 1, 0]);   // a, b, c, d
    assert_eq!(switches(&sequence), 3);
}

#[test]
fn srpt_counts_only_uncached_prompt_tokens() {
    let mut s = engine(128, true, &[], None);
    let long = s.admit(request(0, 4096, 1)).unwrap();
    let short = s.admit(request(1, 1024, 1)).unwrap();
    for (handle, consumed) in [(long, 3584), (short, 0)] {
        let r = s.request_mut(handle).unwrap();
        r.phase = RequestPhase::Prefill;
        r.consumed_tokens = consumed;
    }
    assert_eq!(s.srpt_first(), Some(long.slot as usize));       // 512 uncached tokens left vs 1024
}

#[test]
fn in_flight_leader_blocks_other_prompts_but_not_decode() {
    let mut s = engine(128, true, &[], None);
    let decoder = s.admit(request(9, 128, 64)).unwrap();
    for _ in 0..50 {
        if s.request(decoder).unwrap().phase == RequestPhase::Decode {
            break;
        }
        run_step(&mut s);
    }
    assert_eq!(s.request(decoder).unwrap().phase, RequestPhase::Decode);
    let a = s.admit(request(0, 2048, 1)).unwrap();
    let b = s.admit(request(1, 2048, 1)).unwrap();
    // Run until a's prompt has started (a is the SRPT leader: earlier arrival, lower slot).
    for _ in 0..200 {
        if s.request(a).unwrap().consumed_tokens > 0 {
            break;
        }
        run_step(&mut s);
    }
    assert!(s.request(a).unwrap().consumed_tokens > 0 && s.request(b).unwrap().consumed_tokens == 0);
    // Submit one of a's chunks and keep it in flight; decode-only steps may follow, b's prompt may not.
    let mut in_flight = Vec::new();
    loop {
        let plan = s.plan().expect("a schedulable step");
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let lanes = prefill_lanes(&plan);
        in_flight.push((plan, fence));
        if lanes.contains(&a) {
            break;
        }
        let (plan, fence) = in_flight.pop().unwrap();
        let records = records(&s, &plan, fence);
        s.advance_to(s.now_ns + MS).unwrap();
        s.complete(fence, &records, Default::default()).unwrap();
    }
    assert!(s.request(a).unwrap().pending > 0);
    if let Some(next) = s.plan() {
        assert!(!prefill_lanes(&next).contains(&b), "no other prompt runs while the leader's chunk is in flight");
        assert!(prefill_lanes(&next).is_empty());
    }
    let values = s.lane_values(0).0;
    let mut out = Vec::new();
    s.candidates(&mut out, &values, 0.0);
    assert!(!out.iter().any(|c| c.kind == GraphKind::Prefill), "no prefill candidate behind an in-flight leader");
    assert!(out.iter().any(|c| matches!(c.kind, GraphKind::Decode | GraphKind::Speculative) && c.mask & (1 << decoder.slot) != 0),
            "decode stays schedulable");
}

#[test]
fn an_absent_leader_that_is_not_in_flight_does_not_stall_other_prompts() {
    // Catalog 128/256 with the 256 graph unmeasured (exploration): the leader (100 tokens left, same work at 128)
    // is not offered on the 256 graph although nothing of it is in flight; the other prompt must still be offered there.
    let mut s = engine(256, true, &[(256, 0)], None);
    let leader = s.admit(request(0, 100, 1)).unwrap();
    let other = s.admit(request(1, 1000, 1)).unwrap();
    for handle in [leader, other] {
        s.request_mut(handle).unwrap().phase = RequestPhase::Prefill;
    }
    assert_eq!(s.srpt_first(), Some(leader.slot as usize));
    assert_eq!(s.request(leader).unwrap().pending, 0);
    let values = s.lane_values(0).0;
    let mut out = Vec::new();
    s.candidates(&mut out, &values, 0.0);
    let prefill: Vec<_> = out.iter().filter(|c| c.kind == GraphKind::Prefill).map(|c| (c.capacity, c.mask)).collect();
    assert!(prefill.contains(&(128, 1 << leader.slot)), "{prefill:?}");
    assert!(prefill.contains(&(256, 1 << other.slot)), "the 256 graph still serves the other prompt: {prefill:?}");
}

#[test]
fn aging_promotes_a_long_prompt_behind_short_arrivals_and_the_leader_is_not_preempted() {
    let max_wait = 8;
    let mut s = engine(128, true, &[], Some((max_wait, 32)));
    let long = s.admit(request(0, 4096, 1)).unwrap();
    let mut tag = 1;
    let mut sequence = Vec::new();
    for _ in 0..400 {
        // Keep the other three slots busy with fresh 128-token prompts (one chunk each).
        while s.slots.iter().filter(|e| e.request.is_some()).count() < 4 {
            s.admit(request(tag, 128, 1)).unwrap();
            tag += 1;
        }
        sequence.extend(prefill_lanes(&run_step(&mut s)));
        if s.request(long).map_or(true, |r| r.consumed_tokens >= r.prompt_tokens as u64) {
            break;
        }
    }
    let first = sequence.iter().position(|h| *h == long).expect("the long prompt is scheduled");
    assert!(first <= max_wait as usize, "long prompt started after {first} other chunks (bound {max_wait})");
    let last = sequence.iter().rposition(|h| *h == long).unwrap();
    assert_eq!(last - first + 1, 4096 / 128, "once started, no short arrival preempts the long prompt");
    assert!(sequence[first..=last].iter().all(|h| *h == long));
}

#[test]
fn decode_only_scheduling_is_unchanged_by_the_knob() {
    let mut plans = Vec::new();
    for srpt in [false, true] {
        let mut s = engine(128, srpt, &[], None);
        let handles: Vec<_> = (0..4).map(|i| s.admit(request(i, 64 + 32 * i as usize, 24)).unwrap()).collect();
        // Prompts of one chunk each run first in both policies (order may differ); then compare decode steps.
        for _ in 0..50 {
            if handles.iter().all(|h| s.request(*h).unwrap().phase == RequestPhase::Decode) {
                break;
            }
            run_step(&mut s);
        }
        let mut steps = Vec::new();
        for _ in 0..200 {
            if s.plan().is_none() {
                break;
            }
            let plan = run_step(&mut s);
            steps.push((plan.step.kind, plan.step.batch_capacity, plan.step.token_capacity,
                        plan.step.lanes.iter().map(|l| (l.request.slot, l.tokens)).collect::<Vec<_>>()));
        }
        plans.push(steps);
    }
    assert_eq!(plans[0], plans[1]);
}
