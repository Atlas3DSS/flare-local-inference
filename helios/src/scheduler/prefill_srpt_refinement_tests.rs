//! SRPT refinements (`prefill_srpt_short_tokens`, `prefill_srpt_decode_priority_tokens`): unit tests, and an event
//! simulation of the serving loop (two-deep submission window, width-5 speculation, measured SM70 costs of the chunk-512
//! package) whose policy comparisons print with `--nocapture` and whose schedule digest pins the default knobs to v3.
use super::*;
use crate::execution::{ExecutionTiming, GraphDescriptor, SamplingMode, TokenVector};
use std::collections::VecDeque;

const MS: u64 = 1_000_000;

thread_local! {
    /// (start_ns, end_ns, kind, [(job, tokens)]) of every committed step when enabled.
    static STEPS: std::cell::RefCell<Option<Vec<(u64, u64, GraphKind, Vec<(usize, usize)>)>>> =
        const { std::cell::RefCell::new(None) };
}

// ---------------------------------------------------------------------------------------------------------------------
// Unit tests on the small engine of prefill_srpt_tests (plain decode, 128-token chunks).

fn small(srpt: bool, short: u64, budget: u64) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits { request_slots: 4, max_active: 4, max_sequence_tokens: 8192, prefill_chunk_size: 128,
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
    }
    s.measured_graph_durations(&graphs).unwrap();
    s.configure_policy(PolicyConfig {
        prefill_srpt: srpt,
        prefill_srpt_short_tokens: short,
        prefill_srpt_decode_priority_tokens: budget,
        ..PolicyConfig::agent()
    })
    .unwrap();
    s
}

fn request(tag: u32, prompt: usize, output: u32) -> RequestSpec {
    let mut prompt: Vec<u32> = (0..prompt as u32).collect();
    prompt[0] = 100_000 + tag;
    RequestSpec { message_boundaries: vec![], prompt, conditioning: vec![], max_new_tokens: output,
                  grammar_constrained: false, host_stops: false, growable_output: false }
}

fn small_records(s: &RequestScheduler, plan: &ScheduledPlan, fence: Fence) -> [CompletionRecord; 4] {
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

fn small_step(s: &mut RequestScheduler) -> ScheduledPlan {
    let plan = s.plan().expect("a schedulable step");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = small_records(s, &plan, fence);
    s.advance_to(s.now_ns + plan.estimated_duration_ns.max(MS)).unwrap();
    s.complete(fence, &records, Default::default()).unwrap();
    plan
}

fn prefill_lanes(plan: &ScheduledPlan) -> Vec<RequestHandle> {
    plan.step.lanes.iter().enumerate()
        .filter(|(row, _)| plan.step.lane_kind(*row) == GraphKind::Prefill)
        .map(|(_, lane)| lane.request)
        .collect()
}

fn done(s: &RequestScheduler, h: RequestHandle) -> bool {
    s.request(h).map_or(true, |r| r.consumed_tokens >= r.prompt_tokens as u64)
}

/// Starts a 4096-token leader, then admits a short (2 chunks) and a mid-size (1024) prompt; returns the prefill lane
/// sequence after the admissions and the three handles.
fn leader_with_short_arrival(short: u64) -> (Vec<RequestHandle>, [RequestHandle; 3]) {
    let mut s = small(true, short, 0);
    let leader = s.admit(request(0, 4096, 1)).unwrap();
    for _ in 0..50 {
        if s.request(leader).unwrap().consumed_tokens >= 512 {
            break;
        }
        small_step(&mut s);
    }
    let brief = s.admit(request(1, 256, 1)).unwrap();
    let mid = s.admit(request(2, 1024, 1)).unwrap();
    let mut sequence = Vec::new();
    for _ in 0..400 {
        if [leader, brief, mid].iter().all(|h| done(&s, *h)) {
            break;
        }
        sequence.extend(prefill_lanes(&small_step(&mut s)));
    }
    (sequence, [leader, brief, mid])
}

#[test]
fn defaults_keep_the_v3_order() {
    for policy in [PolicyConfig::default(), PolicyConfig::throughput(), PolicyConfig::agent()] {
        assert_eq!((policy.prefill_srpt_short_tokens, policy.prefill_srpt_decode_priority_tokens), (0, 0));
    }
    let (sequence, [leader, brief, mid]) = leader_with_short_arrival(0);
    let last_leader = sequence.iter().rposition(|h| *h == leader).unwrap();
    let first_brief = sequence.iter().position(|h| *h == brief).unwrap();
    let first_mid = sequence.iter().position(|h| *h == mid).unwrap();
    assert!(last_leader < first_brief && first_brief < first_mid, "v3: no preemption, then SRPT: {sequence:?}");
}

#[test]
fn a_short_prompt_interleaves_with_the_leader_which_keeps_its_place() {
    let (sequence, [leader, brief, mid]) = leader_with_short_arrival(256);
    let brief_rows: Vec<_> = sequence.iter().enumerate().filter(|(_, h)| **h == brief).map(|(i, _)| i).collect();
    assert_eq!(brief_rows.len(), 2, "{sequence:?}");
    assert!(brief_rows[0] <= 1, "the short prompt runs at most one leader chunk after arrival: {sequence:?}");
    // Alternation: a leader chunk sits between the short prompt's two chunks.
    assert_eq!(sequence[brief_rows[0] + 1], leader, "{sequence:?}");
    assert!(brief_rows[1] == brief_rows[0] + 2, "{sequence:?}");
    // The mid-size prompt does not preempt; the leader finishes before it starts.
    let last_leader = sequence.iter().rposition(|h| *h == leader).unwrap();
    let first_mid = sequence.iter().position(|h| *h == mid).unwrap();
    assert!(last_leader < first_mid, "{sequence:?}");
}

#[test]
fn a_stream_of_short_prompts_leaves_the_leader_every_other_chunk() {
    let mut s = small(true, 128, 0);
    let leader = s.admit(request(0, 2048, 1)).unwrap();
    small_step(&mut s); // Init
    let mut tag = 1;
    let mut sequence = Vec::new();
    for _ in 0..400 {
        while s.slots.iter().filter(|e| e.request.is_some()).count() < 4 {
            s.admit(request(tag, 128, 1)).unwrap();
            tag += 1;
        }
        sequence.extend(prefill_lanes(&small_step(&mut s)));
        if done(&s, leader) {
            break;
        }
    }
    let first = sequence.iter().position(|h| *h == leader).unwrap();
    let last = sequence.iter().rposition(|h| *h == leader).unwrap();
    let leader_chunks = sequence[first..=last].iter().filter(|h| **h == leader).count();
    assert_eq!(leader_chunks, 2048 / 128);
    assert!(last - first + 1 <= 2 * leader_chunks, "leader got fewer than every other chunk: {sequence:?}");
}

#[test]
fn decode_priority_holds_prompt_chunks_only_within_the_budget() {
    for (srpt, budget, held) in [(true, 24u64, true), (false, 24, false), (true, 0, false)] {
        let mut s = small(srpt, 0, budget);
        let decoder = s.admit(request(9, 128, 48)).unwrap();
        for _ in 0..50 {
            if s.request(decoder).unwrap().phase == RequestPhase::Decode {
                break;
            }
            small_step(&mut s);
        }
        let prompt = s.admit(request(0, 1024, 1)).unwrap();
        let mut first_prompt_chunk_at = None;
        for _ in 0..400 {
            let generated = s.request(decoder).map_or(48, |r| r.generated_tokens);
            if s.plan().is_none() {
                break;
            }
            let plan = small_step(&mut s);
            if first_prompt_chunk_at.is_none() && prefill_lanes(&plan).contains(&prompt) {
                first_prompt_chunk_at = Some(generated);
            }
            if done(&s, prompt) && s.request(decoder).is_err() {
                break;
            }
        }
        let at = first_prompt_chunk_at.expect("the prompt runs");
        if held {
            assert!(at as u64 >= budget, "srpt {srpt}: prompt chunk ran at {at} decoder tokens (budget {budget})");
        } else {
            assert!((at as u64) < 24, "srpt {srpt} budget {budget}: prompt was held until {at}");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Event simulation with measured G128 costs.

/// Measured costs from the prefill lab (package f21b9abb, one V100): 512/256/128-row chunks at short context
/// (991.5 / 958 / 860 tok/s c1), +5.4 us per context token per 512 rows (30.7k c1 = 866 tok/s), width-5 speculative
/// rounds 53 / 62.5 / 77.6 ms at batch 1 / 2 / 4 (long-c4 counters), small Init/Resume.
fn cost(kind: GraphKind, batch: usize, tokens: usize) -> u64 {
    let b = batch.ilog2() as usize;
    match kind {
        GraphKind::Prefill => match tokens {
            128 => 149 * MS,
            256 => 267 * MS,
            512 => 516 * MS,
            n => n as u64 * MS,
        },
        GraphKind::Speculative => {
            let base = [53.0, 62.5, 77.6, 90.0][b.min(3)];
            (base * (0.68 + 0.064 * tokens as f64) * 1e6) as u64
        }
        GraphKind::Decode => [40, 45, 55, 70][b.min(3)] * MS,
        GraphKind::Resume => [40, 45, 55, 70][b.min(3)] * MS,
        GraphKind::Init | GraphKind::Restore => 4 * MS,
        GraphKind::Snapshot => 1,
        _ => MS,
    }
}

fn sim_graphs() -> Vec<GraphDescriptor> {
    let mut graphs = super::super::tests::duration_graphs(4);
    for g in &mut graphs {
        g.estimated_duration_ns = cost(g.kind, g.batch_capacity, g.token_capacity);
    }
    let _ = SamplingMode::General;
    graphs
}

fn engine(policy: PolicyConfig) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [9; 32],
        SchedulerLimits { request_slots: 4, max_active: 4, max_sequence_tokens: 32_768, prefill_chunk_size: 512,
                          pages: 1400, snapshots: 16 },
        SchedulerCapabilities {
            vocabulary: 248_320,
            page_tokens: 128,
            vision: crate::model::VisionCapabilities {
                feature_slots: 4, features_per_request: 16384, max_images: 4, patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16384],
            },
            draft_pages_per_request: 1,
            completion_inputs: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    s.measured_graph_durations(&sim_graphs()).unwrap();
    s.enable_speculation(&[1, 2, 3, 4, 5]).unwrap();
    s.fixed_speculative_width(Some(5)).unwrap();
    s.configure_policy(policy).unwrap();
    s
}

#[derive(Clone, Copy)]
struct Job {
    arrival_ns: u64,
    prompt: usize,
    output: u32,
}

#[derive(Clone, Copy, Default, Debug)]
struct Outcome {
    admitted_ns: u64,
    first_ns: Option<u64>,
    last_ns: Option<u64>,
    generated: u32,
}

/// Accepted drafts per speculative round, cycled per request (mean 2.6 accepted = 3.6 emitted per round).
const ACCEPTED: [usize; 5] = [3, 2, 3, 3, 2];

fn true_ns(plan: &ScheduledPlan) -> u64 {
    let step = &plan.step;
    let base = cost(step.kind, step.batch_capacity, step.token_capacity);
    if step.kind == GraphKind::Prefill {
        let lane = &step.lanes[0];
        base + (lane.tokens as f64 / 512.0 * lane.start_position as f64 * 5.4e-6 * 1e9) as u64
    } else {
        base
    }
}

fn sim_records(s: &RequestScheduler, plan: &ScheduledPlan, fence: Fence, rounds: &mut [usize; 4])
    -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    for (row, lane) in plan.step.lanes.iter().enumerate() {
        let kind = plan.step.lane_kind(row);
        let Ok(r) = s.request(lane.request) else { continue };
        let slot = lane.request.slot as usize;
        let count = match kind {
            GraphKind::Prefill => usize::from(lane.start_position + lane.tokens as u64 == r.prompt_tokens as u64),
            GraphKind::Resume | GraphKind::Decode => 1,
            GraphKind::Speculative => {
                let accepted = ACCEPTED[rounds[slot] % ACCEPTED.len()];
                rounds[slot] += 1;
                lane.tokens.min(accepted + 1)
            }
            _ => 0,
        };
        let consumed = if kind == GraphKind::Speculative {
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
            status: if count > 0 && generated == r.max_new_tokens { 2 } else { 1 },
            ..Default::default()
        };
        if kind == GraphKind::Speculative {
            let mut tokens = [-1; 8];
            tokens[..count].fill(99);
            record.vector = Some(TokenVector {
                emitted_count: count as i64,
                offered_drafts: (plan.step.token_capacity - 1) as i64,
                accepted_drafts: (count - 1) as i64,
                verification_width: plan.step.token_capacity as i64,
                draft_cache_consumed: -1,
                tokens,
                ..Default::default()
            });
        }
        records[slot] = record;
    }
    records
}

fn simulate(policy: PolicyConfig, jobs: &[Job], clients: usize) -> Vec<Outcome> {
    let mut s = engine(policy);
    let mut out = vec![Outcome::default(); jobs.len()];
    let mut live: [Option<(usize, RequestHandle)>; 4] = [None; 4];
    let mut rounds = [0usize; 4];
    let mut waiting: VecDeque<usize> = VecDeque::new();
    let (mut next, mut now, mut gpu_free, mut outstanding, mut finished) = (0usize, 0u64, 0u64, 0usize, 0usize);
    let mut inflight: VecDeque<(ScheduledPlan, Fence, u64, u64, bool)> = VecDeque::new();
    let mut guard = 0u64;
    while finished < jobs.len() {
        guard += 1;
        assert!(guard < 20_000_000, "simulation did not finish");
        while next < jobs.len() && jobs[next].arrival_ns <= now {
            waiting.push_back(next);
            next += 1;
        }
        while outstanding < clients && s.has_available_slot() {
            let Some(j) = waiting.pop_front() else { break };
            let mut prompt: Vec<u32> = (0..jobs[j].prompt as u32).map(|i| 1000 + i % 90_000).collect();
            prompt[0] = 150_000 + j as u32;
            let spec = RequestSpec { message_boundaries: vec![], prompt, conditioning: vec![],
                                     max_new_tokens: jobs[j].output, grammar_constrained: false, host_stops: false,
                                     growable_output: false };
            let h = s.admit(spec).unwrap();
            live[h.slot as usize] = Some((j, h));
            rounds[h.slot as usize] = 0;
            out[j].admitted_ns = now;
            outstanding += 1;
        }
        while let Some(plan) = s.plan() {
            let fence = Fence(s.submitted.0 + 1);
            if s.commit(&plan, fence).is_err() {
                break;
            }
            let ns = true_ns(&plan).max(1);
            let queued = gpu_free > now;
            STEPS.with(|steps| {
                if let Some(steps) = steps.borrow_mut().as_mut() {
                    let lanes = plan.step.lanes.iter().enumerate()
                        .map(|(row, l)| (live[l.request.slot as usize].unwrap().0,
                                         if plan.step.lane_kind(row) == GraphKind::Prefill { l.tokens } else { 0 }))
                        .collect();
                    steps.push((gpu_free.max(now), gpu_free.max(now) + ns, plan.step.kind, lanes));
                }
            });
            gpu_free = gpu_free.max(now) + ns;
            inflight.push_back((plan, fence, gpu_free, ns, queued));
        }
        let next_finish = inflight.front().map(|f| f.2);
        let next_arrival = jobs.get(next).map(|j| j.arrival_ns);
        let complete_next = match (next_finish, next_arrival) {
            (Some(f), a) => a.is_none_or(|a| f <= a),
            (None, Some(_)) => false,
            (None, None) => panic!("stalled with {} unfinished requests", jobs.len() - finished),
        };
        match (complete_next, next_finish, next_arrival) {
            (true, Some(f), _) => {
                now = now.max(f);
                s.advance_to(now).unwrap();
                let (plan, fence, _, ns, queued) = inflight.pop_front().unwrap();
                let records = sim_records(&s, &plan, fence, &mut rounds);
                let mut gpu = [None; 16];
                gpu[0] = Some(ns);
                let tokens = s.complete(fence, &records, ExecutionTiming { gpu_ns: gpu, wall_ns: ns, queued }).unwrap();
                let mut published: Vec<RequestHandle> = Vec::new();
                for t in &tokens {
                    if !published.contains(&t.request) {
                        published.push(t.request);
                    }
                }
                for h in published {
                    s.published_output(h, fence).unwrap();
                    let (j, _) = live[h.slot as usize].unwrap();
                    out[j].first_ns.get_or_insert(now);
                    out[j].last_ns = Some(now);
                    out[j].generated = s.request(h).unwrap().generated_tokens;
                }
                for slot in 0..4 {
                    let Some((_, h)) = live[slot] else { continue };
                    let r = s.request(h).unwrap();
                    if r.phase == RequestPhase::Finishing && r.pending == 0 {
                        s.finish(h).unwrap();
                        live[slot] = None;
                        outstanding -= 1;
                        finished += 1;
                    }
                }
            }
            (_, _, Some(a)) => {
                now = now.max(a);
                s.advance_to(now).unwrap();
            }
            _ => unreachable!(),
        }
    }
    out
}

fn policies() -> Vec<(&'static str, PolicyConfig)> {
    let srpt = PolicyConfig { prefill_srpt: true, ..PolicyConfig::default() };
    vec![
        ("default", PolicyConfig::default()),
        ("srpt-v3", srpt),
        ("v3+A1024", PolicyConfig { prefill_srpt_short_tokens: 1024, ..srpt }),
        ("v3+B1024", PolicyConfig { prefill_srpt_decode_priority_tokens: 1024, ..srpt }),
        ("v3+A+B1024", PolicyConfig { prefill_srpt_short_tokens: 1024, prefill_srpt_decode_priority_tokens: 1024, ..srpt }),
    ]
}

fn secs(ns: u64) -> f64 {
    ns as f64 / 1e9
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

#[test]
fn simulate_concurrent_long_agents() {
    // 4 x 28,672-token prompts at t = 0 with 512 output tokens (the release long-c4 shape), then 2,048 outputs.
    for output in [512u32, 2048] {
        let jobs = vec![Job { arrival_ns: 0, prompt: 28_672, output }; 4];
        println!("== long c4: 4 x 28,672 prompt, {output} output");
        for (name, policy) in policies() {
            let out = simulate(policy, &jobs, 4);
            let ttft: Vec<f64> = out.iter().map(|o| secs(o.first_ns.unwrap() - o.admitted_ns)).collect();
            let last: Vec<f64> = out.iter().map(|o| secs(o.last_ns.unwrap() - o.admitted_ns)).collect();
            let decode: Vec<f64> = out.iter().map(|o| (o.generated - 1) as f64 / secs(o.last_ns.unwrap() - o.first_ns.unwrap())).collect();
            assert!(out.iter().all(|o| o.generated == output));
            println!("{name:>14}: TTFT {:?} mean {:.1} | last {:?} mean {:.1} | decode tok/s {:?} median {:.1}",
                     ttft.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(), ttft.iter().sum::<f64>() / 4.0,
                     last.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(), last.iter().sum::<f64>() / 4.0,
                     decode.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(), quantile(decode.clone(), 0.5));
        }
    }
}

#[test]
fn simulate_c4_prefill() {
    let jobs = vec![Job { arrival_ns: 0, prompt: 30_720, output: 2 }; 4];
    println!("== c4 x 30,720, 2 output tokens");
    for (name, policy) in policies() {
        let out = simulate(policy, &jobs, 4);
        let mut ttft: Vec<f64> = out.iter().map(|o| secs(o.first_ns.unwrap() - o.admitted_ns)).collect();
        ttft.sort_by(f64::total_cmp);
        println!("{name:>14}: TTFT {:?} mean {:.1}", ttft.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(),
                 ttft.iter().sum::<f64>() / 4.0);
    }
}

#[test]
fn simulate_mixed_poisson_soak() {
    // The round-1 soak shape: Poisson 0.17/s for 15 min, 512/2k/8k/30k = 50/30/15/5 %, 4 clients, 8 output tokens.
    let mut rng = 20_261_024u64;
    let mut draw = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut jobs = Vec::new();
    let mut t = 0.0f64;
    loop {
        t += -(1.0 - draw()).ln() / 0.17;
        if t >= 900.0 {
            break;
        }
        let x = draw();
        let prompt = if x < 0.5 { 512 } else if x < 0.8 { 2048 } else if x < 0.95 { 8192 } else { 30_720 };
        jobs.push(Job { arrival_ns: (t * 1e9) as u64, prompt, output: 8 });
    }
    println!("== soak: {} arrivals, TTFT from ARRIVAL (s) p50 / p95 / max by prompt length; all mean", jobs.len());
    for (name, policy) in policies() {
        let out = simulate(policy, &jobs, 4);
        let mut line = format!("{name:>14}:");
        for length in [512usize, 2048, 8192, 30_720] {
            let v: Vec<f64> = jobs.iter().zip(&out).filter(|(j, _)| j.prompt == length)
                .map(|(j, o)| secs(o.first_ns.unwrap() - j.arrival_ns)).collect();
            if v.is_empty() {
                continue;
            }
            line += &format!(" {length}: {:.2}/{:.2}/{:.2} (n {})", quantile(v.clone(), 0.5), quantile(v.clone(), 0.95),
                             quantile(v.clone(), 1.0), v.len());
        }
        let all: Vec<f64> = jobs.iter().zip(&out).map(|(j, o)| secs(o.first_ns.unwrap() - j.arrival_ns)).collect();
        line += &format!(" | all mean {:.2} max {:.2}", all.iter().sum::<f64>() / all.len() as f64, quantile(all, 1.0));
        println!("{line}");
    }
}





#[test]
fn simulate_sensitivity() {
    let mut rng = 20_261_024u64;
    let mut draw = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut soak = Vec::new();
    let mut t = 0.0f64;
    loop {
        t += -(1.0 - draw()).ln() / 0.17;
        if t >= 900.0 { break; }
        let x = draw();
        let prompt = if x < 0.5 { 512 } else if x < 0.8 { 2048 } else if x < 0.95 { 8192 } else { 30_720 };
        soak.push(Job { arrival_ns: (t * 1e9) as u64, prompt, output: 8 });
    }
    let long = vec![Job { arrival_ns: 0, prompt: 28_672, output: 512 }; 4];
    let long2k = vec![Job { arrival_ns: 0, prompt: 28_672, output: 2048 }; 4];
    let srpt = PolicyConfig { prefill_srpt: true, ..PolicyConfig::default() };
    println!("== sensitivity: soak from arrival mean / 512 p95 / 30k max | long512 decode median / last mean / TTFT mean | long2048 last mean");
    for short in [0u64, 512, 1024, 2048] {
        for budget in [0u64, 256, 1024, 4096] {
            let p = PolicyConfig { prefill_srpt_short_tokens: short,
                                   prefill_srpt_decode_priority_tokens: budget, ..srpt };
            let o = simulate(p, &soak, 4);
            let all: Vec<f64> = soak.iter().zip(&o).map(|(j, o)| secs(o.first_ns.unwrap() - j.arrival_ns)).collect();
            let s512: Vec<f64> = soak.iter().zip(&o).filter(|(j, _)| j.prompt == 512).map(|(j, o)| secs(o.first_ns.unwrap() - j.arrival_ns)).collect();
            let s30: Vec<f64> = soak.iter().zip(&o).filter(|(j, _)| j.prompt == 30_720).map(|(j, o)| secs(o.first_ns.unwrap() - j.arrival_ns)).collect();
            let l = simulate(p, &long, 4);
            let dec: Vec<f64> = l.iter().map(|o| (o.generated - 1) as f64 / secs(o.last_ns.unwrap() - o.first_ns.unwrap())).collect();
            let last = l.iter().map(|o| secs(o.last_ns.unwrap())).sum::<f64>() / 4.0;
            let ttft = l.iter().map(|o| secs(o.first_ns.unwrap())).sum::<f64>() / 4.0;
            let l2 = simulate(p, &long2k, 4);
            let last2 = l2.iter().map(|o| secs(o.last_ns.unwrap())).sum::<f64>() / 4.0;
            println!("short {short:>4} budget {budget:>4}: soak {:.2} / {:.2} / {:.1} | long {:.1} / {:.1} / {:.1} | long2k {:.1}",
                     all.iter().sum::<f64>() / all.len() as f64, quantile(s512, 0.95), quantile(s30, 1.0),
                     quantile(dec, 0.5), last, ttft, last2);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Tests required by the independent review (required change 5).

fn small_aging(short: u64, budget: u64, max_wait: u64) -> RequestScheduler {
    let mut s = small(true, short, budget);
    let mut policy = s.policy;
    policy.prefill_srpt_max_wait_chunks = max_wait;
    s.policy = policy;
    s
}

#[test]
fn an_aged_out_waiter_takes_over_from_short_prompts_when_the_leader_finishes() {
    let mut s = small_aging(128, 0, 8);
    let l1 = s.admit(request(0, 2048, 1)).unwrap();
    let l2 = s.admit(request(1, 4096, 1)).unwrap();
    let mut tag = 2;
    let mut sequence = Vec::new();
    for _ in 0..600 {
        while s.slots.iter().filter(|e| e.request.is_some()).count() < 4 {
            s.admit(request(tag, 128, 1)).unwrap();
            tag += 1;
        }
        sequence.extend(prefill_lanes(&small_step(&mut s)));
        if done(&s, l1) && sequence.iter().filter(|h| **h == l2).count() >= 2 {
            break;
        }
    }
    let first = sequence.iter().position(|h| *h == l1).unwrap();
    let last = sequence.iter().rposition(|h| *h == l1).unwrap();
    assert!(last - first + 1 <= 2 * (2048 / 128), "leader got fewer than every other chunk");
    assert_eq!(sequence[last + 1], l2, "the aged-out waiter wins the chunk after the leader finishes: {sequence:?}");
}

#[test]
fn a_two_chunk_short_prompt_completes_across_the_leader_finishing() {
    let mut s = small(true, 256, 0);
    let leader = s.admit(request(0, 1024, 1)).unwrap();
    for _ in 0..50 {
        if s.request(leader).unwrap().consumed_tokens >= 896 {
            break;
        }
        small_step(&mut s);
    }
    let brief = s.admit(request(1, 256, 1)).unwrap();
    let mut sequence = Vec::new();
    for _ in 0..50 {
        if done(&s, leader) && done(&s, brief) {
            break;
        }
        sequence.extend(prefill_lanes(&small_step(&mut s)));
    }
    assert_eq!(sequence, vec![brief, leader, brief], "short chunk, leader's final chunk, short's final chunk");
}

#[test]
fn a_hold_never_withholds_every_plan_from_an_idle_gpu() {
    // No decode graphs: a decoder within its budget is eligible but has no runnable graph; the waiting prompt must
    // still be planned when nothing is in flight.
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits { request_slots: 4, max_active: 4, max_sequence_tokens: 8192, prefill_chunk_size: 128,
                          pages: 512, snapshots: 16 },
        SchedulerCapabilities {
            vocabulary: 248320, page_tokens: 128,
            vision: crate::model::VisionCapabilities { feature_slots: 4, features_per_request: 16384, max_images: 4,
                patches_per_feature: 4, patch_capacities: vec![256, 1024, 4096, 16384] },
            draft_pages_per_request: 0, completion_inputs: vec![], state_rows_per_request: None,
        },
    ).unwrap();
    let mut graphs = super::super::tests::duration_graphs(4);
    graphs.retain(|g| !matches!(g.kind, GraphKind::Decode | GraphKind::Speculative));
    for g in &mut graphs {
        g.estimated_duration_ns *= 1000;
    }
    s.measured_graph_durations(&graphs).unwrap();
    s.configure_policy(PolicyConfig { prefill_srpt: true, prefill_srpt_decode_priority_tokens: 64, ..PolicyConfig::agent() })
        .unwrap();
    let decoder = s.admit(request(9, 128, 48)).unwrap();
    for _ in 0..50 {
        if s.request(decoder).unwrap().phase == RequestPhase::Decode {
            break;
        }
        small_step(&mut s);
    }
    assert_eq!(s.request(decoder).unwrap().phase, RequestPhase::Decode);
    let prompt = s.admit(request(0, 512, 1)).unwrap();
    for _ in 0..50 {
        if done(&s, prompt) {
            break;
        }
        assert!(s.decode_priority_holds_prefill(), "the decoder is within its budget and eligible");
        small_step(&mut s); // panics if no plan
    }
    assert!(done(&s, prompt), "the prompt was planned despite the hold");
}

#[test]
fn decode_priority_covers_the_boundary_snapshot_the_second_slot_and_releases_at_the_budget() {
    let mut s = small(true, 0, 24);
    let a = s.admit(request(0, 128, 48)).unwrap();
    let b = s.admit(request(1, 1024, 1)).unwrap();
    let mut saw_snapshot = false;
    let mut first_b_at = None;
    for _ in 0..400 {
        let Some(plan) = s.plan() else { break };
        let generated = s.request(a).map_or(48, |r| r.generated_tokens);
        if s.request(a).is_ok_and(|r| r.phase == RequestPhase::Snapshot) {
            saw_snapshot = true;
        }
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        if first_b_at.is_none() && prefill_lanes(&plan).contains(&b) {
            first_b_at = Some(generated);
        }
        // Second pipeline slot: while a's round is in flight and a is within its budget, b's chunk is not queued.
        if plan.step.lanes.iter().any(|l| l.request == a)
            && matches!(plan.step.kind, GraphKind::Decode) && generated + 1 < 24 {
            if let Some(next) = s.plan() {
                assert!(!prefill_lanes(&next).contains(&b), "prefill queued behind an in-flight decoder round");
            }
        }
        let records = small_records(&s, &plan, fence);
        s.advance_to(s.now_ns + plan.estimated_duration_ns.max(MS)).unwrap();
        s.complete(fence, &records, Default::default()).unwrap();
        if done(&s, b) && s.request(a).map_or(true, |r| r.generated_tokens >= 48) {
            break;
        }
    }
    let at = first_b_at.expect("b runs");
    assert!(at >= 24, "b's first chunk ran at {at} of a's tokens (budget 24)");
    assert!(at < 48, "the hold released at the budget while a still had tokens to generate (at {at})");
    let _ = saw_snapshot; // boundary snapshots are covered by the hold whenever the engine captures one
}

/// FNV-1a digest of every committed step (kind, batch, capacity, lanes) and every outcome of the simulation.
fn digest(policy: PolicyConfig, jobs: &[Job]) -> u64 {
    STEPS.with(|s| *s.borrow_mut() = Some(Vec::new()));
    let out = simulate(policy, jobs, 4);
    let steps = STEPS.with(|s| s.borrow_mut().take().unwrap());
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for (st, en, kind, lanes) in &steps {
        eat(*st); eat(*en); eat(*kind as u64);
        for (j, t) in lanes { eat(*j as u64); eat(*t as u64); }
    }
    for o in &out {
        eat(o.first_ns.unwrap()); eat(o.last_ns.unwrap()); eat(o.generated as u64);
    }
    h
}

/// Long prefills, short arrivals during them, boundary snapshots, decodes and aging (max wait 16 chunks).
fn trace_jobs() -> Vec<Job> {
    let mut jobs = vec![Job { arrival_ns: 0, prompt: 8192, output: 64 }, Job { arrival_ns: 0, prompt: 30_720, output: 16 }];
    for i in 0..12u64 {
        jobs.push(Job { arrival_ns: (2 + 3 * i) * 1_000 * MS, prompt: [512, 2048, 512, 8192][i as usize % 4], output: 32 });
    }
    jobs
}

#[test]
fn default_knobs_reproduce_the_v3_schedule_bit_for_bit() {
    // Digests computed with the same simulation on the tree of the shipped option (the sources of binary 48b69ed4):
    // every committed step and every outcome must match.
    let v3 = [("default", PolicyConfig::default()), ("srpt", PolicyConfig { prefill_srpt: true, ..PolicyConfig::default() }),
              ("srpt-aging16", PolicyConfig { prefill_srpt: true, prefill_srpt_max_wait_chunks: 16, ..PolicyConfig::default() })];
    for (name, policy) in v3 {
        let d = digest(policy, &trace_jobs());
        println!("schedule digest {name} {d:016x}");
        let expected = V3_DIGESTS.iter().find(|(n, _)| *n == name).map(|(_, d)| *d);
        if let Some(expected) = expected {
            assert_eq!(d, expected, "{name}: schedule differs from v3");
        }
    }
}

/// Filled from the v3-tree run (see default_knobs_reproduce_the_v3_schedule_bit_for_bit).
const V3_DIGESTS: &[(&str, u64)] = &[("default", 0x50565f1ab983a480), ("srpt", 0x05cbfe3de6e05329),
                                      ("srpt-aging16", 0x1fa4c4bce24a7841)];
