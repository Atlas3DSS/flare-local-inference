fn combined_engine(active: usize) -> RequestScheduler {
    let mut s = engine(active);
    s.configure_policy(PolicyConfig::throughput()).unwrap();
    let graphs = combined_graphs(active);
    s.graph_durations(&graphs).unwrap();
    for graph in &graphs {
        s.costs.set_measured(&ScheduledStep { draft_tokens: graph.draft_tokens, kind: graph.kind,
            batch_capacity: graph.batch_capacity, token_capacity: graph.token_capacity,
            packed_token_capacity: graph.packed_token_capacity, mixed: None, lanes: vec![],
            requires_feedback: true }, if graph.kind == GraphKind::PackedSpeculative {
                1_000_000 + graph.draft_tokens as u64 * 100_000 + graph.packed_token_capacity as u64 * 20_000
            } else { 10_000_000 });
    }
    s
}

fn combined_graphs(active: usize) -> Vec<GraphDescriptor> {
    let mut graphs = super::super::tests::duration_graphs(active);
    graphs.retain(|g| g.kind != GraphKind::Speculative || g.token_capacity == 1);
    for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= active.next_power_of_two()) {
        let mut capacities = vec![batch, batch * 2, batch * 4, (batch * 6).min(128), (batch * 8).min(128)];
        capacities.sort_unstable(); capacities.dedup();
        for depth in 1..=7 {
            for &tokens in &capacities {
                graphs.push(GraphDescriptor { name: format!("combined-{batch}-{depth}-{tokens}"),
                    kind: GraphKind::PackedSpeculative, draft_tokens: depth,
                    sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                    token_capacity: 8, packed_token_capacity: tokens, prefill_capacity: 0,
                    estimated_duration_ns: 0, input_bindings: vec![] });
            }
        }
    }
    graphs
}

fn combined_plan(s: &RequestScheduler, handles: &[RequestHandle], limits: &[usize], budget: usize) -> ScheduledPlan {
    ScheduledPlan { origin: PlanOrigin::Ordinary, selected_cost: s.selected_cost(cost::Key { drafts: 7, kind: GraphKind::PackedSpeculative, batch: handles.len().next_power_of_two(), tokens: 8, packed: budget, prefill: 0 }), owner: s.plan_owner.clone(),
        epochs: handles.iter().map(|h| s.request(*h).unwrap().epoch).collect(),
        estimated_duration_ns: 1_000_000, selection_value_rate: 0.0,
        selection_log_scale: 0.0, verification_budget: None,
        step: ScheduledStep { kind: GraphKind::PackedSpeculative, draft_tokens: 7,
            batch_capacity: handles.len().next_power_of_two(), token_capacity: 8,
            packed_token_capacity: budget, mixed: None, requires_feedback: true,
            lanes: handles.iter().zip(limits).map(|(&h, &limit)| ScheduledLane {
                request: h, start_position: s.request(h).unwrap().consumed_tokens, tokens: limit,
            }).collect() } }
}

fn combined_records(s: &RequestScheduler, plan: &ScheduledPlan, offered: &[i64]) -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    for (lane, &offers) in plan.lanes.iter().zip(offered) {
        let request = s.request(lane.request).unwrap();
        let mut vector = TokenVector { emitted_count: 1, offered_drafts: offers,
            verification_width: 8, ..Default::default() };
        vector.tokens[0] = 100;
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64, submission_sequence: 1,
            consumed_tokens: request.consumed_tokens as i64 + 1,
            generated_tokens: request.generated_tokens as i64 + 1,
            token_id: 100, status: 1, vector: Some(vector), ..Default::default()
        };
    }
    records
}

#[test]
fn packed_combined_selects_captured_depth_and_budget_before_sampling() {
    for active in [1, 8, 32] {
        let mut s = combined_engine(active);
        let handles: Vec<_> = (0..active).map(|_| ready(&mut s)).collect();
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::PackedSpeculative);
        assert_eq!(plan.lanes.len(), active);
        assert!((1..=7).contains(&plan.draft_tokens));
        assert_eq!(plan.token_capacity, 8);
        assert!(plan.packed_token_capacity >= active && plan.packed_token_capacity <= 128);
        assert!(plan.lanes.iter().all(|lane| lane.tokens == plan.draft_tokens + 1));
        assert!(handles.iter().all(|h| !s.owns_proposal(s.request(*h).unwrap())));
        // Distinct depth costs cannot collapse into the same B/W/P key.
        let shape = s.costs.packed_round_shapes().find(|g| g.batch == active && g.drafts == 1).unwrap();
        let step = ScheduledStep { kind: GraphKind::PackedSpeculative, draft_tokens: 1,
            batch_capacity: active, token_capacity: 8, packed_token_capacity: shape.tokens,
            mixed: None, lanes: vec![], requires_feedback: true };
        s.costs.set_measured(&step, 1);
        assert_eq!(s.plan().unwrap().draft_tokens, 1);
    }
}

#[test]
fn packed_combined_invalid_feedback_is_atomic_for_every_lane_and_forecast() {
    for failure in 0..6 {
        let mut s = combined_engine(4);
        let handles: Vec<_> = (0..4).map(|_| ready(&mut s)).collect();
        let limits = if failure == 0 { [8, 8, 8, 2] } else { [8; 4] };
        let plan = combined_plan(&s, &handles, &limits, 8);
        s.commit(&plan, Fence(1)).unwrap();
        let before: Vec<_> = handles.iter().map(|h| {
            let r = s.request(*h).unwrap();
            (r.generated_tokens, r.consumed_tokens, r.scheduled_consumed, r.pending,
             r.private_pages.clone(), r.reserved_pages.clone())
        }).collect();
        let mut records = combined_records(&s, &plan, &[0, 0, 0, 0]);
        let mut confidence = [[0.5; 7]; 4];
        let last = &mut records[handles[3].slot as usize];
        match failure {
            0 => last.vector.as_mut().unwrap().offered_drafts = 2,
            1 => for h in &handles { records[h.slot as usize].vector.as_mut().unwrap().offered_drafts = 2; },
            2 => confidence[3][0] = f32::NAN,
            3 => last.generation += 1,
            4 => last.vector.as_mut().unwrap().verification_width = 7,
            _ => last.vector.as_mut().unwrap().emitted_count = 2,
        }
        assert!(s.complete_packed_round(Fence(1), &records, Default::default(), &confidence).is_err());
        assert!(s.submissions.contains_key(&Fence(1)));
        for (h, expected) in handles.iter().zip(before) {
            let r = s.request(*h).unwrap();
            assert_eq!((r.generated_tokens, r.consumed_tokens, r.scheduled_consumed, r.pending,
                r.private_pages.clone(), r.reserved_pages.clone()), expected);
        }
        assert!(s.split.as_ref().unwrap().forecast.iter().all(Option::is_none));
    }
}

#[test]
fn packed_combined_commits_once_with_frozen_reservations_and_terminal_forecasts() {
    let mut s = combined_engine(4);
    let handles: Vec<_> = (0..4).map(|_| ready(&mut s)).collect();
    let plan = combined_plan(&s, &handles, &[1, 3, 4, 8], 12);
    s.commit(&plan, Fence(1)).unwrap();
    assert!(s.plan().is_none(), "terminal feedback is the sole continuation fence");
    for lane in &plan.lanes {
        let r = s.request(lane.request).unwrap();
        assert_eq!(r.scheduled_consumed, lane.start_position + lane.tokens as u64);
        assert!(r.pending > 0);
    }
    let records = combined_records(&s, &plan, &[0, 1, 2, 3]);
    let generated = s.complete_packed_round(Fence(1), &records, Default::default(), &[[0.8; 7]; 4]).unwrap();
    assert_eq!(generated.len(), 4);
    for h in &handles {
        let r = s.request(*h).unwrap();
        assert_eq!((r.generated_tokens, r.consumed_tokens, r.scheduled_consumed, r.pending), (2, 66, 66, 0));
        let forecast = s.split.as_ref().unwrap().forecast[h.slot as usize].unwrap();
        assert_eq!(forecast.request, *h);
        assert_eq!(forecast.observed_at[0], if *h == handles[0] { 0 } else { 1 });
    }
    assert!(s.complete_packed_round(Fence(1), &records, Default::default(), &[[0.8; 7]; 4]).is_err());
}

#[test]
fn packed_combined_cancellation_holds_worst_case_pages_and_slot_until_terminal_fence() {
    let mut s = combined_engine(1);
    let h = s.admit(super::super::tests::spec(127, 64)).unwrap();
    let r = s.request_mut(h).unwrap();
    r.phase = RequestPhase::Decode; r.consumed_tokens = 127;
    r.scheduled_consumed = 127; r.generated_tokens = 1;
    let plan = combined_plan(&s, &[h], &[8], 8);
    let records = combined_records(&s, &plan, &[0]);
    s.commit(&plan, Fence(1)).unwrap();
    assert_eq!(s.request(h).unwrap().scheduled_consumed, 135);
    assert_eq!(s.request(h).unwrap().private_pages.len(), 4,
        "three worst-case target pages plus the declared draft page are reserved before GPU allocation");
    s.cancel(h).unwrap();
    assert!(s.request(h).is_err());
    assert!(s.admit(super::super::tests::spec(127, 64)).is_err());
    let output = s.complete_packed_round(Fence(1), &records, Default::default(), &[[0.9; 7]]).unwrap();
    assert!(output.is_empty());
    let replacement = s.admit(super::super::tests::spec(127, 64)).unwrap();
    assert_eq!(replacement.slot, h.slot);
    assert_ne!(replacement.generation, h.generation);
    assert_eq!(s.request(replacement).unwrap().generated_tokens, 0);
    assert!(s.split.as_ref().unwrap().forecast[replacement.slot as usize].is_none());
}

fn matched_c32_planner(gpu: bool) -> RequestScheduler {
    let mut s = super::super::tests::scheduler(32, 1024, 64);
    s.configure_policy(PolicyConfig::throughput()).unwrap();
    s.enable_speculation(&(1..=6).collect::<Vec<_>>()).unwrap();
    s.split_speculation(6).unwrap();
    let mut graphs = super::super::tests::duration_graphs(32);
    graphs.retain(|g| g.kind != GraphKind::Speculative || g.token_capacity == 1);
    for batch in [1, 2, 4, 8, 16, 32] {
        for depth in 1..=5 {
            if !gpu {
                graphs.push(GraphDescriptor { name: format!("proposal-{batch}-{depth}"),
                    kind: GraphKind::Proposal, draft_tokens: 0, sampling_mode: SamplingMode::General,
                    handle: 1, batch_capacity: batch, token_capacity: depth + 1,
                    packed_token_capacity: 0, prefill_capacity: 0, estimated_duration_ns: 0, input_bindings: vec![] });
            }
            for tokens in (1..=6).map(|n| batch * n).filter(|n| *n <= 128) {
                if gpu || depth == 1 {
                    graphs.push(GraphDescriptor { name: format!("target-{batch}-{depth}-{tokens}"),
                        kind: if gpu { GraphKind::PackedSpeculative } else { GraphKind::Verification },
                        draft_tokens: if gpu { depth } else { 0 }, sampling_mode: SamplingMode::General,
                        handle: 1, batch_capacity: batch, token_capacity: 6, packed_token_capacity: tokens,
                        prefill_capacity: 0, estimated_duration_ns: 0, input_bindings: vec![] });
                }
            }
        }
    }
    s.graph_durations(&graphs).unwrap();
    for graph in &graphs {
        let proposal = |depth: usize| 400_000 + depth as u64 * 500_000;
        let target = |tokens: usize| 7_000_000 + graph.batch_capacity as u64 * 100_000 + tokens as u64 * 150_000;
        let duration = match graph.kind {
            GraphKind::Proposal => proposal(graph.token_capacity - 1),
            GraphKind::Verification => target(graph.packed_token_capacity),
            GraphKind::PackedSpeculative => proposal(graph.draft_tokens) + target(graph.packed_token_capacity),
            _ => 10_000_000 + graph.batch_capacity as u64 * 100_000,
        };
        s.costs.set_measured(&ScheduledStep { kind: graph.kind, draft_tokens: graph.draft_tokens,
            batch_capacity: graph.batch_capacity, token_capacity: graph.token_capacity,
            packed_token_capacity: graph.packed_token_capacity, mixed: None, lanes: vec![],
            requires_feedback: true }, duration);
    }
    s.advance_to(10_000_000).unwrap();
    for index in 0..32 {
        let h = s.admit(super::super::tests::spec(65, 384)).unwrap();
        let r = s.request_mut(h).unwrap();
        r.phase = RequestPhase::Decode; r.consumed_tokens = 161;
        r.scheduled_consumed = 161; r.generated_tokens = 97;
        r.accounting.last_progress_ns = 10_000_000 - index as u64 * 10_000;
        r.accounting.service.tokens = Some(utility::ServicePrice::initial(0.06 + (index % 3) as f64 * 0.01));
        let confidence = std::array::from_fn(|position| 0.5 + ((index * 17 + position * 7) % 40) as f32 / 100.0);
        s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast { request: h,
            prefixes: survival(&confidence), observed_at: std::array::from_fn(|position| 97 - (position % 3) as u32 * 16) });
    }
    s
}

#[test]
fn packed_combined_duration_reference_covers_the_requested_cohort_rows() {
    let cpu = matched_c32_planner(false);
    let gpu = matched_c32_planner(true);
    for width in 2..=4 {
        assert_eq!(gpu.proposal_round_duration(32, width), cpu.proposal_round_duration(32, width),
            "equal summed graph costs must price the same requested cohort capacity at width {width}");
    }
}

#[test]
#[ignore = "CPU full-planner ABBA with equal state and sum-matched graph costs"]
fn packed_combined_full_c32_planning_cpu_cost() {
    let cpu = matched_c32_planner(false);
    let gpu = matched_c32_planner(true);
    let describe = |s: &RequestScheduler, gpu: bool| {
        let p = s.plan().unwrap();
        (p.batch_capacity, if gpu { p.draft_tokens } else { p.token_capacity - 1 },
            if gpu { Some(p.packed_token_capacity) } else { p.verification_budget },
            p.lanes.iter().map(|l| (l.request, l.tokens)).collect::<Vec<_>>(), p.selection_value_rate.to_bits())
    };
    eprintln!("same_state_plans cpu={:?} gpu={:?}", describe(&cpu, false), describe(&gpu, true));
    for optimized in [false, true, true, false] {
        let scheduler = if optimized { &gpu } else { &cpu };
        for _ in 0..10 { std::hint::black_box(scheduler.plan()); }
        let start = std::time::Instant::now();
        for _ in 0..100 { std::hint::black_box(scheduler.plan()); }
        eprintln!("gpu_prefix={optimized} us/full_plan={:.6}", start.elapsed().as_secs_f64() * 1e6 / 100.0);
    }
}


#[test]
fn packed_cohort_selection_prices_long_prefixes_and_includes_width_one_rows() {
    let mut s = combined_engine(4);
    let handles: Vec<_> = (0..3).map(|_| ready(&mut s)).collect();
    for (&h, remaining) in handles.iter().zip([2, 6, 6]) {
        s.request_mut(h).unwrap().max_new_tokens = 1 + remaining;
        s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast {
            request: h, prefixes: [1.; 7], observed_at: [1; 7],
        });
    }
    let mut values = [LaneValue::default(); REQUEST_LANES];
    for &h in &handles { values[h.slot as usize].output_token = 1.; }
    let mut candidates = Vec::new();
    s.packed_round_candidates(&mut candidates, &values, 0.);
    let best = candidates.iter().filter(|c| c.batch == 2 && c.draft_tokens == 5
        && c.packed_token_capacity == 12).max_by(|a,b| a.rate.compare(b.rate)).unwrap();
    assert_eq!(best.mask, (1 << handles[1].slot) | (1 << handles[2].slot));
    assert_eq!(best.rate.score, ValueRate::new(best.duration_ns, 12., 0.).score);
    let h = handles[0];
    s.request_mut(h).unwrap().host_stops = true;
    values[h.slot as usize].output_token = 100.;
    candidates.clear();
    s.packed_round_candidates(&mut candidates, &values, 0.);
    let best = candidates.iter().filter(|c| c.batch == 2 && c.draft_tokens == 5
        && c.packed_token_capacity == 12).max_by(|a,b| a.rate.compare(b.rate)).unwrap();
    assert_ne!(best.mask & (1 << h.slot), 0);
    assert_eq!(best.rate.score, ValueRate::new(best.duration_ns, 106., 0.).score);
    assert_eq!(s.speculative_tokens(s.request(h).unwrap(), best.draft_tokens + 1), 1);
    values[h.slot as usize].output_token = 1.;
    values[h.slot as usize].age_price = 1.;
    values[h.slot as usize].age_ms = 100.;
    values[h.slot as usize].reference_delivery_ms = 10.;
    candidates.clear();
    s.packed_round_candidates(&mut candidates, &values, 0.);
    let best = candidates.iter().filter(|c| c.batch == 2 && c.draft_tokens == 5
        && c.packed_token_capacity == 12).max_by(|a,b| a.rate.compare(b.rate)).unwrap();
    assert_ne!(best.mask & (1 << h.slot), 0, "aged width-one rows retain delivery utility");
}

#[test]
fn packed_candidates_keep_full_cost_with_padding_and_respect_retained_ownership() {
    let mut s = combined_engine(4);
    let handles: Vec<_> = (0..2).map(|_| ready(&mut s)).collect();
    retain(&mut s, handles[0], [1.; 7]);
    s.request_mut(handles[1]).unwrap().max_new_tokens = 2;
    let (values, age, _) = s.lane_values(0);
    let mut candidates = Vec::new();
    s.packed_round_candidates(&mut candidates, &values, age);
    assert!(candidates.iter().all(|c| c.mask == 1 << handles[1].slot));
    let padded = candidates.iter().find(|c| c.batch == 4 && c.draft_tokens == 7
        && c.packed_token_capacity == 32).unwrap();
    assert_eq!(padded.duration_ns, 2_340_000);
    assert_eq!(padded.rate.score, ValueRate::new(padded.duration_ns,
        values[handles[1].slot as usize].decode(1., padded.duration_ns as f64 / 1e6), age).score);
}

#[test]
#[ignore = "manual host planning latency measurement"]
fn packed_planning_latency_at_concurrency() {
    for active in [32, 64] {
        for packed in [false, true] {
            let mut s = engine(active);
            s.configure_policy(PolicyConfig::throughput()).unwrap();
            let mut graphs = combined_graphs(active);
            if !packed { graphs.retain(|g| g.kind != GraphKind::PackedSpeculative); }
            for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= active) {
                for width in 2..=8 {
                    for kind in [GraphKind::Proposal, GraphKind::Verification, GraphKind::Speculative] {
                        if kind != GraphKind::Proposal && batch * width > 128 { continue; }
                        graphs.push(GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0,
                            name: format!("{kind:?}-{batch}-{width}"), kind, sampling_mode: SamplingMode::General,
                            handle: 1, batch_capacity: batch, token_capacity: width, prefill_capacity: 0,
                            estimated_duration_ns: 0, input_bindings: vec![] });
                    }
                }
            }
            s.speculative_graphs(&graphs.iter().filter(|g| g.kind == GraphKind::Speculative)
                .map(|g| (g.batch_capacity, g.token_capacity)).collect::<Vec<_>>()).unwrap();
            s.graph_durations(&graphs).unwrap();
            for graph in &graphs {
                s.costs.set_measured(&ScheduledStep { draft_tokens: graph.draft_tokens, kind: graph.kind,
                    batch_capacity: graph.batch_capacity, token_capacity: graph.token_capacity,
                    packed_token_capacity: graph.packed_token_capacity, mixed: None, lanes: vec![],
                    requires_feedback: true }, 10_000_000);
            }
            for index in 0..active {
                let h = ready(&mut s);
                s.request_mut(h).unwrap().max_new_tokens = 1 + (index % 8 + 1) as u32;
                s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast {
                    request: h, prefixes: survival(&[0.5 + (index % 5) as f32 * 0.1; 7]), observed_at: [1; 7],
                });
            }
            std::hint::black_box(s.plan());
            let start = std::time::Instant::now();
            for _ in 0..20 { std::hint::black_box(s.plan()); }
            eprintln!("active={active} packed={packed} packed_shapes={} us/full_plan={:.3}",
                s.costs.packed_round_shapes().count(), start.elapsed().as_secs_f64() * 1e6 / 20.);
        }
    }
}

#[test]
fn packed_joined_split_and_mixed_candidates_compete_with_frozen_proposal_ownership() {
    let mut s = engine(2);
    s.configure_policy(PolicyConfig::throughput()).unwrap();
    s.speculative_graphs(&[(1, 1), (1, 8), (2, 1), (2, 8)]).unwrap();
    s.mixed_graphs(&[(2, 8, 64)]).unwrap();
    let mut graphs = combined_graphs(2);
    for kind in [GraphKind::Speculative, GraphKind::Proposal, GraphKind::Verification, GraphKind::Mixed] {
        graphs.push(GraphDescriptor { draft_tokens: 0, name: format!("{kind:?}-wide"), kind,
            sampling_mode: SamplingMode::General, handle: 1, batch_capacity: 2, token_capacity: 8,
            packed_token_capacity: 0, prefill_capacity: if kind == GraphKind::Mixed { 64 } else { 0 },
            estimated_duration_ns: 0, input_bindings: vec![] });
    }
    s.graph_durations(&graphs).unwrap();
    for graph in &graphs {
        s.costs.set_measured(&ScheduledStep { draft_tokens: graph.draft_tokens, kind: graph.kind,
            batch_capacity: graph.batch_capacity, token_capacity: graph.token_capacity,
            packed_token_capacity: graph.packed_token_capacity,
            mixed: (graph.kind == GraphKind::Mixed).then_some(MixedStep { prefill_capacity: 64, prefill_row: None, speculative: true }),
            lanes: vec![], requires_feedback: true }, 10_000_000);
    }
    let handles: Vec<_> = (0..2).map(|_| ready(&mut s)).collect();
    let (values, age, _) = s.lane_values(0);
    let mut candidates = Vec::new();
    s.candidates(&mut candidates, &values, age);
    for kind in [GraphKind::Speculative, GraphKind::Proposal, GraphKind::PackedSpeculative, GraphKind::Mixed] {
        assert!(candidates.iter().any(|c| c.kind == kind), "missing {kind:?}");
    }
    let joined = ScheduledStep { draft_tokens: 0, kind: GraphKind::Speculative, batch_capacity: 2,
        token_capacity: 8, packed_token_capacity: 0, mixed: None, lanes: vec![], requires_feedback: true };
    s.costs.set_measured(&joined, 100);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
    s.costs.set_measured(&joined, 10_000_000);
    for kind in [GraphKind::Proposal, GraphKind::Verification] {
        s.costs.set_measured(&ScheduledStep { kind, ..joined.clone() }, 100);
    }
    assert_eq!(s.plan().unwrap().kind, GraphKind::Proposal);
    retain(&mut s, handles[0], [0.9; 7]);
    candidates.clear();
    s.candidates(&mut candidates, &values, age);
    for candidate in &candidates {
        if matches!(candidate.kind, GraphKind::Speculative | GraphKind::Proposal | GraphKind::PackedSpeculative | GraphKind::Mixed) {
            assert_eq!(candidate.mask & (1 << handles[0].slot), 0);
        }
    }
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Verification
        && c.mask & (1 << handles[0].slot) != 0));
}

#[test]
#[ignore = "manual host planning latency measurement for six-position verification"]
fn packed_planning_latency_c8_p32() {
    for packed in [false, true] {
        let mut s = super::super::tests::scheduler(8, 16384, 256);
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        s.enable_speculation(&(1..=6).collect::<Vec<_>>()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(8);
        graphs.retain(|g| g.kind != GraphKind::Speculative && g.kind != GraphKind::Vision
            && !(g.kind == GraphKind::Prefill && g.token_capacity > 4096));
        let mut shapes = Vec::new();
        for batch in [1, 2, 4, 8] {
            for width in 1..=6 {
                if width > 1 {
                    graphs.push(GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0,
                        name: format!("proposal-{batch}-{width}"), kind: GraphKind::Proposal,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: width, prefill_capacity: 0, estimated_duration_ns: 0, input_bindings: vec![] });
                }
                if batch * width > 32 { continue; }
                shapes.push((batch, width));
                for kind in [GraphKind::Speculative, GraphKind::Verification] {
                    graphs.push(GraphDescriptor { draft_tokens: 0,
                        packed_token_capacity: if kind == GraphKind::Verification { batch * width } else { 0 },
                        name: format!("{kind:?}-{batch}-{width}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: if kind == GraphKind::Verification { 6 } else { width },
                        prefill_capacity: 0, estimated_duration_ns: 0, input_bindings: vec![] });
                }
                if packed {
                    for depth in 1..=5 {
                        graphs.push(GraphDescriptor { draft_tokens: depth, packed_token_capacity: batch * width,
                            name: format!("packed-{batch}-{depth}-{width}"), kind: GraphKind::PackedSpeculative,
                            sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                            token_capacity: 6, prefill_capacity: 0, estimated_duration_ns: 0, input_bindings: vec![] });
                    }
                }
            }
        }
        s.speculative_graphs(&shapes).unwrap();
        s.split_speculation(6).unwrap();
        s.graph_durations(&graphs).unwrap();
        for graph in &graphs {
            s.costs.set_measured(&ScheduledStep { draft_tokens: graph.draft_tokens, kind: graph.kind,
                batch_capacity: graph.batch_capacity, token_capacity: graph.token_capacity,
                packed_token_capacity: graph.packed_token_capacity, mixed: None, lanes: vec![],
                requires_feedback: true }, 20_000_000);
        }
        for index in 0..8 {
            let h = ready(&mut s);
            s.request_mut(h).unwrap().max_new_tokens = 1 + (index % 6 + 1) as u32;
            s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast {
                request: h, prefixes: survival(&[0.5 + (index % 5) as f32 * 0.1; 7]), observed_at: [1; 7],
            });
        }
        std::hint::black_box(s.plan());
        let start = std::time::Instant::now();
        for _ in 0..100 { std::hint::black_box(s.plan()); }
        eprintln!("active=8 D5 W6 P32 packed={packed} graphs={} packed_shapes={} us/full_plan={:.3}",
            graphs.len(), s.costs.packed_round_shapes().count(), start.elapsed().as_secs_f64() * 1e6 / 100.);
    }
}

#[test]
fn packed_duration_envelopes_require_a_captured_batch_capacity() {
    let split = matched_c32_planner(false);
    let packed = matched_c32_planner(true);
    assert!(split.costs.duration(GraphKind::Verification, 1, 2).is_some());
    assert!(packed.costs.packed_round_duration(1, 2).is_some());
    for batch in [0, 3, 6, 65] {
        assert_eq!(split.costs.duration(GraphKind::Verification, batch, 2), None);
        assert_eq!(packed.costs.packed_round_duration(batch, 2), None);
    }
}
