//! Serving-loop progress under KV pressure: agents with growing conversations on the
//! depth-4 pool, a host store that pins each new checkpoint until its copy-out completes,
//! and the worker's admission gate, which retries a refused request only after
//! `RequestScheduler::admission_key` changes.

use super::*;

const PAGE_BYTES: f64 = 8_912_896.0;
const ROW_BYTES: f64 = 174_215_424.0;
const SYSTEM: usize = 4230;
const HEADER: usize = 5;
const CAP: usize = 31_000;

/// The served configuration at `--mtp-draft-tokens <width - 1>`: 4 request slots, 32k context,
/// 128-token pages and prefill chunks, one draft page per request, and 32 snapshot rows of
/// 174,215,424 bytes aliased onto the top of a pool of 8,912,896-byte pages.
fn served_scheduler(pages: u32, width: usize) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [9; 32],
        SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 32_768,
            prefill_chunk_size: 128,
            pages,
            snapshots: 32,
        },
        SchedulerCapabilities {
            vocabulary: 248_320,
            page_tokens: 128,
            vision: crate::model::VisionCapabilities {
                feature_slots: 4,
                features_per_request: 16384,
                max_images: 4,
                patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16384],
            },
            draft_pages_per_request: 1,
            completion_inputs: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    s.measured_graph_durations(&super::tests::duration_graphs(4))
        .unwrap();
    s.enable_speculation(&(1..=width).collect::<Vec<_>>()).unwrap();
    s.fixed_speculative_width(Some(width)).unwrap();
    let row = ROW_BYTES / PAGE_BYTES;
    let aliases: Vec<_> = (0..32)
        .map(|i| {
            let high = pages as f64 - i as f64 * row;
            (high - row).floor() as u32..(high.ceil() as u32).min(pages)
        })
        .collect();
    s.share_snapshot_pages(&aliases).unwrap();
    s.enable_publications();
    s
}

/// Executes one planned step to completion with every lane succeeding.
fn run_step(s: &mut RequestScheduler) -> Option<ScheduledPlan> {
    let step = s.plan()?;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let r = s.request(lane.request).unwrap();
        let end = lane.start_position + lane.tokens as u64;
        let emitted = match step.kind {
            GraphKind::Prefill => usize::from(end == r.prompt_tokens as u64),
            GraphKind::Decode | GraphKind::Resume => 1,
            GraphKind::Speculative => step
                .token_capacity
                .min((r.max_new_tokens - r.generated_tokens) as usize),
            _ => 0,
        };
        let consumed = if step.kind == GraphKind::Speculative {
            r.consumed_tokens + emitted as u64
        } else {
            end
        };
        let generated = r.generated_tokens + emitted as u32;
        let token = if emitted > 0 { 900 } else { -1 };
        let mut record = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: token,
            status: if emitted > 0 && generated == r.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
        if step.kind == GraphKind::Speculative {
            let mut vector = crate::execution::TokenVector {
                emitted_count: emitted as i64,
                offered_drafts: (step.token_capacity - 1) as i64,
                accepted_drafts: emitted.min(step.token_capacity - 1) as i64,
                verification_width: step.token_capacity as i64,
                draft_cache_consumed: consumed as i64 - 1,
                ..Default::default()
            };
            vector.tokens[..emitted].fill(token);
            record.vector = Some(vector);
        }
        records[lane.request.slot as usize] = record;
    }
    s.complete(fence, &records, Default::default()).unwrap();
    Some(step)
}

/// A request waiting for admission, as the worker keeps it.
struct Waiting {
    agent: usize,
    spec: RequestSpec,
    blocked: Option<AdmissionKey>,
    pinned: Option<CheckpointId>,
}

/// The serving loop of `server::runtime::Worker` reduced to its scheduler calls: admission
/// with the capacity gate (parked readmissions first, then the queue in arrival order),
/// one step completed per pass, finishing and parking preempted requests, and a serial
/// host-store copy-out that pins each new checkpoint until its copy completes.
struct Worker {
    s: RequestScheduler,
    queue: VecDeque<Waiting>,
    parked: VecDeque<Waiting>,
    active: [Option<(usize, RequestHandle)>; 4],
    copies: VecDeque<(f64, CheckpointId)>,
    copy_free: f64,
    copy_seconds_per_page: f64,
    now: f64,
    refusals: usize,
    idle_refusals: usize,
}

enum Pass {
    Progress(Vec<(usize, usize)>),
    /// Nothing runs, no copy is in flight, and every waiting request is blocked.
    Stalled,
    Idle,
}

impl Worker {
    fn new(s: RequestScheduler, copy_ms_per_page: f64) -> Self {
        Worker {
            s,
            queue: VecDeque::new(),
            parked: VecDeque::new(),
            active: [None; 4],
            copies: VecDeque::new(),
            copy_free: 0.0,
            copy_seconds_per_page: copy_ms_per_page / 1000.0,
            now: 0.0,
            refusals: 0,
            idle_refusals: 0,
        }
    }

    fn submit(&mut self, agent: usize, spec: RequestSpec) {
        self.queue.push_back(Waiting { agent, spec, blocked: None, pinned: None });
    }

    fn waiting(&self) -> usize {
        self.queue.len() + self.parked.len()
    }

    fn running(&self) -> usize {
        self.active.iter().flatten().count()
    }

    /// One pass of the loop; returns the (agent, generated tokens) of finished requests.
    fn pass(&mut self) -> Pass {
        let s = &mut self.s;
        let mut progress = false;
        while s.has_available_slot() {
            let key = s.admission_key();
            let (mut waiting, position, parked) =
                match self.parked.iter().position(|w| w.blocked != Some(key)) {
                    Some(i) => (self.parked.remove(i).unwrap(), i, true),
                    None => match self.queue.iter().position(|w| w.blocked != Some(key)) {
                        Some(i) => (self.queue.remove(i).unwrap(), i, false),
                        None => break,
                    },
                };
            match s.admit(waiting.spec.clone()) {
                Ok(handle) => {
                    if let Some(id) = waiting.pinned.take() {
                        s.release_readmission(id).unwrap();
                    }
                    self.active[handle.slot as usize] = Some((waiting.agent, handle));
                    progress = true;
                }
                Err(crate::Error::Admission(crate::AdmissionError::Capacity(_))) => {
                    self.refusals += 1;
                    let idle = self.active.iter().all(Option::is_none);
                    self.idle_refusals += usize::from(idle);
                    waiting.blocked = Some(key);
                    if parked {
                        self.parked.insert(position, waiting);
                    } else {
                        self.queue.insert(position, waiting);
                    }
                    if idle && self.parked.iter().any(|w| w.pinned.is_some()) {
                        for w in &mut self.parked {
                            if let Some(id) = w.pinned.take() {
                                s.release_readmission(id).unwrap();
                            }
                            w.blocked = None;
                        }
                        progress = true;
                    }
                }
                Err(error) => panic!("unexpected admission error: {error}"),
            }
        }
        if let Some(step) = run_step(s) {
            progress = true;
            self.now += match step.kind {
                GraphKind::Prefill => 0.030,
                GraphKind::Speculative | GraphKind::Decode => 0.050 + 0.008 * step.lanes.len() as f64,
                _ => 0.004,
            };
        }
        let mut finished = Vec::new();
        for slot in 0..4 {
            let Some((agent, handle)) = self.active[slot] else { continue };
            let live = s.request(handle).unwrap();
            if live.phase != RequestPhase::Finishing || live.last_use() > s.completed() {
                continue;
            }
            progress = true;
            if live.preempted() {
                // The runtime readmits the history with the remaining output limit.
                let spec = RequestSpec {
                    message_boundaries: live.message_boundaries.clone(),
                    prompt: live.tokens().to_vec(),
                    conditioning: vec![],
                    max_new_tokens: (live.max_new_tokens - live.generated_tokens).max(1),
                    grammar_constrained: false,
                    host_stops: false,
                    growable_output: true,
                };
                let prefix = live.prefix();
                let pinned = s
                    .finish(handle)
                    .unwrap()
                    .filter(|id| Some(*id) != prefix)
                    .filter(|id| s.holds_readmission(*id));
                self.parked.push_back(Waiting { agent, spec, blocked: None, pinned });
            } else {
                finished.push((agent, live.generated_tokens as usize));
                s.finish(handle).unwrap();
            }
            self.active[slot] = None;
        }
        for publication in s.take_publications() {
            progress = true;
            if s.pin_checkpoint(publication.id).is_err() {
                continue;
            }
            self.copy_free = self.copy_free.max(self.now)
                + 0.060
                + self.copy_seconds_per_page * publication.tail_pages as f64;
            self.copies.push_back((self.copy_free, publication.id));
        }
        while self.copies.front().is_some_and(|(at, _)| *at <= self.now) {
            let (_, id) = self.copies.pop_front().unwrap();
            let _ = s.unpin_checkpoint(id);
            progress = true;
        }
        if progress {
            return Pass::Progress(finished);
        }
        if let Some(&(at, _)) = self.copies.front() {
            // The executor is idle until the next copy-out completes.
            self.now = self.now.max(at);
            return Pass::Progress(finished);
        }
        if self.waiting() > 0 { Pass::Stalled } else { Pass::Idle }
    }
}

/// A conversation replayed like the agent traces: one shared system message, a long first
/// user message, then per turn a re-rendered assistant message and a tool result, each a
/// message endpoint, plus a generation header the next turn renders differently.
struct Agent {
    id: u32,
    rng: u64,
    /// Rendered history through the last message endpoint, without the header.
    history: Vec<u32>,
    boundaries: Vec<usize>,
    turns: usize,
}

impl Agent {
    fn new(id: u32, seed: u64) -> Self {
        let mut agent = Agent {
            id,
            rng: seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(id as u64 + 1),
            history: (0..SYSTEM as u32).map(|i| 1000 + i).collect(),
            boundaries: vec![SYSTEM],
            turns: 0,
        };
        let user = 3000 + agent.draw(600);
        agent.append(user);
        agent
    }
    fn draw(&mut self, bound: usize) -> usize {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng >> 33) % bound as u64) as usize
    }
    fn append(&mut self, tokens: usize) {
        // Agent-specific content; ids depend on the turn so re-rendered messages differ.
        let base = 20_000 + self.id * 40_000 + (self.turns as u32 % 7) * 5_000;
        let start = self.history.len() as u32;
        self.history
            .extend((0..tokens as u32).map(|i| base + (start + i) % 5_000));
        self.boundaries.push(self.history.len());
    }
    fn request(&mut self) -> RequestSpec {
        let mut prompt = self.history.clone();
        prompt.extend(std::iter::repeat_n(7u32, HEADER));
        RequestSpec {
            message_boundaries: self.boundaries.clone(),
            prompt,
            conditioning: vec![],
            max_new_tokens: 40 + self.draw(500) as u32,
            grammar_constrained: false,
            host_stops: false,
            growable_output: true,
        }
    }
    /// Appends the re-rendered assistant reply and the next tool result; above the context
    /// cap the window shifts back to the first two messages plus the newest ones.
    fn advance(&mut self, generated: usize) {
        self.turns += 1;
        self.append(generated.max(8) * 3 / 2);
        let tool = if self.draw(4) == 0 {
            1000 + self.draw(6000)
        } else {
            30 + self.draw(700)
        };
        self.append(tool);
        if self.history.len() + 1024 > CAP {
            let head = self.boundaries[1];
            let keep = self.history.len() - CAP / 2;
            let cut = self.boundaries.iter().copied().find(|&b| b >= keep).unwrap();
            let mut history = self.history[..head].to_vec();
            history.extend_from_slice(&self.history[cut..]);
            let mut boundaries = vec![SYSTEM, head];
            boundaries.extend(
                self.boundaries
                    .iter()
                    .filter(|&&b| b > cut)
                    .map(|&b| b - cut + head),
            );
            self.history = history;
            self.boundaries = boundaries;
        }
    }
}

#[derive(Debug)]
struct Outcome {
    turns: usize,
    stalled: bool,
    waiting: usize,
    running: usize,
    free_pages: usize,
    cached_pages: usize,
    refusals: usize,
    idle_refusals: usize,
    seconds: f64,
}

/// Closed-loop replay: each agent sends its next turn when the previous one finishes, until
/// every agent completed `turns` turns or the loop stalls.
fn replay(pages: u32, width: usize, agents: usize, turns: usize, seed: u64, copy_ms_per_page: f64) -> Outcome {
    let mut worker = Worker::new(served_scheduler(pages, width), copy_ms_per_page);
    let mut agents: Vec<_> = (0..agents as u32).map(|i| Agent::new(i, seed)).collect();
    for (index, agent) in agents.iter_mut().enumerate() {
        worker.submit(index, agent.request());
    }
    let mut completed = 0;
    let stalled = loop {
        match worker.pass() {
            Pass::Progress(finished) => {
                for (agent, generated) in finished {
                    completed += 1;
                    agents[agent].advance(generated);
                    if agents[agent].turns < turns {
                        let spec = agents[agent].request();
                        worker.submit(agent, spec);
                    }
                }
            }
            Pass::Stalled => break true,
            Pass::Idle => break false,
        }
    };
    let stats = worker.s.stats();
    Outcome {
        turns: completed,
        stalled,
        waiting: worker.waiting(),
        running: worker.running(),
        free_pages: stats.free_pages,
        cached_pages: stats.cached_pages,
        refusals: worker.refusals,
        idle_refusals: worker.idle_refusals,
        seconds: worker.now,
    }
}

/// A long conversation sharing the system message: system and history endpoints, header.
fn long_request(tag: u32, prompt: usize) -> RequestSpec {
    let mut tokens: Vec<u32> = (0..SYSTEM as u32).map(|i| 1000 + i).collect();
    tokens.extend((SYSTEM..prompt).map(|i| 20_000 + tag * 40_000 + (i % 5_000) as u32));
    RequestSpec {
        message_boundaries: vec![SYSTEM, prompt - HEADER],
        prompt: tokens,
        conditioning: vec![],
        max_new_tokens: 16,
        grammar_constrained: false,
        host_stops: false,
        growable_output: true,
    }
}

#[test]
fn idle_admission_retries_once_host_copies_release_their_checkpoints() {
    // Copy-outs outlast the requests, as when the host store is writing through to disk.
    let mut w = Worker::new(served_scheduler(760, 5), 1000.0);
    for tag in 0..2 {
        w.submit(tag, long_request(tag as u32, 31_000));
        while w.waiting() + w.running() > 0 {
            assert!(matches!(w.pass(), Pass::Progress(_)));
        }
    }
    assert_eq!(w.refusals, 0);
    assert!(w.copies.len() >= 6);
    // Both conversations' chains are still pinned for copy-out, so neither 31k request fits
    // beside them, and nothing runs whose completion would change the admission key.
    for tag in 2..4 {
        w.submit(tag, long_request(tag as u32, 31_000));
    }
    let mut finished = 0;
    for _ in 0..20_000 {
        match w.pass() {
            Pass::Progress(done) => finished += done.len(),
            Pass::Stalled => panic!(
                "no progress: {} waiting, {} free pages, {} cached pages, {} idle refusals",
                w.waiting(),
                w.s.stats().free_pages,
                w.s.stats().cached_pages,
                w.idle_refusals
            ),
            Pass::Idle => break,
        }
    }
    assert_eq!(finished, 2);
    assert!(w.idle_refusals >= 2);
}

#[test]
fn four_agents_on_the_depth4_pool_keep_making_progress() {
    // Seed 4 reaches a state where all four agents wait with nothing running after 59 turns
    // when a released copy-out pin does not reopen admission; the depth-3 pool is the control.
    for (pages, width) in [(760, 5), (840, 4)] {
        let outcome = replay(pages, width, 4, 20, 4, 8.0);
        assert!(!outcome.stalled, "{pages} pages, width {width}: {outcome:?}");
        assert_eq!(outcome.turns, 80, "{outcome:?}");
    }
}

#[test]
#[ignore = "exploration: prints replay outcomes across seeds, pools and copy-out speeds"]
fn explore_replay_outcomes() {
    for (pages, width) in [(760u32, 5usize), (840, 4)] {
        for copy in [1.0, 3.0, 8.0] {
            let mut stalls = 0;
            for seed in 0..12u64 {
                let o = replay(pages, width, 4, 45, seed, copy);
                stalls += usize::from(o.stalled);
                eprintln!(
                    "pages {pages} width {width} copy {copy}ms seed {seed}: turns {} stalled {} \
                     waiting {} running {} free {} cached {} refusals {} idle refusals {} {:.0} s",
                    o.turns, o.stalled, o.waiting, o.running, o.free_pages, o.cached_pages,
                    o.refusals, o.idle_refusals, o.seconds
                );
            }
            eprintln!("== pages {pages} width {width} copy {copy}: {stalls}/12 stalled");
        }
    }
}
