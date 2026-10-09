use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

use tokio::sync::mpsc as async_mpsc;

use crate::{
    Result, Shutdown,
    cache::ConditioningEvent,
    execution::{ExecutionManifest, GraphExecutor, GraphKind, SamplingMode},
    model::{ModelConfig, ModelContract, SamplingParameters},
    scheduler::{RequestHandle, RequestPhase, RequestScheduler, RequestSpec, ScheduledStep},
};

pub struct Prompt<M: ModelContract> {
    pub tokens: Vec<u32>,
    pub data: M::PromptData,
    pub images: Option<crate::frontend::images::ProcessedImages>,
    pub conditioning: Vec<ConditioningEvent>,
    pub sampling: SamplingParameters,
    pub max_new_tokens: u32,
    pub stop_sequences: Vec<Vec<u32>>,
    pub stop_strings: Vec<String>,
    pub grammar: Option<crate::grammar::RequestGrammar>,
}

impl<M: ModelContract> Prompt<M> {
    fn has_host_stops(&self) -> bool {
        !self.stop_sequences.is_empty() || !self.stop_strings.is_empty()
    }
}

#[derive(Clone, Debug)]
pub enum GenerationEvent {
    Token(u32),
    Protocol(Vec<crate::output::Piece>),
    ObservedToken {
        token: u32,
        timestamp_ns: Option<u64>,
    },
    Finished {
        prompt_tokens: usize,
        generated_tokens: u32,
        reused_prompt_tokens: u64,
        reason: &'static str,
        termination: crate::output::Termination,
    },
    AdmissionRejected(String),
    Error(String),
}

pub struct Generation {
    pub events: async_mpsc::Receiver<GenerationEvent>,
    cancelled: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    pub(super) enqueue_ns: Option<u64>,
}

impl Generation {
    pub fn failure(&self) -> String {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap_or_else(|| "model worker closed without a terminal completion".into())
    }
}
impl Drop for Generation {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

struct Incoming<M: ModelContract> {
    prompt: Prompt<M>,
    events: async_mpsc::Sender<GenerationEvent>,
    cancelled: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    metrics: Arc<super::metrics::Metrics>,
    qualification: bool,
    observation: Option<bool>,
}
impl<M: ModelContract> Incoming<M> {
    fn fail(&self, error: String) {
        self.finish_error(error, GenerationEvent::Error);
    }
    fn reject_admission(&self, error: String) {
        self.finish_error(error, GenerationEvent::AdmissionRejected);
    }
    fn finish_error(&self, error: String, event: fn(String) -> GenerationEvent) {
        let mut failure = self.failure.lock().unwrap_or_else(|e| e.into_inner());
        if failure.is_some() {
            return;
        }
        *failure = Some(error.clone());
        self.metrics.failed.fetch_add(1, Ordering::Relaxed);
        let _ = self.events.try_send(event(error));
    }
    fn emit_token(&self, token: u32) {
        self.emit(GenerationEvent::Token(token));
    }
    fn observe_token(&self, token: u32) {
        if let Some(timestamps) = self.observation {
            let timestamp_ns = timestamps.then(super::observation::monotonic_ns);
            self.emit(GenerationEvent::ObservedToken {
                token,
                timestamp_ns,
            });
        }
    }
    fn emit(&self, event: GenerationEvent) {
        if self.events.try_send(event).is_err() {
            self.fail("generation consumer exceeded the bounded output queue".into());
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

struct Active<'a, M: ModelContract> {
    request: RequestHandle,
    incoming: Incoming<M>,
    last_token: Option<u32>,
    next_image: usize,
    stops: super::stops::TokenStops,
    text_stops: Option<super::stops::TextStops<'a>>,
}

impl<M: ModelContract> Active<'_, M> {
    fn emit_output(&mut self, token: u32, eos: &[u32]) -> Result<bool> {
        if let Some(stops) = &mut self.text_stops {
            if eos.contains(&token) {
                return Ok(false);
            }
            let (pieces, matched) = stops.push(token)?;
            if !pieces.is_empty() {
                self.incoming.emit(GenerationEvent::Protocol(pieces));
            }
            Ok(matched)
        } else {
            let (visible, matched) = self.stops.push(token);
            for token in visible {
                self.incoming.emit_token(token);
            }
            Ok(matched)
        }
    }
}

pub struct ModelRuntime<M: ModelContract> {
    sender: mpsc::SyncSender<Incoming<M>>,
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    config: Arc<ModelConfig>,
    event_capacity: usize,
    thread: Mutex<Option<JoinHandle<Shutdown>>>,
    shutdown: Mutex<Option<Shutdown>>,
    finished: Arc<(Mutex<bool>, Condvar)>,
    metrics: Arc<super::metrics::Metrics>,
}

impl<M: ModelContract> ModelRuntime<M> {
    /// # Safety
    /// The caller retains every prepared graph/context/allocation until close
    /// confirms drain, and grants this runtime exclusive graph mutation access.
    pub unsafe fn start(
        manifest: ExecutionManifest,
        config: ModelConfig,
        queue_capacity: usize,
        event_capacity: usize,
        frontend: Arc<crate::frontend::Frontend>,
    ) -> Result<Self> {
        M::validate(&config, &manifest)?;
        super::observation::initialize_clock();
        if queue_capacity == 0 || queue_capacity > 1024 || !(2..=4096).contains(&event_capacity) {
            return Err("invalid native request/event queue capacity".into());
        }
        let config = Arc::new(config);
        let (sender, receiver) = mpsc::sync_channel(queue_capacity);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker_health = healthy.clone();
        let worker_config = config.clone();
        let finished = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_finished = finished.clone();
        let metrics = Arc::new(super::metrics::Metrics::default());
        let worker_metrics = metrics.clone();
        let thread = thread::Builder::new()
            .name("sparkgap-model".into())
            .spawn(move || {
                // SAFETY: start's caller retains all prepared resources through
                // confirmed drain. This thread exclusively owns the executor;
                // construction binds both streams to the manifest's context.
                let mut executor = match unsafe { GraphExecutor::new(manifest) } {
                    Ok(executor) => executor,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.clone()));
                        return Shutdown {
                            drained: true,
                            error: Some(error),
                        };
                    }
                };
                let mut scheduler = match RequestScheduler::new(
                    worker_config.identity,
                    worker_config.limits,
                    M::scheduler_capabilities(&worker_config),
                ) {
                    Ok(scheduler) => scheduler,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.clone()));
                        let mut shutdown = executor.close();
                        shutdown.error = Some(error);
                        return shutdown;
                    }
                };
                if let Err(error) = scheduler
                    .enable_speculation(&worker_config.speculative_widths)
                    .and_then(|()| {
                        scheduler.proposal_source(worker_config.speculative_proposal_source)
                    })
                    .and_then(|()| {
                        scheduler.fixed_speculative_width(worker_config.speculative_fixed_width)
                    })
                {
                    let _ = ready_tx.send(Err(error.clone()));
                    let mut shutdown = executor.close();
                    shutdown.error = Some(error);
                    return shutdown;
                }
                let mut active: [Option<Active<M>>; 4] = std::array::from_fn(|_| None);
                let mut pending = VecDeque::with_capacity(queue_capacity);
                worker_metrics.scheduler(scheduler.stats());
                worker_health.store(true, Ordering::Release);
                let _ = ready_tx.send(Ok(()));
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_loop(
                        &worker_config,
                        &frontend,
                        &mut executor,
                        &mut scheduler,
                        &receiver,
                        &worker_stop,
                        &mut active,
                        &mut pending,
                        queue_capacity,
                        &worker_metrics,
                    )
                }));
                worker_health.store(false, Ordering::Release);
                let error = match run {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error),
                    Err(_) => Some("native model worker panicked".into()),
                };
                let final_error = error
                    .clone()
                    .unwrap_or_else(|| "model runtime stopped".into());
                for request in active.iter().flatten() {
                    request.incoming.fail(final_error.clone());
                }
                for request in &pending {
                    request.fail(final_error.clone());
                }
                while let Ok(request) = receiver.try_recv() {
                    request.fail(final_error.clone());
                }
                let mut shutdown = executor.close();
                if shutdown.drained {
                    let _ = scheduler.release_after_shutdown(&shutdown);
                    worker_metrics.scheduler(scheduler.stats());
                }
                worker_metrics.pending.store(0, Ordering::Relaxed);
                if shutdown.error.is_none() {
                    shutdown.error = error;
                }
                *worker_finished.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
                worker_finished.1.notify_all();
                shutdown
            })
            .map_err(|error| format!("cannot start native model thread: {error}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender,
                stop,
                healthy,
                config,
                event_capacity,
                thread: Mutex::new(Some(thread)),
                shutdown: Mutex::new(None),
                finished,
                metrics,
            }),
            result => {
                let _ = thread.join();
                Err(match result {
                    Ok(Err(error)) => error,
                    _ => "model worker exited before readiness".into(),
                })
            }
        }
    }

    pub fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && !self.stop.load(Ordering::Acquire)
    }
    pub(super) fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub(super) fn metrics(&self) -> &super::metrics::Metrics {
        &self.metrics
    }
    pub fn wait(&self, timeout: std::time::Duration) -> bool {
        let finished = self.finished.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .finished
            .1
            .wait_timeout_while(finished, timeout, |finished| !*finished);
        self.healthy()
    }
    pub fn generate(&self, prompt: Prompt<M>) -> Result<Generation> {
        self.enqueue(prompt, false, None)
    }

    pub(super) fn generate_observed(
        &self,
        prompt: Prompt<M>,
        timestamps: bool,
    ) -> Result<Generation> {
        self.enqueue(prompt, false, Some(timestamps))
    }

    pub(super) fn generate_qualification(&self, prompt: Prompt<M>) -> Result<Generation> {
        super::qualification::validate_prompt(&prompt)?;
        self.enqueue(prompt, true, None)
    }

    fn enqueue(
        &self,
        prompt: Prompt<M>,
        qualification: bool,
        observation: Option<bool>,
    ) -> Result<Generation> {
        if !self.healthy() {
            return Err("prepared model runtime is unavailable".into());
        }
        prompt.sampling.validate()?;
        super::stops::TokenStops::new(
            prompt.stop_sequences.clone(),
            self.config.contract.vocabulary,
        )?;
        super::stops::validate_strings(
            &prompt.stop_strings,
            !prompt.stop_sequences.is_empty(),
            prompt.grammar.is_some(),
        )?;
        M::validate_prompt(&self.config, &prompt)?;
        let (events, receiver) = async_mpsc::channel(self.event_capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let incoming = Incoming {
            prompt,
            events,
            cancelled: cancelled.clone(),
            failure: failure.clone(),
            metrics: self.metrics.clone(),
            qualification,
            observation,
        };
        let enqueue_ns = observation
            .filter(|timestamps| *timestamps)
            .map(|_| super::observation::monotonic_ns());
        self.sender.try_send(incoming).map_err(|error| {
            match error {
                mpsc::TrySendError::Full(_) => "native request queue is full",
                mpsc::TrySendError::Disconnected(_) => "native model worker disconnected",
            }
            .to_owned()
        })?;
        Ok(Generation {
            events: receiver,
            cancelled,
            failure,
            enqueue_ns,
        })
    }
    pub fn close(&self) -> Shutdown {
        // Join serialization precedes the cached result lock; request callbacks never use either.
        let mut thread = self.thread.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(shutdown) = self
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return shutdown;
        }
        self.stop.store(true, Ordering::Release);
        let shutdown = thread
            .take()
            .map(|thread| {
                thread.join().unwrap_or_else(|_| Shutdown {
                    drained: false,
                    error: Some("model thread panicked outside its drain boundary".into()),
                })
            })
            .unwrap_or(Shutdown {
                drained: false,
                error: Some("missing model worker shutdown state".into()),
            });
        *self.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(shutdown.clone());
        shutdown
    }
}
impl<M: ModelContract> Drop for ModelRuntime<M> {
    fn drop(&mut self) {
        self.close();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_loop<'a, M: ModelContract>(
    config: &ModelConfig,
    frontend: &'a crate::frontend::Frontend,
    executor: &mut GraphExecutor,
    scheduler: &mut RequestScheduler,
    receiver: &mpsc::Receiver<Incoming<M>>,
    stop: &AtomicBool,
    active: &mut [Option<Active<'a, M>>; 4],
    pending: &mut VecDeque<Incoming<M>>,
    queue_capacity: usize,
    metrics: &super::metrics::Metrics,
) -> Result<()> {
    let mut admissions: [Option<M::Admission>; 4] = std::array::from_fn(|_| None);
    let mut inputs = M::Inputs::default();
    while !stop.load(Ordering::Acquire) {
        let mut progress = false;
        pending.retain(|request| {
            let cancelled = request.cancelled.load(Ordering::Acquire);
            if cancelled {
                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                progress = true;
            }
            !cancelled
        });
        while pending.len() < queue_capacity {
            match receiver.try_recv() {
                Ok(request) => {
                    pending.push_back(request);
                    progress = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }
        for slot in 0..4 {
            if active[slot]
                .as_ref()
                .is_some_and(|a| a.incoming.cancelled.load(Ordering::Acquire))
            {
                scheduler.cancel(active[slot].take().unwrap().request)?;
                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                admissions[slot] = None;
                progress = true;
            }
        }
        while scheduler.has_available_slot() {
            let Some(incoming) = pending.pop_front() else {
                break;
            };
            if incoming.cancelled.load(Ordering::Acquire) {
                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let spec = RequestSpec {
                prompt: incoming.prompt.tokens.clone(),
                conditioning: incoming.prompt.conditioning.clone(),
                max_new_tokens: incoming.prompt.max_new_tokens,
                grammar_constrained: incoming.prompt.grammar.is_some(),
                host_stops: incoming.prompt.has_host_stops(),
            };
            let admission = if incoming.qualification {
                scheduler.admit_cold(spec)
            } else {
                scheduler.admit(spec)
            };
            let handle = match admission {
                Ok(handle) => handle,
                Err(error) => {
                    incoming.reject_admission(error);
                    continue;
                }
            };
            let slot = handle.slot as usize;
            metrics.admitted.fetch_add(1, Ordering::Relaxed);
            if let Some(images) = &incoming.prompt.images {
                if let Err(error) = scheduler.set_vision_images(
                    handle,
                    &images
                        .geometry
                        .iter()
                        .map(|geometry| geometry.patches)
                        .collect::<Vec<_>>(),
                ) {
                    scheduler.cancel(handle)?;
                    incoming.fail(error);
                    continue;
                }
            }
            admissions[slot] = Some(M::admission(
                config,
                handle,
                &incoming.prompt,
                incoming.qualification,
            )?);
            active[slot] = Some(Active {
                request: handle,
                stops: super::stops::TokenStops::new(
                    incoming.prompt.stop_sequences.clone(),
                    config.contract.vocabulary,
                )?,
                text_stops: if incoming.prompt.stop_strings.is_empty() {
                    None
                } else {
                    Some(super::stops::TextStops::new(
                        frontend,
                        incoming.prompt.stop_strings.clone(),
                    )?)
                },
                incoming,
                last_token: None,
                next_image: 0,
            });
            progress = true;
        }
        while let Some(completion) = executor.poll()? {
            let kind = executor.manifest().graphs[completion.graph_index].kind;
            for record in &completion.records {
                if record.status != 0 {
                    M::validate_completion(config, kind, record)?;
                }
            }
            if matches!(
                kind,
                GraphKind::Prefill | GraphKind::Decode | GraphKind::Speculative
            ) {
                for request in active.iter().flatten() {
                    let record = completion.records[request.request.slot as usize];
                    if (record.status == 1 || record.status == 2)
                        && record.generation == request.request.generation as i64
                    {
                        let boundary = M::candidate_boundary(&record);
                        let prefix = scheduler
                            .request(request.request)?
                            .prefix()
                            .map(|id| {
                                scheduler
                                    .cache()
                                    .checkpoint(id)
                                    .map(|entry| entry.prefix().consumed_tokens)
                            })
                            .transpose()?
                            .unwrap_or(0);
                        if boundary > prefix {
                            scheduler.record_candidate(
                                request.request,
                                boundary,
                                completion.fence,
                            )?;
                        }
                    }
                }
            }
            let measured_step = completion
                .elapsed_ns
                .map(|_| scheduler.submitted_step(completion.fence).cloned())
                .transpose()?;
            let generated = scheduler.complete(completion.fence, &completion.records)?;
            if let (Some(step), Some(nanos)) = (measured_step, completion.elapsed_ns) {
                scheduler.observe_round(&step, nanos, generated.len())?;
                metrics.speculative_completed(
                    step.batch_capacity,
                    step.token_capacity,
                    nanos,
                    &completion.records,
                );
            }
            let mut stopped = Vec::new();
            for token in generated {
                if let Some(request) = active[token.request.slot as usize]
                    .as_mut()
                    .filter(|a| a.request == token.request)
                {
                    request.incoming.observe_token(token.token);
                    request.last_token = Some(token.token);
                    if request.incoming.qualification {
                        request.incoming.emit_token(token.token);
                        continue;
                    }
                    if let Some(grammar) = &mut request.incoming.prompt.grammar {
                        if let Err(error) = grammar.advance(token.token) {
                            request
                                .incoming
                                .fail(format!("grammar advancement failed: {error}"));
                            stopped.push(token.request);
                            continue;
                        }
                        M::grammar_updated(
                            admissions[token.request.slot as usize].as_mut().unwrap(),
                            grammar.mask(),
                        );
                    }
                    let stop = match request.emit_output(token.token, &config.eos_token_ids) {
                        Ok(matched) => matched,
                        Err(error) => {
                            request.incoming.fail(error);
                            stopped.push(token.request);
                            continue;
                        }
                    };
                    if stop {
                        metrics.completed.fetch_add(1, Ordering::Relaxed);
                        let live = scheduler.request(token.request)?;
                        if request
                            .incoming
                            .events
                            .try_send(GenerationEvent::Finished {
                                prompt_tokens: live.prompt_tokens,
                                generated_tokens: live.generated_tokens,
                                reused_prompt_tokens: live.reused_prompt_tokens(),
                                reason: "stop",
                                termination: crate::output::Termination::UserStop,
                            })
                            .is_err()
                        {
                            request.incoming.fail(
                                "generation consumer exceeded the bounded output queue".into(),
                            );
                        }
                        stopped.push(token.request);
                    }
                }
            }
            for handle in stopped {
                scheduler.cancel(handle)?;
                active[handle.slot as usize] = None;
                admissions[handle.slot as usize] = None;
            }
            progress = true;
        }
        for slot in 0..4 {
            if let Some(request) = &mut active[slot] {
                let live = scheduler.request(request.request)?;
                if live.phase == RequestPhase::Finishing && live.last_use() <= scheduler.completed()
                {
                    if let Some(error) = live.error_code {
                        request
                            .incoming
                            .fail(format!("prepared model reported device error {error}"));
                    } else {
                        let user_stop = if let Some(stops) = &mut request.text_stops {
                            let eos = request
                                .last_token
                                .filter(|token| config.eos_token_ids.contains(token));
                            match stops.finish(eos) {
                                Ok((pieces, matched)) => {
                                    if !pieces.is_empty() {
                                        request.incoming.emit(GenerationEvent::Protocol(pieces));
                                    }
                                    matched
                                }
                                Err(error) => {
                                    request.incoming.fail(error);
                                    scheduler.cancel(request.request)?;
                                    active[slot] = None;
                                    admissions[slot] = None;
                                    progress = true;
                                    continue;
                                }
                            }
                        } else {
                            false
                        };
                        metrics.completed.fetch_add(1, Ordering::Relaxed);
                        for token in request.stops.finish() {
                            if request
                                .incoming
                                .events
                                .try_send(GenerationEvent::Token(token))
                                .is_err()
                            {
                                request.incoming.fail(
                                    "generation consumer exceeded the bounded output queue".into(),
                                );
                            }
                        }
                        let reason = if user_stop
                            || (!request.incoming.qualification
                                && request
                                    .last_token
                                    .is_some_and(|token| config.eos_token_ids.contains(&token)))
                        {
                            "stop"
                        } else {
                            "length"
                        };
                        if request
                            .incoming
                            .events
                            .try_send(GenerationEvent::Finished {
                                prompt_tokens: live.prompt_tokens,
                                generated_tokens: live.generated_tokens,
                                reused_prompt_tokens: live.reused_prompt_tokens(),
                                reason,
                                termination: if user_stop {
                                    crate::output::Termination::UserStop
                                } else if reason == "stop" {
                                    crate::output::Termination::Eos
                                } else {
                                    crate::output::Termination::OutputLimit
                                },
                            })
                            .is_err()
                        {
                            request.incoming.fail(
                                "generation consumer exceeded the bounded output queue".into(),
                            );
                        }
                        if user_stop {
                            scheduler.cancel(request.request)?;
                            active[slot] = None;
                            admissions[slot] = None;
                            progress = true;
                            continue;
                        }
                    }
                    scheduler.finish(request.request)?;
                    active[slot] = None;
                    admissions[slot] = None;
                    progress = true;
                }
            }
        }
        if config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
            && executor.outstanding() == 0
        {
            for slot in 0..active.len() {
                let Some(request) = active[slot].as_ref() else {
                    continue;
                };
                let live = scheduler.request(request.request)?;
                if live.phase != RequestPhase::Decode || live.copy_proposal().is_some() {
                    continue;
                }
                let proposal = prepare_copy_proposal(
                    live.tokens(),
                    live.consumed_tokens,
                    request.incoming.prompt.grammar.as_ref(),
                    if config.speculative_copy && !live.host_stops() {
                        7
                    } else {
                        0
                    },
                    &config.eos_token_ids,
                );
                match proposal {
                    Ok(proposal) => scheduler.set_copy_proposal(request.request, proposal)?,
                    Err(error) => {
                        request
                            .incoming
                            .fail(format!("copy grammar preparation failed: {error}"));
                        scheduler.cancel(request.request)?;
                        active[slot] = None;
                        admissions[slot] = None;
                        progress = true;
                    }
                }
            }
        }
        if let Some(step) = scheduler.next_step() {
            if executor.can_submit(step.grammar_constrained) {
                validate_sampling_admissions::<M>(&step, &admissions)?;
                let graph = M::graph(
                    config,
                    executor.manifest(),
                    step.kind,
                    step.batch_capacity,
                    step.token_capacity,
                    SamplingMode::General,
                )?;
                let fence = executor.next_fence()?;
                scheduler.record_submission(&step, fence)?;
                if step.kind == GraphKind::Vision {
                    let request = active[step.lanes[0].request.slot as usize]
                        .as_mut()
                        .ok_or("missing active vision request")?;
                    M::pack_vision(
                        &mut inputs,
                        config,
                        request.request,
                        request
                            .incoming
                            .prompt
                            .images
                            .as_ref()
                            .ok_or("missing prepared vision inputs")?,
                        request.next_image,
                    )?;
                    request.next_image += 1;
                } else {
                    M::pack(&mut inputs, config, scheduler, &step, &admissions)?;
                }
                if executor.submit(graph, &M::uploads(&inputs), step.grammar_constrained)?
                    != Some(fence)
                {
                    return Err("executor refused a reserved scheduler submission".into());
                }
                observe_submission(metrics, scheduler, &step, executor.outstanding())?;
                progress = true;
            }
        }
        if progress {
            metrics.scheduler(scheduler.stats());
            metrics
                .pending
                .store(pending.len() as u64, Ordering::Relaxed);
        }
        if !progress {
            if executor.outstanding() > 0 {
                std::hint::spin_loop();
                thread::yield_now();
            } else {
                match receiver.recv_timeout(std::time::Duration::from_millis(2)) {
                    Ok(request) => pending.push_back(request),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                }
            }
        }
    }
    Ok(())
}

fn prepare_copy_proposal(
    tokens: &[u32],
    consumed: u64,
    grammar: Option<&crate::grammar::RequestGrammar>,
    maximum: usize,
    eos_ids: &[u32],
) -> Result<crate::scheduler::speculation::CopyProposal> {
    let (mut ids, mut count) = crate::scheduler::speculation::copy_proposal(tokens, maximum);
    if let Some(end) = ids[..count]
        .iter()
        .position(|id| eos_ids.contains(&(*id as u32)))
    {
        count = end;
    }
    let grammar_masks = if let Some(grammar) = grammar {
        let tokens: Vec<_> = ids[..count].iter().map(|id| *id as u32).collect();
        let (legal, masks) = grammar.copy_masks(&tokens)?;
        count = legal;
        masks
    } else {
        Vec::new()
    };
    ids[count..].fill(-1);
    Ok(crate::scheduler::speculation::CopyProposal {
        consumed,
        tokens: ids,
        count,
        grammar_masks,
    })
}

fn validate_sampling_admissions<M: ModelContract>(
    step: &ScheduledStep,
    admissions: &[Option<M::Admission>; 4],
) -> Result<()> {
    if !step.kind.samples() {
        return Ok(());
    }
    if step.lanes.is_empty() {
        return Err("sampling submission has no request lanes".into());
    }
    for lane in &step.lanes {
        admissions
            .get(lane.request.slot as usize)
            .and_then(Option::as_ref)
            .filter(|admission| M::admission_handle(admission) == lane.request)
            .ok_or("sampling submission lacks its matching admission")?;
    }
    Ok(())
}

fn observe_submission(
    metrics: &super::metrics::Metrics,
    scheduler: &RequestScheduler,
    step: &ScheduledStep,
    outstanding: usize,
) -> Result<()> {
    metrics.launched(step.kind);
    let (mut constrained, mut unconstrained) = (false, false);
    for lane in &step.lanes {
        if scheduler.request(lane.request)?.grammar_constrained {
            constrained = true;
        } else {
            unconstrained = true;
        }
    }
    if constrained {
        metrics.constrained_submission(
            outstanding,
            matches!(step.kind, GraphKind::Decode | GraphKind::Speculative) && unconstrained,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen4exp::{PromptData, Qwen4Exp, inputs::Admission};
    use crate::scheduler::SchedulerCapabilities;

    #[test]
    fn general_sampler_handles_zero_positive_and_mixed_temperatures_with_matching_epochs() {
        use crate::scheduler::ScheduledLane;

        let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|slot| {
            Some(Admission {
                handle: RequestHandle {
                    slot: slot as u32,
                    generation: 17 + slot as u64,
                },
                sampling: SamplingParameters {
                    temperature: if slot == 0 { 1. } else { -0. },
                    top_k: 1,
                    ..Default::default()
                },
                max_new_tokens: 16,
                prompt: vec![7],
                positions: vec![[0; 3]],
                vision_token_indices: vec![-1],
                grammar_mask: None,
                ignore_eos: false,
            })
        });
        let lane = |slot: usize| ScheduledLane {
            request: RequestHandle {
                slot: slot as u32,
                generation: 17 + slot as u64,
            },
            start_position: 64,
            tokens: 1,
        };
        for kind in [
            GraphKind::Prefill,
            GraphKind::Decode,
            GraphKind::Resume,
            GraphKind::Speculative,
        ] {
            let mut step = ScheduledStep {
                kind,
                batch_capacity: 4,
                token_capacity: 1,
                lanes: vec![lane(3), lane(1)],
                grammar_constrained: true,
            };
            validate_sampling_admissions::<Qwen4Exp>(&step, &admissions).unwrap();
            let before = step.clone();
            step.lanes.push(lane(0));
            validate_sampling_admissions::<Qwen4Exp>(&step, &admissions).unwrap();
            step.lanes.remove(2);
            assert_eq!(step, before);
            step.lanes.reverse();
            validate_sampling_admissions::<Qwen4Exp>(&step, &admissions).unwrap();
            step.lanes[0].request.generation += 1;
            assert!(
                validate_sampling_admissions::<Qwen4Exp>(&step, &admissions)
                    .unwrap_err()
                    .contains("matching admission")
            );
        }
        let mut step = ScheduledStep {
            kind: GraphKind::Decode,
            batch_capacity: 1,
            token_capacity: 1,
            lanes: vec![lane(0)],
            grammar_constrained: false,
        };
        validate_sampling_admissions::<Qwen4Exp>(&step, &admissions).unwrap();
        admissions[0] = None;
        assert!(validate_sampling_admissions::<Qwen4Exp>(&step, &admissions).is_err());
        for kind in [
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Vision,
        ] {
            step.kind = kind;
            validate_sampling_admissions::<Qwen4Exp>(&step, &admissions).unwrap();
        }
    }

    #[test]
    fn copy_preparation_trims_eos_and_explicit_disable_has_no_proposals() {
        let history = [1, 2, 3, 4, 11, 12, 13, 99, 1, 2, 3, 4];
        let copy = prepare_copy_proposal(&history, 11, None, 7, &[12]).unwrap();
        assert_eq!(copy.count, 1);
        assert_eq!(copy.tokens, [11, -1, -1, -1, -1, -1, -1]);
        assert_eq!(copy.consumed, 11);
        assert!(copy.grammar_masks.is_empty());
        let disabled = prepare_copy_proposal(&history, 11, None, 0, &[12]).unwrap();
        assert_eq!((disabled.count, disabled.tokens), (0, [-1; 7]));
        let absent = prepare_copy_proposal(&[1, 2, 3, 4], 3, None, 7, &[]).unwrap();
        assert_eq!(absent.count, 0);
    }

    #[test]
    fn grammar_metrics_count_submitted_lanes_and_ignore_unconstrained_pipeline_depth() {
        use crate::{cache::Fence, execution::CompletionRecord, scheduler::SchedulerLimits};

        fn submit(
            scheduler: &mut RequestScheduler,
            metrics: &super::super::metrics::Metrics,
            sequence: u64,
            outstanding: usize,
        ) -> ScheduledStep {
            let step = scheduler.next_step().unwrap();
            let fence = Fence(sequence);
            scheduler.record_submission(&step, fence).unwrap();
            observe_submission(metrics, scheduler, &step, outstanding).unwrap();
            let mut records = [CompletionRecord::default(); 4];
            for lane in &step.lanes {
                let live = scheduler.request(lane.request).unwrap();
                let samples = matches!(step.kind, GraphKind::Prefill | GraphKind::Decode);
                records[lane.request.slot as usize] = CompletionRecord {
                    generation: lane.request.generation as i64,
                    submission_sequence: sequence as i64,
                    consumed_tokens: (lane.start_position + lane.tokens as u64) as i64,
                    generated_tokens: live.generated_tokens as i64 + i64::from(samples),
                    token_id: if samples { 17 } else { -1 },
                    status: 1,
                    ..CompletionRecord::default()
                };
            }
            scheduler.complete(fence, &records).unwrap();
            step
        }

        let metrics = super::super::metrics::Metrics::default();
        let mut scheduler = RequestScheduler::new(
            [5; 32],
            SchedulerLimits {
                max_active: 4,
                max_sequence_tokens: 64,
                prefill_chunk_size: 64,
                pages: 4,
                snapshots: 4,
            },
            SchedulerCapabilities {
                vocabulary: 248320,
                vision: crate::model::VisionCapabilities {
                    features_per_request: 16384,
                    max_images: 4,
                    patches_per_feature: 4,
                    patch_capacities: vec![256, 1024, 4096, 16384],
                },
                live_state_base: 1,
                candidate_state_base: 5,
                draft_pages_per_request: 1,
            },
        )
        .unwrap();
        let mut constrained = None;
        for grammar_constrained in [true, false, false] {
            let handle = scheduler
                .admit(RequestSpec {
                    prompt: vec![7],
                    conditioning: vec![],
                    max_new_tokens: 20,
                    grammar_constrained,
                    host_stops: false,
                })
                .unwrap();
            if grammar_constrained {
                constrained = Some(handle);
            }
        }
        assert_eq!(submit(&mut scheduler, &metrics, 1, 1).kind, GraphKind::Init);
        assert_eq!(
            submit(&mut scheduler, &metrics, 2, 1).kind,
            GraphKind::Prefill
        );
        let only_constrained = submit(&mut scheduler, &metrics, 3, 1);
        assert_eq!(only_constrained.kind, GraphKind::Decode);
        assert_eq!(only_constrained.lanes.len(), 1);
        assert_eq!(scheduler.stats().active_requests, 3);
        assert!(
            metrics
                .render()
                .contains("sparkgap_mixed_grammar_decode_steps_total 0\n")
        );

        assert_eq!(
            submit(&mut scheduler, &metrics, 4, 1).kind,
            GraphKind::Prefill
        );
        let mixed = submit(&mut scheduler, &metrics, 5, 1);
        assert_eq!(mixed.kind, GraphKind::Decode);
        assert_eq!(mixed.lanes.len(), 2);
        assert!(
            metrics
                .render()
                .contains("sparkgap_mixed_grammar_decode_steps_total 1\n")
        );
        assert_eq!(
            submit(&mut scheduler, &metrics, 6, 1).kind,
            GraphKind::Prefill
        );
        scheduler.cancel(constrained.unwrap()).unwrap();
        let unconstrained = submit(&mut scheduler, &metrics, 7, 2);
        assert_eq!(unconstrained.kind, GraphKind::Decode);
        assert_eq!(unconstrained.lanes.len(), 2);
        assert!(!unconstrained.grammar_constrained);
        let text = metrics.render();
        assert!(text.contains("sparkgap_mixed_grammar_decode_steps_total 1\n"));
        assert!(text.contains("sparkgap_constrained_max_inflight 1\n"));
        assert!(text.contains("sparkgap_graph_launches_total{kind=\"decode\"} 3\n"));
    }

    fn stream(capacity: usize) -> (Incoming<Qwen4Exp>, Generation) {
        let (events, receiver) = async_mpsc::channel(capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let incoming = Incoming {
            prompt: Prompt {
                tokens: vec![7],
                data: PromptData {
                    positions: vec![[0; 3]],
                    vision_token_indices: vec![-1],
                },
                images: None,
                conditioning: vec![],
                sampling: SamplingParameters::default(),
                max_new_tokens: 10,
                stop_sequences: vec![],
                stop_strings: vec![],
                grammar: None,
            },
            events,
            cancelled: cancelled.clone(),
            failure: failure.clone(),
            metrics: Arc::new(super::super::metrics::Metrics::default()),
            qualification: false,
            observation: None,
        };
        (
            incoming,
            Generation {
                events: receiver,
                cancelled,
                failure,
                enqueue_ns: None,
            },
        )
    }

    #[test]
    fn string_and_token_stops_both_require_the_host_stop_scheduler_barrier() {
        let (mut incoming, _) = stream(4);
        assert!(!incoming.prompt.has_host_stops());
        incoming.prompt.stop_strings.push("END".into());
        assert!(incoming.prompt.has_host_stops());
        incoming.prompt.stop_strings.clear();
        incoming.prompt.stop_sequences.push(vec![1, 2]);
        assert!(incoming.prompt.has_host_stops());
    }

    #[test]
    fn worker_string_stop_keeps_raw_final_token_and_suppresses_subtoken_suffix() {
        let frontend = super::super::stops::string_tests::frontend();
        let (mut incoming, mut generation) = stream(16);
        incoming.observation = Some(false);
        incoming.prompt.stop_strings = vec!["END".into()];
        let mut request = Active {
            request: RequestHandle {
                slot: 0,
                generation: 1,
            },
            incoming,
            last_token: None,
            next_image: 0,
            stops: super::super::stops::TokenStops::new(vec![], 248320).unwrap(),
            text_stops: Some(
                super::super::stops::TextStops::new(&frontend, vec!["END".into()]).unwrap(),
            ),
        };
        request.incoming.observe_token(0);
        assert!(!request.emit_output(0, &[248046]).unwrap());
        request.incoming.observe_token(1);
        assert!(request.emit_output(1, &[248046]).unwrap());
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 0,
                timestamp_ns: None
            })
        ));
        assert!(
            matches!(generation.events.try_recv(), Ok(GenerationEvent::Protocol(pieces)) if pieces == vec![crate::output::Piece::Token(0),crate::output::Piece::Text("before".into())])
        );
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 1,
                timestamp_ns: None
            })
        ));
        assert!(generation.events.try_recv().is_err());
        assert_eq!(
            request.text_stops.as_mut().unwrap().finish(None).unwrap(),
            (vec![], true)
        );
    }

    #[test]
    fn saturated_output_cancels_without_blocking_and_preserves_first_failure() {
        let (incoming, mut generation) = stream(2);
        incoming.emit_token(11);
        incoming.emit_token(12);
        assert!(!incoming.cancelled.load(Ordering::Acquire));
        incoming.emit_token(13);
        incoming.emit_token(14);
        incoming.fail("a later shutdown error".into());
        assert!(incoming.cancelled.load(Ordering::Acquire));
        assert_eq!(incoming.metrics.failed.load(Ordering::Relaxed), 1);
        assert_eq!(
            generation.failure(),
            "generation consumer exceeded the bounded output queue"
        );
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(11))
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(12))
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn observations_keep_raw_stop_and_eos_ids_without_changing_visible_tokens() {
        let (mut incoming, mut generation) = stream(16);
        incoming.observe_token(9);
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        incoming.observation = Some(false);
        let mut stops =
            super::super::stops::TokenStops::new(vec![vec![10, 248046]], 248320).unwrap();
        for token in [9, 10, 248046] {
            incoming.observe_token(token);
            let (visible, _) = stops.push(token);
            for value in visible {
                incoming.emit_token(value);
            }
        }
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 9,
                timestamp_ns: None
            })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(9))
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 10,
                timestamp_ns: None
            })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 248046,
                timestamp_ns: None
            })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        assert!(!incoming.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn observation_timestamps_are_monotonic_and_backpressure_remains_bounded() {
        let (mut incoming, mut generation) = stream(2);
        incoming.observation = Some(true);
        let start = super::super::observation::monotonic_ns();
        incoming.observe_token(248044);
        incoming.observe_token(17);
        incoming.observe_token(18);
        assert!(incoming.cancelled.load(Ordering::Acquire));
        let mut previous = start;
        for expected in [248044, 17] {
            let GenerationEvent::ObservedToken {
                token,
                timestamp_ns: Some(timestamp),
            } = generation.events.try_recv().unwrap()
            else {
                panic!("missing raw observation")
            };
            assert_eq!(token, expected);
            assert!(timestamp >= previous);
            previous = timestamp;
        }
        assert!(generation.failure().contains("bounded output queue"));
    }

    #[test]
    fn disconnected_output_marks_cancellation_and_retains_failure_for_the_reader() {
        let (incoming, generation) = stream(2);
        let failure = generation.failure.clone();
        drop(generation);
        assert!(incoming.cancelled.load(Ordering::Acquire));
        incoming.emit_token(11);
        assert_eq!(incoming.metrics.failed.load(Ordering::Relaxed), 1);
        assert!(
            failure
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .contains("bounded output queue")
        );
    }

    #[test]
    fn resource_rejection_is_distinct_from_execution_failure_and_retains_first_error() {
        let mut scheduler = RequestScheduler::new(
            [7; 32],
            crate::scheduler::SchedulerLimits {
                max_active: 2,
                max_sequence_tokens: 64,
                prefill_chunk_size: 64,
                pages: 1,
                snapshots: 2,
            },
            SchedulerCapabilities {
                vocabulary: 248320,
                vision: crate::model::VisionCapabilities {
                    features_per_request: 16384,
                    max_images: 4,
                    patches_per_feature: 4,
                    patch_capacities: vec![256, 1024, 4096, 16384],
                },
                live_state_base: 1,
                candidate_state_base: 5,
                draft_pages_per_request: 1,
            },
        )
        .unwrap();
        let spec = || RequestSpec {
            prompt: vec![7],
            conditioning: vec![],
            max_new_tokens: 10,
            grammar_constrained: false,
            host_stops: false,
        };
        let active = scheduler.admit(spec()).unwrap();
        let rejected = scheduler.admit(spec()).unwrap_err();
        let (incoming, mut generation) = stream(2);
        incoming.reject_admission(rejected.clone());
        incoming.fail("later runtime error".into());
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::AdmissionRejected(message)) if message == rejected
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        assert_eq!(generation.failure(), rejected);
        assert_eq!(incoming.metrics.failed.load(Ordering::Relaxed), 1);
        assert_eq!(scheduler.stats().active_requests, 1);
        assert_eq!(scheduler.free_page_count(), 0);
        assert_eq!(scheduler.free_snapshot_count(), 1);

        let (execution, mut generation) = stream(2);
        execution.fail(rejected.clone());
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Error(message)) if message == rejected
        ));
        scheduler.cancel(active).unwrap();
        assert_eq!(scheduler.free_page_count(), 1);
        assert_eq!(scheduler.free_snapshot_count(), 2);
    }
}
