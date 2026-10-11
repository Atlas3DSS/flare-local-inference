//! Native HTTP admission, bounded generation streams and direct graph scheduling.

mod config;
mod dashboard;
mod listener;
mod metrics;
mod observation;
pub(crate) mod output;
mod qualification;
mod request;
pub use request::{PreparedRequest, prepare as prepare_request};
mod runtime;
mod stops;
pub use config::Bootstrap;
pub use runtime::{Generation, GenerationEvent, ModelRuntime, Prompt};

#[cfg(feature = "python")]
pub(crate) fn restore_terminal_for_exit() {
    dashboard::restore_for_exit();
}

use crate::{
    AdmissionError, Error, ExecutionError, Failure, Result, Shutdown,
    execution::ExecutionManifest,
    frontend::{Frontend, images::ImageProcessor},
    model::{ModelConfig, ModelContract},
};
use axum::{
    Extension, Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub struct ServerConfig {
    pub bind: SocketAddr,
    pub queue_capacity: usize,
    pub event_capacity: usize,
    pub frontend_workers: usize,
    pub max_body_bytes: usize,
    pub qualification_token_trace: bool,
    /// Bearer token required by every route except `/health`; `None` admits all callers.
    pub api_key: Option<String>,
    /// Names requests may use for the model in addition to its identifier.
    pub served_names: Vec<String>,
    /// Setting that raises the per-request context, appended to context rejections.
    pub context_hint: String,
    /// Setting that raises request or page capacity, appended to capacity rejections.
    pub capacity_hint: String,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8000".parse().unwrap(),
            queue_capacity: 16,
            event_capacity: 128,
            frontend_workers: 2,
            max_body_bytes: 32 * 1024 * 1024,
            qualification_token_trace: false,
            api_key: None,
            served_names: Vec::new(),
            context_hint: String::new(),
            capacity_hint: String::new(),
        }
    }
}
const GENERATED_WINDOW: Duration = Duration::from_secs(60);
const GENERATED_SAMPLE_PERIOD: Duration = Duration::from_secs(5);
struct HttpState<M: ModelContract> {
    model: Arc<ModelRuntime<M>>,
    frontend: Arc<Frontend>,
    images: Option<Arc<ImageProcessor>>,
    grammar: Arc<crate::grammar::GrammarCompiler>,
    frontend_slots: Arc<Semaphore>,
    ingress_slots: Arc<Semaphore>,
    frontend_capacity: usize,
    ingress_capacity: usize,
    ids: AtomicU64,
    api_key: Option<String>,
    served_names: Vec<String>,
    context_hint: String,
    capacity_hint: String,
    started: Instant,
    generated: Mutex<VecDeque<(Instant, u64)>>,
}
impl<M: ModelContract> HttpState<M> {
    /// Records the generated-token counter, retaining one sample at or before the window start.
    fn sample_generated(&self) {
        let now = Instant::now();
        let tokens = self.model.metrics().snapshot(false).generated_tokens;
        let mut samples = self.generated.lock().unwrap_or_else(|e| e.into_inner());
        samples.push_back((now, tokens));
        while samples.len() > 2 && now.duration_since(samples[1].0) >= GENERATED_WINDOW {
            samples.pop_front();
        }
    }

    fn generated_last_minute(&self) -> u64 {
        let samples = self.generated.lock().unwrap_or_else(|e| e.into_inner());
        match (samples.front(), samples.back()) {
            (Some(first), Some(last)) => last.1.saturating_sub(first.1),
            _ => 0,
        }
    }

    /// Reports an admission failure, naming the setting that raises an exceeded capacity.
    fn rejection(&self, failure: Error) -> Response {
        let hint = match &failure {
            Error::Admission(AdmissionError::Context(_)) => &self.context_hint,
            Error::Admission(AdmissionError::Capacity(_)) => &self.capacity_hint,
            _ => return typed_error(failure),
        };
        if hint.is_empty() {
            return typed_error(failure);
        }
        error(status(&failure), format!("{failure}; {hint}"))
    }
}

pub struct Server<M: ModelContract> {
    address: SocketAddr,
    model: Arc<ModelRuntime<M>>,
    control: Mutex<HttpControl>,
    shutdown: Mutex<Option<Shutdown>>,
    dashboard: Mutex<Option<dashboard::Dashboard>>,
}
struct HttpControl {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<()>>>,
}

pub struct StartFailure {
    pub message: Error,
    pub drained: bool,
}
impl From<Error> for StartFailure {
    fn from(message: Error) -> Self {
        Self {
            message,
            drained: true,
        }
    }
}
impl From<&str> for StartFailure {
    fn from(message: &str) -> Self {
        Error::from(message).into()
    }
}
impl<M: ModelContract> Server<M> {
    /// # Safety
    /// The caller owns stable, exclusive prepared resources until close confirms
    /// both CUDA streams drained. No Python code runs on either native worker.
    pub unsafe fn start(
        manifest: ExecutionManifest,
        model: ModelConfig,
        frontend: Frontend,
        images: Option<ImageProcessor>,
        config: ServerConfig,
        peers: Option<Vec<std::os::unix::net::UnixStream>>,
    ) -> std::result::Result<Self, StartFailure> {
        if model.contract.vision.enabled() != images.is_some() {
            return Err("image processor must match prepared vision capabilities".into());
        }
        if config.qualification_token_trace && !config.bind.ip().is_loopback() {
            return Err("token qualification requires a loopback listener".into());
        }
        if !(1..=64).contains(&config.frontend_workers)
            || config.max_body_bytes == 0
            || config.max_body_bytes > 256 * 1024 * 1024
        {
            return Err("invalid HTTP admission limits".into());
        }
        let grammar = Arc::new(crate::grammar::GrammarCompiler::new(
            frontend.tokenizer(),
            model.contract.vocabulary,
            &model.eos_token_ids,
        )?);
        let ingress_capacity = config
            .queue_capacity
            .checked_add(model.limits.max_active)
            .ok_or("HTTP request capacity overflows")?;
        let frontend = Arc::new(frontend);
        // SAFETY: The caller grants exclusive graph/storage access and retains
        // their owners until drain. This server propagates that obligation to the
        // model worker and reports uncertain drain through StartFailure or close.
        let model = Arc::new(unsafe {
            ModelRuntime::<M>::start(
                manifest,
                model,
                config.queue_capacity,
                config.event_capacity,
                frontend.clone(),
                peers,
            )?
        });
        let state = Arc::new(HttpState {
            model: model.clone(),
            frontend,
            images: images.map(Arc::new),
            grammar,
            frontend_slots: Arc::new(Semaphore::new(config.frontend_workers)),
            ingress_slots: Arc::new(Semaphore::new(ingress_capacity)),
            frontend_capacity: config.frontend_workers,
            ingress_capacity,
            ids: AtomicU64::new(1),
            api_key: config.api_key.clone(),
            served_names: config.served_names.clone(),
            context_hint: config.context_hint.clone(),
            capacity_hint: config.capacity_hint.clone(),
            started: Instant::now(),
            generated: Mutex::new(VecDeque::new()),
        });
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let runtime_model = model.clone();
        let thread = match thread::Builder::new()
            .name("flare-http".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot create HTTP runtime: {error}"), error))))?;
                runtime.block_on(async move {
                    let shutdown_state = state.clone();
                    let sampler = state.clone();
                    tokio::spawn(async move {
                        let mut ticks = tokio::time::interval(GENERATED_SAMPLE_PERIOD);
                        loop {
                            ticks.tick().await;
                            sampler.sample_generated();
                        }
                    });
                    let mut router = Router::new()
                        .route("/metrics", get(metrics::<M>))
                        .route("/v1/models", get(models::<M>))
                        .route("/v1/chat/completions", post(chat::<M>).layer(middleware::from_fn_with_state(state.clone(), admit_http::<M>)));
                    if config.qualification_token_trace {
                        router = router.route("/__qualify/generate", post(qualification::generate::<M>)
                            .layer(middleware::from_fn_with_state(state.clone(), admit_http::<M>)));
                    }
                    let router = router
                        .route_layer(middleware::from_fn_with_state(state.clone(), authorize::<M>))
                        .route("/health", get(health::<M>))
                        .layer(DefaultBodyLimit::max(config.max_body_bytes))
                        .with_state(state);
                    let listener = match tokio::net::TcpListener::bind(config.bind).await {
                        Ok(listener) => listener,
                        Err(error) => {
                            let error = Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot bind HTTP listener: {error}"), error)));
                            let _ = ready_tx.send(Err(error.clone()));
                            return Err(error);
                        }
                    };
                    let address = listener.local_addr().map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?;
                    let listener = listener::HttpListener::new(listener);
                    let _ = ready_tx.send(Ok(address));
                    let (drain_tx, drain_rx) = oneshot::channel();
                    let mut serving = tokio::spawn(async move {
                        axum::serve(listener, router)
                            .with_graceful_shutdown(async {
                                let _ = drain_rx.await;
                            })
                            .await
                    });
                    tokio::select! {
                        result=&mut serving=>{runtime_model.request_stop();return result.map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?.map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))));},
                        _=shutdown_rx=>{}
                    }
                    shutdown_state.frontend_slots.close();
                    shutdown_state.ingress_slots.close();
                    let _ = drain_tx.send(());
                    match tokio::time::timeout(Duration::from_secs(5), &mut serving).await {
                        Ok(result) => result
                            .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?
                            .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error)))),
                        Err(_) => {
                            serving.abort();
                            let _ = serving.await;
                            Ok(())
                        }
                    }
                })
            }) {
            Ok(thread) => thread,
            Err(error) => {
                let shutdown = model.close();
                return Err(StartFailure {
                    message: Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot start HTTP thread: {error}"), error))),
                    drained: shutdown.drained,
                });
            }
        };
        match ready_rx.recv() {
            Ok(Ok(address)) => Ok(Self {
                address,
                model,
                control: Mutex::new(HttpControl {
                    stop: Some(shutdown_tx),
                    thread: Some(thread),
                }),
                shutdown: Mutex::new(None),
                dashboard: Mutex::new(None),
            }),
            result => {
                let returned = thread.join().ok().and_then(Result::err);
                let shutdown = model.close();
                Err(StartFailure {
                    message: match result {
                        Ok(Err(error)) => error,
                        Err(disconnected) => readiness_failure(
                            "HTTP thread exited before readiness",
                            disconnected,
                            returned,
                        ),
                        Ok(Ok(_)) => unreachable!("ready success handled above"),
                    },
                    drained: shutdown.drained,
                })
            }
        }
    }
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn healthy(&self) -> bool {
        self.model.healthy()
    }
    pub fn wait(&self, timeout: Duration) -> bool {
        self.model.wait(timeout)
    }
    pub fn start_dashboard(&self) -> std::io::Result<bool> {
        let _control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        let mut dashboard = self.dashboard.lock().unwrap_or_else(|e| e.into_inner());
        if dashboard.is_none() && self.healthy() {
            *dashboard = dashboard::Dashboard::start(
                self.model.metrics_handle(),
                format!("{} · {}", self.model.config().model_id, self.address),
                self.model.config().has_draft_cache(),
            )?;
        }
        Ok(dashboard.is_some())
    }
    pub fn close(&self) -> Shutdown {
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        self.dashboard
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(shutdown) = self
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return shutdown;
        }
        if let Some(stop) = control.stop.take() {
            let _ = stop.send(());
        }
        let mut shutdown = self.model.close();
        if let Some(thread) = control.thread.take() {
            match thread.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    shutdown.error.get_or_insert(error);
                }
                Err(_) => {
                    shutdown.error.get_or_insert(Error::execution(
                        ExecutionError::Worker,
                        "HTTP thread panicked",
                    ));
                }
            }
        }
        *self.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(shutdown.clone());
        shutdown
    }
}
impl<M: ModelContract> Drop for Server<M> {
    fn drop(&mut self) {
        self.close();
    }
}

async fn health<M: ModelContract>(State(state): State<Arc<HttpState<M>>>) -> Response {
    if state.model.healthy() {
        Json(json!({"status":"ok"})).into_response()
    } else {
        error(
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        )
    }
}
async fn models<M: ModelContract>(State(state): State<Arc<HttpState<M>>>) -> Response {
    let identifier = &state.model.config().model_id;
    let data: Vec<Value> = std::iter::once(identifier)
        .chain(state.served_names.iter().filter(|name| *name != identifier))
        .map(|id| json!({"id":id,"object":"model","owned_by":"local"}))
        .collect();
    Json(json!({"object":"list","data":data})).into_response()
}
async fn metrics<M: ModelContract>(State(state): State<Arc<HttpState<M>>>) -> Response {
    let mut text = state.model.metrics().render();
    for (name, help, value) in [
        (
            "http_inflight_requests",
            "HTTP requests holding admission capacity, including streamed responses.",
            state.ingress_capacity - state.ingress_slots.available_permits(),
        ),
        (
            "http_request_capacity",
            "Maximum simultaneous admitted HTTP requests.",
            state.ingress_capacity,
        ),
        (
            "frontend_active_workers",
            "CPU workers preparing tokenizer, image and grammar inputs.",
            state.frontend_capacity - state.frontend_slots.available_permits(),
        ),
        (
            "frontend_worker_capacity",
            "Maximum simultaneous CPU preparation workers.",
            state.frontend_capacity,
        ),
        (
            "uptime_seconds",
            "Seconds since the HTTP listener started.",
            state.started.elapsed().as_secs() as usize,
        ),
        (
            "generated_tokens_last_minute",
            "Committed generated tokens during the most recent minute, sampled every five seconds.",
            state.generated_last_minute() as usize,
        ),
    ] {
        metrics::metric(&mut text, name, "gauge", help, value as u64);
    }
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        text,
    )
        .into_response()
}
fn readiness_failure(
    message: &'static str,
    disconnected: std::sync::mpsc::RecvError,
    returned: Option<Error>,
) -> Error {
    let cause = match returned {
        Some(error) => Failure::caused_by(message, error),
        None => Failure::caused_by(message, disconnected),
    };
    ExecutionError::Worker(cause).into()
}

fn status(failure: &Error) -> StatusCode {
    match failure {
        Error::Admission(AdmissionError::Capacity(_) | AdmissionError::Unavailable(_)) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        Error::Admission(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn typed_error(failure: Error) -> Response {
    error(status(&failure), failure)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
}

async fn authorize<M: ModelContract>(
    State(state): State<Arc<HttpState<M>>>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(key) = &state.api_key {
        let presented = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.as_bytes().strip_prefix(b"Bearer "));
        if !presented.is_some_and(|presented| constant_time_eq(presented, key.as_bytes())) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error":{"message":"missing or invalid API key","type":"authentication_error"}})),
            )
                .into_response();
        }
    }
    next.run(request).await
}

fn error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"invalid_request_error"}})),
    )
        .into_response()
}

async fn admit_http<M: ModelContract>(
    State(state): State<Arc<HttpState<M>>>,
    mut request: Request,
    next: Next,
) -> Response {
    if !state.model.healthy() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        );
    }
    let ingress = match state.ingress_slots.clone().try_acquire_owned() {
        Ok(permit) => Arc::new(permit),
        Err(_) => {
            let mut message = format!(
                "native HTTP admission is full: {} requests are active or queued",
                state.ingress_capacity
            );
            if !state.capacity_hint.is_empty() {
                message = format!("{message}; {}", state.capacity_hint);
            }
            return error(StatusCode::TOO_MANY_REQUESTS, message);
        }
    };
    request.extensions_mut().insert(ingress);
    next.run(request).await
}

async fn chat<M: ModelContract>(
    State(state): State<Arc<HttpState<M>>>,
    Extension(ingress): Extension<Arc<OwnedSemaphorePermit>>,
    Json(mut body): Json<Value>,
) -> Response {
    let ingress_ns = (body.get("return_token_timestamps") == Some(&Value::Bool(true)))
        .then(observation::monotonic_ns);
    // Requests may use any served name; preparation sees the model's identifier.
    if let Some(model) = body.get_mut("model") {
        let served = model
            .as_str()
            .is_some_and(|name| state.served_names.iter().any(|served| served == name));
        if served {
            *model = Value::String(state.model.config().model_id.clone());
        }
    }
    let permit = match state.frontend_slots.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "native frontend is shutting down",
            );
        }
    };
    let prepare_state = state.clone();
    let preparation_ingress = ingress.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _ingress = preparation_ingress;
        request::prepare::<M>(
            &prepare_state.frontend,
            prepare_state.images.as_deref(),
            &prepare_state.grammar,
            prepare_state.model.config(),
            body,
        )
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(message)) => return state.rejection(message),
        Err(error_value) => {
            return error(StatusCode::INTERNAL_SERVER_ERROR, error_value.to_string());
        }
    };
    drop(prepared.rendered_prompt);
    let observation = prepared.observation;
    let prompt_token_ids = observation
        .token_ids
        .then(|| prepared.prompt.tokens.clone());
    let submitted = if observation.token_ids {
        state
            .model
            .generate_observed(prepared.prompt, observation.timestamps)
    } else {
        state.model.generate(prepared.prompt)
    };
    let generation = match submitted {
        Ok(generation) => generation,
        Err(message) => return state.rejection(message),
    };
    let enqueue_ns = generation.enqueue_ns;
    let id = format!("chatcmpl-{}", state.ids.fetch_add(1, Ordering::Relaxed));
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if prepared.stream {
        let stream = async_stream::stream! {
            let _ingress=ingress;
            let mut generation=generation;
            let mut presentation=output::Presentation::<M::Output>::new(&state.frontend, &state.model.config().eos_token_ids, prepared.output, id.clone());
            let mut first=chunk(&id,created,&state.model.config().model_id,json!({"role":"assistant"}),Value::Null,observation.include_usage);
            if let Some(ids)=&prompt_token_ids{first["prompt_token_ids"]=json!(ids);}
            yield Ok::<_,Infallible>(Event::default().data(first.to_string()));
            let mut terminal=false;
            while let Some(event)=generation.events.recv().await{
                match event{
                    GenerationEvent::ObservedToken{token,timestamp_ns}=>{
                        yield Ok(Event::default().data(observed_chunk(&id,created,&state.model.config().model_id,token,timestamp_ns,observation.include_usage).to_string()));
                    },
                    GenerationEvent::Token(token)=>{
                        match presentation.push(token){
                            Ok(deltas)=>for delta in deltas{yield Ok(Event::default().data(chunk(&id,created,&state.model.config().model_id,delta,Value::Null,observation.include_usage).to_string()));},
                            Err(message)=>{presentation.abort(crate::output::Termination::ExecutionError);yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));terminal=true;break;}
                        }
                    },
                    GenerationEvent::Protocol(pieces)=>{
                        match presentation.push_protocol(pieces){
                            Ok(deltas)=>for delta in deltas{yield Ok(Event::default().data(chunk(&id,created,&state.model.config().model_id,delta,Value::Null,observation.include_usage).to_string()));},
                            Err(message)=>{presentation.abort(crate::output::Termination::ExecutionError);yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));terminal=true;break;}
                        }
                    },
                    GenerationEvent::Finished{prompt_tokens,generated_tokens,reused_prompt_tokens,reason,termination}=>{
                        let deltas=presentation.finish(termination);
                        match deltas{Ok(deltas)=>for delta in deltas{yield Ok(Event::default().data(chunk(&id,created,&state.model.config().model_id,delta,Value::Null,observation.include_usage).to_string()));},Err(message)=>{presentation.abort(crate::output::Termination::ExecutionError);yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));terminal=true;break;}}
                        let reason=if termination != crate::output::Termination::UserStop && presentation.has_tools(){"tool_calls"}else{reason};
                        let mut last=chunk(&id,created,&state.model.config().model_id,json!({}),json!(reason),observation.include_usage);
                        observation.finish(&mut last,prompt_tokens,generated_tokens,reused_prompt_tokens,(ingress_ns,enqueue_ns),true);
                        yield Ok(Event::default().data(last.to_string()));
                        if observation.include_usage{
                            let mut usage=chunk(&id,created,&state.model.config().model_id,json!({}),Value::Null,true);
                            usage["choices"]=json!([]);
                            usage["usage"]=observation::usage(prompt_tokens,generated_tokens);
                            yield Ok(Event::default().data(usage.to_string()));
                        }
                        yield Ok(Event::default().data("[DONE]"));terminal=true;break;
                    },
                    GenerationEvent::AdmissionRejected(message)|GenerationEvent::Error(message)=>{presentation.abort(crate::output::Termination::ExecutionError);yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));terminal=true;break;}
                }
            }
            if !terminal{presentation.abort(crate::output::Termination::ExecutionError);yield Ok(Event::default().data(json!({"error":{"message":generation.failure()}}).to_string()));}
        };
        Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response()
    } else {
        let mut generation = generation;
        let mut presentation = output::Presentation::<M::Output>::new(
            &state.frontend,
            &state.model.config().eos_token_ids,
            prepared.output,
            id.clone(),
        );
        let mut raw_tokens = Vec::new();
        let mut token_timestamps = Vec::new();
        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::ObservedToken {
                    token,
                    timestamp_ns,
                } => {
                    raw_tokens.push(token);
                    if let Some(timestamp) = timestamp_ns {
                        token_timestamps.push(timestamp);
                    }
                }
                GenerationEvent::Token(token) => {
                    if let Err(message) = presentation.push(token) {
                        presentation.abort(crate::output::Termination::ExecutionError);
                        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
                    }
                }
                GenerationEvent::Protocol(pieces) => {
                    if let Err(message) = presentation.push_protocol(pieces) {
                        presentation.abort(crate::output::Termination::ExecutionError);
                        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
                    }
                }
                GenerationEvent::Error(message) => {
                    presentation.abort(crate::output::Termination::ExecutionError);
                    return error(StatusCode::INTERNAL_SERVER_ERROR, message);
                }
                GenerationEvent::AdmissionRejected(message) => {
                    presentation.abort(crate::output::Termination::ExecutionError);
                    return state.rejection(message);
                }
                GenerationEvent::Finished {
                    prompt_tokens,
                    generated_tokens,
                    reason,
                    reused_prompt_tokens,
                    termination,
                } => {
                    if let Err(message) = presentation.finish(termination) {
                        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
                    }
                    let reason = if termination != crate::output::Termination::UserStop
                        && presentation.has_tools()
                    {
                        "tool_calls"
                    } else {
                        reason
                    };
                    let mut response = json!({"id":id,"object":"chat.completion","created":created,"model":state.model.config().model_id,"choices":[{"index":0,"message":presentation.message(),"finish_reason":reason}],"usage":observation::usage(prompt_tokens,generated_tokens)});
                    if let Some(ids) = &prompt_token_ids {
                        response["prompt_token_ids"] = json!(ids);
                        response["choices"][0]["token_ids"] = json!(raw_tokens);
                    }
                    if observation.timestamps {
                        response["choices"][0]["token_timestamps_ns"] = json!(token_timestamps);
                    }
                    observation.finish(
                        &mut response,
                        prompt_tokens,
                        generated_tokens,
                        reused_prompt_tokens,
                        (ingress_ns, enqueue_ns),
                        false,
                    );
                    return Json(response).into_response();
                }
            }
        }
        presentation.abort(crate::output::Termination::ExecutionError);
        error(StatusCode::INTERNAL_SERVER_ERROR, generation.failure())
    }
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Value,
    include_usage: bool,
) -> Value {
    let mut value = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    if include_usage {
        value["usage"] = Value::Null;
    }
    value
}

fn observed_chunk(
    id: &str,
    created: u64,
    model: &str,
    token: u32,
    timestamp_ns: Option<u64>,
    include_usage: bool,
) -> Value {
    let mut value = chunk(id, created, model, json!({}), Value::Null, include_usage);
    value["choices"][0]["token_ids"] = json!([token]);
    if let Some(timestamp) = timestamp_ns {
        value["choices"][0]["token_timestamps_ns"] = json!([timestamp]);
    }
    value
}

#[cfg(test)]
mod observation_tests {
    use super::*;

    #[test]
    fn readiness_failure_keeps_worker_cause_with_unchanged_fallback_message() {
        use std::error::Error as _;
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let joined = std::thread::spawn(move || -> Result<()> {
            drop(tx);
            Err(ExecutionError::Worker(Failure::caused_by(
                "cannot create HTTP runtime: denied",
                std::io::Error::from_raw_os_error(13),
            ))
            .into())
        });
        let disconnected = rx.recv().unwrap_err();
        let returned = joined.join().unwrap().err();
        let error = readiness_failure(
            "HTTP thread exited before readiness",
            disconnected,
            returned,
        );
        assert!(matches!(error, Error::Execution(ExecutionError::Worker(_))));
        assert_eq!(error.to_string(), "HTTP thread exited before readiness");
        let worker = error.source().unwrap().downcast_ref::<Error>().unwrap();
        assert_eq!(
            worker
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(13)
        );
    }

    #[tokio::test]
    async fn typed_status_and_wire_messages_preserve_admission_and_execution_boundaries() {
        let cases = [
            (
                Error::admission(AdmissionError::Invalid, "invalid sampling parameters"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(
                    AdmissionError::Feature,
                    "parallel_tool_calls=false is not implemented",
                ),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Grammar, "invalid grammar"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Image, "invalid image base64"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Capacity, "native request queue is full"),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                Error::admission(AdmissionError::Context, "prompt exceeds the context"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(
                    AdmissionError::Unavailable,
                    "prepared model runtime is unavailable",
                ),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                Error::execution(ExecutionError::Poisoned, "executor failed"),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (failure, status) in cases {
            let message = failure.to_string();
            let response = typed_error(failure);
            assert_eq!(response.status(), status);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["message"], message);
        }
    }

    #[test]
    fn raw_metadata_chunks_never_duplicate_text_or_hide_eos() {
        let ordinary = chunk(
            "x",
            1,
            "model",
            json!({"content":"answer"}),
            Value::Null,
            false,
        );
        assert!(ordinary.get("usage").is_none());
        assert!(ordinary["choices"][0].get("token_ids").is_none());
        let observed = observed_chunk("x", 1, "model", 248046, Some(42), true);
        assert_eq!(observed["choices"][0]["delta"], json!({}));
        assert_eq!(observed["choices"][0]["token_ids"], json!([248046]));
        assert_eq!(observed["choices"][0]["token_timestamps_ns"], json!([42]));
        assert!(observed["choices"][0]["finish_reason"].is_null());
        assert!(observed["usage"].is_null());
        let ids_only = observed_chunk("x", 1, "model", 7, None, false);
        assert!(ids_only["choices"][0].get("token_timestamps_ns").is_none());
    }
}
