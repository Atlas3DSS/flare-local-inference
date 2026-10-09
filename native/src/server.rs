//! Native HTTP admission, bounded generation streams and direct graph scheduling.

mod config;
mod listener;
mod metrics;
mod observation;
pub(crate) mod output;
mod qualification;
mod request;
pub use crate::model::OutputConfig;
pub use request::{PreparedRequest, prepare as prepare_request};
mod runtime;
mod stops;
pub use config::Bootstrap;
pub use runtime::{Generation, GenerationEvent, ModelRuntime, Prompt};

use crate::{
    Result, Shutdown,
    execution::ExecutionManifest,
    frontend::{Frontend, images::ImageProcessor},
    model::ModelConfig,
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
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub struct ServerConfig {
    pub bind: SocketAddr,
    pub queue_capacity: usize,
    pub event_capacity: usize,
    pub frontend_workers: usize,
    pub max_body_bytes: usize,
    pub qualification_token_trace: bool,
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
        }
    }
}
struct HttpState {
    model: Arc<ModelRuntime>,
    frontend: Arc<Frontend>,
    images: Arc<ImageProcessor>,
    grammar: Arc<crate::grammar::GrammarCompiler>,
    frontend_slots: Arc<Semaphore>,
    ingress_slots: Arc<Semaphore>,
    frontend_capacity: usize,
    ingress_capacity: usize,
    ids: AtomicU64,
}

pub struct Server {
    address: SocketAddr,
    model: Arc<ModelRuntime>,
    control: Mutex<HttpControl>,
    shutdown: Mutex<Option<Shutdown>>,
}
struct HttpControl {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<()>>>,
}

pub struct StartFailure {
    pub message: String,
    pub drained: bool,
}
impl From<String> for StartFailure {
    fn from(message: String) -> Self {
        Self {
            message,
            drained: true,
        }
    }
}
impl From<&str> for StartFailure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}
impl Server {
    /// # Safety
    /// The caller owns stable, exclusive prepared resources until close confirms
    /// both CUDA streams drained. No Python code runs on either native worker.
    pub unsafe fn start(
        manifest: ExecutionManifest,
        model: ModelConfig,
        frontend: Frontend,
        images: ImageProcessor,
        config: ServerConfig,
    ) -> std::result::Result<Self, StartFailure> {
        if config.qualification_token_trace && !config.bind.ip().is_loopback() {
            return Err("token qualification requires a loopback listener".into());
        }
        if !(1..=32).contains(&config.frontend_workers)
            || config.max_body_bytes == 0
            || config.max_body_bytes > 128 * 1024 * 1024
        {
            return Err("invalid HTTP admission limits".into());
        }
        let grammar = Arc::new(crate::grammar::GrammarCompiler::new(
            frontend.tokenizer(),
            crate::model::VOCABULARY,
            &model.eos_token_ids,
        )?);
        let ingress_capacity = config
            .queue_capacity
            .checked_add(model.limits.max_active)
            .ok_or("HTTP request capacity overflows")?;
        let model = Arc::new(unsafe {
            ModelRuntime::start(
                manifest,
                model,
                config.queue_capacity,
                config.event_capacity,
            )?
        });
        let state = Arc::new(HttpState {
            model: model.clone(),
            frontend: Arc::new(frontend),
            images: Arc::new(images),
            grammar,
            frontend_slots: Arc::new(Semaphore::new(config.frontend_workers)),
            ingress_slots: Arc::new(Semaphore::new(ingress_capacity)),
            frontend_capacity: config.frontend_workers,
            ingress_capacity,
            ids: AtomicU64::new(1),
        });
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let runtime_model = model.clone();
        let thread = match thread::Builder::new()
            .name("sparkgap-http".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|error| format!("cannot create HTTP runtime: {error}"))?;
                runtime.block_on(async move {
                    let shutdown_state = state.clone();
                    let mut router = Router::new()
                        .route("/health", get(health))
                        .route("/metrics", get(metrics))
                        .route("/v1/models", get(models))
                        .route("/v1/chat/completions", post(chat).layer(middleware::from_fn_with_state(state.clone(), admit_http)));
                    if config.qualification_token_trace {
                        router = router.route("/__qualify/generate", post(qualification::generate)
                            .layer(middleware::from_fn_with_state(state.clone(), admit_http)));
                    }
                    let router = router
                        .layer(DefaultBodyLimit::max(config.max_body_bytes))
                        .with_state(state);
                    let listener = match tokio::net::TcpListener::bind(config.bind).await {
                        Ok(listener) => listener,
                        Err(error) => {
                            let error = format!("cannot bind HTTP listener: {error}");
                            let _ = ready_tx.send(Err(error.clone()));
                            return Err(error);
                        }
                    };
                    let address = listener.local_addr().map_err(|error| error.to_string())?;
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
                        result=&mut serving=>{runtime_model.request_stop();return result.map_err(|error|error.to_string())?.map_err(|error|error.to_string());},
                        _=shutdown_rx=>{}
                    }
                    shutdown_state.frontend_slots.close();
                    shutdown_state.ingress_slots.close();
                    let _ = drain_tx.send(());
                    match tokio::time::timeout(Duration::from_secs(5), &mut serving).await {
                        Ok(result) => result
                            .map_err(|error| error.to_string())?
                            .map_err(|error| error.to_string()),
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
                    message: format!("cannot start HTTP thread: {error}"),
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
            }),
            result => {
                let _ = thread.join();
                let shutdown = model.close();
                Err(StartFailure {
                    message: match result {
                        Ok(Err(error)) => error,
                        _ => "HTTP thread exited before readiness".into(),
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
    pub fn close(&self) -> Shutdown {
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner());
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
                    shutdown.error.get_or_insert("HTTP thread panicked".into());
                }
            }
        }
        *self.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.close();
    }
}

async fn health(State(state): State<Arc<HttpState>>) -> Response {
    if state.model.healthy() {
        Json(json!({"status":"ok"})).into_response()
    } else {
        error(
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        )
    }
}
async fn models(State(state): State<Arc<HttpState>>) -> Response {
    Json(json!({"object":"list","data":[{"id":state.model.config().model_id,"object":"model","owned_by":"local"}]})).into_response()
}
async fn metrics(State(state): State<Arc<HttpState>>) -> Response {
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
    ] {
        metrics::metric(&mut text, name, "gauge", help, value as u64);
    }
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        text,
    )
        .into_response()
}
fn error(status: StatusCode, message: impl AsRef<str>) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.as_ref(),"type":"invalid_request_error"}})),
    )
        .into_response()
}

async fn admit_http(
    State(state): State<Arc<HttpState>>,
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
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "native HTTP admission is full",
            );
        }
    };
    request.extensions_mut().insert(ingress);
    next.run(request).await
}

async fn chat(
    State(state): State<Arc<HttpState>>,
    Extension(ingress): Extension<Arc<OwnedSemaphorePermit>>,
    Json(body): Json<Value>,
) -> Response {
    let ingress_ns = (body.get("return_token_timestamps") == Some(&Value::Bool(true)))
        .then(observation::monotonic_ns);
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
        request::prepare(
            &prepare_state.frontend,
            &prepare_state.images,
            &prepare_state.grammar,
            prepare_state.model.config(),
            body,
        )
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(message)) => return error(StatusCode::BAD_REQUEST, message),
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
        Err(message) => return error(StatusCode::SERVICE_UNAVAILABLE, message),
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
            let mut presentation=output::Presentation::<crate::model::OutputProtocol>::new(&state.frontend, &state.model.config().eos_token_ids, prepared.output, id.clone());
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
                    GenerationEvent::Finished{prompt_tokens,generated_tokens,reused_prompt_tokens,reason,termination}=>{
                        let deltas=presentation.finish(termination);
                        match deltas{Ok(deltas)=>for delta in deltas{yield Ok(Event::default().data(chunk(&id,created,&state.model.config().model_id,delta,Value::Null,observation.include_usage).to_string()));},Err(message)=>{presentation.abort(crate::output::Termination::ExecutionError);yield Ok(Event::default().data(json!({"error":{"message":message}}).to_string()));terminal=true;break;}}
                        let reason=if presentation.has_tools(){"tool_calls"}else{reason};
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
        let mut presentation = output::Presentation::<crate::model::OutputProtocol>::new(
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
                GenerationEvent::Error(message) => {
                    presentation.abort(crate::output::Termination::ExecutionError);
                    return error(StatusCode::INTERNAL_SERVER_ERROR, message);
                }
                GenerationEvent::AdmissionRejected(message) => {
                    presentation.abort(crate::output::Termination::ExecutionError);
                    return error(StatusCode::SERVICE_UNAVAILABLE, message);
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
                    let reason = if presentation.has_tools() {
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
