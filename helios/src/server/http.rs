use super::protocol::{Encoder, Frame, PromptUsage, Protocol};
use super::*;
use futures_util::StreamExt;
use std::{io::Read, task::Poll};

pub(super) fn error(
    protocol: Protocol,
    status: StatusCode,
    message: impl std::fmt::Display,
) -> Response {
    (status, Json(protocol.error(status, message))).into_response()
}
pub(super) async fn no_websocket() -> Response {
    error(
        Protocol::Responses,
        StatusCode::BAD_REQUEST,
        "WebSockets are not supported; use POST /v1/responses with HTTP/SSE",
    )
}
fn rejection(state: &HttpState, protocol: Protocol, failure: Error) -> Response {
    let (status, message) = rejection_details(state, failure);
    error(protocol, status, message)
}
fn rejection_details(state: &HttpState, failure: Error) -> (StatusCode, String) {
    let hint = match &failure {
        Error::Admission(AdmissionError::Context(_)) => &state.context_hint,
        Error::Admission(AdmissionError::Capacity(_)) => &state.capacity_hint,
        _ => "",
    };
    let message = if hint.is_empty() {
        failure.to_string()
    } else {
        format!("{failure}; {hint}")
    };
    (status(&failure), message)
}
fn decode_body(
    bytes: &[u8],
    encoding: &str,
    limit: usize,
) -> std::result::Result<Value, (StatusCode, String)> {
    let invalid = |e: std::io::Error| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid compressed request: {e}"),
        )
    };
    let reader: Box<dyn Read + '_> = match encoding {
        "" | "identity" => Box::new(bytes),
        "gzip" => Box::new(flate2::read::MultiGzDecoder::new(bytes)),
        "deflate" => Box::new(flate2::read::ZlibDecoder::new(bytes)),
        "br" => Box::new(brotli::Decompressor::new(bytes, 4096)),
        "zstd" => {
            let mut reader = zstd::stream::read::Decoder::new(bytes).map_err(invalid)?;
            reader.window_log_max(25).map_err(invalid)?;
            Box::new(reader)
        }
        _ => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported Content-Encoding".into(),
            ));
        }
    };
    let mut decoded = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut decoded)
        .map_err(invalid)?;
    if decoded.len() > limit {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "decoded request exceeds body limit".into(),
        ));
    }
    serde_json::from_slice(&decoded).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid JSON request: {e}"),
        )
    })
}
enum Delivery {
    Frame(Frame),
    Complete(Value),
    Failed(Error, Frame),
}

pub(super) async fn generate(
    State(state): State<Arc<HttpState>>,
    Extension(ingress): Extension<Arc<OwnedSemaphorePermit>>,
    request: Request,
) -> Response {
    let ingress_ns = observation::monotonic_ns();
    let protocol = Protocol::from_path(request.uri().path());
    let count = request.uri().path() == "/v1/messages/count_tokens";
    let metric = if count {
        4
    } else {
        match protocol {
            Protocol::Chat => 0,
            Protocol::Completions => 1,
            Protocol::Messages => 2,
            Protocol::Responses => 3,
        }
    };
    state.model.metrics().http_requests[metric].fetch_add(1, Ordering::Relaxed);

    if protocol == Protocol::Messages
        && request
            .headers()
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok())
            != Some("2023-06-01")
    {
        return error(
            protocol,
            StatusCode::BAD_REQUEST,
            "anthropic-version must be 2023-06-01",
        );
    }
    let encoding = request
        .headers()
        .get("content-encoding")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("identity")
        .to_owned();
    let bytes = match axum::body::to_bytes(request.into_body(), state.max_body_bytes).await {
        Ok(bytes) => bytes,
        Err(e) => return error(protocol, StatusCode::PAYLOAD_TOO_LARGE, e),
    };
    let permit = match state.frontend_slots.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return error(
                protocol,
                StatusCode::SERVICE_UNAVAILABLE,
                "native frontend is shutting down",
            );
        }
    };
    let prepare_state = state.clone();
    let preparation_ingress = ingress.clone();
    let preparation = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _ingress = preparation_ingress;
        let body = decode_body(&bytes, &encoding, prepare_state.max_body_bytes)?;
        let decoded = protocol::decode(protocol, body, count)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let tool_grammars = decoded
            .request
            .tools
            .iter()
            .filter_map(|tool| {
                tool.grammar.as_ref().map(|grammar| {
                    prepare_state
                        .grammar
                        .compile_text(&grammar.syntax, &grammar.source)
                        .map(|compiled| (tool.native_name(), compiled))
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
                })
            })
            .collect::<std::result::Result<std::collections::HashMap<_, _>, _>>()?;
        let size = decoded
            .request
            .generation_count()
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        if size > prepare_state.ingress_capacity {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "generation group of {size} exceeds capacity {}",
                    prepare_state.ingress_capacity
                ),
            ));
        }
        let extra = if size > 1 {
            Some(
                prepare_state
                    .ingress_slots
                    .clone()
                    .try_acquire_many_owned((size - 1) as u32)
                    .map_err(|_| {
                        (
                            StatusCode::TOO_MANY_REQUESTS,
                            "native generation admission is full".into(),
                        )
                    })?,
            )
        } else {
            None
        };
        let mut prepared = Vec::with_capacity(size);
        for index in 0..size {
            prepared.push(
                request::prepare_inference(
                    &prepare_state.frontend,
                    prepare_state.images.as_deref(),
                    &prepare_state.grammar,
                    prepare_state.model.config(),
                    &decoded.request,
                    index,
                    count,
                )
                .map_err(|e| {
                    rejection_details(
                        &prepare_state,
                        match e {
                            Error::Admission(_) => e,
                            _ => e.admission_context(AdmissionError::Invalid),
                        },
                    )
                })?,
            );
        }
        Ok((decoded, prepared, extra, tool_grammars))
    })
    .await;
    let (decoded, prepared, extra, tool_grammars) = match preparation {
        Ok(Ok(v)) => v,
        Ok(Err((status, message))) => return error(protocol, status, message),
        Err(e) => return error(protocol, StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    if count {
        return Json(json!({"input_tokens":prepared[0].prompt.tokens.len()})).into_response();
    }
    let prompt_usage = prepared
        .iter()
        .step_by(decoded.request.choices)
        .map(|p| PromptUsage {
            tokens: p.prompt.tokens.len(),
            image_tokens: p.prompt.images.as_ref().map_or(0, |images| {
                images.geometry.iter().map(|geometry| geometry.tokens).sum()
            }),
        })
        .collect();
    let observation = decoded.observation;
    let ingress_ns = observation.timestamps.then_some(ingress_ns);
    let prompt_ids = observation
        .token_ids
        .then(|| prepared[0].prompt.tokens.clone());
    let size = prepared.len();
    let mut configs = vec![];
    let mut echoes = vec![];
    let mut prompts = vec![];
    let mut zero = vec![];
    for (i, p) in prepared.into_iter().enumerate() {
        configs.push(p.output);
        echoes.push(p.echo);
        if p.prompt.max_new_tokens == 0 {
            zero.push(i);
        } else {
            prompts.push(p.prompt);
        }
    }
    let submitted = if prompts.is_empty() {
        Ok(vec![])
    } else if observation.token_ids && size == 1 {
        state
            .model
            .generate_observed(prompts.remove(0), observation.timestamps)
            .map(|g| vec![g])
    } else {
        state.model.generate_many(prompts)
    };
    let generation = match submitted {
        Ok(g) => g,
        Err(e) => return rejection(&state, protocol, e),
    };
    let enqueue_ns = generation.first().and_then(|g| g.enqueue_ns);
    let mut iter = generation.into_iter();
    let mut generations: Vec<Option<Generation>> = (0..size)
        .map(|i| if zero.contains(&i) { None } else { iter.next() })
        .collect();
    let prefix = match protocol {
        Protocol::Chat => "chatcmpl",
        Protocol::Completions => "cmpl",
        Protocol::Messages => "msg",
        Protocol::Responses => "resp",
    };
    let id = format!(
        "{prefix}-{}-{}",
        std::process::id(),
        state.ids.fetch_add(1, Ordering::Relaxed)
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut encoder = Encoder::new(
        protocol,
        id.clone(),
        state.model.config().model_id.clone(),
        created,
        decoded.request.choices,
        prompt_usage,
        decoded.request.tools,
        decoded.request.tool_choice,
        decoded.request.parallel_tool_calls,
        observation.include_usage,
        decoded.metadata,
    );
    encoder.set_tool_grammars(tool_grammars);
    let streaming = decoded.stream;
    let stream_state = state.clone();
    let delivery = async_stream::stream! {
        let state = stream_state;
        let _ingress = ingress;
        let _extra = extra;
        let mut decoders: Vec<_> = configs
            .into_iter()
            .map(|config| {
                output::Decoder::<crate::output::OutputPipeline>::new(
                    &state.frontend,
                    &state.model.config().eos_token_ids,
                    config,
                )
            })
            .collect();
        let mut raw_tokens = Vec::new();
        let mut timestamps = Vec::new();
        let mut reused_total = 0;
        for mut frame in encoder.start() {
            if let Some(ids) = &prompt_ids {
                frame.data["prompt_token_ids"] = json!(ids);
            }
            yield Delivery::Frame(frame);
        }
        for (i, echo) in echoes.into_iter().enumerate() {
            if decoded.echo && !echo.is_empty() {
                match encoder.apply(i, vec![crate::output::Event::Content(echo)]) {
                    Ok(frames) => {
                        for frame in frames {
                            yield Delivery::Frame(frame)
                        }
                    }
                    Err(e) => {
                        let frame = encoder.failure(&e);
                        yield Delivery::Failed(e, frame);
                        return;
                    }
                }
            }
        }
        for i in zero {
            match encoder.finish(i, crate::output::Termination::OutputLimit, 0, 0, None) {
                Ok(frames) => {
                    for frame in frames {
                        yield Delivery::Frame(frame)
                    }
                }
                Err(e) => {
                    let frame = encoder.failure(&e);
                    yield Delivery::Failed(e, frame);
                    return;
                }
            }
        }
        let mut cursor = 0;
        while generations.iter().any(Option::is_some) {
            let (index, event) = std::future::poll_fn(|cx| {
                for offset in 0..size {
                    let i = (cursor + offset) % size;
                    if let Some(generation) = &mut generations[i] {
                        match generation.events.poll_recv(cx) {
                            Poll::Ready(Some(event)) => {
                                cursor = (i + 1) % size;
                                return Poll::Ready((i, event));
                            }
                            Poll::Ready(None) => {
                                return Poll::Ready((i, GenerationEvent::Error(generation.failure())));
                            }
                            Poll::Pending => {}
                        }
                    }
                }
                Poll::Pending
            })
            .await;
            let result = match event {
                GenerationEvent::Token(token) => decoders[index]
                    .push(token)
                    .and_then(|events| encoder.apply(index, events)),
                GenerationEvent::Protocol(pieces) => decoders[index]
                    .push_protocol(pieces)
                    .and_then(|events| encoder.apply(index, events)),
                GenerationEvent::ObservedToken {
                    token,
                    timestamp_ns,
                } => {
                    raw_tokens.push(token);
                    if let Some(timestamp) = timestamp_ns {
                        timestamps.push(timestamp);
                    }
                    if protocol == Protocol::Chat {
                        yield Delivery::Frame(Frame {
                            event: None,
                            data: super::observed_chunk(
                                &id,
                                created,
                                &state.model.config().model_id,
                                token,
                                timestamp_ns,
                                observation.include_usage,
                            ),
                            done: false,
                        });
                    }
                    Ok(vec![])
                }
                GenerationEvent::Finished {
                    generated_tokens,
                    reused_prompt_tokens,
                    termination,
                    stop_sequence,
                    ..
                } => {
                    reused_total += reused_prompt_tokens;
                    let result = decoders[index]
                        .finish(termination)
                        .and_then(|events| encoder.apply(index, events))
                        .and_then(|mut frames| {
                            let mut final_frames = encoder.finish(
                                index,
                                termination,
                                generated_tokens,
                                reused_prompt_tokens,
                                stop_sequence,
                            )?;
                            if protocol == Protocol::Chat {
                                for frame in &mut final_frames {
                                    if !frame.done
                                        && frame.data["choices"][0]["finish_reason"].is_string()
                                    {
                                        observation.finish(
                                            &mut frame.data,
                                            &encoder.usage_value(),
                                            reused_prompt_tokens,
                                            (ingress_ns, enqueue_ns),
                                            true,
                                        );
                                    }
                                }
                            }
                            frames.extend(final_frames);
                            Ok(frames)
                        });
                    generations[index] = None;
                    result
                }
                GenerationEvent::AdmissionRejected(e) | GenerationEvent::Error(e) => Err(e),
            };
            match result {
                Ok(frames) => {
                    for frame in frames {
                        yield Delivery::Frame(frame)
                    }
                }
                Err(e) => {
                    let frame = encoder.failure(&e);
                    yield Delivery::Failed(e, frame);
                    return;
                }
            }
        }
        let mut response = encoder.response();
        if protocol == Protocol::Chat {
            if let Some(ids) = prompt_ids {
                response["prompt_token_ids"] = json!(ids);
                response["choices"][0]["token_ids"] = json!(raw_tokens);
            }
            if observation.timestamps {
                response["choices"][0]["token_timestamps_ns"] = json!(timestamps);
            }
            observation.finish(
                &mut response,
                &encoder.usage_value(),
                reused_total,
                (ingress_ns, enqueue_ns),
                false,
            );
        }
        yield Delivery::Complete(response);
    };
    if streaming {
        let events = delivery.filter_map(|event| async move {
            match event {
                Delivery::Frame(frame) => Some(Ok::<_, Infallible>(frame.sse())),
                Delivery::Failed(error, frame) => {
                    eprintln!("streaming generation failed: {error}");
                    Some(Ok::<_, Infallible>(frame.sse()))
                }
                Delivery::Complete(_) => None,
            }
        });
        Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response()
    } else {
        futures_util::pin_mut!(delivery);
        while let Some(event) = delivery.next().await {
            match event {
                Delivery::Complete(body) => return Json(body).into_response(),
                Delivery::Failed(e, _) => return rejection(&state, protocol, e),
                _ => {}
            }
        }
        error(
            protocol,
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation ended without a response",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn compressed_bodies_share_json_and_decoded_size_limits() {
        let body = br#"{"input":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}"#;
        let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip.write_all(body).unwrap();
        let mut deflate =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        deflate.write_all(body).unwrap();
        let mut brotli = Vec::new();
        {
            let mut writer = brotli::CompressorWriter::new(&mut brotli, 4096, 5, 22);
            writer.write_all(body).unwrap();
        }
        for (encoding, data) in [
            ("identity", body.to_vec()),
            ("gzip", gzip.finish().unwrap()),
            ("deflate", deflate.finish().unwrap()),
            ("br", brotli),
            ("zstd", zstd::stream::encode_all(&body[..], 1).unwrap()),
        ] {
            assert_eq!(
                decode_body(&data, encoding, 100).unwrap(),
                json!({"input":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"})
            );
            assert_eq!(
                decode_body(&data, encoding, 10).unwrap_err().0,
                StatusCode::PAYLOAD_TOO_LARGE
            );
        }
        assert_eq!(
            decode_body(b"{}", "unknown", 100).unwrap_err().0,
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
        assert_eq!(
            decode_body(b"not gzip", "gzip", 100).unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            decode_body(b"{", "identity", 100).unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
    }
}
