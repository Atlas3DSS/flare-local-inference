//! Opt-in, loopback-only observation of cold greedy generation through the server.

use super::*;

pub(super) fn validate_prompt(prompt: &Prompt) -> Result<()> {
    let sampling = &prompt.sampling;
    if prompt.max_new_tokens != 64
        || sampling.temperature != 0.
        || sampling.repetition_penalty != 1.
        || sampling.presence_penalty != 0.
        || sampling.frequency_penalty != 0.
        || !prompt.stop_sequences.is_empty()
        || prompt.grammar.is_some()
        || prompt.images.is_some()
        || !prompt.conditioning.is_empty()
        || prompt.vision_token_indices.iter().any(|index| *index != -1)
    {
        return Err("token qualification requires text, temperature=0, max_tokens=64, neutral penalties, and no grammar or token stops".into());
    }
    Ok(())
}

fn envelope(body: Value) -> Result<(String, Value, Vec<u32>)> {
    if body.get("schema_version").and_then(Value::as_u64) != Some(1)
        || body.get("ignore_eos") != Some(&Value::Bool(true))
    {
        return Err("token qualification requires schema_version=1 and ignore_eos=true".into());
    }
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty() && id.len() <= 256)
        .ok_or("token qualification requires a nonempty ID of at most 256 bytes")?
        .to_owned();
    let request = body
        .get("request")
        .filter(|request| request.is_object())
        .ok_or("token qualification requires a chat request object")?
        .clone();
    let expected: Vec<u32> = serde_json::from_value(
        body.get("expected_prompt_tokens")
            .ok_or("token qualification requires expected_prompt_tokens")?
            .clone(),
    )
    .map_err(|error| format!("invalid expected_prompt_tokens: {error}"))?;
    if expected.is_empty()
        || expected
            .iter()
            .any(|token| *token as usize >= crate::model::VOCABULARY)
    {
        return Err("expected prompt requires nonempty, valid vocabulary IDs".into());
    }
    Ok((id, request, expected))
}

pub(super) async fn generate(
    State(state): State<Arc<HttpState>>,
    Extension(ingress): Extension<Arc<OwnedSemaphorePermit>>,
    Json(body): Json<Value>,
) -> Response {
    let (id, body, expected) = match envelope(body) {
        Ok(value) => value,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let permit = match state.frontend_slots.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "native frontend is shutting down",
            );
        }
    };
    let preparation_state = state.clone();
    let preparation_ingress = ingress.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _ingress = preparation_ingress;
        let prepared = request::prepare(
            &preparation_state.frontend,
            &preparation_state.images,
            &preparation_state.grammar,
            preparation_state.model.config(),
            body,
        )?;
        validate_prepared(&prepared, &expected)?;
        Ok::<_, String>(prepared)
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(message)) => return error(StatusCode::BAD_REQUEST, message),
        Err(message) => return error(StatusCode::INTERNAL_SERVER_ERROR, message.to_string()),
    };
    let prompt_tokens = prepared.prompt.tokens.clone();
    let mut generation = match state.model.generate_qualification(prepared.prompt) {
        Ok(generation) => generation,
        Err(message) => return error(StatusCode::SERVICE_UNAVAILABLE, message),
    };
    let _ingress = ingress;
    let mut tokens = Vec::with_capacity(64);
    while let Some(event) = generation.events.recv().await {
        match event {
            GenerationEvent::ObservedToken { .. } => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "unexpected observation event in qualification stream",
                );
            }
            GenerationEvent::Token(token) => {
                if tokens.len() == 64 {
                    return error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "qualification generated more than 64 tokens",
                    );
                }
                tokens.push(token);
            }
            GenerationEvent::AdmissionRejected(message) => {
                return error(StatusCode::SERVICE_UNAVAILABLE, message);
            }
            GenerationEvent::Error(message) => {
                return error(StatusCode::INTERNAL_SERVER_ERROR, message);
            }
            GenerationEvent::Finished {
                prompt_tokens: count,
                generated_tokens,
                reused_prompt_tokens,
                reason,
            } => {
                if count != prompt_tokens.len()
                    || generated_tokens != 64
                    || tokens.len() != 64
                    || reused_prompt_tokens != 0
                    || reason != "length"
                {
                    return error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "qualification completion violates cold 64-token contract",
                    );
                }
                let identity: String = state
                    .model
                    .config()
                    .identity
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                return Json(json!({"schema_version":1,"id":id,"model_id":state.model.config().model_id,
                    "model_identity":identity,"prompt_token_ids":prompt_tokens,"generated_token_ids":tokens,
                    "rendered_prompt":prepared.rendered_prompt,"ignore_eos":true,"cold":true,
                    "reused_prompt_tokens":reused_prompt_tokens,"finish_reason":reason,
                    "usage":{"prompt_tokens":count,"completion_tokens":generated_tokens,"total_tokens":count+64}})).into_response();
            }
        }
    }
    error(StatusCode::INTERNAL_SERVER_ERROR, generation.failure())
}

fn validate_prepared(prepared: &PreparedRequest, expected: &[u32]) -> Result<()> {
    if prepared.stream {
        return Err("token qualification requires stream=false".into());
    }
    if prepared.prompt.tokens != expected {
        return Err("native rendered prompt token IDs differ from the qualification corpus".into());
    }
    validate_prompt(&prepared.prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SamplingParameters;

    fn prepared() -> PreparedRequest {
        PreparedRequest {
            prompt: Prompt {
                tokens: vec![760, 6511],
                positions: vec![[0; 3], [1; 3]],
                vision_token_indices: vec![-1; 2],
                images: None,
                conditioning: vec![],
                sampling: SamplingParameters {
                    temperature: 0.,
                    ..Default::default()
                },
                max_new_tokens: 64,
                stop_sequences: vec![],
                grammar: None,
            },
            stream: false,
            output: OutputConfig {
                verbatim: false,
                reasoning: true,
                tools: vec![json!({"type":"function","function":{"name":"search"}})],
            },
            rendered_prompt: "unchanged tool-bearing prompt".into(),
            observation: Default::default(),
        }
    }

    #[test]
    fn prompt_match_and_sampling_contract_are_required_before_enqueue() {
        let mut prepared = prepared();
        assert!(validate_prepared(&prepared, &[760, 6511]).is_ok());
        assert!(
            validate_prepared(&prepared, &[760, 6512])
                .unwrap_err()
                .contains("token IDs differ")
        );
        prepared.stream = true;
        assert!(validate_prepared(&prepared, &[760, 6511]).is_err());
        prepared.stream = false;
        prepared.prompt.stop_sequences.push(vec![248046]);
        assert!(validate_prepared(&prepared, &[760, 6511]).is_err());
        prepared.prompt.stop_sequences.clear();
        prepared.prompt.max_new_tokens = 63;
        assert!(validate_prepared(&prepared, &[760, 6511]).is_err());
        prepared.prompt.max_new_tokens = 64;
        prepared.prompt.sampling.temperature = 1.;
        assert!(validate_prepared(&prepared, &[760, 6511]).is_err());
        prepared.prompt.sampling.temperature = 0.;
        prepared.prompt.sampling.repetition_penalty = 1.1;
        assert!(validate_prepared(&prepared, &[760, 6511]).is_err());
        prepared.prompt.sampling.repetition_penalty = 1.;
        prepared.prompt.vision_token_indices[0] = 0;
        assert!(validate_prepared(&prepared, &[760, 6511]).is_err());
    }

    #[test]
    fn explicit_qualification_envelope_preserves_chat_controls() {
        let request = json!({"messages":[],"tools":[{"function":{"name":"search"}}],
            "continue_final_message":true,"temperature":0,"max_tokens":64});
        let body = json!({"schema_version":1,"ignore_eos":true,"id":"tool-case",
            "request":request,"expected_prompt_tokens":[760,6511]});
        let (id, actual, tokens) = envelope(body.clone()).unwrap();
        assert_eq!(id, "tool-case");
        assert_eq!(actual, request);
        assert_eq!(tokens, [760, 6511]);
        for field in [
            "schema_version",
            "ignore_eos",
            "id",
            "request",
            "expected_prompt_tokens",
        ] {
            let mut invalid = body.clone();
            invalid.as_object_mut().unwrap().remove(field);
            assert!(envelope(invalid).is_err());
        }
    }
}
