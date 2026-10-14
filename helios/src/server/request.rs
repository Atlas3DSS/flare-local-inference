use super::runtime::Prompt;
use crate::{
    AdmissionError, Error, Failure, FrontendError, Result,
    frontend::{Frontend, images::ImageProcessor},
    model::{ModelConfig, ModelContract, SamplingParameters},
};
use base64::Engine;
use serde_json::{Map, Value};

pub struct PreparedRequest<M: ModelContract> {
    pub prompt: Prompt<M>,
    pub stream: bool,
    pub output: M::OutputConfig,
    pub rendered_prompt: String,
}

pub fn prepare<M: ModelContract>(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    compiler: &crate::grammar::GrammarCompiler,
    config: &ModelConfig,
    body: Value,
) -> Result<PreparedRequest<M>> {
    prepare_inner(frontend, processor, compiler, config, body).map_err(|error| match error {
        Error::Admission(_) => error,
        Error::Frontend(FrontendError::ImageDecode(_)) => {
            error.admission_context(AdmissionError::Image)
        }
        _ => error.admission_context(AdmissionError::Invalid),
    })
}

fn prepare_inner<M: ModelContract>(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    compiler: &crate::grammar::GrammarCompiler,
    config: &ModelConfig,
    body: Value,
) -> Result<PreparedRequest<M>> {
    if body.get("model").and_then(Value::as_str) != Some(config.model_id.as_str()) {
        return Err("requested model is not loaded".into());
    }
    let decoded = super::protocol::decode(super::protocol::Protocol::Chat, body, false)?;
    let prepared = prepare_inference::<M>(
        frontend,
        processor,
        compiler,
        config,
        &decoded.request,
        0,
        false,
    )?;
    Ok(PreparedRequest {
        prompt: prepared.prompt,
        output: prepared.output.ok_or("chat requires parsed output")?,
        stream: decoded.stream,
        rendered_prompt: prepared.rendered_prompt,
    })
}

pub(super) struct PreparedInference<M: ModelContract> {
    pub prompt: Prompt<M>,
    pub output: Option<M::OutputConfig>,
    pub rendered_prompt: String,
    pub echo: String,
}

pub(super) fn prepare_inference<M: ModelContract>(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    compiler: &crate::grammar::GrammarCompiler,
    config: &ModelConfig,
    request: &super::inference::Request,
    index: usize,
    count_only: bool,
) -> Result<PreparedInference<M>> {
    use super::inference::{Constraint, Input, RawPrompt, ToolChoice};
    let opts = &request.options;
    let mut object = serde_json::to_value(&opts.sampling)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    if let Some(limit) = opts.output_limit {
        object.insert("max_tokens".into(), limit.into());
    }
    let object = &object;
    let structured = opts.constraint.is_some();
    let (mut normalized, raw) = match &request.input {
        Input::Conversation(messages) => (
            super::inference::template_messages(messages, request.tool_choice)?,
            None,
        ),
        Input::Raw(prompts) => (vec![], Some(&prompts[index / request.choices])),
    };
    frontend.normalize_inline_system_messages(&mut normalized)?;
    if raw.is_none() {
        if let Some(Constraint::Json(schema)) = &opts.constraint {
            super::inference::append_system_instruction(
                &mut normalized,
                &format!(
                    "Return the final answer as a JSON value matching this schema. Do not wrap it in Markdown fences or add text outside the JSON value. Tool calls may precede the final answer.\nJSON schema:\n{schema}"
                ),
            )?;
        }
    }
    let mut images = Vec::new();
    for message in &mut normalized {
        if !message.is_object() {
            return Err("messages must contain objects".into());
        }
        if let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) {
            for part in parts {
                let kind = part
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or("message content part requires type")?;
                match kind {
                    "text" => {
                        if !part.get("text").is_some_and(Value::is_string) {
                            return Err("text content must be a string".into());
                        }
                    }
                    "image_url" => {
                        let processor = processor.ok_or_else(|| {
                            Error::admission(
                                AdmissionError::Feature,
                                "prepared model does not support image inputs",
                            )
                        })?;
                        let url = part
                            .get("image_url")
                            .and_then(|value| value.get("url"))
                            .and_then(Value::as_str)
                            .ok_or("image_url.url must be a string")?;
                        let (data, jpeg) =
                            if let Some(data) = url.strip_prefix("data:image/png;base64,") {
                                (data, false)
                            } else if let Some(data) = url.strip_prefix("data:image/jpeg;base64,") {
                                (data, true)
                            } else {
                                return Err(Error::admission(
                                    AdmissionError::Image,
                                    "image input requires a base64 PNG or JPEG data URL",
                                ));
                            };
                        if data.len() > processor.max_encoded_bytes().div_ceil(3).saturating_mul(4)
                        {
                            return Err(Error::admission(
                                AdmissionError::Image,
                                "encoded image exceeds the configured byte limit",
                            ));
                        }
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(data)
                            .map_err(|error| {
                                Error::Admission(AdmissionError::Image(Failure::caused_by(
                                    format!("invalid image base64: {error}"),
                                    error,
                                )))
                            })?;
                        images.push(if jpeg {
                            processor
                                .decode_jpeg(&bytes)
                                .map_err(|error| error.admission_context(AdmissionError::Image))?
                        } else {
                            processor
                                .decode_png(&bytes)
                                .map_err(|error| error.admission_context(AdmissionError::Image))?
                        });
                        *part = serde_json::json!({"type":"image"});
                    }
                    _ => {}
                }
            }
        }
    }
    let mut options = opts.template.clone();
    if !options.contains_key("reasoning_effort") {
        if let Some(effort) = opts.effort {
            use super::inference::Effort;
            let candidates: &[&str] = match effort {
                Effort::Low => &["low"],
                Effort::High => &["high", "medium"],
                Effort::Max => &["max", "xhigh", "high", "medium"],
            };
            options.insert(
                "reasoning_effort".into(),
                frontend.resolve_effort(candidates)?.into(),
            );
        }
    }

    if request.tool_choice == ToolChoice::None {
        options.insert("tools".into(), Value::Array(vec![]));
    } else if !request.tools.is_empty() {
        options.insert("tools".into(), Value::Array(request.tools.iter().map(|tool| {
            let description = if let Some(grammar) = &tool.grammar {
                format!("{}\nPut the complete tool input in the input string. It must match this {} grammar exactly:\n{}", tool.description, grammar.syntax, grammar.source)
            } else { tool.description.clone() };
            serde_json::json!({"type":"function", "function":{"name":tool.native_name(), "description":description, "parameters":tool.parameters}})
        }).collect()));
    }
    if !options.contains_key("add_generation_prompt")
        && requests_continuation(options.get("continue_final_message"))
    {
        options.insert("add_generation_prompt".into(), Value::Bool(false));
    }
    let (rendered, echo) = if let Some(raw) = raw {
        let (mut token_ids, echo) = match raw {
            RawPrompt::Text(text) => (frontend.encode(text, true)?, text.clone()),
            RawPrompt::Tokens(ids) => {
                if ids.is_empty()
                    || ids
                        .iter()
                        .any(|id| *id as usize >= config.contract.vocabulary)
                {
                    return Err("token prompt must contain valid vocabulary IDs".into());
                }
                (ids.clone(), frontend.decode(ids, false)?)
            }
        };
        if token_ids.is_empty() {
            token_ids.push(
                frontend
                    .bos_token_id()
                    .ok_or("empty prompt requires a packaged BOS token")?,
            );
        }
        (
            crate::frontend::RenderedPrompt {
                text: frontend.decode(&token_ids, false)?,
                token_ids,
            },
            echo,
        )
    } else {
        (frontend.render_chat(&normalized, &options)?, String::new())
    };
    let output = if raw.is_some() {
        None
    } else {
        Some(M::prepare_output(
            frontend,
            &rendered,
            &normalized,
            &options,
            structured,
        )?)
    };
    use crate::output::ModelOutput;
    let envelope = output.as_ref().and_then(M::Output::grammar_envelope);
    let grammar = match &opts.constraint {
        Some(Constraint::Json(schema)) => {
            Some(compiler.compile_json_schema_in(schema, envelope.as_ref())?)
        }
        Some(Constraint::Lark(source)) => {
            Some(compiler.compile_lark_in(source, envelope.as_ref())?)
        }
        None => None,
    };
    let crate::model::PreparedInput {
        tokens,
        data,
        conditioning,
        images,
    } = M::prepare_input(frontend, processor, &rendered, images)?;
    let system_boundary = if raw.is_some() {
        None
    } else {
        M::system_boundary(frontend, &tokens, &normalized, &options)
    };
    let mut sampling = sampling_parameters(object, &config.sampling_defaults)?;
    sampling.seed = sampling.seed.wrapping_add(index as u64);
    let stop_sequences = opts.token_stops.clone();
    super::stops::TokenStops::new(stop_sequences.clone(), config.contract.vocabulary)?;
    let stop_strings = opts.stops.clone();
    super::stops::validate_strings(&stop_strings, !stop_sequences.is_empty(), grammar.is_some())?;
    if grammar.is_some() && !stop_sequences.is_empty() {
        return Err("structured output cannot be combined with custom token stops".into());
    }
    if grammar.is_some()
        && requests_continuation(
            options
                .get("continue_final_message")
                .or_else(|| frontend.template_default("continue_final_message")),
        )
    {
        return Err("structured output cannot continue an existing assistant message".into());
    }
    let max_new_tokens = if count_only || (raw.is_some() && opts.output_limit == Some(0)) {
        if tokens.len() > config.limits.max_sequence_tokens {
            return Err(Error::admission(
                AdmissionError::Context,
                "prompt exceeds model context",
            ));
        }
        0
    } else {
        output_limit(
            object,
            config.max_new_tokens,
            tokens.len(),
            config.limits.max_sequence_tokens,
        )?
    };
    let ignore_eos = opts.ignore_eos;
    if ignore_eos && grammar.is_some() {
        return Err("structured output cannot ignore end-of-sequence tokens".into());
    }
    Ok(PreparedInference {
        prompt: Prompt {
            tokens,
            system_boundary,
            data,
            images,
            conditioning,
            sampling,
            max_new_tokens,
            stop_sequences,
            stop_strings,
            grammar,
            ignore_eos,
        },
        output,
        rendered_prompt: rendered.text,
        echo,
    })
}

fn output_limit(
    object: &Map<String, Value>,
    configured: u32,
    prompt: usize,
    context: usize,
) -> Result<u32> {
    let remaining = context.saturating_sub(prompt);
    let requested = object
        .get("max_completion_tokens")
        .filter(|v| !v.is_null())
        .or_else(|| object.get("max_tokens").filter(|v| !v.is_null()));
    let limit = match requested {
        Some(value) => value
            .as_u64()
            .ok_or("output limit must be a nonnegative integer")?,
        None => (configured as usize).min(remaining) as u64,
    };
    let max_new_tokens = u32::try_from(limit).map_err(|error| {
        Error::Admission(AdmissionError::Invalid(Failure::caused_by(
            "output limit exceeds uint32",
            error,
        )))
    })?;
    if max_new_tokens == 0 || max_new_tokens > i32::MAX as u32 {
        return Err("output limit must be between 1 and 2147483647 tokens".into());
    }
    if prompt
        .checked_add(max_new_tokens as usize)
        .is_none_or(|count| count > context)
    {
        return Err(Error::admission(
            AdmissionError::Context,
            format!(
                "prompt of {prompt} tokens plus an output limit of {max_new_tokens} tokens exceeds this server's context of {context} tokens"
            ),
        ));
    }
    Ok(max_new_tokens)
}

fn requests_continuation(value: Option<&Value>) -> bool {
    value.is_some_and(|value| {
        value == &Value::Bool(true) || value.as_str().is_some_and(|field| !field.is_empty())
    })
}

pub(super) fn sampling_parameters(
    object: &Map<String, Value>,
    defaults: &SamplingParameters,
) -> Result<SamplingParameters> {
    let sampling = SamplingParameters {
        seed: integer(object, "seed", defaults.seed)?,
        temperature: number(object, "temperature", defaults.temperature)?,
        top_p: number(object, "top_p", defaults.top_p)?,
        min_p: number(object, "min_p", defaults.min_p)?,
        repetition_penalty: number(object, "repetition_penalty", defaults.repetition_penalty)?,
        presence_penalty: number(object, "presence_penalty", defaults.presence_penalty)?,
        frequency_penalty: number(object, "frequency_penalty", defaults.frequency_penalty)?,
        top_k: object
            .get("top_k")
            .map(|value| {
                value
                    .as_i64()
                    .and_then(|v| i32::try_from(v).ok())
                    .ok_or("top_k must be int32")
            })
            .transpose()?
            .unwrap_or(defaults.top_k),
    };
    sampling.validate()?;
    Ok(sampling)
}

#[cfg(test)]
fn boolean(object: &Map<String, Value>, name: &str, default: bool) -> Result<bool> {
    object
        .get(name)
        .map(|v| {
            v.as_bool().ok_or_else(|| {
                Error::admission(AdmissionError::Invalid, format!("{name} must be boolean"))
            })
        })
        .unwrap_or(Ok(default))
}
fn integer(object: &Map<String, Value>, name: &str, default: u64) -> Result<u64> {
    object
        .get(name)
        .map(|v| {
            v.as_u64().ok_or_else(|| {
                Error::admission(
                    AdmissionError::Invalid,
                    format!("{name} must be a nonnegative integer"),
                )
            })
        })
        .unwrap_or(Ok(default))
}
fn number(object: &Map<String, Value>, name: &str, default: f32) -> Result<f32> {
    object
        .get(name)
        .map(|v| {
            v.as_f64()
                .filter(|v| v.is_finite() && v.abs() <= f32::MAX as f64)
                .map(|v| v as f32)
                .ok_or_else(|| {
                    Error::admission(
                        AdmissionError::Invalid,
                        format!("{name} must be finite float32"),
                    )
                })
        })
        .unwrap_or(Ok(default))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn omitted_output_limit_uses_remaining_context_and_explicit_limits_win() {
        let limit = |value: Value, configured, prompt| {
            output_limit(value.as_object().unwrap(), configured, prompt, 262144)
        };
        assert_eq!(limit(serde_json::json!({}), 262144, 144).unwrap(), 262000);
        assert_eq!(
            limit(serde_json::json!({"max_tokens":null}), 262144, 144).unwrap(),
            262000
        );
        assert_eq!(limit(serde_json::json!({}), 1024, 144).unwrap(), 1024);
        assert_eq!(limit(serde_json::json!({}), 1024, 262140).unwrap(), 4);
        assert_eq!(
            limit(serde_json::json!({"max_tokens":4096}), 1024, 144).unwrap(),
            4096
        );
        assert_eq!(
            limit(
                serde_json::json!({"max_tokens":32,"max_completion_tokens":64}),
                262144,
                144
            )
            .unwrap(),
            64
        );
        assert_eq!(
            limit(
                serde_json::json!({"max_tokens":32,"max_completion_tokens":null}),
                262144,
                144
            )
            .unwrap(),
            32
        );
        for body in [
            serde_json::json!({"max_tokens":0}),
            serde_json::json!({"max_tokens":-1}),
            serde_json::json!({"max_tokens":262144}),
            serde_json::json!({"max_tokens":"32"}),
        ] {
            assert!(limit(body, 262144, 144).is_err());
        }
        assert!(limit(serde_json::json!({}), 262144, 262144).is_err());
        assert!(limit(serde_json::json!({}), 262144, 262145).is_err());
    }
    #[test]
    fn explicit_zero_values_survive_defaults_and_nonfinite_casts_fail() {
        let values = serde_json::json!({"temperature":0,"seed":0,"stream":false,"large":1e100});
        let object = values.as_object().unwrap();
        assert_eq!(number(object, "temperature", 1.).unwrap(), 0.);
        assert_eq!(integer(object, "seed", 42).unwrap(), 0);
        assert!(!boolean(object, "stream", true).unwrap());
        assert!(number(object, "large", 0.).is_err());
    }

    #[test]
    fn request_sampling_overrides_only_explicit_fields_in_prepared_defaults() {
        let defaults = SamplingParameters {
            seed: 123,
            temperature: 1.,
            top_p: 0.95,
            top_k: -1,
            ..SamplingParameters::default()
        };
        let empty = sampling_parameters(&Map::new(), &defaults).unwrap();
        assert_eq!(empty.top_k, -1);
        assert_eq!(empty.temperature, 1.);
        let values = serde_json::json!({"seed":0,"temperature":0.5,"top_p":1,"top_k":0});
        let explicit = sampling_parameters(values.as_object().unwrap(), &defaults).unwrap();
        assert_eq!(explicit.seed, 0);
        assert_eq!(explicit.temperature, 0.5);
        assert_eq!(explicit.top_k, 0);
        assert_eq!(explicit.top_p, 1.);
        assert_eq!(explicit.repetition_penalty, 1.);
        assert_eq!(defaults.seed, 123);
        assert_eq!(defaults.temperature, 1.);
    }

    #[test]
    fn zero_temperature_is_rejected_without_replacing_it_with_the_default() {
        for temperature in [0., -0.] {
            let values = serde_json::json!({"temperature": temperature});
            let error =
                sampling_parameters(values.as_object().unwrap(), &SamplingParameters::default())
                    .unwrap_err();
            assert!(matches!(
                error,
                Error::Admission(AdmissionError::Invalid(_))
            ));
            assert!(error.to_string().contains(
                "greedy decoding incorrectly collapses the model’s probability distribution"
            ));
        }
    }
}
