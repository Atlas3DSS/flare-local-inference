use super::runtime::Prompt;
use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{Frontend, images::ImageProcessor},
    model::{ModelConfig, SamplingParameters},
};
use base64::Engine;
use serde_json::{Map, Value};

pub struct PreparedRequest {
    pub prompt: Prompt,
    pub stream: bool,
    pub output: crate::model::OutputConfig,
    pub rendered_prompt: String,
    pub(crate) observation: super::observation::ObservationOptions,
}

pub fn prepare(
    frontend: &Frontend,
    processor: &ImageProcessor,
    compiler: &crate::grammar::GrammarCompiler,
    config: &ModelConfig,
    body: Value,
) -> Result<PreparedRequest> {
    let object = body.as_object().ok_or("request must be a JSON object")?;
    let observation = super::observation::ObservationOptions::parse(object)?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or("model must be a string")?;
    if model != config.model_id {
        return Err(format!("model {model:?} is not loaded"));
    }
    for name in [
        "stop",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "guided_json",
    ] {
        if object
            .get(name)
            .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
        {
            return Err(format!("request option {name} is not implemented"));
        }
    }
    let tools_disabled = match object.get("tool_choice").filter(|value| !value.is_null()) {
        None => false,
        Some(Value::String(choice)) if choice == "auto" => false,
        Some(Value::String(choice)) if choice == "none" => true,
        _ => {
            return Err(
                "tool_choice supports auto or none; forced tool selection is not implemented"
                    .into(),
            );
        }
    };
    if object
        .get("parallel_tool_calls")
        .is_some_and(|value| value != &Value::Bool(true))
    {
        return Err("parallel_tool_calls=false is not implemented".into());
    }
    if object
        .get("n")
        .is_some_and(|value| value.as_u64() != Some(1))
    {
        return Err("n must be 1".into());
    }
    let stream = boolean(object, "stream", false)?;
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    let mut normalized = messages.clone();
    let mut images = Vec::new();
    for message in &mut normalized {
        if !message.is_object() {
            return Err("messages must contain objects".into());
        }
        normalize_tool_arguments(message)?;
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
                        let url = part
                            .get("image_url")
                            .and_then(|value| value.get("url"))
                            .and_then(Value::as_str)
                            .ok_or("image_url.url must be a string")?;
                        let data = url
                            .strip_prefix("data:image/png;base64,")
                            .ok_or("image input requires a base64 PNG data URL")?;
                        // The HTTP body bound applies before base64 or PNG allocation.
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(data)
                            .map_err(|error| format!("invalid image base64: {error}"))?;
                        images.push(processor.decode_png(&bytes)?);
                        *part = serde_json::json!({"type":"image"});
                    }
                    _ => {}
                }
            }
        }
    }
    let mut options = match object.get("chat_template_kwargs") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(options)) => options.clone(),
        _ => return Err("chat_template_kwargs must be an object".into()),
    };
    for name in [
        "tools",
        "documents",
        "enable_thinking",
        "reasoning_effort",
        "continue_final_message",
        "add_generation_prompt",
    ] {
        if let Some(value) = object.get(name) {
            options.insert(name.into(), value.clone());
        }
    }
    let grammar = compile_grammar(compiler, object)?;
    if tools_disabled {
        options.insert("tools".into(), Value::Array(vec![]));
    }
    if grammar.is_some() {
        if options
            .get("tools")
            .or_else(|| frontend.template_default("tools"))
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty())
        {
            return Err("structured output cannot be combined with tool-call generation".into());
        }
        if options.get("enable_thinking") == Some(&Value::Bool(true)) {
            return Err("structured output requires enable_thinking=false".into());
        }
        options.insert("enable_thinking".into(), Value::Bool(false));
    }
    if !options.contains_key("add_generation_prompt")
        && requests_continuation(options.get("continue_final_message"))
    {
        options.insert("add_generation_prompt".into(), Value::Bool(false));
    }
    let rendered = frontend.render_chat(&normalized, &options)?;
    let output = crate::model::prepare_output(
        frontend,
        &rendered,
        &normalized,
        &options,
        grammar.is_some(),
    );
    let (tokens, positions, vision_token_indices, conditioning, images) = if images.is_empty() {
        if rendered
            .token_ids
            .iter()
            .any(|id| *id == 248056 || *id == 248057)
        {
            return Err("image/video placeholders require matching supported image inputs".into());
        }
        let positions = (0..rendered.token_ids.len())
            .map(|position| [position as i64; 3])
            .collect();
        let indices = vec![-1; rendered.token_ids.len()];
        (rendered.token_ids, positions, indices, Vec::new(), None)
    } else {
        let processed = processor.process(&images)?;
        let expanded = processor.expand_prompt(frontend, &rendered.text, &processed)?;
        let positions = (0..expanded.token_ids.len())
            .map(|position| {
                [
                    expanded.position_ids[1][position],
                    expanded.position_ids[2][position],
                    expanded.position_ids[3][position],
                ]
            })
            .collect();
        let mut conditioning = Vec::with_capacity(images.len());
        let mut image = 0;
        use sha2::{Digest, Sha256};
        for (position, kind) in expanded.mm_token_type_ids.iter().enumerate() {
            if *kind == 1 && (position == 0 || expanded.mm_token_type_ids[position - 1] != 1) {
                let source = &images[image];
                let mut hash = Sha256::new();
                hash.update(b"sparkgap-rgb-image-v1\0");
                hash.update((source.width() as u64).to_le_bytes());
                hash.update((source.height() as u64).to_le_bytes());
                hash.update(source.pixels());
                conditioning.push(ConditioningEvent {
                    position: position as u64,
                    identity: hash.finalize().into(),
                });
                image += 1;
            }
        }
        if image != images.len() {
            return Err("expanded image spans do not match input image count".into());
        }
        let mut index = 0;
        let indices = expanded
            .mm_token_type_ids
            .iter()
            .map(|kind| {
                if *kind == 1 {
                    let value = index;
                    index += 1;
                    value
                } else {
                    -1
                }
            })
            .collect();
        (
            expanded.token_ids,
            positions,
            indices,
            conditioning,
            Some(processed),
        )
    };
    let sampling = sampling_parameters(object, &config.sampling_defaults)?;
    let mut stop_sequences: Vec<Vec<u32>> = object
        .get("stop_token_sequences")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid stop_token_sequences: {error}"))
        })
        .transpose()?
        .unwrap_or_default();
    if let Some(value) = object.get("stop_token_ids") {
        let ids: Vec<u32> = serde_json::from_value(value.clone())
            .map_err(|error| format!("invalid stop_token_ids: {error}"))?;
        stop_sequences.extend(ids.into_iter().map(|id| vec![id]));
    }
    super::stops::TokenStops::new(stop_sequences.clone())?;
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
    let max_new_tokens = u32::try_from(if object.contains_key("max_completion_tokens") {
        integer(
            object,
            "max_completion_tokens",
            config.max_new_tokens as u64,
        )?
    } else {
        integer(object, "max_tokens", config.max_new_tokens as u64)?
    })
    .map_err(|_| "output limit exceeds uint32")?;
    if max_new_tokens == 0
        || max_new_tokens > i32::MAX as u32
        || tokens
            .len()
            .checked_add(max_new_tokens as usize)
            .is_none_or(|count| count > config.limits.max_sequence_tokens)
    {
        return Err("prompt plus output limit exceeds prepared context capacity".into());
    }
    Ok(PreparedRequest {
        prompt: Prompt {
            tokens,
            positions,
            vision_token_indices,
            images,
            conditioning,
            sampling,
            max_new_tokens,
            stop_sequences,
            grammar,
        },
        stream,
        output,
        rendered_prompt: rendered.text,
        observation,
    })
}

fn requests_continuation(value: Option<&Value>) -> bool {
    value.is_some_and(|value| {
        value == &Value::Bool(true) || value.as_str().is_some_and(|field| !field.is_empty())
    })
}

fn normalize_tool_arguments(message: &mut Value) -> Result<()> {
    let Some(calls) = message
        .get_mut("tool_calls")
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };
    for call in calls
        .as_array_mut()
        .ok_or("message tool_calls must be an array")?
    {
        let function = call
            .get_mut("function")
            .and_then(Value::as_object_mut)
            .ok_or("tool call function must be an object")?;
        let arguments = function
            .get_mut("arguments")
            .ok_or("tool call function.arguments is required")?;
        if let Some(encoded) = arguments.as_str() {
            *arguments = serde_json::from_str(encoded)
                .map_err(|error| format!("invalid tool call arguments JSON: {error}"))?;
        }
        if !arguments.is_object() {
            return Err("tool call function.arguments must encode a JSON object".into());
        }
    }
    Ok(())
}

fn compile_grammar(
    compiler: &crate::grammar::GrammarCompiler,
    object: &Map<String, Value>,
) -> Result<Option<crate::grammar::RequestGrammar>> {
    let response = object
        .get("response_format")
        .filter(|value| !value.is_null());
    let lark = object.get("grammar").filter(|value| !value.is_null());
    if response.is_some() && lark.is_some() {
        return Err("response_format and grammar are mutually exclusive".into());
    }
    if let Some(lark) = lark {
        return compiler
            .compile_lark(lark.as_str().ok_or("grammar must be a Lark string")?)
            .map(Some);
    }
    let Some(response) = response else {
        return Ok(None);
    };
    match response.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => compiler
            .compile_json_schema(&serde_json::json!({"type":"object"}))
            .map(Some),
        Some("json_schema") => {
            let schema = response
                .get("json_schema")
                .and_then(|value| value.get("schema"))
                .ok_or("response_format.json_schema.schema is required")?;
            compiler.compile_json_schema(schema).map(Some)
        }
        _ => Err("response_format.type must be text, json_object or json_schema".into()),
    }
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

fn boolean(object: &Map<String, Value>, name: &str, default: bool) -> Result<bool> {
    object
        .get(name)
        .map(|v| v.as_bool().ok_or_else(|| format!("{name} must be boolean")))
        .unwrap_or(Ok(default))
}
fn integer(object: &Map<String, Value>, name: &str, default: u64) -> Result<u64> {
    object
        .get(name)
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| format!("{name} must be a nonnegative integer"))
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
                .ok_or_else(|| format!("{name} must be finite float32"))
        })
        .unwrap_or(Ok(default))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn response_tool_arguments_round_trip_without_changing_json_types() {
        let expected = serde_json::json!({"query":"café 世界","options":{"count":2,"enabled":true,"none":null,"items":[1,"✓"]}});
        let mut message = serde_json::json!({"role":"assistant","tool_calls":[{"type":"function","id":"call_1","function":{"name":"search","arguments":expected.to_string()}}]});
        normalize_tool_arguments(&mut message).unwrap();
        assert_eq!(message["tool_calls"][0]["function"]["arguments"], expected);
        assert_eq!(message["tool_calls"][0]["id"], "call_1");
        normalize_tool_arguments(&mut message).unwrap();
        for invalid in ["{", "null", "[]", "\"text\"", "1"] {
            let mut message =
                serde_json::json!({"tool_calls":[{"function":{"arguments":invalid}}]});
            assert!(normalize_tool_arguments(&mut message).is_err());
        }
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
}
