use super::{Decoded, Protocol};
use crate::{
    Result,
    server::{inference::*, observation::ObservationOptions},
};
use serde_json::{Value, json};

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string").into())
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a [Value]> {
    v.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{key} must be an array").into())
}
fn optional_id(v: &Value, key: &str) -> Result<Option<String>> {
    v.get(key)
        .filter(|v| !v.is_null())
        .map(|_| field(v, key).map(str::to_owned))
        .transpose()
}
fn flag(v: &Value, key: &str, default: bool) -> Result<bool> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(v)) => Ok(*v),
        _ => Err(format!("{key} must be boolean").into()),
    }
}
fn present(v: &Value, key: &str) -> bool {
    v.get(key).is_some_and(|v| !v.is_null())
}
fn reject(v: &Value, keys: &[&str]) -> Result<()> {
    for key in keys {
        if present(v, key) && v[key] != false {
            return Err(unsupported(format!("{key} is not supported")));
        }
    }
    Ok(())
}
fn strings(v: Option<&Value>) -> Result<Vec<String>> {
    match v {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "stop strings must be strings".into())
            })
            .collect(),
        _ => Err("stops must be a string or an array of strings".into()),
    }
}
fn arguments(v: &Value) -> Result<Value> {
    let value = if let Some(text) = v.as_str() {
        serde_json::from_str(text).map_err(|e| format!("invalid tool arguments JSON: {e}"))?
    } else {
        v.clone()
    };
    if !value.is_object() {
        return Err("tool arguments must be a JSON object".into());
    }
    Ok(value)
}
fn qualified(name: &str, namespace: Option<&str>) -> String {
    namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}__{name}"))
}
fn plain_parts(value: &Value, protocol: Protocol) -> Result<Vec<Part>> {
    if value.is_null() {
        return Ok(vec![]);
    }
    if let Some(text) = value.as_str() {
        return Ok(vec![Part::Text(text.into())]);
    }
    value
        .as_array()
        .ok_or(if protocol == Protocol::Chat {
            "Unexpected content type."
        } else {
            "content must be text or an array"
        })?
        .iter()
        .map(|part| match field(part, "type")? {
            "text" | "input_text" | "output_text" => Ok(Part::Text(field(part, "text")?.into())),
            "image_url" => Ok(Part::Image(field(&part["image_url"], "url")?.into())),
            "input_image" => Ok(Part::Image(field(part, "image_url")?.into())),
            "image" if protocol == Protocol::Messages => {
                let source = &part["source"];
                if field(source, "type")? != "base64" {
                    return Err(unsupported("images require base64 PNG or JPEG"));
                }
                let media = field(source, "media_type")?;
                if !["image/png", "image/jpeg"].contains(&media) {
                    return Err(unsupported("images require PNG or JPEG"));
                }
                Ok(Part::Image(format!(
                    "data:{media};base64,{}",
                    field(source, "data")?
                )))
            }
            kind => Err(unsupported(if protocol == Protocol::Chat {
                "Unexpected item type in content.".into()
            } else {
                format!("unsupported content type {kind}")
            })),
        })
        .collect()
}
fn thinking(text: &str, signature: Option<&str>) -> Result<Part> {
    if let Some(signature) = signature.filter(|s| s.starts_with("flare-thinking-v1:")) {
        if signature != super::encode::thinking_signature(text) {
            return Err("thinking marker does not match text".into());
        }
    }
    Ok(Part::Reasoning(text.into()))
}
fn chat_messages(body: &Value) -> Result<Vec<Message>> {
    array(body, "messages")?
        .iter()
        .map(|v| {
            let role = serde_json::from_value(v["role"].clone())
                .map_err(|_| "unsupported message role")?;
            let mut parts = vec![];
            if let Some(text) = v.get("reasoning_content").filter(|v| !v.is_null()) {
                parts.push(Part::Reasoning(
                    text.as_str()
                        .ok_or("reasoning_content must be text")?
                        .into(),
                ));
            }
            if role == Role::Tool {
                parts.push(Part::ToolResult {
                    id: optional_id(v, "tool_call_id")?,
                    content: plain_parts(&v["content"], Protocol::Chat)?,
                    is_error: flag(v, "is_error", false)?,
                });
            } else {
                parts.extend(plain_parts(&v["content"], Protocol::Chat)?);
                if present(v, "tool_calls") {
                    for call in array(v, "tool_calls")? {
                        parts.push(Part::ToolCall {
                            id: optional_id(call, "id")?,
                            name: field(&call["function"], "name")?.into(),
                            arguments: arguments(&call["function"]["arguments"])?,
                        });
                    }
                }
            }
            Ok(Message {
                role,
                parts,
                phase: v.get("phase").and_then(Value::as_str).map(str::to_owned),
            })
        })
        .collect()
}
fn anthropic_messages(body: &Value) -> Result<Vec<Message>> {
    let mut messages = vec![];
    if present(body, "system") {
        messages.push(Message {
            role: Role::System,
            parts: plain_parts(&body["system"], Protocol::Messages)?,
            phase: None,
        });
    }
    for v in array(body, "messages")? {
        let role = match field(v, "role")? {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => return Err("Messages roles must be user or assistant".into()),
        };
        let mut parts = vec![];
        if let Some(blocks) = v["content"].as_array() {
            for block in blocks {
                parts.extend(match field(block, "type")? {
                    "thinking" => vec![thinking(
                        field(block, "thinking")?,
                        block.get("signature").and_then(Value::as_str),
                    )?],
                    "tool_use" => vec![Part::ToolCall {
                        id: Some(field(block, "id")?.into()),
                        name: field(block, "name")?.into(),
                        arguments: arguments(&block["input"])?,
                    }],
                    "tool_result" => vec![Part::ToolResult {
                        id: Some(field(block, "tool_use_id")?.into()),
                        content: plain_parts(&block["content"], Protocol::Messages)?,
                        is_error: flag(block, "is_error", false)?,
                    }],
                    _ => plain_parts(&json!([block]), Protocol::Messages)?,
                });
            }
        } else {
            parts = plain_parts(&v["content"], Protocol::Messages)?;
        }
        messages.push(Message {
            role,
            parts,
            phase: None,
        });
    }
    Ok(messages)
}
fn responses_messages(body: &Value) -> Result<Vec<Message>> {
    let mut messages = vec![];
    if present(body, "instructions") {
        messages.push(Message {
            role: Role::System,
            parts: vec![Part::Text(field(body, "instructions")?.into())],
            phase: None,
        });
    }
    let items = if let Some(text) = body["input"].as_str() {
        vec![json!({"role":"user","content":text})]
    } else {
        array(body, "input")?.to_vec()
    };
    for item in items {
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        let (role, parts) = match kind {
            "message" => (
                serde_json::from_value(item["role"].clone()).map_err(|_| "invalid input role")?,
                plain_parts(&item["content"], Protocol::Responses)?,
            ),
            "function_call" => (
                Role::Assistant,
                vec![Part::ToolCall {
                    id: Some(field(&item, "call_id")?.into()),
                    name: qualified(
                        field(&item, "name")?,
                        item.get("namespace").and_then(Value::as_str),
                    ),
                    arguments: arguments(&item["arguments"])?,
                }],
            ),
            "custom_tool_call" => (
                Role::Assistant,
                vec![Part::ToolCall {
                    id: Some(field(&item, "call_id")?.into()),
                    name: qualified(
                        field(&item, "name")?,
                        item.get("namespace").and_then(Value::as_str),
                    ),
                    arguments: json!({"input":field(&item,"input")?}),
                }],
            ),
            "function_call_output" | "custom_tool_call_output" => (
                Role::Tool,
                vec![Part::ToolResult {
                    id: Some(field(&item, "call_id")?.into()),
                    content: plain_parts(&item["output"], Protocol::Responses)?,
                    is_error: false,
                }],
            ),
            "reasoning" => {
                let source = item
                    .get("content")
                    .and_then(Value::as_array)
                    .filter(|v| !v.is_empty())
                    .or_else(|| item.get("summary").and_then(Value::as_array));
                let mut parts = vec![];
                if let Some(source) = source {
                    for part in source {
                        let text = field(part, "text")?;
                        if !text.is_empty() {
                            parts.push(Part::Reasoning(text.into()));
                        }
                    }
                }
                if parts.is_empty() && present(&item, "encrypted_content") {
                    return Err(unsupported(
                        "encrypted-only reasoning history cannot be rendered",
                    ));
                }
                (Role::Assistant, parts)
            }
            _ => {
                return Err(unsupported(format!(
                    "unsupported Responses input item {kind}"
                )));
            }
        };
        // Reasoning and tool calls belong to the same assistant turn as its text.
        if role == Role::Assistant && messages.last().is_some_and(|m| m.role == Role::Assistant) {
            messages.last_mut().unwrap().parts.extend(parts);
        } else {
            messages.push(Message {
                role,
                parts,
                phase: item.get("phase").and_then(Value::as_str).map(str::to_owned),
            });
        }
    }
    Ok(messages)
}
fn tool(v: &Value, protocol: Protocol, namespace: Option<&str>) -> Result<Tool> {
    if protocol == Protocol::Messages && present(v, "type") && v["type"] != "custom" {
        return Err(unsupported(
            "only client tools are supported; hosted tool types cannot execute locally",
        ));
    }
    let custom = protocol == Protocol::Responses && v["type"] == "custom";
    let definition = if protocol == Protocol::Chat {
        &v["function"]
    } else {
        v
    };
    if flag(definition, "strict", false)? {
        return Err(unsupported(
            "strict tool schemas require constrained tool generation",
        ));
    }
    let grammar = if custom && present(v, "format") && v["format"]["type"] != "text" {
        if v["format"]["type"] != "grammar"
            || !["lark", "regex"].contains(&field(&v["format"], "syntax")?)
        {
            return Err(unsupported(
                "custom tool format must be text, Lark, or regex",
            ));
        }
        Some(ToolGrammar {
            syntax: field(&v["format"], "syntax")?.into(),
            source: field(&v["format"], "definition")?.into(),
        })
    } else {
        None
    };
    if protocol != Protocol::Messages && !custom && v["type"] != "function" {
        return Err(unsupported(
            "only client function and text custom tools are supported",
        ));
    }
    let parameters = if custom {
        json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"]})
    } else {
        definition
            .get(if protocol == Protocol::Messages {
                "input_schema"
            } else {
                "parameters"
            })
            .cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}}))
    };
    Ok(Tool {
        name: field(definition, "name")?.into(),
        namespace: namespace.map(str::to_owned),
        description: definition
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
        parameters,
        custom,
        grammar,
    })
}
fn options(body: &Value, protocol: Protocol) -> Result<Options> {
    let mut options = Options::default();
    options.sampling = serde_json::from_value(body.clone())
        .map_err(|e| format!("invalid sampling options: {e}"))?;
    let limit = match protocol {
        Protocol::Responses => body.get("max_output_tokens"),
        Protocol::Chat => body
            .get("max_completion_tokens")
            .filter(|v| !v.is_null())
            .or_else(|| body.get("max_tokens")),
        _ => body.get("max_tokens"),
    };
    options.output_limit = limit
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .ok_or("output limit must be a nonnegative integer")
        })
        .transpose()?;
    options.stops = strings(body.get(if protocol == Protocol::Messages {
        "stop_sequences"
    } else {
        "stop"
    }))?;
    if present(body, "stop_token_sequences") {
        options.token_stops = serde_json::from_value(body["stop_token_sequences"].clone())
            .map_err(|_| "invalid stop_token_sequences")?;
    }
    if present(body, "stop_token_ids") {
        let ids: Vec<u32> = serde_json::from_value(body["stop_token_ids"].clone())
            .map_err(|_| "invalid stop_token_ids")?;
        options
            .token_stops
            .extend(ids.into_iter().map(|id| vec![id]));
    }
    options.ignore_eos = flag(body, "ignore_eos", false)?;
    let format = match protocol {
        Protocol::Responses => &body["text"]["format"],
        Protocol::Messages => &body["output_config"]["format"],
        _ => &body["response_format"],
    };
    if !format.is_null() {
        options.constraint = match field(format, "type")? {
            "text" => None,
            "json_object" => Some(Constraint::Json(json!({"type":"object"}))),
            "json_schema" => Some(Constraint::Json(
                if protocol == Protocol::Chat {
                    &format["json_schema"]["schema"]
                } else {
                    &format["schema"]
                }
                .clone(),
            )),
            _ => return Err(unsupported("unsupported output format")),
        };
    }
    if present(body, "grammar") {
        if !format.is_null() {
            return Err("grammar and output format are mutually exclusive".into());
        }
        options.constraint = Some(Constraint::Lark(field(body, "grammar")?.into()));
    }
    let effort = if protocol == Protocol::Responses {
        body["reasoning"].get("effort")
    } else if protocol == Protocol::Messages {
        body["output_config"].get("effort")
    } else {
        None
    };
    if let Some(effort) = effort.filter(|v| !v.is_null()) {
        let native = match effort.as_str() {
            Some("none") => {
                options
                    .template
                    .insert("enable_thinking".into(), false.into());
                None
            }
            Some("minimal" | "low") => Some(Effort::Low),
            Some("medium" | "high") => Some(Effort::High),
            Some("xhigh" | "max") => Some(Effort::Max),
            _ => return Err("unsupported reasoning effort".into()),
        };
        if let Some(native) = native {
            options.effort = Some(native);
            if protocol == Protocol::Responses {
                options
                    .template
                    .insert("enable_thinking".into(), true.into());
            }
        }
    }
    if protocol == Protocol::Messages && present(body, "thinking") {
        let enabled = match field(&body["thinking"], "type")? {
            "disabled" => false,
            "enabled" | "adaptive" => true,
            _ => return Err("unsupported thinking mode".into()),
        };
        options
            .template
            .insert("enable_thinking".into(), enabled.into());
        if let Some(budget) = body["thinking"].get("budget_tokens") {
            let budget = budget
                .as_u64()
                .filter(|v| *v > 0)
                .ok_or("thinking budget must be positive")?;
            options.effort.get_or_insert(if budget <= 5000 {
                Effort::Low
            } else if budget <= 20000 {
                Effort::High
            } else {
                Effort::Max
            });
        }
    }
    for key in [
        "documents",
        "enable_thinking",
        "reasoning_effort",
        "continue_final_message",
        "add_generation_prompt",
    ] {
        if let Some(value) = body.get(key) {
            options.template.insert(key.into(), value.clone());
        }
    }
    if present(body, "chat_template_kwargs") {
        options.template.extend(
            body["chat_template_kwargs"]
                .as_object()
                .ok_or("chat_template_kwargs must be an object")?
                .clone(),
        );
    }
    Ok(options)
}
fn raw_prompts(value: &Value) -> Result<Vec<RawPrompt>> {
    if let Some(text) = value.as_str() {
        return Ok(vec![RawPrompt::Text(text.into())]);
    }
    let items = value
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("prompt must be text or a nonempty prompt/token array")?;
    if items[0].is_number() {
        return Ok(vec![RawPrompt::Tokens(
            serde_json::from_value(value.clone()).map_err(|_| "prompt token IDs must be uint32")?,
        )]);
    }
    if !items.iter().all(|v| v.is_string()) && !items.iter().all(|v| v.is_array()) {
        return Err("prompt batch must contain either only strings or only token arrays".into());
    }
    items
        .iter()
        .map(|v| {
            if let Some(text) = v.as_str() {
                Ok(RawPrompt::Text(text.into()))
            } else {
                let ids: Vec<u32> = serde_json::from_value(v.clone())
                    .map_err(|_| "prompt batch must contain only text or token arrays")?;
                if ids.is_empty() {
                    return Err("token prompts cannot be empty".into());
                }
                Ok(RawPrompt::Tokens(ids))
            }
        })
        .collect()
}
pub(crate) fn decode(protocol: Protocol, body: Value, count: bool) -> Result<Decoded> {
    body.as_object().ok_or("request must be a JSON object")?;
    reject(
        &body,
        &["logprobs", "top_logprobs", "logit_bias", "guided_json"],
    )?;
    if protocol == Protocol::Responses {
        reject(
            &body,
            &[
                "previous_response_id",
                "conversation",
                "background",
                "store",
            ],
        )?;
        if present(&body, "truncation") && body["truncation"] != "disabled" {
            return Err(unsupported("automatic truncation is not supported"));
        }
        if present(&body, "service_tier")
            && body["service_tier"] != "auto"
            && body["service_tier"] != "default"
        {
            return Err(unsupported("service tiers are not supported"));
        }
    }
    if protocol == Protocol::Completions {
        reject(&body, &["suffix"])?;
    }
    let choices = body
        .get("n")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or("n must be positive")
        })
        .transpose()?
        .unwrap_or(1);
    if protocol != Protocol::Completions && choices != 1 {
        return Err("n must be 1".into());
    }
    if present(&body, "best_of") && body["best_of"].as_u64() != Some(choices as u64) {
        return Err(unsupported(
            "best_of must equal n; ranked sampling is not supported",
        ));
    }
    let tools_disabled = match body.get("tool_choice").filter(|v| !v.is_null()) {
        None => false,
        Some(v) if v == "auto" || v["type"] == "auto" => false,
        Some(v) if v == "none" || v["type"] == "none" => true,
        _ => {
            return Err(unsupported(
                "tool_choice supports auto or none; forced tool selection is not implemented",
            ));
        }
    };
    if !flag(&body, "parallel_tool_calls", true)?
        || flag(&body["tool_choice"], "disable_parallel_tool_use", false)?
    {
        return Err(unsupported(
            "disabling parallel tool calls requires constrained tool generation",
        ));
    }
    reject(
        &body,
        &[
            "audio",
            "mcp_servers",
            "container",
            "prediction",
            "max_tool_calls",
        ],
    )?;
    if present(&body, "context_management") {
        let context = &body["context_management"];
        let preserve = protocol == Protocol::Messages
            && context.as_object().is_some_and(|v| v.len() == 1)
            && context["edits"].as_array().is_some_and(|edits| {
                edits
                    .iter()
                    .all(|edit| edit == &json!({"type":"clear_thinking_20251015","keep":"all"}))
            });
        if !preserve {
            return Err(unsupported(
                "context_management supports only preserving all thinking; server-side context edits are not supported",
            ));
        }
    }
    if protocol == Protocol::Completions {
        reject(&body, &["tools", "tool_choice"])?;
    }
    let mut tools = vec![];
    if present(&body, "tools") {
        for v in array(&body, "tools")? {
            if protocol == Protocol::Responses && v["type"] == "namespace" {
                let namespace = field(v, "name")?;
                for nested in array(v, "tools")? {
                    let mut t = tool(nested, protocol, Some(namespace))?;
                    if let Some(description) = v["description"].as_str() {
                        t.description = format!("{description}\n{}", t.description);
                    }
                    tools.push(t);
                }
            } else {
                tools.push(tool(v, protocol, None)?);
            }
        }
    }
    let mut options = options(&body, protocol)?;
    if protocol == Protocol::Messages && !count && options.output_limit.is_none() {
        return Err("max_tokens is required".into());
    }
    let input = match protocol {
        Protocol::Chat => Input::Conversation(chat_messages(&body)?),
        Protocol::Messages => {
            let messages = anthropic_messages(&body)?;
            if messages.last().is_some_and(|m| m.role == Role::Assistant) {
                options
                    .template
                    .entry("continue_final_message")
                    .or_insert(json!(true));
            }
            Input::Conversation(messages)
        }
        Protocol::Responses => Input::Conversation(responses_messages(&body)?),
        Protocol::Completions => Input::Raw(raw_prompts(&body["prompt"])?),
    };
    let request = Request {
        input,
        options,
        tools,
        tools_disabled,
        choices,
    };
    request.validate()?;
    let observation = ObservationOptions::parse(body.as_object().unwrap())?;
    if protocol != Protocol::Chat && (observation.token_ids || observation.timestamps) {
        return Err(unsupported(
            "token observation extensions require /v1/chat/completions",
        ));
    }
    Ok(Decoded {
        request,
        stream: flag(&body, "stream", false)?,
        echo: flag(&body, "echo", false)?,
        observation,
        metadata: body.get("metadata").cloned().unwrap_or_else(|| json!({})),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn string_stops(value: Option<&Value>) -> Result<Vec<String>> {
        let decoded = decode(
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"x"}],"stop":value}),
            false,
        )?;
        crate::server::stops::validate_strings(&decoded.request.options.stops, false, false)?;
        Ok(decoded.request.options.stops)
    }
    fn normalize_tool_arguments(message: &mut Value) -> Result<()> {
        let decoded = chat_messages(&json!({"messages":[message]}))?;
        *message = template_messages(&decoded)?.remove(0);
        Ok(())
    }
    #[test]
    fn standard_string_stop_values_are_validated_without_coercion() {
        assert!(string_stops(None).unwrap().is_empty());
        assert!(string_stops(Some(&Value::Null)).unwrap().is_empty());
        assert_eq!(
            string_stops(Some(&serde_json::json!("\nEND_EDIT"))).unwrap(),
            vec!["\nEND_EDIT"]
        );
        assert_eq!(
            string_stops(Some(&serde_json::json!(["a", "世界"]))).unwrap(),
            vec!["a", "世界"]
        );
        for value in [
            serde_json::json!(false),
            serde_json::json!(1),
            serde_json::json!(["ok", 2]),
            serde_json::json!(""),
            serde_json::json!(["a", "b", "c", "d", "e"]),
        ] {
            assert!(string_stops(Some(&value)).is_err(), "{value}");
        }
    }
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
}
