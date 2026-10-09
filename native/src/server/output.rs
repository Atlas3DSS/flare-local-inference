use crate::Result;
use serde_json::{Map, Value, json};

#[derive(Clone)]
pub struct OutputConfig {
    pub verbatim: bool,
    pub reasoning: bool,
    pub tools: Vec<Value>,
}

enum Mode {
    Verbatim,
    Content,
    Reasoning,
    Tool,
}

/// Checkpoint <think> and Qwen Coder function-tag presentation.
pub(super) struct OutputParser {
    mode: Mode,
    buffer: String,
    trim_newlines: usize,
    tools: Vec<Value>,
    id: String,
    content: String,
    reasoning: String,
    calls: Vec<Value>,
}

impl OutputParser {
    pub fn new(config: OutputConfig, id: String) -> Self {
        Self {
            mode: if config.verbatim {
                Mode::Verbatim
            } else if config.reasoning {
                Mode::Reasoning
            } else {
                Mode::Content
            },
            buffer: String::new(),
            trim_newlines: 0,
            tools: config.tools,
            id,
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
        }
    }
    pub fn has_tools(&self) -> bool {
        !self.calls.is_empty()
    }
    pub fn message(&self) -> Value {
        let mut message = json!({"role":"assistant","content":if self.content.is_empty(){Value::Null}else{Value::String(self.content.clone())}});
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(self.reasoning.clone());
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = Value::Array(self.calls.clone());
        }
        message
    }
    fn text(&mut self, text: String, reasoning: bool, output: &mut Vec<Value>) {
        if text.is_empty() {
            return;
        }
        if reasoning {
            self.reasoning.push_str(&text);
            output.push(json!({"reasoning_content":text}));
        } else {
            self.content.push_str(&text);
            output.push(json!({"content":text}));
        }
    }
    pub fn push(&mut self, text: &str) -> Result<Vec<Value>> {
        self.buffer.push_str(text);
        let mut output = Vec::new();
        loop {
            while self.trim_newlines > 0 {
                if self.buffer.is_empty() {
                    return Ok(output);
                }
                if self.buffer.starts_with('\n') {
                    self.buffer.remove(0);
                    self.trim_newlines -= 1;
                } else {
                    self.trim_newlines = 0;
                }
            }
            match self.mode {
                Mode::Verbatim => {
                    let text = std::mem::take(&mut self.buffer);
                    self.text(text, false, &mut output);
                    return Ok(output);
                }
                Mode::Tool => {
                    if self.buffer.len() > 1024 * 1024 {
                        return Err("tool call exceeds the 1 MiB parser bound".into());
                    }
                    let Some(end) = self.buffer.find("</tool_call>") else {
                        return Ok(output);
                    };
                    let (name, arguments) = parse_tool(&self.buffer[..end], &self.tools)?;
                    self.buffer.drain(..end + "</tool_call>".len());
                    let index = self.calls.len();
                    let id = format!("call_{}_{}", self.id, index);
                    let call = json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}});
                    self.calls.push(call.clone());
                    let mut delta = call;
                    delta["index"] = json!(index);
                    output.push(json!({"tool_calls":[delta]}));
                    self.mode = Mode::Content;
                }
                Mode::Reasoning => {
                    if let Some(end) = self.buffer.find("</think>") {
                        let text = self.buffer[..end]
                            .strip_suffix('\n')
                            .unwrap_or(&self.buffer[..end])
                            .to_owned();
                        self.text(text, true, &mut output);
                        self.buffer.drain(..end + "</think>".len());
                        self.mode = Mode::Content;
                        self.trim_newlines = 2;
                    } else {
                        let hold = partial_marker(&self.buffer, &["</think>", "\n</think>"]);
                        let visible = self.buffer.len() - hold;
                        let text = self.buffer.drain(..visible).collect();
                        self.text(text, true, &mut output);
                        return Ok(output);
                    }
                }
                Mode::Content => {
                    let mut marker = self.buffer.find("<think>").map(|at| (at, false));
                    if !self.tools.is_empty() {
                        if let Some(at) = self.buffer.find("<tool_call>") {
                            if marker.is_none_or(|(first, _)| at < first) {
                                marker = Some((at, true));
                            }
                        }
                    }
                    if let Some((at, tool)) = marker {
                        let prefix = &self.buffer[..at];
                        let prefix = if tool {
                            if self.calls.is_empty() {
                                prefix.strip_suffix("\n\n").unwrap_or(prefix)
                            } else {
                                prefix.strip_suffix('\n').unwrap_or(prefix)
                            }
                        } else {
                            prefix
                        };
                        let text = prefix.to_owned();
                        self.text(text, false, &mut output);
                        self.buffer.drain(
                            ..at + if tool {
                                "<tool_call>".len()
                            } else {
                                "<think>".len()
                            },
                        );
                        self.mode = if tool { Mode::Tool } else { Mode::Reasoning };
                        if !tool {
                            self.trim_newlines = 1;
                        }
                    } else {
                        let markers: &[&str] = if self.tools.is_empty() {
                            &["<think>"]
                        } else {
                            &["<think>", "<tool_call>", "\n\n<tool_call>", "\n<tool_call>"]
                        };
                        let hold = partial_marker(&self.buffer, markers);
                        let visible = self.buffer.len() - hold;
                        let text = self.buffer.drain(..visible).collect();
                        self.text(text, false, &mut output);
                        return Ok(output);
                    }
                }
            }
        }
    }
    pub fn finish(&mut self) -> Result<Vec<Value>> {
        let mut output = self.push("")?;
        if matches!(self.mode, Mode::Tool) {
            return Err("incomplete Qwen Coder tool call".into());
        }
        let text = std::mem::take(&mut self.buffer);
        self.text(text, matches!(self.mode, Mode::Reasoning), &mut output);
        Ok(output)
    }
}

fn partial_marker(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| {
            (1..marker.len())
                .rev()
                .filter(|length| text.ends_with(&marker[..*length]))
                .take(1)
        })
        .max()
        .unwrap_or(0)
}

fn parse_tool(body: &str, tools: &[Value]) -> Result<(String, String)> {
    let body = body
        .trim()
        .strip_prefix("<function=")
        .ok_or("tool call requires a function tag")?;
    let (name, mut rest) = body.split_once('>').ok_or("unterminated function name")?;
    if name.is_empty() {
        return Err("tool function name is empty".into());
    }
    let function = tools
        .iter()
        .filter_map(|tool| tool.get("function"))
        .find(|function| function.get("name").and_then(Value::as_str) == Some(name))
        .ok_or_else(|| format!("model called unregistered tool {name:?}"))?;
    let properties = function
        .get("parameters")
        .and_then(|parameters| parameters.get("properties"))
        .and_then(Value::as_object);
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix("</function>") {
            if !tail.trim().is_empty() {
                return Err("unexpected content after tool function".into());
            }
            break;
        }
        rest = rest
            .strip_prefix("<parameter=")
            .ok_or("tool function requires parameter or closing tag")?;
        let (parameter, tail) = rest
            .split_once('>')
            .ok_or("unterminated tool parameter name")?;
        if parameter.is_empty() || arguments.contains_key(parameter) {
            return Err("tool parameter names must be nonempty and unique".into());
        }
        let (value, tail) = tail
            .split_once("</parameter>")
            .ok_or("unterminated tool parameter value")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let schema = properties.and_then(|properties| properties.get(parameter));
        arguments.insert(parameter.into(), parameter_value(value, schema)?);
        rest = tail;
    }
    Ok((
        name.into(),
        serde_json::to_string(&arguments).map_err(|error| error.to_string())?,
    ))
}

fn parameter_value(raw: &str, schema: Option<&Value>) -> Result<Value> {
    let Some(schema) = schema else {
        return Ok(Value::String(raw.into()));
    };
    let types = schema.get("type");
    if types == Some(&Value::String("string".into())) {
        return Ok(Value::String(raw.into()));
    }
    if types.is_none() {
        for key in ["anyOf", "oneOf"] {
            if let Some(branches) = schema.get(key).and_then(Value::as_array) {
                for branch in branches {
                    if let Ok(value) = parameter_value(raw, Some(branch)) {
                        return Ok(value);
                    }
                }
                return Err("tool parameter does not match any declared type".into());
            }
        }
        return Ok(Value::String(raw.into()));
    }
    let kinds: Vec<_> = match types {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => return Err("invalid tool parameter type declaration".into()),
    };
    let parsed: std::result::Result<Value, _> = serde_json::from_str(raw);
    for kind in kinds {
        if kind == "string" {
            return Ok(Value::String(raw.into()));
        }
        if let Ok(value) = &parsed {
            let valid = match kind {
                "null" => value.is_null(),
                "boolean" => value.is_boolean(),
                "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
                "number" => value.is_number(),
                "array" => value.is_array(),
                "object" => value.is_object(),
                _ => false,
            };
            if valid {
                return Ok(value.clone());
            }
        }
    }
    Err("tool parameter value does not match its declared JSON type".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> OutputConfig {
        OutputConfig {
            verbatim: false,
            reasoning: true,
            tools: vec![
                json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"city":{"type":"string"},"n":{"type":"integer"},"data":{"type":"object"}}}}}),
            ],
        }
    }
    fn sample() -> &'static str {
        "café 世界\n</think>\n\nSearching.\n\n<tool_call>\n<function=lookup>\n<parameter=city>\n Paris \n</parameter>\n<parameter=n>\n3\n</parameter>\n<parameter=data>\n{\"flag\":true,\"nothing\":null}\n</parameter>\n</function>\n</tool_call>"
    }
    #[test]
    fn every_utf8_split_preserves_reasoning_tools_and_parameter_types() {
        let mut whole = OutputParser::new(config(), "x".into());
        whole.push(sample()).unwrap();
        whole.finish().unwrap();
        let expected = whole.message();
        assert_eq!(expected["reasoning_content"], "café 世界");
        assert_eq!(expected["content"], "Searching.");
        let args: Value = serde_json::from_str(
            expected["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            args,
            json!({"city":" Paris ","n":3,"data":{"flag":true,"nothing":null}})
        );
        for split in sample().char_indices().map(|(at, _)| at) {
            let mut parser = OutputParser::new(config(), "x".into());
            parser.push(&sample()[..split]).unwrap();
            parser.push(&sample()[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.message(), expected, "split {split}");
        }
        let mut parser = OutputParser::new(config(), "x".into());
        for ch in sample().chars() {
            parser.push(&ch.to_string()).unwrap();
        }
        parser.finish().unwrap();
        assert_eq!(parser.message(), expected);
    }
    #[test]
    fn incomplete_unknown_and_duplicate_calls_fail() {
        let mut parser = OutputParser::new(
            OutputConfig {
                verbatim: false,
                reasoning: false,
                ..config()
            },
            "x".into(),
        );
        parser.push("<tool_call><function=lookup>").unwrap();
        assert!(parser.finish().is_err());
        assert!(parse_tool("<function=unknown></function>", &config().tools).is_err());
        assert!(
            parse_tool(
                "<function=lookup><parameter=n>1</parameter><parameter=n>2</parameter></function>",
                &config().tools
            )
            .is_err()
        );
    }
    #[test]
    fn tag_like_text_is_preserved_without_registered_tools() {
        let mut parser = OutputParser::new(
            OutputConfig {
                verbatim: false,
                reasoning: false,
                tools: vec![],
            },
            "x".into(),
        );
        parser.push("Use <tool_call> as an example.\n").unwrap();
        parser.finish().unwrap();
        assert_eq!(
            parser.message()["content"],
            "Use <tool_call> as an example.\n"
        );
    }
    #[test]
    fn canonical_checkpoint_template_tool_record_preserves_nested_arguments() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/frontend/qwen38.json")).unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == "tool_calls")
            .unwrap();
        let rendered = case["rendered"].as_str().unwrap();
        let assistant = rendered.rsplit_once("<|im_start|>assistant\n").unwrap().1;
        let end = assistant.find("</tool_call>").unwrap() + "</tool_call>".len();
        let mut parser = OutputParser::new(
            OutputConfig {
                reasoning: false,
                tools: case["options"]["tools"].as_array().unwrap().clone(),
                verbatim: false,
            },
            "canonical".into(),
        );
        for character in assistant[..end].chars() {
            parser.push(&character.to_string()).unwrap();
        }
        parser.finish().unwrap();
        let message = parser.message();
        assert_eq!(message["reasoning_content"], "Need two sources.");
        let arguments: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let expected = &case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message.get("tool_calls").is_some())
            .unwrap()["tool_calls"][0]["function"]["arguments"];
        assert_eq!(&arguments, expected);
    }
    #[test]
    fn structured_content_preserves_literal_reasoning_and_tool_tags() {
        let text = r#"{"literal":"<think>café 世界</think><tool_call><function=lookup></function></tool_call>"}"#;
        for split in text.char_indices().map(|(index, _)| index) {
            let mut parser = OutputParser::new(
                OutputConfig {
                    verbatim: true,
                    ..config()
                },
                "json".into(),
            );
            parser.push(&text[..split]).unwrap();
            parser.push(&text[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.message()["content"], text);
            assert!(parser.message().get("reasoning_content").is_none());
            assert!(!parser.has_tools());
        }
    }
}
