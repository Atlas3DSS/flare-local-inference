//! Qwen reasoning delimiters and Coder function-tag serialization.

pub use crate::output::tagged::OutputConfig;
use crate::{
    Result,
    frontend::{Frontend, RenderedPrompt},
    output::tagged::{TaggedOutput, ToolDialect, parameter_value},
};
use serde_json::{Map, Value};

pub struct QwenCoder;
impl ToolDialect for QwenCoder {
    const NAME: &'static str = "Qwen Coder";
    const TRIM_NEWLINES: bool = true;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";
    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        Ok(vec![parse_tool(body, tools)?])
    }
}
pub type QwenOutput = TaggedOutput<QwenCoder>;

/// Initialize from the open assistant envelope, never an earlier message body.
pub fn prepare_output(
    frontend: &Frontend,
    rendered: &RenderedPrompt,
    messages: &[Value],
    options: &Map<String, Value>,
    structured: bool,
) -> OutputConfig {
    let effective = |name: &str| {
        options
            .get(name)
            .or_else(|| frontend.template_default(name))
    };
    let continuation = effective("continue_final_message")
        .is_some_and(|v| v == &Value::Bool(true) || v.as_str().is_some_and(|s| !s.is_empty()));
    let generation = effective("add_generation_prompt") == Some(&Value::Bool(true));
    let assistant_continuation =
        continuation && messages.last().is_some_and(|m| m["role"] == "assistant");
    let prefix = if generation || assistant_continuation {
        rendered
            .text
            .rsplit_once("<|im_start|>assistant\n")
            .map(|(_, suffix)| suffix)
            .filter(|suffix| !suffix.contains("<|im_end|>"))
    } else {
        None
    };
    let reasoning = prefix.is_some_and(|text| {
        text.rfind("<think>")
            .is_some_and(|open| text.rfind("</think>").is_none_or(|close| open > close))
    });
    OutputConfig {
        verbatim: structured,
        reasoning,
        tools: effective("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    }
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
        serde_json::to_string(&arguments).map_err(|error| {
            crate::FrontendError::Input(crate::Failure::caused_by(error.to_string(), error))
        })?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;
    struct Harness {
        parser: QwenOutput,
        output: Accumulator,
    }
    impl Harness {
        fn new(config: OutputConfig, id: String) -> Self {
            Self {
                parser: QwenOutput::new(config),
                output: Accumulator::new(id),
            }
        }
        fn push(&mut self, text: &str) -> Result<Vec<Value>> {
            self.output.apply(self.parser.push(Input::Text(text))?)
        }
        fn finish(&mut self) -> Result<Vec<Value>> {
            let result = self.output.apply(self.parser.finish(Termination::Eos)?)?;
            self.output.finish()?;
            Ok(result)
        }
        fn message(&self) -> Value {
            self.output.message()
        }
        fn has_tools(&self) -> bool {
            self.output.has_tools()
        }
    }

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
        let mut whole = Harness::new(config(), "x".into());
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
            let mut parser = Harness::new(config(), "x".into());
            parser.push(&sample()[..split]).unwrap();
            parser.push(&sample()[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.message(), expected, "split {split}");
        }
        let mut parser = Harness::new(config(), "x".into());
        for ch in sample().chars() {
            parser.push(&ch.to_string()).unwrap();
        }
        parser.finish().unwrap();
        assert_eq!(parser.message(), expected);
    }
    #[test]
    fn incomplete_unknown_and_duplicate_calls_fail() {
        let mut parser = Harness::new(
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
        let mut parser = Harness::new(
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
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/frontend/qwen38.json"
        )))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == "tool_calls")
            .unwrap();
        let rendered = case["rendered"].as_str().unwrap();
        let assistant = rendered.rsplit_once("<|im_start|>assistant\n").unwrap().1;
        let end = assistant.find("</tool_call>").unwrap() + "</tool_call>".len();
        let mut parser = Harness::new(
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
            let mut parser = Harness::new(
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
    #[test]
    fn terminal_cause_preserves_incomplete_call_errors_and_never_fabricates_calls() {
        for cause in [Termination::Eos, Termination::OutputLimit] {
            let mut parser = QwenOutput::new(OutputConfig {
                reasoning: false,
                ..config()
            });
            assert!(
                parser
                    .push(Input::Text("<tool_call><function=lookup>"))
                    .unwrap()
                    .is_empty()
            );
            assert!(parser.finish(cause).is_err());
        }
        for cause in [
            Termination::Cancelled,
            Termination::ExecutionError,
            Termination::UserStop,
        ] {
            let mut parser = QwenOutput::new(OutputConfig {
                reasoning: false,
                ..config()
            });
            parser
                .push(Input::Text("<tool_call><function=lookup>"))
                .unwrap();
            assert!(parser.finish(cause).unwrap().is_empty());
        }
    }

    #[test]
    fn initial_reasoning_uses_only_an_open_assistant_prefix() {
        let frontend = crate::server::output::tests::frontend(&["x"]);
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/frontend/qwen38.json"
        )))
        .unwrap();
        for (name, reasoning) in [
            ("generation_prompt", true),
            ("thinking_disabled", false),
            ("assistant_continuation", false),
            ("reasoning_continuation", true),
            ("continuation_trailing_spaces", false),
            ("continuation_text_blocks", false),
        ] {
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == name)
                .unwrap();
            let rendered = RenderedPrompt {
                text: case["rendered"].as_str().unwrap().into(),
                token_ids: vec![],
            };
            let output = prepare_output(
                &frontend,
                &rendered,
                case["messages"].as_array().unwrap(),
                case["options"].as_object().unwrap(),
                false,
            );
            assert_eq!(output.reasoning, reasoning, "{name}");
        }
        for role in ["user", "tool"] {
            let messages = vec![json!({"role":role,"content":"literal <think>"})];
            for (options, suffix, expected) in [
                (json!({}), "", false),
                (
                    json!({"add_generation_prompt":true}),
                    "<|im_start|>assistant\n",
                    false,
                ),
                (
                    json!({"add_generation_prompt":true}),
                    "<|im_start|>assistant\n<think>\n",
                    true,
                ),
            ] {
                let rendered = RenderedPrompt {
                    text: format!("<|im_start|>{role}\nliteral <think><|im_end|>\n{suffix}"),
                    token_ids: vec![],
                };
                assert_eq!(
                    prepare_output(
                        &frontend,
                        &rendered,
                        &messages,
                        options.as_object().unwrap(),
                        false
                    )
                    .reasoning,
                    expected
                );
            }
        }
        let messages = vec![json!({"role":"user","content":"<think>"})];
        let rendered = RenderedPrompt {
            text: "<|im_start|>user\n<think>".into(),
            token_ids: vec![],
        };
        assert!(
            !prepare_output(
                &frontend,
                &rendered,
                &messages,
                json!({"continue_final_message":true}).as_object().unwrap(),
                false
            )
            .reasoning
        );
    }
}
