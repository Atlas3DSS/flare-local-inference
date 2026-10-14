//! GLM think envelopes and raw-string/JSON argument tags from the chat template.

use crate::{
    Result,
    frontend::{Frontend, RenderedPrompt},
    output::tagged::{TaggedOutput, ToolDialect, parameter_value},
};
use serde_json::{Map, Value};

pub use crate::output::tagged::OutputConfig;

pub struct GlmArgumentTags;
impl ToolDialect for GlmArgumentTags {
    const NAME: &'static str = "GLM";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let body = body.trim();
        let split = body.find("<arg_key>").unwrap_or(body.len());
        let name = body[..split].trim();
        let mut rest = &body[split..];
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
        while !rest.trim().is_empty() {
            let key = rest
                .trim_start()
                .strip_prefix("<arg_key>")
                .ok_or("GLM tool call requires an argument key")?;
            let (key, tail) = key
                .split_once("</arg_key>")
                .ok_or("unterminated GLM argument key")?;
            if key.is_empty() || arguments.contains_key(key) {
                return Err("GLM argument names must be nonempty and unique".into());
            }
            let value = tail
                .trim_start()
                .strip_prefix("<arg_value>")
                .ok_or("GLM argument requires a value tag")?;
            let (value, tail) = value
                .split_once("</arg_value>")
                .ok_or("unterminated GLM argument value")?;
            arguments.insert(
                key.to_owned(),
                parameter_value(value, properties.and_then(|p| p.get(key)))?,
            );
            rest = tail;
        }
        Ok(vec![(
            name.to_owned(),
            serde_json::to_string(&arguments).map_err(|error| {
                crate::FrontendError::Input(crate::Failure::caused_by(error.to_string(), error))
            })?,
        )])
    }
}

pub type GlmOutput = TaggedOutput<GlmArgumentTags>;

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
    let prefix = if generation
        || (continuation && messages.last().is_some_and(|m| m["role"] == "assistant"))
    {
        rendered
            .text
            .rsplit_once("<|assistant|>")
            .map(|(_, text)| text)
            .filter(|text| {
                !["<|user|>", "<|system|>", "<|observation|>"]
                    .iter()
                    .any(|marker| text.contains(marker))
            })
    } else {
        None
    };
    OutputConfig {
        verbatim: structured,
        reasoning: prefix.is_some_and(|text| {
            text.rfind("<think>")
                .is_some_and(|open| text.rfind("</think>").is_none_or(|close| open > close))
        }),
        tools: effective("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        output::{Input, ModelOutput, Termination},
        server::output::Accumulator,
    };
    use serde_json::json;

    fn config() -> OutputConfig {
        OutputConfig {
            verbatim: false,
            reasoning: true,
            tools: vec![json!({"type":"function", "function": {
            "name":"run", "parameters":{"type":"object", "properties":{
                "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
                "items":{"type":"array"}, "options":{"type":"object"}, "empty":{"type":"null"}
            }}}})],
        }
    }

    #[test]
    fn split_streams_preserve_reasoning_and_decode_typed_tool_arguments() {
        let text = "\nreason α\n</think>\nAnswer\n<tool_call>run<arg_key>text</arg_key><arg_value>\nhello α\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value><arg_key>options</arg_key><arg_value>{\"x\":1}</arg_value><arg_key>empty</arg_key><arg_value>null</arg_value></tool_call><tool_call>run</tool_call>\nend";
        for boundary in text
            .char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
        {
            let mut parser = GlmOutput::new(config());
            let mut output = Accumulator::new("test".into());
            output
                .apply(parser.push(Input::Text(&text[..boundary])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[boundary..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "\nreason α\n");
            assert_eq!(message["content"], "\nAnswer\n\nend");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0]["function"]["name"], "run");
            let arguments: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(
                arguments,
                json!({"text":"\nhello α\n", "count":2, "enabled":true, "items":[1,2], "options":{"x":1}, "empty":null})
            );
            assert_eq!(calls[1]["function"]["arguments"], "{}");
        }
    }

    #[test]
    fn malformed_and_unregistered_calls_fail_without_coercion() {
        for body in [
            "unknown",
            "run<arg_key></arg_key><arg_value>x</arg_value>",
            "run<arg_key>count</arg_key><arg_value>two</arg_value>",
            "run<arg_key>text</arg_key><arg_value>a</arg_value><arg_key>text</arg_key><arg_value>b</arg_value>",
            "run<arg_key>text</arg_key><arg_value>unfinished",
            "run<arg_key>x</arg_key>text",
        ] {
            assert!(
                GlmArgumentTags::parse(body, &config().tools).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn truncation_and_structured_output_keep_protocol_semantics() {
        let mut parser = GlmOutput::new(config());
        parser
            .push(Input::Text("</think><tool_call>run<arg_key>text</arg_key>"))
            .unwrap();
        assert!(parser.finish(Termination::OutputLimit).unwrap().is_empty());
        let mut parser = GlmOutput::new(OutputConfig {
            verbatim: true,
            ..config()
        });
        assert_eq!(
            parser
                .push(Input::Text("<think>{\"x\":1}</think>"))
                .unwrap(),
            vec![crate::output::Event::Content(
                "<think>{\"x\":1}</think>".into()
            )]
        );
    }
}
