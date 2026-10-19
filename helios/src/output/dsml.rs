use crate::Result;
use serde_json::{Map, Value};

pub(crate) fn parse(body: &str, tools: &[Value], separator: &str) -> Result<Vec<(String, String)>> {
    let mut rest = body.trim();
    let mut calls = Vec::new();
    while !rest.is_empty() {
        let invoke = rest
            .strip_prefix(&format!("<｜DSML｜{separator}invoke name=\""))
            .ok_or("DSML call requires an invoke tag")?;
        let (name, tail) = invoke
            .split_once("\">")
            .ok_or("unterminated DSML function name")?;
        if name.is_empty()
            || !tools
                .iter()
                .any(|tool| tool["function"]["name"].as_str() == Some(name))
        {
            return Err(format!("model called unregistered tool {name:?}").into());
        }
        let (mut parameters, tail) = tail
            .split_once(&format!("</｜DSML｜{separator}invoke>"))
            .ok_or("unterminated DSML invocation")?;
        let mut arguments = Map::new();
        while !parameters.trim().is_empty() {
            let parameter = parameters
                .trim_start()
                .strip_prefix(&format!("<｜DSML｜{separator}parameter name=\""))
                .ok_or("DSML invocation requires parameter tags")?;
            let (key, parameter) = parameter
                .split_once("\" string=\"")
                .ok_or("DSML parameter requires a string attribute")?;
            let (string, parameter) = parameter
                .split_once("\">")
                .ok_or("unterminated DSML parameter attributes")?;
            let (raw, next) = parameter
                .split_once(&format!("</｜DSML｜{separator}parameter>"))
                .ok_or("unterminated DSML parameter")?;
            if key.is_empty() || arguments.contains_key(key) {
                return Err("DSML parameter names must be nonempty and unique".into());
            }
            let value = match string {
                "true" => Value::String(raw.into()),
                "false" => serde_json::from_str(raw)
                    .map_err(|e| format!("invalid DSML JSON parameter: {e}"))?,
                _ => return Err("DSML string attribute must be true or false".into()),
            };
            arguments.insert(key.into(), value);
            parameters = next;
        }
        calls.push((name.into(), Value::Object(arguments).to_string()));
        rest = tail.trim_start();
    }
    if calls.is_empty() {
        return Err("DSML calls block is empty".into());
    }
    Ok(calls)
}

pub struct Dsml<const SPACED: bool>;
impl<const SPACED: bool> crate::output::tagged::ToolDialect for Dsml<SPACED> {
    const NAME: &'static str = "DSML";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = if SPACED {
        "<｜DSML｜ calls>"
    } else {
        "<｜DSML｜tool_calls>"
    };
    const CLOSE: &'static str = if SPACED {
        "</｜DSML｜ calls>"
    } else {
        "</｜DSML｜tool_calls>"
    };
    fn parse(body: &str, tools: &[serde_json::Value]) -> crate::Result<Vec<(String, String)>> {
        parse(body, tools, if SPACED { " " } else { "" })
    }
}

#[cfg(test)]
mod deepseek41_syntax_tests {
    use super::*;
    use crate::output::tagged::{OutputConfig, TaggedOutput};
    type DeepSeekOutput = TaggedOutput<Dsml<true>>;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">東京</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter></｜DSML｜ invoke>\n</｜DSML｜ calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "thought");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\":\"東京\"}"
            );
            assert_eq!(
                message["tool_calls"][1]["function"]["arguments"],
                "{\"n\":3}"
            );
        }
    }
}

#[cfg(test)]
mod deepseek4_syntax_tests {
    use super::*;
    use crate::output::tagged::{OutputConfig, TaggedOutput};
    type DeepSeekOutput = TaggedOutput<Dsml<false>>;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"city\" string=\"true\">東京</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"n\" string=\"false\">3</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "thought");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\":\"東京\"}"
            );
            assert_eq!(
                message["tool_calls"][1]["function"]["arguments"],
                "{\"n\":3}"
            );
        }
    }
}

#[cfg(test)]
mod prefixed_tool_tests {
    use super::*;
    use crate::output::tagged::{OutputConfig, OutputStart, TaggedOutput};
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    fn check<const SPACED: bool>() {
        let separator = if SPACED { " " } else { "" };
        let close = if SPACED {
            "</｜DSML｜ calls>"
        } else {
            "</｜DSML｜tool_calls>"
        };
        let text = format!(
            "\n<｜DSML｜{separator}invoke name=\"lookup\"><｜DSML｜{separator}parameter name=\"city\" string=\"true\">東京</｜DSML｜{separator}parameter></｜DSML｜{separator}invoke>{close}"
        );
        for split in text
            .char_indices()
            .map(|(offset, _)| offset)
            .chain([text.len()])
        {
            let mut parser = TaggedOutput::<Dsml<SPACED>>::new(OutputConfig {
                structured: true,
                start: OutputStart::ToolBody,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["tool_calls"][0]["function"]["name"], "lookup");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\":\"東京\"}"
            );
            assert!(message.get("reasoning_content").is_none_or(Value::is_null));
        }
    }

    #[test]
    fn compact_tool_body_accepts_every_utf8_split() {
        check::<false>();
    }

    #[test]
    fn spaced_tool_body_accepts_every_utf8_split() {
        check::<true>();
    }
}
