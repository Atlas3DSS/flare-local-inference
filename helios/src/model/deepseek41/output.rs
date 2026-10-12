//! Streaming DeepSeek Markup Language calls and think-tag reasoning.

pub use crate::output::tagged::OutputConfig;
use crate::{
    Result,
    output::tagged::{TaggedOutput, ToolDialect},
};
use serde_json::{Map, Value};

pub struct Dsml;
pub type DeepSeekOutput = TaggedOutput<Dsml>;

impl ToolDialect for Dsml {
    const NAME: &'static str = "DeepSeek DSML";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<｜DSML｜ calls>";
    const CLOSE: &'static str = "</｜DSML｜ calls>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let mut rest = body.trim();
        let mut calls = Vec::new();
        while !rest.is_empty() {
            let invoke = rest
                .strip_prefix("<｜DSML｜ invoke name=\"")
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
                .split_once("</｜DSML｜ invoke>")
                .ok_or("unterminated DSML invocation")?;
            let mut arguments = Map::new();
            while !parameters.trim().is_empty() {
                let parameter = parameters
                    .trim_start()
                    .strip_prefix("<｜DSML｜ parameter name=\"")
                    .ok_or("DSML invocation requires parameter tags")?;
                let (key, parameter) = parameter
                    .split_once("\" string=\"")
                    .ok_or("DSML parameter requires a string attribute")?;
                let (string, parameter) = parameter
                    .split_once("\">")
                    .ok_or("unterminated DSML parameter attributes")?;
                let (raw, next) = parameter
                    .split_once("</｜DSML｜ parameter>")
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">東京</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter></｜DSML｜ invoke>\n</｜DSML｜ calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                verbatim: false,
                reasoning: true,
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
