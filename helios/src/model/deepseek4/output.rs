//! Streaming DeepSeek Markup Language calls and think-tag reasoning.

pub use crate::output::tagged::OutputConfig;
use crate::{
    Result,
    output::tagged::{TaggedOutput, ToolDialect},
};
use serde_json::Value;

pub struct Dsml;
pub type DeepSeekOutput = TaggedOutput<Dsml>;

impl ToolDialect for Dsml {
    const NAME: &'static str = "DeepSeek DSML";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<｜DSML｜tool_calls>";
    const CLOSE: &'static str = "</｜DSML｜tool_calls>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        crate::output::dsml::parse(body, tools, "")
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
        let text = "thought</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"city\" string=\"true\">東京</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"n\" string=\"false\">3</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>";
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
