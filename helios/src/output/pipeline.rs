//! Independent prepared syntax selection for streamed reasoning and tool calls.

use super::{
    Event, Input, ModelOutput, Termination,
    tagged::{OutputConfig as TaggedConfig, TaggedOutput},
};
use crate::{
    Result,
    model::contract::{FrontendContract, ToolParserKind},
};

#[derive(Clone)]
pub struct OutputConfig {
    pub syntax: FrontendContract,
    pub content: TaggedConfig,
    pub control_tokens: Vec<(u32, String)>,
}

pub enum OutputPipeline {
    FunctionTags(TaggedOutput<super::chatml::FunctionTags<true>>),
    CompactFunctionTags(TaggedOutput<super::chatml::FunctionTags<false>>),
    ArgumentTags(TaggedOutput<super::argument_tags::ArgumentTags>),
    Dsml(TaggedOutput<super::dsml::Dsml<false>>),
    SpacedDsml(TaggedOutput<super::dsml::Dsml<true>>),
    Xtml(super::xtml::XtmlOutput),
    Text(TaggedOutput<PlainText>),
}

pub struct PlainText;
impl super::tagged::ToolDialect for PlainText {
    const NAME: &'static str = "plain text";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "";
    const CLOSE: &'static str = "";
    fn parse(_: &str, _: &[serde_json::Value]) -> Result<Vec<(String, String)>> {
        Err("tool parsing is not configured".into())
    }
}

macro_rules! dispatch {
    ($self:expr, $parser:ident => $body:expr) => {
        match $self {
            OutputPipeline::FunctionTags($parser) => $body,
            OutputPipeline::CompactFunctionTags($parser) => $body,
            OutputPipeline::ArgumentTags($parser) => $body,
            OutputPipeline::Dsml($parser) => $body,
            OutputPipeline::SpacedDsml($parser) => $body,
            OutputPipeline::Xtml($parser) => $body,
            OutputPipeline::Text($parser) => $body,
        }
    };
}

impl OutputPipeline {
    pub fn new(config: OutputConfig) -> Self {
        let reasoning = config.syntax.reasoning.as_ref();
        match config
            .syntax
            .tools
            .as_ref()
            .map(|t| (t.kind, t.wrapping_newlines, t.tag_separator.as_str()))
        {
            Some((ToolParserKind::QwenCoder, 0, _)) => Self::CompactFunctionTags(
                TaggedOutput::new(config.content).with_reasoning(reasoning),
            ),
            Some((ToolParserKind::QwenCoder, _, _)) => {
                Self::FunctionTags(TaggedOutput::new(config.content).with_reasoning(reasoning))
            }
            Some((ToolParserKind::GlmArgumentTags, _, _)) => {
                Self::ArgumentTags(TaggedOutput::new(config.content).with_reasoning(reasoning))
            }
            Some((ToolParserKind::Dsml, _, " ")) => {
                Self::SpacedDsml(TaggedOutput::new(config.content).with_reasoning(reasoning))
            }
            Some((ToolParserKind::Dsml, _, _)) => {
                Self::Dsml(TaggedOutput::new(config.content).with_reasoning(reasoning))
            }
            Some((ToolParserKind::Xtml, _, _)) => Self::Xtml(super::xtml::XtmlOutput::new(
                config.content,
                config.control_tokens,
            )),
            None => Self::Text(TaggedOutput::new(config.content).with_reasoning(reasoning)),
        }
    }

    pub fn grammar_envelope(config: &OutputConfig) -> Option<crate::grammar::OutputEnvelope> {
        let mut envelope = match config
            .syntax
            .tools
            .as_ref()
            .map(|t| (t.kind, t.wrapping_newlines, t.tag_separator.as_str()))
        {
            Some((ToolParserKind::QwenCoder, 0, _)) => {
                TaggedOutput::<super::chatml::FunctionTags<false>>::grammar_envelope(
                    &config.content,
                )
            }
            Some((ToolParserKind::QwenCoder, _, _)) => {
                TaggedOutput::<super::chatml::FunctionTags<true>>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::GlmArgumentTags, _, _)) => {
                TaggedOutput::<super::argument_tags::ArgumentTags>::grammar_envelope(
                    &config.content,
                )
            }
            Some((ToolParserKind::Dsml, _, " ")) => {
                TaggedOutput::<super::dsml::Dsml<true>>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::Dsml, _, _)) => {
                TaggedOutput::<super::dsml::Dsml<false>>::grammar_envelope(&config.content)
            }
            Some((ToolParserKind::Xtml, _, _)) => Some(crate::grammar::OutputEnvelope {
                reasoning_end: config
                    .content
                    .start
                    .eq(&super::tagged::OutputStart::Reasoning)
                    .then_some("<|close|>think<|sep|><|open|>response<|sep|>"),
                tool: (!config.content.tools.is_empty())
                    .then_some((super::xtml::TOOLS_OPEN, super::xtml::TOOLS_CLOSE)),
                function: None,
                tool_ends_reasoning: false,
                response: Some(("", super::xtml::RESPONSE_CLOSE)),
                tool_body_start: false,
            }),
            None => Some(crate::grammar::OutputEnvelope {
                reasoning_end: (config.content.start == super::tagged::OutputStart::Reasoning)
                    .then_some("</think>"),
                tool: None,
                function: None,
                tool_ends_reasoning: false,
                response: None,
                tool_body_start: false,
            }),
        }?;
        envelope.tool_ends_reasoning = config.syntax.reasoning.as_ref().is_some_and(|r| {
            envelope
                .tool
                .is_some_and(|(open, _)| r.terminal_markers.iter().any(|m| m == open))
        });
        Some(envelope)
    }

    pub fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        dispatch!(self, parser => parser.push(input))
    }
    pub fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        dispatch!(self, parser => parser.finish(cause))
    }
}

impl ModelOutput for OutputPipeline {
    type Config = OutputConfig;
    fn new(config: Self::Config) -> Self {
        OutputPipeline::new(config)
    }
    fn grammar_envelope(config: &Self::Config) -> Option<crate::grammar::OutputEnvelope> {
        OutputPipeline::grammar_envelope(config)
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        OutputPipeline::push(self, input)
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        OutputPipeline::finish(self, cause)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::contract::{ReasoningParser, ReasoningParserKind, ToolParser};
    use crate::server::output::Accumulator;
    use serde_json::json;

    fn run(
        syntax: FrontendContract,
        text: &str,
        tools: Vec<serde_json::Value>,
        structured: bool,
    ) -> serde_json::Value {
        let mut parser = OutputPipeline::new(OutputConfig {
            syntax,
            control_tokens: vec![],
            content: TaggedConfig {
                structured,
                start: crate::output::tagged::OutputStart::Content,
                tools,
            },
        });
        let mut output = Accumulator::new("contract".into());
        for character in text.chars() {
            output
                .apply(parser.push(Input::Text(&character.to_string())).unwrap())
                .unwrap();
        }
        output
            .apply(parser.finish(Termination::Eos).unwrap())
            .unwrap();
        output.finish().unwrap();
        output.message()
    }

    #[test]
    fn absent_parsers_preserve_plain_and_structured_content() {
        for structured in [false, true] {
            let text = "{\"literal\":\"<think>content</think><tool_call>\"}";
            assert_eq!(
                run(
                    crate::model::testing::metadata().frontend,
                    text,
                    vec![],
                    structured
                )["content"],
                text
            );
        }
    }

    #[test]
    fn tool_parser_does_not_implicitly_enable_reasoning() {
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::QwenCoder,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        let tools = vec![json!({"type":"function","function":{"name":"lookup"}})];
        let result = run(
            syntax.clone(),
            "<think>literal</think><tool_call><function=lookup></function></tool_call>",
            tools,
            false,
        );
        assert_eq!(result["content"], "<think>literal</think>");
        assert!(result.get("reasoning_content").is_none());
        assert_eq!(result["tool_calls"][0]["function"]["name"], "lookup");
        syntax.tools = None;
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ThinkTags,
            open: "<think>".into(),
            close: "</think>".into(),
            wrapping_newlines: 0,
            terminal_markers: vec![],
        });
        let result = run(syntax, "<think>plan</think>answer", vec![], false);
        assert_eq!(result["reasoning_content"], "plan");
        assert_eq!(result["content"], "answer");
    }
}
