//! Incremental think/tool envelopes with statically selected model payload syntax.

use super::{Event, Input, ModelOutput, Progress, ReasoningParser, Termination, ToolCallParser};
use crate::Result;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
pub struct OutputConfig {
    pub verbatim: bool,
    pub reasoning: bool,
    pub tools: Vec<Value>,
}

pub trait ToolDialect {
    const NAME: &'static str;
    const TRIM_NEWLINES: bool;
    fn parse(body: &str, tools: &[Value]) -> Result<(String, String)>;
}

pub struct ThinkReasoning<D: ToolDialect> {
    dialect: std::marker::PhantomData<D>,
    buffer: String,
    trim_newlines: usize,
}
impl<D: ToolDialect> ReasoningParser for ThinkReasoning<D> {
    fn push(&mut self, input: Input<'_>) -> Result<Progress> {
        let Input::Text(text) = input else {
            return Ok(Progress::default());
        };
        self.buffer.push_str(text);
        while self.trim_newlines > 0 && !self.buffer.is_empty() {
            if self.buffer.starts_with('\n') {
                self.buffer.remove(0);
                self.trim_newlines -= 1;
            } else {
                self.trim_newlines = 0;
            }
        }
        let mut progress = Progress::default();
        if let Some(end) = self.buffer.find("</think>") {
            let text = &self.buffer[..end];
            let text = if D::TRIM_NEWLINES {
                text.strip_suffix('\n').unwrap_or(text)
            } else {
                text
            }
            .to_owned();
            if !text.is_empty() {
                progress.events.push(Event::Reasoning(text));
            }
            self.buffer.drain(..end + "</think>".len());
            progress.remainder = Some(std::mem::take(&mut self.buffer));
        } else {
            let hold = partial_marker(&self.buffer, &["</think>", "\n</think>"]);
            let visible = self.buffer.len() - hold;
            let text: String = self.buffer.drain(..visible).collect();
            if !text.is_empty() {
                progress.events.push(Event::Reasoning(text));
            }
        }
        Ok(progress)
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        let text = std::mem::take(&mut self.buffer);
        Ok(
            if text.is_empty()
                || matches!(cause, Termination::Cancelled | Termination::ExecutionError)
            {
                vec![]
            } else {
                vec![Event::Reasoning(text)]
            },
        )
    }
}

pub struct TaggedTools<D: ToolDialect> {
    dialect: std::marker::PhantomData<D>,
    buffer: String,
    tools: Arc<[Value]>,
    index: usize,
}
impl<D: ToolDialect> ToolCallParser for TaggedTools<D> {
    fn push(&mut self, input: Input<'_>) -> Result<Progress> {
        let Input::Text(text) = input else {
            return Ok(Progress::default());
        };
        self.buffer.push_str(text);
        if self.buffer.len() > 1024 * 1024 {
            return Err("tool call exceeds the 1 MiB parser bound".into());
        }
        let Some(end) = self.buffer.find("</tool_call>") else {
            return Ok(Progress::default());
        };
        let (name, arguments) = D::parse(&self.buffer[..end], &self.tools)?;
        self.buffer.drain(..end + "</tool_call>".len());
        Ok(Progress {
            events: vec![
                Event::ToolBegin {
                    index: self.index,
                    name,
                    id: None,
                },
                Event::ToolArguments {
                    index: self.index,
                    text: arguments,
                },
                Event::ToolEnd { index: self.index },
            ],
            remainder: Some(std::mem::take(&mut self.buffer)),
        })
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        if matches!(
            cause,
            Termination::Cancelled | Termination::ExecutionError | Termination::UserStop
        ) {
            self.buffer.clear();
            return Ok(vec![]);
        }
        Err(format!("incomplete {} tool call", D::NAME).into())
    }
}

enum Mode {
    Verbatim,
    Content,
    Reasoning,
    Tool,
}

pub struct TaggedOutput<D: ToolDialect> {
    mode: Mode,
    buffer: String,
    trim_newlines: usize,
    tools: Arc<[Value]>,
    reasoning: ThinkReasoning<D>,
    tool: Option<TaggedTools<D>>,
    calls: usize,
}
impl<D: ToolDialect> ModelOutput for TaggedOutput<D> {
    type Config = OutputConfig;
    type Reasoning = ThinkReasoning<D>;
    type Tools = TaggedTools<D>;
    fn new(config: OutputConfig) -> Self {
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
            tools: config.tools.into(),
            reasoning: ThinkReasoning {
                buffer: String::new(),
                trim_newlines: 0,
                dialect: std::marker::PhantomData,
            },
            tool: None,
            calls: 0,
        }
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        if let Input::Token(_) = input {
            return match self.mode {
                Mode::Reasoning => self.reasoning.push(input).map(|p| p.events),
                Mode::Tool => self.tool.as_mut().unwrap().push(input).map(|p| p.events),
                _ => Ok(vec![]),
            };
        }
        let Input::Text(text) = input else {
            unreachable!()
        };
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
                    if !text.is_empty() {
                        output.push(Event::Content(text));
                    }
                    return Ok(output);
                }
                Mode::Reasoning | Mode::Tool => {
                    let text = std::mem::take(&mut self.buffer);
                    let reasoning = matches!(self.mode, Mode::Reasoning);
                    let progress = if reasoning {
                        self.reasoning.push(Input::Text(&text))?
                    } else {
                        self.tool.as_mut().unwrap().push(Input::Text(&text))?
                    };
                    output.extend(progress.events);
                    let Some(remainder) = progress.remainder else {
                        return Ok(output);
                    };
                    self.buffer = remainder;
                    self.mode = Mode::Content;
                    if reasoning {
                        self.trim_newlines = if D::TRIM_NEWLINES { 2 } else { 0 };
                    } else {
                        self.calls += 1;
                        self.tool = None;
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
                        let prefix = if tool && D::TRIM_NEWLINES {
                            if self.calls == 0 {
                                prefix.strip_suffix("\n\n").unwrap_or(prefix)
                            } else {
                                prefix.strip_suffix('\n').unwrap_or(prefix)
                            }
                        } else {
                            prefix
                        };
                        if !prefix.is_empty() {
                            output.push(Event::Content(prefix.to_owned()));
                        }
                        self.buffer.drain(
                            ..at + if tool {
                                "<tool_call>".len()
                            } else {
                                "<think>".len()
                            },
                        );
                        self.mode = if tool { Mode::Tool } else { Mode::Reasoning };
                        if tool {
                            self.tool = Some(TaggedTools {
                                dialect: std::marker::PhantomData,
                                buffer: String::new(),
                                tools: self.tools.clone(),
                                index: self.calls,
                            });
                        } else {
                            self.reasoning = ThinkReasoning {
                                dialect: std::marker::PhantomData,
                                buffer: String::new(),
                                trim_newlines: u8::from(D::TRIM_NEWLINES) as usize,
                            };
                        }
                    } else {
                        let markers: &[&str] = if self.tools.is_empty() {
                            &["<think>"]
                        } else {
                            &["<think>", "<tool_call>", "\n\n<tool_call>", "\n<tool_call>"]
                        };
                        let hold = partial_marker(&self.buffer, markers);
                        let text: String = self.buffer.drain(..self.buffer.len() - hold).collect();
                        if !text.is_empty() {
                            output.push(Event::Content(text));
                        }
                        return Ok(output);
                    }
                }
            }
        }
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.buffer.clear();
            return match self.mode {
                Mode::Reasoning => self.reasoning.finish(cause),
                Mode::Tool => self.tool.as_mut().unwrap().finish(cause),
                _ => Ok(vec![]),
            };
        }
        let mut output = self.push(Input::Text(""))?;
        match self.mode {
            Mode::Reasoning => output.extend(self.reasoning.finish(cause)?),
            Mode::Tool => output.extend(self.tool.as_mut().unwrap().finish(cause)?),
            _ => {
                let text = std::mem::take(&mut self.buffer);
                if !text.is_empty() {
                    output.push(Event::Content(text));
                }
            }
        }
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

pub(crate) fn parameter_value(raw: &str, schema: Option<&Value>) -> Result<Value> {
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
