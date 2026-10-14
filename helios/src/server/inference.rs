//! Protocol-neutral inputs lowered only at the model frontend boundary.

use crate::{AdmissionError, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Conversation(Vec<Message>),
    Raw(Vec<RawPrompt>),
}
#[derive(Clone, Debug, PartialEq)]
pub enum RawPrompt {
    Text(String),
    Tokens(Vec<u32>),
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
    pub phase: Option<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Text(String),
    Image(String),
    Reasoning(String),
    ToolCall {
        id: Option<String>,
        name: String,
        arguments: Value,
    },
    ToolResult {
        id: Option<String>,
        content: Vec<Part>,
        is_error: bool,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub namespace: Option<String>,
    pub description: String,
    pub parameters: Value,
    pub custom: bool,
    pub grammar: Option<ToolGrammar>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ToolGrammar {
    pub syntax: String,
    pub source: String,
}
impl Tool {
    pub fn native_name(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{ns}__{}", self.name),
            None => self.name.clone(),
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Sampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    Json(Value),
    Lark(String),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Effort {
    Low,
    High,
    Max,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    pub sampling: Sampling,
    pub effort: Option<Effort>,
    pub output_limit: Option<u64>,
    pub stops: Vec<String>,
    pub token_stops: Vec<Vec<u32>>,
    pub constraint: Option<Constraint>,
    pub ignore_eos: bool,
    pub template: Map<String, Value>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub input: Input,
    pub options: Options,
    pub tools: Vec<Tool>,
    pub tools_disabled: bool,
    pub choices: usize,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        let mut names = std::collections::HashSet::new();
        for tool in &self.tools {
            if tool.name.is_empty()
                || !tool.parameters.is_object()
                || !names.insert(tool.native_name())
            {
                return Err(
                    "tool names must be nonempty and unambiguous, with object schemas".into(),
                );
            }
        }
        if self.options.constraint.is_some() && !self.tools_disabled && !self.tools.is_empty() {
            return Err(unsupported(
                "structured output cannot be combined with tool-call generation",
            ));
        }
        Ok(())
    }
    pub fn generation_count(&self) -> Result<usize> {
        let prompts = match &self.input {
            Input::Raw(prompts) => prompts.len(),
            _ => 1,
        };
        prompts
            .checked_mul(self.choices)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                Error::from("prompt and choice counts must be positive and fit in memory")
            })
    }
}
pub fn unsupported(message: impl Into<String>) -> Error {
    Error::admission(AdmissionError::Feature, message.into())
}

fn content(parts: &[Part]) -> Result<Value> {
    let mut values = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => values.push(json!({"type":"text","text":text})),
            Part::Image(url) => values.push(json!({"type":"image_url","image_url":{"url":url}})),
            _ => return Err("tool result content must contain text or images".into()),
        }
    }
    if values.is_empty() {
        Ok(Value::String(String::new()))
    } else if values.len() == 1 && values[0]["type"] == "text" {
        Ok(values[0]["text"].clone())
    } else {
        Ok(Value::Array(values))
    }
}

/// Converts semantic parts to the message context consumed by prepared templates.
pub fn template_messages(messages: &[Message]) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let lower_instructions = messages
        .iter()
        .any(|message| message.role == Role::Developer);
    let instructions: Vec<_> = messages
        .iter()
        .filter(|m| lower_instructions && matches!(m.role, Role::System | Role::Developer))
        .collect();
    if !instructions.is_empty() {
        let labeled = instructions.len() > 1 || instructions[0].role == Role::Developer;
        let mut sections = Vec::new();
        for instruction in instructions {
            let mut text = String::new();
            for part in &instruction.parts {
                let Part::Text(value) = part else {
                    return Err("system and developer instructions require text".into());
                };
                text.push_str(value);
            }
            if labeled {
                let role = if instruction.role == Role::System {
                    "System"
                } else {
                    "Developer"
                };
                sections.push(format!("{role} instructions:\n{text}"));
            } else {
                sections.push(text);
            }
        }
        out.push(json!({"role":"system","content":sections.join("\n\n")}));
    }
    for message in messages {
        if lower_instructions && matches!(message.role, Role::System | Role::Developer) {
            continue;
        }
        let before = out.len();
        let mut current = json!({"role": message.role, "content": ""});
        let mut ordinary = Vec::new();
        let flush =
            |current: &mut Value, ordinary: &mut Vec<Part>, out: &mut Vec<Value>| -> Result<()> {
                current["content"] = content(ordinary)?;
                ordinary.clear();
                if current["content"] != ""
                    || current.get("reasoning_content").is_some()
                    || current.get("tool_calls").is_some()
                {
                    out.push(current.clone());
                }
                *current = json!({"role": message.role, "content":""});
                Ok(())
            };
        for part in &message.parts {
            match part {
                Part::Text(_) | Part::Image(_) => {
                    if current.get("tool_calls").is_some() {
                        flush(&mut current, &mut ordinary, &mut out)?;
                    }
                    ordinary.push(part.clone());
                }
                Part::Reasoning(text) => {
                    if !ordinary.is_empty() || current.get("tool_calls").is_some() {
                        flush(&mut current, &mut ordinary, &mut out)?;
                    }
                    let previous = current
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    current["reasoning_content"] = json!(format!("{previous}{text}"));
                }
                Part::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    if current.get("tool_calls").is_none() {
                        current["tool_calls"] = json!([]);
                    }
                    let mut call =
                        json!({"type":"function","function":{"name":name,"arguments":arguments}});
                    if let Some(id) = id {
                        call["id"] = json!(id);
                    }
                    current["tool_calls"].as_array_mut().unwrap().push(call);
                }
                Part::ToolResult {
                    id,
                    content: parts,
                    is_error,
                } => {
                    flush(&mut current, &mut ordinary, &mut out)?;
                    let mut parts = parts.clone();
                    if *is_error {
                        parts.insert(0, Part::Text("Tool execution failed:".into()));
                    }
                    let mut result =
                        json!({"role":"tool","content":content(&parts)?,"is_error":is_error});
                    if let Some(id) = id {
                        result["tool_call_id"] = json!(id);
                    }
                    out.push(result);
                }
            }
        }
        flush(&mut current, &mut ordinary, &mut out)?;
        if before == out.len() {
            out.push(json!({"role":message.role,"content":""}));
        }
    }
    Ok(out)
}
