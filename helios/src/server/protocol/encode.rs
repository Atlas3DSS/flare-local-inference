use super::Protocol;
use crate::{
    Result,
    output::{Event, Termination},
    server::{
        inference::{Tool, ToolChoice},
        output::{Accumulator, Block, Change, ResponseOutput},
    },
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(super) fn thinking_signature(text: &str) -> String {
    format!("flare-thinking-v1:{:x}", Sha256::digest(text.as_bytes()))
}
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub event: Option<String>,
    pub data: Value,
    pub done: bool,
}
impl Frame {
    fn json(data: Value) -> Self {
        Self {
            event: None,
            data,
            done: false,
        }
    }
    fn named(name: &str, data: Value) -> Self {
        Self {
            event: Some(name.into()),
            data,
            done: false,
        }
    }
    pub fn sse(self) -> axum::response::sse::Event {
        let mut event = axum::response::sse::Event::default();
        if let Some(name) = self.event {
            event = event.event(name);
        }
        event.data(if self.done {
            "[DONE]".into()
        } else {
            self.data.to_string()
        })
    }
}
#[derive(Clone, Default)]
struct Usage {
    generated: u32,
    cached: u64,
}
#[derive(Clone, Copy, Default)]
pub(crate) struct PromptUsage {
    pub tokens: usize,
    pub image_tokens: usize,
}
pub(crate) struct Encoder {
    pub protocol: Protocol,
    pub id: String,
    pub model: String,
    pub created: u64,
    pub include_usage: bool,
    continuous_usage: bool,
    pub outputs: Vec<ResponseOutput>,
    chats: Vec<Accumulator>,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    tool_grammars: std::collections::HashMap<String, crate::grammar::TextGrammar>,
    prompts: Vec<PromptUsage>,
    choices_per_prompt: usize,
    usage: Vec<Usage>,
    reasons: Vec<Option<&'static str>>,
    stop: Option<String>,
    sequence: u64,
    metadata: Value,
    incomplete_item: Option<usize>,
}
impl Encoder {
    pub fn new(
        protocol: Protocol,
        id: String,
        model: String,
        created: u64,
        choices_per_prompt: usize,
        prompts: Vec<PromptUsage>,
        tools: Vec<Tool>,
        tool_choice: ToolChoice,
        parallel_tool_calls: bool,
        include_usage: bool,
        metadata: Value,
    ) -> Self {
        let choices = prompts.len() * choices_per_prompt;
        Self {
            protocol,
            chats: (0..choices)
                .map(|i| Accumulator::new(format!("{id}_{i}")))
                .collect(),
            outputs: (0..choices).map(|_| ResponseOutput::default()).collect(),
            usage: vec![Usage::default(); choices],
            reasons: vec![None; choices],
            id,
            model,
            created,
            tools,
            tool_choice,
            parallel_tool_calls,
            tool_grammars: Default::default(),
            prompts,
            choices_per_prompt,
            include_usage,
            continuous_usage: false,
            stop: None,
            sequence: 0,
            metadata,
            incomplete_item: None,
        }
    }
    pub fn set_tool_grammars(
        &mut self,
        grammars: std::collections::HashMap<String, crate::grammar::TextGrammar>,
    ) {
        self.tool_grammars = grammars;
    }
    pub fn set_continuous_usage(&mut self, enabled: bool) {
        self.continuous_usage = self.include_usage && enabled;
    }

    /// Count raw generated tokens before text decoding can merge or hide them.
    pub fn observe_generated_token(&mut self, index: usize) {
        self.usage[index].generated += 1;
    }
    fn named(&mut self, name: &str, mut data: Value) -> Frame {
        data["type"] = json!(name);
        if self.protocol == Protocol::Responses {
            data["sequence_number"] = json!(self.sequence);
            self.sequence += 1;
        }
        Frame::named(name, data)
    }
    fn chunk(&self, index: usize, delta: Value, finish: Value) -> Frame {
        let mut data = if self.protocol == Protocol::Completions {
            json!({"id":self.id,"object":"text_completion","created":self.created,"model":self.model,"choices":[{"index":index,"text":delta.as_str().unwrap_or(""),"logprobs":null,"finish_reason":finish}]})
        } else {
            json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[{"index":index,"delta":delta,"finish_reason":finish}]})
        };
        if self.include_usage {
            data["usage"] = if self.continuous_usage { self.usage_value() } else { Value::Null };
        }
        Frame::json(data)
    }
    pub fn start(&mut self) -> Vec<Frame> {
        match self.protocol {
            Protocol::Chat => vec![self.chunk(0, json!({"role":"assistant"}), Value::Null)],
            Protocol::Completions => vec![],
            Protocol::Messages => {
                let mut message = self.response();
                message["content"] = json!([]);
                message["stop_reason"] = Value::Null;
                message["usage"]["output_tokens"] = json!(0);
                vec![self.named("message_start", json!({"message":message}))]
            }
            Protocol::Responses => {
                let response = self.response();
                vec![
                    self.named("response.created", json!({"response":response})),
                    self.named("response.in_progress", json!({"response":self.response()})),
                ]
            }
        }
    }
    fn tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.native_name() == name)
    }
    fn item_id(&self, index: usize) -> String {
        format!("item_{}_{index}", self.id)
    }
    fn item(&self, index: usize, complete: bool) -> Value {
        let status = if self.incomplete_item == Some(index) {
            "incomplete"
        } else if complete {
            "completed"
        } else {
            "in_progress"
        };
        let id = self.item_id(index);
        match &self.outputs[0].blocks[index] {
            Block::Text(text) => {
                json!({"id":id,"type":"message","role":"assistant","status":status,"content":if complete{json!([{"type":"output_text","text":text,"annotations":[]}])}else{json!([])}})
            }
            Block::Reasoning(text) => {
                json!({"id":id,"type":"reasoning","status":status,"summary":[],"content":if complete{json!([{"type":"reasoning_text","text":text}])}else{json!([])}})
            }
            Block::Tool {
                id: call_id,
                name,
                arguments,
                ..
            } => {
                let tool = self.tool(name);
                if tool.is_some_and(|t| t.custom) {
                    let input = if complete {
                        serde_json::from_str::<Value>(arguments)
                            .ok()
                            .and_then(|v| v.get("input").cloned())
                            .unwrap_or(Value::Null)
                    } else {
                        json!("")
                    };
                    let mut item = json!({"id":id,"type":"custom_tool_call","status":status,"call_id":call_id,"name":tool.unwrap().name,"input":input});
                    if let Some(namespace) = tool.and_then(|t| t.namespace.as_ref()) {
                        item["namespace"] = json!(namespace);
                    }
                    item
                } else {
                    let mut item = json!({"id":id,"type":"function_call","status":status,"call_id":call_id,"name":tool.map(|t|t.name.as_str()).unwrap_or(name),"arguments":if complete{arguments.as_str()}else{""}});
                    if let Some(namespace) = tool.and_then(|t| t.namespace.as_ref()) {
                        item["namespace"] = json!(namespace);
                    }
                    item
                }
            }
        }
    }
    fn anthropic_block(&self, index: usize, complete: bool) -> Value {
        match &self.outputs[0].blocks[index] {
            Block::Text(text) => json!({"type":"text","text":if complete{text.as_str()}else{""}}),
            Block::Reasoning(text) => {
                json!({"type":"thinking","thinking":if complete{text.as_str()}else{""},"signature":if complete{thinking_signature(text)}else{String::new()}})
            }
            Block::Tool {
                id,
                name,
                arguments,
                ..
            } => {
                json!({"type":"tool_use","id":id,"name":name,"input":if complete{serde_json::from_str::<Value>(arguments).unwrap_or(json!({}))}else{json!({})}})
            }
        }
    }
    fn change(&mut self, change: Change) -> Result<Vec<Frame>> {
        let mut frames = vec![];
        match (self.protocol, change) {
            (Protocol::Messages, Change::Start(i)) => {
                let block = self.anthropic_block(i, false);
                frames.push(self.named(
                    "content_block_start",
                    json!({"index":i,"content_block":block}),
                ));
            }
            (Protocol::Messages, Change::Delta(i, text)) => {
                let delta = match &self.outputs[0].blocks[i] {
                    Block::Text(_) => json!({"type":"text_delta","text":text}),
                    Block::Reasoning(_) => json!({"type":"thinking_delta","thinking":text}),
                    Block::Tool { .. } => json!({"type":"input_json_delta","partial_json":text}),
                };
                frames.push(self.named("content_block_delta", json!({"index":i,"delta":delta})));
            }
            (Protocol::Messages, Change::End(i)) => {
                if let Block::Reasoning(text) = &self.outputs[0].blocks[i] {
                    frames.push(self.named("content_block_delta",json!({"index":i,"delta":{"type":"signature_delta","signature":thinking_signature(text)}})));
                }
                frames.push(self.named("content_block_stop", json!({"index":i})));
            }
            (Protocol::Responses, Change::Start(i)) => {
                frames.push(self.named(
                    "response.output_item.added",
                    json!({"output_index":i,"item":self.item(i,false)}),
                ));
                if matches!(
                    &self.outputs[0].blocks[i],
                    Block::Text(_) | Block::Reasoning(_)
                ) {
                    let kind = if matches!(&self.outputs[0].blocks[i], Block::Text(_)) {
                        "output_text"
                    } else {
                        "reasoning_text"
                    };
                    let part = if kind == "output_text" {
                        json!({"type":kind,"text":"","annotations":[]})
                    } else {
                        json!({"type":kind,"text":""})
                    };
                    frames.push(self.named("response.content_part.added",json!({"item_id":self.item_id(i),"output_index":i,"content_index":0,"part":part})));
                }
            }
            (Protocol::Responses, Change::Delta(i, text)) => {
                let name = match &self.outputs[0].blocks[i] {
                    Block::Text(_) => "response.output_text.delta",
                    Block::Reasoning(_) => "response.reasoning_text.delta",
                    Block::Tool { name, .. } => {
                        if self.tool(name).is_some_and(|t| t.custom) {
                            return Ok(frames);
                        }
                        "response.function_call_arguments.delta"
                    }
                };
                let mut data = json!({"item_id":self.item_id(i),"output_index":i,"delta":text});
                if !matches!(&self.outputs[0].blocks[i], Block::Tool { .. }) {
                    data["content_index"] = json!(0);
                }
                frames.push(self.named(name, data));
            }
            (Protocol::Responses, Change::End(i)) => {
                if let Block::Tool {
                    name, arguments, ..
                } = &self.outputs[0].blocks[i]
                {
                    if self.tool(name).is_some_and(|tool| tool.custom) {
                        let value: Value = serde_json::from_str(arguments)
                            .map_err(|e| format!("invalid custom tool arguments: {e}"))?;
                        if !value.get("input").is_some_and(Value::is_string) {
                            return Err("custom tool input must be text".into());
                        }
                        if let Some(grammar) = self.tool_grammars.get(name) {
                            grammar.validate(value["input"].as_str().unwrap())?;
                        }
                    }
                }
                let item = self.item(i, true);
                match &self.outputs[0].blocks[i] {
                    Block::Text(text) | Block::Reasoning(text) => {
                        let name = if matches!(&self.outputs[0].blocks[i], Block::Text(_)) {
                            "response.output_text.done"
                        } else {
                            "response.reasoning_text.done"
                        };
                        frames.push(self.named(name,json!({"item_id":self.item_id(i),"output_index":i,"content_index":0,"text":text})));
                        frames.push(self.named("response.content_part.done",json!({"item_id":self.item_id(i),"output_index":i,"content_index":0,"part":item["content"][0]})));
                    }
                    Block::Tool { .. } => {
                        if item["type"] == "custom_tool_call" {
                            if !item["input"].is_string() {
                                return Err("custom tool input must be text".into());
                            }
                            frames.push(self.named("response.custom_tool_call_input.delta",json!({"item_id":self.item_id(i),"output_index":i,"delta":item["input"]})));
                            frames.push(self.named("response.custom_tool_call_input.done",json!({"item_id":self.item_id(i),"output_index":i,"input":item["input"]})));
                        } else {
                            frames.push(self.named("response.function_call_arguments.done",json!({"item_id":self.item_id(i),"output_index":i,"arguments":item["arguments"]})));
                        }
                    }
                }
                frames.push(self.named(
                    "response.output_item.done",
                    json!({"output_index":i,"item":item}),
                ));
            }
            _ => {}
        }
        Ok(frames)
    }
    pub fn apply(&mut self, index: usize, events: Vec<Event>) -> Result<Vec<Frame>> {
        let mut frames = vec![];
        for event in events {
            if let Event::ToolBegin { name, .. } = &event {
                if !self.parallel_tool_calls && self.outputs[index].has_tools() {
                    return Err("parallel tool calls are disabled, but the model produced more than one tool call".into());
                }
                if (self.protocol != Protocol::Chat || self.tool_choice == ToolChoice::Required)
                    && self.tool(name).is_none()
                {
                    return Err(format!("model emitted undeclared tool {name}").into());
                }
            }
            if self.protocol == Protocol::Chat {
                for delta in self.chats[index].apply(vec![event.clone()])? {
                    frames.push(self.chunk(index, delta, Value::Null));
                }
            } else if self.protocol == Protocol::Completions {
                if let Event::Content(text) = &event {
                    frames.push(self.chunk(index, json!(text), Value::Null));
                }
            }
            for change in self.outputs[index].apply(event, &self.id)? {
                frames.extend(self.change(change)?);
            }
        }
        Ok(frames)
    }
    pub fn finish(
        &mut self,
        index: usize,
        cause: Termination,
        generated: u32,
        cached: u64,
        stop: Option<String>,
    ) -> Result<Vec<Frame>> {
        let mut frames = vec![];
        if self.protocol == Protocol::Chat {
            self.chats[index].finish()?;
        }
        for change in self.outputs[index].finish()? {
            if self.protocol == Protocol::Responses && cause == Termination::OutputLimit {
                if let Change::End(i) = change {
                    self.incomplete_item = Some(i);
                }
            }
            frames.extend(self.change(change)?);
        }
        if self.tool_choice == ToolChoice::Required && !self.outputs[index].has_tools() {
            return Err("required tool choice was not satisfied: the model did not produce a completed tool call".into());
        }
        let reason = if cause == Termination::OutputLimit {
            "length"
        } else if cause != Termination::UserStop && self.outputs[index].has_tools() {
            "tool_calls"
        } else {
            "stop"
        };
        self.reasons[index] = Some(reason);
        self.usage[index] = Usage { generated, cached };
        self.stop = stop;
        if matches!(self.protocol, Protocol::Chat | Protocol::Completions) {
            frames.push(self.chunk(
                index,
                if self.protocol == Protocol::Chat {
                    json!({})
                } else {
                    json!("")
                },
                json!(reason),
            ));
        }
        if self.reasons.iter().all(Option::is_some) {
            match self.protocol {
                Protocol::Chat | Protocol::Completions => {
                    if self.include_usage {
                        let mut frame = self.chunk(0, Value::Null, Value::Null);
                        frame.data["choices"] = json!([]);
                        frame.data["usage"] = self.usage_value();
                        frames.push(frame);
                    }
                    frames.push(Frame {
                        event: None,
                        data: Value::Null,
                        done: true,
                    });
                }
                Protocol::Messages => {
                    let response = self.response();
                    frames.push(self.named("message_delta",json!({"delta":{"stop_reason":response["stop_reason"],"stop_sequence":response["stop_sequence"]},"usage":self.usage_value()})));
                    frames.push(self.named("message_stop", json!({})));
                }
                Protocol::Responses => {
                    let response = self.response();
                    frames.push(self.named(
                        if reason == "length" {
                            "response.incomplete"
                        } else {
                            "response.completed"
                        },
                        json!({"response":response}),
                    ));
                }
            }
        }
        Ok(frames)
    }
    pub fn usage_value(&self) -> Value {
        let prompt_tokens: usize = self.prompts.iter().map(|p| p.tokens).sum();
        let image_tokens: usize = self.prompts.iter().map(|p| p.image_tokens).sum();
        let generated: u64 = self.usage.iter().map(|u| u.generated as u64).sum();
        // Prompt usage counts each input once even with n > 1. Use its first
        // choice's admission lookup, independent of completion order or reuse
        // by later siblings after that choice has published a checkpoint.
        let cached: u64 = self
            .usage
            .iter()
            .step_by(self.choices_per_prompt)
            .map(|u| u.cached)
            .sum();
        match self.protocol {
            Protocol::Chat | Protocol::Completions => {
                let mut details = json!({"cached_tokens":cached});
                if image_tokens > 0 {
                    details["multimodal_tokens"] = json!({"image":image_tokens});
                }
                json!({"prompt_tokens":prompt_tokens,"completion_tokens":generated,"total_tokens":prompt_tokens as u64+generated,"prompt_tokens_details":details})
            }
            Protocol::Messages => {
                json!({"input_tokens":prompt_tokens as u64-cached,"output_tokens":generated,"cache_creation_input_tokens":0,"cache_read_input_tokens":cached})
            }
            Protocol::Responses => {
                json!({"input_tokens":prompt_tokens,"output_tokens":generated,"total_tokens":prompt_tokens as u64+generated,"input_tokens_details":{"cached_tokens":cached}})
            }
        }
    }
    pub fn response(&self) -> Value {
        match self.protocol {
            Protocol::Chat => {
                json!({"id":self.id,"object":"chat.completion","created":self.created,"model":self.model,"choices":[{"index":0,"message":self.chats[0].message(),"finish_reason":self.reasons[0]}],"usage":self.usage_value()})
            }
            Protocol::Completions => {
                json!({"id":self.id,"object":"text_completion","created":self.created,"model":self.model,"choices":self.outputs.iter().enumerate().map(|(i,o)|json!({"index":i,"text":o.text(),"logprobs":null,"finish_reason":self.reasons[i]})).collect::<Vec<_>>(),"usage":self.usage_value()})
            }
            Protocol::Messages => {
                json!({"id":self.id,"type":"message","role":"assistant","model":self.model,"content":(0..self.outputs[0].blocks.len()).map(|i|self.anthropic_block(i,true)).collect::<Vec<_>>(),"stop_reason":self.reasons[0].map(|r|if r=="length"{"max_tokens"}else if r=="tool_calls"{"tool_use"}else if self.stop.is_some(){"stop_sequence"}else{"end_turn"}),"stop_sequence":self.stop,"usage":self.usage_value()})
            }
            Protocol::Responses => {
                let complete = self.reasons[0].is_some();
                let incomplete = self.reasons[0] == Some("length");
                json!({"id":self.id,"object":"response","created_at":self.created,"model":self.model,"status":if !complete{"in_progress"}else if incomplete{"incomplete"}else{"completed"},"error":null,"incomplete_details":if incomplete{json!({"reason":"max_output_tokens"})}else{Value::Null},"output":(0..self.outputs[0].blocks.len()).map(|i|self.item(i,complete)).collect::<Vec<_>>(),"usage":if complete{self.usage_value()}else{Value::Null},"store":false,"metadata":self.metadata,"parallel_tool_calls":self.parallel_tool_calls,"tool_choice":self.tool_choice,"tools":[],"text":{"format":{"type":"text"}}})
            }
        }
    }
    pub fn failure(&mut self, error: &crate::Error) -> Frame {
        let body = self.protocol.failure(error);
        match self.protocol {
            Protocol::Messages => Frame::named("error", body),
            Protocol::Responses => {
                let mut response = self.response();
                response["status"] = json!("failed");
                response["error"] =
                    json!({"code":body["error"]["code"],"message":body["error"]["message"]});
                self.named("response.failed", json!({"response":response}))
            }
            _ => Frame::json(body),
        }
    }
}
